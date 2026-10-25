"""Unit tests for ClipForge transcription, device resolution, and caching logic."""

import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from src.transcriber import (
    check_audio_stream_exists,
    check_cuda_runtime_available,
    compute_file_hash,
    detect_compute_device,
    extract_audio_from_video,
    format_srt_timestamp,
    generate_srt,
    get_cache_path,
    load_cached_transcript,
    save_transcript_to_cache,
    transcribe_video,
    NoAudioStreamError,
    TranscriptionError,
)


class TestTranscriber(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.tmp_path = Path(self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_compute_file_hash(self):
        """Verify hash is deterministic and changes when content changes."""
        f1 = self.tmp_path / "file1.txt"
        f2 = self.tmp_path / "file2.txt"
        f1.write_text("clipforge test data")
        f2.write_text("clipforge test data")

        h1 = compute_file_hash(f1)
        h2 = compute_file_hash(f2)
        self.assertEqual(h1, h2)
        self.assertEqual(len(h1), 16)

        f2.write_text("different data")
        self.assertNotEqual(h1, compute_file_hash(f2))

    def test_detect_compute_device(self):
        """Device detector must return reliable CPU default when CUDA runtime is missing."""
        dev, comp, desc = detect_compute_device()
        self.assertIn(dev, ("cuda", "cpu"))
        self.assertIn(comp, ("float16", "int8", "float32"))
        self.assertIsInstance(desc, str)
        # Verify no exceptions or argument errors in description
        self.assertNotIn("StorageView", desc)
        self.assertNotIn("TypeError", desc)

    def test_check_audio_stream_exists(self):
        """Distinguish videos with and without audio streams."""
        with_audio = self.tmp_path / "with_audio.mp4"
        without_audio = self.tmp_path / "without_audio.mp4"

        # Generate video with audio
        subprocess.run([
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=1:size=160x120:rate=15",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
            "-c:v", "libx264", "-c:a", "aac",
            str(with_audio)
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Generate video without audio (-an)
        subprocess.run([
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=1:size=160x120:rate=15",
            "-an", "-c:v", "libx264",
            str(without_audio)
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        self.assertTrue(check_audio_stream_exists(with_audio))
        self.assertFalse(check_audio_stream_exists(without_audio))

    def test_extract_audio_from_video_no_audio_raises(self):
        """Attempting audio extraction on video without audio must raise NoAudioStreamError."""
        no_audio_vid = self.tmp_path / "mute.mp4"
        subprocess.run([
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=1:size=160x120:rate=15",
            "-an", "-c:v", "libx264",
            str(no_audio_vid)
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        with self.assertRaises(NoAudioStreamError):
            extract_audio_from_video(no_audio_vid)

        with self.assertRaises(NoAudioStreamError):
            transcribe_video(no_audio_vid, model_size="tiny", download_root=Path("temp/models"))

    def test_extract_audio_from_video_success(self):
        """Extract audio to 16kHz mono WAV from valid video."""
        video_path = self.tmp_path / "test_vid.mp4"
        subprocess.run([
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=2:size=320x240:rate=24",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
            "-c:v", "libx264", "-c:a", "aac",
            str(video_path),
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        wav_path = self.tmp_path / "audio.wav"
        extracted = extract_audio_from_video(video_path, wav_path)

        self.assertTrue(extracted.is_file())
        self.assertGreater(extracted.stat().st_size, 0)

    def test_srt_generation(self):
        """Verify SRT timestamp formatting and subtitle block output."""
        self.assertEqual(format_srt_timestamp(0.0), "00:00:00,000")
        self.assertEqual(format_srt_timestamp(65.125), "00:01:05,125")

        segments = [
            {"id": 0, "start": 1.0, "end": 4.5, "text": "Welcome to ClipForge"},
            {"id": 1, "start": 5.0, "end": 7.25, "text": "Next segment here"},
        ]
        srt = generate_srt(segments)
        self.assertIn("00:00:01,000 --> 00:00:04,500", srt)
        self.assertIn("Welcome to ClipForge", srt)
        self.assertIn("00:00:05,000 --> 00:00:07,250", srt)

    def test_transcript_caching(self):
        """Verify cache save, load, and invalidation."""
        cache_dir = self.tmp_path / "transcripts"
        dummy_data = {
            "source_file": "sample.mp4",
            "segments": [{"id": 0, "start": 0.0, "end": 2.0, "text": "Hello world"}],
            "full_text": "Hello world",
        }

        # Cache miss
        self.assertIsNone(load_cached_transcript("hash123", "tiny", cache_dir=cache_dir))

        # Save to cache
        saved_file = save_transcript_to_cache(dummy_data, "hash123", "tiny", cache_dir=cache_dir)
        self.assertTrue(saved_file.is_file())

        # Cache hit
        loaded = load_cached_transcript("hash123", "tiny", cache_dir=cache_dir)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["full_text"], "Hello world")
        self.assertTrue(loaded["from_cache"])

    def test_transcribe_video_pure_tone_empty_speech(self):
        """Tone audio contains no speech: returns status 'no_speech_detected' and segments=[]."""
        tone_vid = self.tmp_path / "tone.mp4"
        subprocess.run([
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=2:size=160x120:rate=25",
            "-f", "lavfi", "-i", "sine=frequency=1000:duration=2",
            "-c:v", "libx264", "-c:a", "aac",
            str(tone_vid),
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        res = transcribe_video(
            tone_vid,
            model_size="tiny",
            device="cpu",
            compute_type="int8",
            cache_dir=self.tmp_path / "transcripts",
            download_root=Path("temp/models"),
        )

        self.assertEqual(res["segments_count"], 0)
        self.assertEqual(res["status"], "no_speech_detected")
        self.assertEqual(res["segments"], [])


if __name__ == "__main__":
    unittest.main()
