"""Unit tests for ClipForge video clipping and timestamp validation."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from src.clipper import (
    batch_create_clips,
    build_ffmpeg_clip_command,
    create_video_clip,
    validate_timestamps,
    TimestampValidationError,
    VideoClipperError,
)
from src.video_info import extract_video_metadata


class TestClipper(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.tmp_path = Path(self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_timestamp_validation_valid(self):
        """Valid ranges should pass without exception."""
        # 0.0 to 10.0 in a 60.0s video
        validate_timestamps(0.0, 10.0, 60.0)
        # fractional seconds
        validate_timestamps(1.25, 3.75, 5.0)
        # up to max boundary
        validate_timestamps(0.0, 30.0, 30.0)

    def test_timestamp_validation_negative_start(self):
        """Negative start time must raise TimestampValidationError."""
        with self.assertRaises(TimestampValidationError) as ctx:
            validate_timestamps(-1.0, 5.0, 30.0)
        self.assertIn("cannot be negative", str(ctx.exception))

    def test_timestamp_validation_end_less_than_start(self):
        """End time <= start time must raise TimestampValidationError."""
        with self.assertRaises(TimestampValidationError) as ctx:
            validate_timestamps(10.0, 5.0, 30.0)
        self.assertIn("must be greater than start time", str(ctx.exception))

        # Equal start and end
        with self.assertRaises(TimestampValidationError):
            validate_timestamps(5.0, 5.0, 30.0)

    def test_timestamp_validation_end_exceeds_duration(self):
        """End time exceeding total video duration must raise TimestampValidationError."""
        with self.assertRaises(TimestampValidationError) as ctx:
            validate_timestamps(10.0, 35.0, 30.0)
        self.assertIn("exceeds source video duration", str(ctx.exception))

    def test_build_ffmpeg_clip_command(self):
        """Verify the constructed argument list is safe and formatted correctly."""
        cmd = build_ffmpeg_clip_command(
            input_path="/tmp/test_in.mp4",
            output_path="/tmp/test_out.mp4",
            start_time=2.5,
            end_time=8.75,
        )

        self.assertIsInstance(cmd, list)
        self.assertEqual(cmd[0], "ffmpeg")
        self.assertIn("-ss", cmd)
        self.assertIn("2.5", cmd)
        self.assertIn("-to", cmd)
        self.assertIn("8.75", cmd)
        self.assertIn("-i", cmd)
        self.assertIn("libx264", cmd)
        self.assertIn("aac", cmd)
        self.assertEqual(cmd[-1], str(Path("/tmp/test_out.mp4").resolve()))

    def test_create_video_clip_real_execution(self):
        """Generate a test video and verify real clipping execution."""
        source = self.tmp_path / "source.mp4"
        dest = self.tmp_path / "exports" / "test_clip.mp4"

        # Generate 4-second source video
        gen_cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=4:size=640x360:rate=25",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=4",
            "-c:v", "libx264", "-c:a", "aac",
            "-pix_fmt", "yuv420p",
            str(source)
        ]
        res = subprocess.run(gen_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertEqual(res.returncode, 0)
        source_original_size = source.stat().st_size

        # Create clip from 1.0s to 3.0s (expected duration ~2.0s)
        clipped_path = create_video_clip(
            input_path=source,
            output_path=dest,
            start_time=1.0,
            end_time=3.0,
            max_duration=4.0,
        )

        self.assertTrue(clipped_path.exists())
        self.assertGreater(clipped_path.stat().st_size, 0)
        # Original video must remain untouched
        self.assertEqual(source.stat().st_size, source_original_size)

        # Inspect generated clip metadata
        meta = extract_video_metadata(clipped_path)
        self.assertGreaterEqual(meta.duration_seconds, 1.9)
        self.assertLessEqual(meta.duration_seconds, 2.2)

    def test_create_video_clip_overwrite_protection(self):
        """Ensure original file is not accepted as output destination."""
        source = self.tmp_path / "source.mp4"
        source.write_bytes(b"dummy")

        with self.assertRaises(VideoClipperError):
            create_video_clip(source, source, start_time=0.0, end_time=2.0)

    def test_create_video_clip_missing_source(self):
        """Nonexistent source must raise FileNotFoundError."""
        missing = self.tmp_path / "missing.mp4"
        dest = self.tmp_path / "out.mp4"

        with self.assertRaises(FileNotFoundError):
            create_video_clip(missing, dest, start_time=0.0, end_time=2.0)

    def test_batch_create_clips_orchestration(self):
        """Test automatic batch clipping on multiple candidates."""
        source = self.tmp_path / "batch_source.mp4"
        gen_cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=5:size=320x240:rate=25",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=5",
            "-c:v", "libx264", "-c:a", "aac",
            str(source)
        ]
        res = subprocess.run(gen_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertEqual(res.returncode, 0)
        orig_size = source.stat().st_size

        # Test batch with 2 valid candidates and 1 out-of-bounds candidate
        candidates = [
            {"title": "Hook Story", "start_time": 0.5, "end_time": 2.5},
            {"title": "Punchline", "start_time": 2.5, "end_time": 4.5},
            {"title": "Invalid Bounds", "start_time": 8.0, "end_time": 12.0},
        ]

        batch_out_dir = self.tmp_path / "batch_exports"
        results = batch_create_clips(
            source_path=source,
            candidates=candidates,
            output_dir=batch_out_dir,
            max_duration=5.0,
        )

        self.assertEqual(len(results), 3)

        # Clip 1 success
        self.assertTrue(results[0]["success"])
        self.assertTrue(Path(results[0]["output_path"]).is_file())
        self.assertGreater(Path(results[0]["output_path"]).stat().st_size, 0)

        # Clip 2 success
        self.assertTrue(results[1]["success"])
        self.assertTrue(Path(results[1]["output_path"]).is_file())

        # Clip 3 failed gracefully
        self.assertFalse(results[2]["success"])
        self.assertIsNone(results[2]["output_path"])
        self.assertIn("exceeds", str(results[2]["error"]))

        # Original source remains untouched
        self.assertEqual(source.stat().st_size, orig_size)

    @patch("src.subtitles.burn_subtitles_to_video")
    def test_batch_create_clips_with_subtitles(self, mock_burn):
        """Verify batch clipping generates SRT and calls burn_subtitles_to_video when requested."""
        source = self.tmp_path / "batch_sub_src.mp4"
        gen_cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=3:size=320x240:rate=25",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
            "-c:v", "libx264", "-c:a", "aac",
            str(source),
        ]
        res = subprocess.run(gen_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertEqual(res.returncode, 0)

        mock_transcriber = MagicMock()
        mock_result = MagicMock()
        mock_result.has_speech = True
        mock_result.to_srt.return_value = "1\n00:00:00,000 --> 00:00:01,000\nHello\n"
        mock_result.segments = [MagicMock(text="Hello")]
        mock_transcriber.is_available.return_value = True
        mock_transcriber.transcribe.return_value = mock_result

        batch_out_dir = self.tmp_path / "batch_sub_exports"
        candidates = [{"title": "Subtitled Clip", "start_time": 0.5, "end_time": 2.0}]

        results = batch_create_clips(
            source_path=source,
            candidates=candidates,
            output_dir=batch_out_dir,
            max_duration=3.0,
            generate_subtitles=True,
            burn_subtitles=True,
            transcription_provider=mock_transcriber,
        )

        self.assertEqual(len(results), 1)
        self.assertTrue(results[0]["success"])
        self.assertTrue(results[0]["has_speech"])
        self.assertIsNotNone(results[0]["srt_path"])
        self.assertTrue(Path(results[0]["srt_path"]).is_file())
        self.assertIn("Hello", Path(results[0]["srt_path"]).read_text())
        mock_burn.assert_called_once()


if __name__ == "__main__":
    unittest.main()
