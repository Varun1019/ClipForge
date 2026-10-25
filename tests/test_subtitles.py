"""Automated unit tests for ClipForge subtitles and transcription subsystem."""

from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

from src.subtitles.base import (
    SubtitleSegment,
    TranscriptionResult,
    WordTimestamp,
    format_timestamp_srt,
)
from src.subtitles.burner import (
    SubtitleBurnError,
    burn_subtitles_to_video,
    escape_ffmpeg_filter_path,
)
from src.subtitles.registry import transcription_registry
from src.subtitles.whisper_provider import FasterWhisperProvider


class TestSubtitles(unittest.TestCase):
    """Test suite for subtitle generation, timestamp math, and FFmpeg hardsubbing."""

    def test_format_timestamp_srt(self):
        """Verify precision conversion of floating seconds into HH:MM:SS,mmm."""
        self.assertEqual(format_timestamp_srt(0.0), "00:00:00,000")
        self.assertEqual(format_timestamp_srt(1.5), "00:00:01,500")
        self.assertEqual(format_timestamp_srt(65.123), "00:01:05,123")
        self.assertEqual(format_timestamp_srt(3661.050), "01:01:01,050")
        # Negative seconds clamped to 0
        self.assertEqual(format_timestamp_srt(-5.0), "00:00:00,000")

    def test_subtitle_segment_shift(self):
        """Verify shifting segment and word timestamps by offset."""
        seg = SubtitleSegment(
            id=1,
            start=2.0,
            end=5.0,
            text="Hello world",
            words=[
                WordTimestamp(word="Hello", start=2.0, end=3.0),
                WordTimestamp(word="world", start=3.1, end=5.0),
            ],
        )
        self.assertEqual(seg.duration, 3.0)

        # Shift by +10.0 seconds
        shifted = seg.shift(offset_secs=10.0)
        self.assertEqual(shifted.start, 12.0)
        self.assertEqual(shifted.end, 15.0)
        self.assertEqual(shifted.words[0].start, 12.0)
        self.assertEqual(shifted.words[0].end, 13.0)
        self.assertEqual(shifted.words[1].start, 13.1)
        self.assertEqual(shifted.words[1].end, 15.0)

    def test_transcription_result_to_srt(self):
        """Verify SRT formatting with blocks, indices, and timestamps."""
        seg1 = SubtitleSegment(id=1, start=1.0, end=3.5, text="First subtitle line")
        seg2 = SubtitleSegment(id=2, start=4.0, end=6.25, text="Second subtitle line")
        res = TranscriptionResult(segments=[seg1, seg2], language="en", duration=7.0)

        self.assertTrue(res.has_speech)
        srt = res.to_srt(offset_secs=0.0)
        self.assertIn("1\n00:00:01,000 --> 00:00:03,500\nFirst subtitle line", srt)
        self.assertIn("2\n00:00:04,000 --> 00:00:06,250\nSecond subtitle line", srt)

        # Test with offset
        srt_offset = res.to_srt(offset_secs=5.0)
        self.assertIn("1\n00:00:06,000 --> 00:00:08,500\nFirst subtitle line", srt_offset)

    def test_transcription_result_empty(self):
        """Verify handling of empty or whitespace-only transcription results."""
        empty_res = TranscriptionResult(segments=[])
        self.assertFalse(empty_res.has_speech)
        self.assertEqual(empty_res.to_srt(), "")

        whitespace_res = TranscriptionResult(
            segments=[SubtitleSegment(id=1, start=0.0, end=1.0, text="   ")]
        )
        self.assertFalse(whitespace_res.has_speech)
        self.assertEqual(whitespace_res.to_srt(), "")

    def test_escape_ffmpeg_filter_path(self):
        """Verify path escaping for FFmpeg filtergraph syntax."""
        p = Path("/home/user/path:with:colons/sub'file.srt")
        escaped = escape_ffmpeg_filter_path(p)
        self.assertIn(r"\:", escaped)
        self.assertIn(r"\'", escaped)

    @patch("subprocess.run")
    def test_burn_subtitles_to_video_success(self, mock_run):
        """Verify FFmpeg subtitle burning subprocess invocation."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_dir = Path(tmpdir)
            video_file = tmp_dir / "test_video.mp4"
            video_file.write_bytes(b"dummy_video_bytes")
            srt_file = tmp_dir / "test_subs.srt"
            srt_file.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8")
            out_file = tmp_dir / "test_out.mp4"

            def side_effect(cmd, **kwargs):
                out_file.write_bytes(b"subtitled_video_bytes")
                return MagicMock(returncode=0)

            mock_run.side_effect = side_effect

            out = burn_subtitles_to_video(
                video_path=video_file,
                srt_path=srt_file,
                output_path=out_file,
            )
            self.assertEqual(out, out_file)
            mock_run.assert_called_once()
            cmd = mock_run.call_args[0][0]
            self.assertEqual(cmd[0], "ffmpeg")
            self.assertIn("-vf", cmd)
            self.assertIn(str(out_file), cmd)

    @patch("subprocess.run")
    def test_burn_subtitles_to_video_ffmpeg_failure(self, mock_run):
        """Verify exception when FFmpeg subtitle burn process fails."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_dir = Path(tmpdir)
            video_file = tmp_dir / "test_video.mp4"
            video_file.write_bytes(b"dummy_video_bytes")
            srt_file = tmp_dir / "test_subs.srt"
            srt_file.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8")
            out_file = tmp_dir / "test_out.mp4"

            mock_run.return_value = MagicMock(returncode=1, stderr="Invalid font filter")

            with self.assertRaises(SubtitleBurnError) as ctx:
                burn_subtitles_to_video(
                    video_path=video_file,
                    srt_path=srt_file,
                    output_path=out_file,
                )
            self.assertIn("FFmpeg failed to burn subtitles", str(ctx.exception))

    @patch("src.subtitles.whisper_provider.FasterWhisperProvider._get_model")
    def test_faster_whisper_transcribe_mock(self, mock_get_model):
        """Test FasterWhisperProvider transcribing mock segments without downloading models."""
        provider = FasterWhisperProvider(model_size="tiny")

        # Mock WhisperModel transcribe output
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model

        mock_seg = MagicMock()
        mock_seg.start = 0.5
        mock_seg.end = 2.8
        mock_seg.text = "This is a test transcript"
        mock_seg.words = [
            MagicMock(word="This", start=0.5, end=0.8, probability=0.98),
            MagicMock(word="is", start=0.9, end=1.1, probability=0.99),
        ]

        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.duration = 3.0

        mock_model.transcribe.return_value = ([mock_seg], mock_info)

        with patch("pathlib.Path.is_file", return_value=True):
            res = provider.transcribe(Path("temp/dummy.mp4"))

        self.assertTrue(res.has_speech)
        self.assertEqual(len(res.segments), 1)
        self.assertEqual(res.segments[0].text, "This is a test transcript")
        self.assertEqual(len(res.segments[0].words), 2)
        self.assertEqual(res.segments[0].words[0].word, "This")

    def test_transcription_registry(self):
        """Verify transcription registry lists and retrieves faster-whisper."""
        provider = transcription_registry.get_provider("faster-whisper")
        self.assertIsNotNone(provider)
        self.assertEqual(provider.provider_id, "faster-whisper")


if __name__ == "__main__":
    unittest.main()
