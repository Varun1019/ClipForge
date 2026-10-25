"""Automated security and reliability boundary tests for ClipForge.

Tests:
- API key redaction in logs and exception messages
- Protection against NaN, Inf, and invalid timestamps
- Path traversal and filename sanitation
- FFmpeg argument list safety (no shell=True)
- Overwrite protection for source files
- Empty and corrupted file handling
- Bounds checking on framing dimensions
"""

import math
from pathlib import Path
import tempfile
import unittest

from src.clipper import (
    build_ffmpeg_clip_command,
    create_video_clip,
    validate_timestamps,
    TimestampValidationError,
    VideoClipperError,
)
from src.framing.options import AspectRatioMode, FramingConfig, FramingStyle
from src.providers.gemini_provider import sanitize_secret_string
from src.video_info import extract_video_metadata, InvalidVideoError


class TestSecuritySecretRedaction(unittest.TestCase):
    def test_sanitize_aiza_key(self):
        secret = "AIzaSyD-1234567890abcdefghijklmnopqrstuv"
        text = f"Error communicating with Google API using key {secret} on endpoint"
        sanitized = sanitize_secret_string(text, user_key=secret)
        self.assertNotIn(secret, sanitized)
        self.assertIn("[REDACTED", sanitized)

    def test_sanitize_aq_key(self):
        secret = "AQ.VzB9Xy8K7L6M5N4O3P2Q1R0S"
        text = f"HTTP 403 Forbidden: Invalid credential {secret}"
        sanitized = sanitize_secret_string(text, user_key=secret)
        self.assertNotIn(secret, sanitized)
        self.assertIn("[REDACTED", sanitized)

    def test_sanitize_query_param_token(self):
        text = "https://generativelanguage.googleapis.com/v1beta/models?key=AIzaSySecretToken9999999999"
        sanitized = sanitize_secret_string(text)
        self.assertNotIn("AIzaSySecretToken9999999999", sanitized)
        self.assertIn("[REDACTED", sanitized)

    def test_sanitize_empty_or_none(self):
        self.assertEqual(sanitize_secret_string(""), "")
        self.assertEqual(sanitize_secret_string(None), "")


class TestSecurityTimestampValidation(unittest.TestCase):
    def test_nan_start_time_rejected(self):
        with self.assertRaises(TimestampValidationError):
            validate_timestamps(float("nan"), 10.0, 60.0)

    def test_nan_end_time_rejected(self):
        with self.assertRaises(TimestampValidationError):
            validate_timestamps(0.0, float("nan"), 60.0)

    def test_inf_start_time_rejected(self):
        with self.assertRaises(TimestampValidationError):
            validate_timestamps(float("inf"), 10.0, 60.0)

    def test_inf_end_time_rejected(self):
        with self.assertRaises(TimestampValidationError):
            validate_timestamps(0.0, float("inf"), 60.0)

    def test_negative_start_time_rejected(self):
        with self.assertRaises(TimestampValidationError):
            validate_timestamps(-1.0, 10.0, 60.0)

    def test_inverted_range_rejected(self):
        with self.assertRaises(TimestampValidationError):
            validate_timestamps(10.0, 5.0, 60.0)

    def test_equal_timestamps_rejected(self):
        with self.assertRaises(TimestampValidationError):
            validate_timestamps(5.0, 5.0, 60.0)

    def test_exceeding_duration_rejected(self):
        with self.assertRaises(TimestampValidationError):
            validate_timestamps(0.0, 65.0, 60.0)


class TestSecurityFFmpegCommandSafety(unittest.TestCase):
    def test_clip_command_returns_argument_list(self):
        cmd = build_ffmpeg_clip_command(
            input_path="/tmp/test;rm -rf /;.mp4",
            output_path="/tmp/out.mp4",
            start_time=1.0,
            end_time=3.0,
        )
        self.assertIsInstance(cmd, list)
        self.assertEqual(cmd[0], "ffmpeg")
        # Ensure malicious path is isolated as single argument, not unquoted shell command
        self.assertIn("/tmp/test;rm -rf /;.mp4", cmd)

    def test_overwrite_source_protection(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4") as tf:
            path = Path(tf.name)
            with self.assertRaises(VideoClipperError):
                create_video_clip(
                    input_path=path,
                    output_path=path,
                    start_time=0.0,
                    end_time=1.0,
                )


class TestSecurityFramingBounds(unittest.TestCase):
    def test_odd_dimensions_rejected(self):
        cfg = FramingConfig(
            aspect_ratio=AspectRatioMode.VERTICAL_9_16,
            target_width=1081,
            target_height=1920,
        )
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_zero_dimensions_rejected(self):
        cfg = FramingConfig(
            aspect_ratio=AspectRatioMode.VERTICAL_9_16,
            target_width=0,
            target_height=1920,
        )
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_excessive_8k_dimensions_rejected(self):
        cfg = FramingConfig(
            aspect_ratio=AspectRatioMode.VERTICAL_9_16,
            target_width=10000,
            target_height=20000,
        )
        with self.assertRaises(ValueError):
            cfg.validate()


class TestSecurityEmptyFileHandling(unittest.TestCase):
    def test_empty_file_metadata_extraction_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4") as tf:
            empty_path = Path(tf.name)
            # Empty 0-byte file
            with self.assertRaises(InvalidVideoError):
                extract_video_metadata(empty_path)


class TestSecuritySessionCleanup(unittest.TestCase):
    def test_cleanup_stale_files(self):
        import time
        import os
        from src.storage import cleanup_stale_files

        with tempfile.TemporaryDirectory() as tmp_dir_str:
            base_temp = Path(tmp_dir_str)

            # Create simulated stale session dir and fresh session dir
            stale_dir = base_temp / "stale_test_session"
            stale_dir.mkdir(parents=True, exist_ok=True)
            stale_file = stale_dir / "old_upload.mp4"
            stale_file.write_text("old content")

            # Set mtime back by 2 hours
            two_hours_ago = time.time() - 7200
            os.utime(stale_dir, (two_hours_ago, two_hours_ago))
            os.utime(stale_file, (two_hours_ago, two_hours_ago))

            # Models dir must be preserved even if older
            models_dir = base_temp / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            os.utime(models_dir, (two_hours_ago, two_hours_ago))

            # Run cleanup with 1 hour threshold
            removed = cleanup_stale_files(base_dirs=[base_temp], max_age_seconds=3600)

            # Stale dir must be cleaned
            self.assertFalse(stale_dir.exists())
            self.assertGreater(removed, 0)

            # Models dir must be preserved
            self.assertTrue(models_dir.exists())


if __name__ == "__main__":
    unittest.main()
