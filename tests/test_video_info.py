"""Unit tests for ClipForge video metadata extraction using standard unittest."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from src.video_info import (
    check_ffprobe_available,
    extract_video_metadata,
    format_duration,
    format_file_size,
    InvalidVideoError,
)


class TestVideoInfo(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.tmp_path = Path(self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_ffprobe_availability(self):
        """Verify that ffprobe is detectable on the system."""
        self.assertTrue(check_ffprobe_available())

    def test_format_helpers(self):
        """Test duration and file size formatting."""
        self.assertEqual(format_file_size(500), "500 B")
        self.assertEqual(format_file_size(1024 * 500), "500.00 KB")
        self.assertEqual(format_file_size(1024 * 1024 * 25), "25.00 MB")
        self.assertEqual(format_file_size(1024 * 1024 * 1024 * 2), "2.00 GB")

        self.assertEqual(format_duration(0), "00:00")
        self.assertEqual(format_duration(45), "00:45")
        self.assertEqual(format_duration(125), "02:05")
        self.assertEqual(format_duration(3665), "01:01:05")

    def test_extract_video_metadata_valid(self):
        """Generate a test MP4 using ffmpeg and verify extracted metadata."""
        video_path = self.tmp_path / "valid_clip.mp4"
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=3:size=1920x1080:rate=24",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
            "-c:v", "libx264", "-c:a", "aac",
            "-pix_fmt", "yuv420p",
            str(video_path)
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertEqual(res.returncode, 0)
        self.assertTrue(video_path.exists())

        meta = extract_video_metadata(video_path, original_filename="MyPodcast.mp4")

        self.assertEqual(meta.filename, "MyPodcast.mp4")
        self.assertEqual(meta.width, 1920)
        self.assertEqual(meta.height, 1080)
        self.assertEqual(meta.resolution_str, "1920 × 1080")
        self.assertGreater(meta.file_size_bytes, 0)
        self.assertGreaterEqual(meta.duration_seconds, 2.9)
        self.assertIn(meta.duration_formatted, ("00:03", "00:02"))
        self.assertEqual(meta.codec, "h264")

    def test_extract_video_metadata_corrupted(self):
        """Verify that invalid/corrupted files raise InvalidVideoError."""
        corrupt_path = self.tmp_path / "corrupt.mp4"
        corrupt_path.write_bytes(b"not a valid mp4 header data garbage 123456789")

        with self.assertRaises(InvalidVideoError):
            extract_video_metadata(corrupt_path)

    def test_extract_video_metadata_missing_file(self):
        """Verify that a nonexistent path raises FileNotFoundError."""
        missing_path = self.tmp_path / "nonexistent.mp4"
        with self.assertRaises(FileNotFoundError):
            extract_video_metadata(missing_path)


if __name__ == "__main__":
    unittest.main()
