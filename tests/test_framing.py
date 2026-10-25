"""Automated unit and integration tests for ClipForge video reframing."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from src.clipper import batch_create_clips
from src.framing.options import (
    AspectRatioMode,
    FramingConfig,
    FramingStyle,
    SQUARE_RESOLUTIONS,
    VERTICAL_RESOLUTIONS,
)
from src.framing.reframe import (
    FramingError,
    build_reframe_filter,
    reframe_video,
)


class TestFramingOptions(unittest.TestCase):
    def test_aspect_ratio_modes(self):
        self.assertEqual(AspectRatioMode.ORIGINAL.value, "original")
        self.assertEqual(AspectRatioMode.VERTICAL_9_16.value, "9:16")
        self.assertEqual(AspectRatioMode.SQUARE_1_1.value, "1:1")
        self.assertIn("Vertical", AspectRatioMode.VERTICAL_9_16.label())
        self.assertIn("Square", AspectRatioMode.SQUARE_1_1.label())

    def test_framing_styles(self):
        self.assertEqual(FramingStyle.BLURRED_BACKGROUND.value, "blur")
        self.assertEqual(FramingStyle.CENTER_CROP.value, "crop")
        self.assertIn("Blurred", FramingStyle.BLURRED_BACKGROUND.label())
        self.assertIn("Center", FramingStyle.CENTER_CROP.label())

    def test_framing_config_defaults(self):
        cfg = FramingConfig()
        self.assertEqual(cfg.aspect_ratio, AspectRatioMode.ORIGINAL)
        self.assertFalse(cfg.is_reframe_needed())
        self.assertEqual(cfg.get_suffix(), "")
        # No validation error on original
        cfg.validate()

    def test_framing_config_suffixes(self):
        v_cfg = FramingConfig(aspect_ratio=AspectRatioMode.VERTICAL_9_16, target_width=1080, target_height=1920)
        self.assertTrue(v_cfg.is_reframe_needed())
        self.assertEqual(v_cfg.get_suffix(), "_vertical")

        s_cfg = FramingConfig(aspect_ratio=AspectRatioMode.SQUARE_1_1, target_width=1080, target_height=1080)
        self.assertTrue(s_cfg.is_reframe_needed())
        self.assertEqual(s_cfg.get_suffix(), "_square")

    def test_framing_config_validation(self):
        # Odd dimensions must fail
        with self.assertRaises(ValueError):
            FramingConfig(
                aspect_ratio=AspectRatioMode.VERTICAL_9_16,
                target_width=1081,
                target_height=1920,
            ).validate()

        with self.assertRaises(ValueError):
            FramingConfig(
                aspect_ratio=AspectRatioMode.VERTICAL_9_16,
                target_width=1080,
                target_height=1921,
            ).validate()

        # Non-positive dimensions must fail
        with self.assertRaises(ValueError):
            FramingConfig(
                aspect_ratio=AspectRatioMode.VERTICAL_9_16,
                target_width=0,
                target_height=1920,
            ).validate()


class TestFramingFilterBuilder(unittest.TestCase):
    def test_center_crop_filter(self):
        cfg = FramingConfig(
            aspect_ratio=AspectRatioMode.VERTICAL_9_16,
            style=FramingStyle.CENTER_CROP,
            target_width=720,
            target_height=1280,
        )
        filter_args, is_complex = build_reframe_filter(cfg)
        self.assertFalse(is_complex)
        self.assertEqual(filter_args[0], "-vf")
        vf_str = filter_args[1]
        self.assertIn("scale=720:1280:force_original_aspect_ratio=increase", vf_str)
        self.assertIn("crop=720:1280", vf_str)

    def test_blurred_background_filter(self):
        cfg = FramingConfig(
            aspect_ratio=AspectRatioMode.VERTICAL_9_16,
            style=FramingStyle.BLURRED_BACKGROUND,
            target_width=1080,
            target_height=1920,
        )
        filter_args, is_complex = build_reframe_filter(cfg)
        self.assertTrue(is_complex)
        self.assertEqual(filter_args[0], "-filter_complex")
        filter_str = filter_args[1]
        self.assertIn("split=2[bg][fg]", filter_str)
        self.assertIn("boxblur=", filter_str)
        self.assertIn("[bg_blur]", filter_str)
        self.assertIn("scale=1080:1920", filter_str)
        self.assertIn("scale=trunc(iw/2)*2:trunc(ih/2)*2[fg_fit]", filter_str)
        self.assertIn("overlay=(W-w)/2:(H-h)/2", filter_str)
        self.assertIn("-map", filter_args)
        self.assertIn("[outv]", filter_args)

    def test_filter_with_subtitles(self):
        with tempfile.NamedTemporaryFile(suffix=".srt", delete=False) as tf:
            tf.write(b"1\n00:00:00,000 --> 00:00:01,000\nHello\n")
            srt_path = Path(tf.name)

        try:
            cfg = FramingConfig(
                aspect_ratio=AspectRatioMode.VERTICAL_9_16,
                style=FramingStyle.BLURRED_BACKGROUND,
                target_width=720,
                target_height=1280,
            )
            filter_args, _ = build_reframe_filter(cfg, srt_path=srt_path)
            filter_str = filter_args[1]
            self.assertIn("subtitles=", filter_str)
        finally:
            if srt_path.is_file():
                srt_path.unlink()


class TestFramingExecution(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ffmpeg_bin = shutil.which("ffmpeg")
        cls.ffprobe_bin = shutil.which("ffprobe")
        if not cls.ffmpeg_bin or not cls.ffprobe_bin:
            raise unittest.SkipTest("ffmpeg or ffprobe not available on system")

    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.test_dir.name)
        self.input_mp4 = self.dir_path / "source_16_9.mp4"

        # Generate a small 640x360 2-second video with audio tone
        cmd = [
            self.ffmpeg_bin,
            "-y",
            "-f", "lavfi", "-i", "testsrc=duration=2:size=640x360:rate=24",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            str(self.input_mp4),
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    def tearDown(self):
        self.test_dir.cleanup()

    def _get_video_dimensions(self, path: Path) -> tuple[int, int]:
        cmd = [
            self.ffprobe_bin,
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=s=x:p=0",
            str(path),
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        w, h = res.stdout.strip().split("x")
        return int(w), int(h)

    def test_reframe_vertical_blur_720x1280(self):
        out_path = self.dir_path / "out_vert_blur.mp4"
        cfg = FramingConfig(
            aspect_ratio=AspectRatioMode.VERTICAL_9_16,
            style=FramingStyle.BLURRED_BACKGROUND,
            target_width=720,
            target_height=1280,
        )
        res = reframe_video(self.input_mp4, out_path, cfg)
        self.assertTrue(res.is_file())
        self.assertGreater(res.stat().st_size, 0)
        w, h = self._get_video_dimensions(res)
        self.assertEqual((w, h), (720, 1280))

    def test_reframe_vertical_crop_720x1280(self):
        out_path = self.dir_path / "out_vert_crop.mp4"
        cfg = FramingConfig(
            aspect_ratio=AspectRatioMode.VERTICAL_9_16,
            style=FramingStyle.CENTER_CROP,
            target_width=720,
            target_height=1280,
        )
        res = reframe_video(self.input_mp4, out_path, cfg)
        self.assertTrue(res.is_file())
        w, h = self._get_video_dimensions(res)
        self.assertEqual((w, h), (720, 1280))

    def test_reframe_square_blur_720x720(self):
        out_path = self.dir_path / "out_square_blur.mp4"
        cfg = FramingConfig(
            aspect_ratio=AspectRatioMode.SQUARE_1_1,
            style=FramingStyle.BLURRED_BACKGROUND,
            target_width=720,
            target_height=720,
        )
        res = reframe_video(self.input_mp4, out_path, cfg)
        self.assertTrue(res.is_file())
        w, h = self._get_video_dimensions(res)
        self.assertEqual((w, h), (720, 720))

    def test_reframe_not_needed_returns_input(self):
        cfg = FramingConfig(aspect_ratio=AspectRatioMode.ORIGINAL)
        res = reframe_video(self.input_mp4, self.dir_path / "not_used.mp4", cfg)
        self.assertEqual(res, self.input_mp4)

    def test_reframe_missing_file_raises(self):
        cfg = FramingConfig(aspect_ratio=AspectRatioMode.VERTICAL_9_16)
        with self.assertRaises(FileNotFoundError):
            reframe_video(self.dir_path / "non_existent.mp4", self.dir_path / "out.mp4", cfg)

    def test_batch_create_clips_with_reframing(self):
        cfg = FramingConfig(
            aspect_ratio=AspectRatioMode.VERTICAL_9_16,
            style=FramingStyle.BLURRED_BACKGROUND,
            target_width=720,
            target_height=1280,
        )
        candidates = [
            {"title": "Test Highlight", "start_time": 0.0, "end_time": 1.5}
        ]
        out_dir = self.dir_path / "batch_out"
        results = batch_create_clips(
            source_path=self.input_mp4,
            candidates=candidates,
            output_dir=out_dir,
            framing_config=cfg,
        )
        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertTrue(r["success"])
        self.assertIsNotNone(r["output_path"])
        self.assertTrue(Path(r["output_path"]).is_file())
        self.assertIsNotNone(r["reframed_path"])
        self.assertTrue(Path(r["reframed_path"]).is_file())

        # Check clean original dimensions vs reframed dimensions
        orig_w, orig_h = self._get_video_dimensions(r["output_path"])
        self.assertEqual((orig_w, orig_h), (640, 360))

        reframed_w, reframed_h = self._get_video_dimensions(r["reframed_path"])
        self.assertEqual((reframed_w, reframed_h), (720, 1280))


if __name__ == "__main__":
    unittest.main()
