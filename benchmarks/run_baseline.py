"""Reproducible baseline performance benchmark for ClipForge local processing.

Measures:
- Metadata extraction time (single vs repeated)
- Sub-clipping time (FFmpeg)
- Reframing time (Vertical 9:16 Blurred vs Center Crop)
- Subtitle burn-in time
- Faster-Whisper transcription time
- End-to-end batch clipping pipeline time
- Peak RAM usage (RUSAGE_SELF)
- Output file sizes and properties
"""

import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import tempfile
import time

# Ensure project root is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.clipper import batch_create_clips, create_video_clip
from src.framing.options import AspectRatioMode, FramingConfig, FramingStyle
from src.framing.reframe import reframe_video
from src.subtitles.burner import burn_subtitles_to_video
from src.subtitles.whisper_provider import FasterWhisperProvider
from src.video_info import extract_video_metadata


def get_peak_ram_mb() -> float:
    # ru_maxrss is in KB on Linux
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_maxrss / 1024.0


def create_test_media(target_dir: Path, duration_secs: int = 10) -> Path:
    target_path = target_dir / "benchmark_source.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-f", "lavfi", "-i", f"testsrc=duration={duration_secs}:size=1280x720:rate=30",
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration_secs}",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        str(target_path),
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return target_path


def run_benchmarks():
    print("=" * 60)
    print("  CLIPFORGE LOCAL PERFORMANCE BENCHMARK (BASELINE)")
    print("=" * 60)
    start_ram = get_peak_ram_mb()
    print(f"Initial Peak RAM: {start_ram:.2f} MB")

    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        source_video = create_test_media(tmp_dir, duration_secs=10)
        source_size_kb = source_video.stat().st_size / 1024.0
        print(f"Generated test source: 1280x720, 10s, {source_size_kb:.1f} KB\n")

        # 1. Metadata Extraction Benchmark
        t0 = time.perf_counter()
        meta = extract_video_metadata(source_video)
        t_meta_first = (time.perf_counter() - t0) * 1000.0

        # Repeated 5 times (simulating Streamlit reruns)
        t0 = time.perf_counter()
        for _ in range(5):
            _ = extract_video_metadata(source_video)
        t_meta_repeat_avg = ((time.perf_counter() - t0) / 5.0) * 1000.0

        print(f"1. Metadata Extraction:")
        print(f"   - First call:           {t_meta_first:.2f} ms")
        print(f"   - Repeated call (avg):   {t_meta_repeat_avg:.2f} ms")
        print(f"   - Codec: {meta.codec}, Duration: {meta.duration_formatted}")

        # 2. FFmpeg Sub-Clipping Benchmark
        clip_out = tmp_dir / "bench_clip_3s.mp4"
        t0 = time.perf_counter()
        create_video_clip(source_video, clip_out, start_time=1.0, end_time=4.0)
        t_clip = (time.perf_counter() - t0) * 1000.0
        clip_size_kb = clip_out.stat().st_size / 1024.0
        print(f"\n2. Sub-Clipping (1.0s -> 4.0s = 3.0s duration):")
        print(f"   - Execution time:       {t_clip:.2f} ms")
        print(f"   - Output clip size:     {clip_size_kb:.1f} KB")

        # 3. Reframing: Center Crop (720x1280 9:16)
        crop_out = tmp_dir / "bench_crop_9_16.mp4"
        cfg_crop = FramingConfig(
            aspect_ratio=AspectRatioMode.VERTICAL_9_16,
            style=FramingStyle.CENTER_CROP,
            target_width=720,
            target_height=1280,
        )
        t0 = time.perf_counter()
        reframe_video(clip_out, crop_out, cfg_crop)
        t_reframe_crop = (time.perf_counter() - t0) * 1000.0
        crop_size_kb = crop_out.stat().st_size / 1024.0
        print(f"\n3. Reframing (9:16 Center Crop 720x1280):")
        print(f"   - Execution time:       {t_reframe_crop:.2f} ms")
        print(f"   - Output size:          {crop_size_kb:.1f} KB")

        # 4. Reframing: Blurred Background (720x1280 9:16)
        blur_out = tmp_dir / "bench_blur_9_16.mp4"
        cfg_blur = FramingConfig(
            aspect_ratio=AspectRatioMode.VERTICAL_9_16,
            style=FramingStyle.BLURRED_BACKGROUND,
            target_width=720,
            target_height=1280,
        )
        t0 = time.perf_counter()
        reframe_video(clip_out, blur_out, cfg_blur)
        t_reframe_blur = (time.perf_counter() - t0) * 1000.0
        blur_size_kb = blur_out.stat().st_size / 1024.0
        print(f"\n4. Reframing (9:16 Blurred Background 720x1280):")
        print(f"   - Execution time:       {t_reframe_blur:.2f} ms")
        print(f"   - Output size:          {blur_size_kb:.1f} KB")

        # 5. Subtitle Burn-in
        srt_file = tmp_dir / "bench.srt"
        srt_file.write_text("1\n00:00:00,000 --> 00:00:02,500\nClipForge Benchmark Subtitle\n")
        sub_out = tmp_dir / "bench_subtitled.mp4"
        t0 = time.perf_counter()
        burn_subtitles_to_video(clip_out, srt_file, sub_out)
        t_burn = (time.perf_counter() - t0) * 1000.0
        sub_size_kb = sub_out.stat().st_size / 1024.0
        print(f"\n5. Subtitle Burning (Hardsub on 3s clip):")
        print(f"   - Execution time:       {t_burn:.2f} ms")
        print(f"   - Output size:          {sub_size_kb:.1f} KB")

        # 6. Faster-Whisper ASR
        transcriber = FasterWhisperProvider(model_size="tiny")
        t0 = time.perf_counter()
        res = transcriber.transcribe(clip_out)
        t_asr = (time.perf_counter() - t0) * 1000.0
        print(f"\n6. Faster-Whisper Transcription (3s clip):")
        print(f"   - Execution time:       {t_asr:.2f} ms")
        print(f"   - Detected speech:      {res.has_speech}")

        # 7. End-to-End Batch Pipeline (1 clip with vertical blur framing + subtitles)
        candidates = [{"title": "Benchmark Candidate", "start_time": 2.0, "end_time": 5.0}]
        batch_out_dir = tmp_dir / "batch_exports"
        t0 = time.perf_counter()
        batch_results = batch_create_clips(
            source_path=source_video,
            candidates=candidates,
            output_dir=batch_out_dir,
            generate_subtitles=True,
            burn_subtitles=True,
            framing_config=cfg_blur,
        )
        t_batch = (time.perf_counter() - t0) * 1000.0
        print(f"\n7. End-to-End Batch Clipping Pipeline (Clip + ASR + Subtitles + 9:16 Reframe):")
        print(f"   - Pipeline total time:  {t_batch:.2f} ms ({t_batch/1000.0:.2f} s)")
        print(f"   - Clips produced:       {len(batch_results)}")

        end_ram = get_peak_ram_mb()
        print(f"\nFinal Peak RAM:            {end_ram:.2f} MB (+{end_ram - start_ram:.2f} MB)")
        print("=" * 60)


if __name__ == "__main__":
    run_benchmarks()
