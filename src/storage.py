"""Session storage, TTL lifecycle, and resource safety management for ClipForge."""

from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import threading
import time
from typing import Collection, Optional, Sequence, Tuple

from src.video_info import format_file_size

# Resource safety configuration
DEFAULT_MIN_FREE_DISK_BYTES = int(float(os.getenv("CLIPFORGE_MIN_FREE_DISK_GB", "2.0")) * 1024 * 1024 * 1024)
DEFAULT_MAX_UPLOAD_MB = int(os.getenv("CLIPFORGE_MAX_UPLOAD_MB", "1024"))
DEFAULT_MAX_VIDEO_DURATION_SECONDS = float(os.getenv("CLIPFORGE_MAX_VIDEO_DURATION_SECONDS", "7200.0"))  # 2 hours
MAX_CONCURRENT_HEAVY_JOBS = int(os.getenv("CLIPFORGE_MAX_CONCURRENT_JOBS", "2"))

_HEAVY_JOB_SEMAPHORE = threading.Semaphore(MAX_CONCURRENT_HEAVY_JOBS)


class CapacityExceededError(Exception):
    """Raised when server processing capacity or concurrency limit is reached."""
    pass


class StorageLimitError(Exception):
    """Raised when disk storage is insufficient for processing."""
    pass


@contextmanager
def acquire_job_slot(timeout: float = 10.0):
    """
    Acquire a processing slot for CPU/GPU heavy operations (rendering / transcription).
    Prevents concurrent jobs from exhausting system RAM and crashing the server.
    """
    acquired = _HEAVY_JOB_SEMAPHORE.acquire(timeout=timeout)
    if not acquired:
        raise CapacityExceededError(
            f"Server processing capacity reached (max {MAX_CONCURRENT_HEAVY_JOBS} concurrent jobs). "
            "Please wait for active rendering to complete and try again."
        )
    try:
        yield
    finally:
        _HEAVY_JOB_SEMAPHORE.release()


def check_disk_space_available(
    path: Path,
    min_free_bytes: int = DEFAULT_MIN_FREE_DISK_BYTES,
) -> Tuple[bool, int, int]:
    """
    Check if the filesystem containing path has sufficient free bytes.
    Returns (is_sufficient, free_bytes, total_bytes).
    """
    try:
        target = path if path.exists() else path.parent
        usage = shutil.disk_usage(target)
        return usage.free >= min_free_bytes, usage.free, usage.total
    except Exception:
        # Fallback if filesystem stats unavailable
        return True, 0, 0


def init_session_workspace(
    base_temp: Path,
    base_exports: Path,
    session_id: str,
) -> Tuple[Path, Path]:
    """Create and return isolated per-session temp and exports directory paths."""
    session_temp = base_temp / session_id
    session_exports = base_exports / session_id
    session_temp.mkdir(parents=True, exist_ok=True)
    session_exports.mkdir(parents=True, exist_ok=True)
    return session_temp, session_exports


def cleanup_stale_files(
    base_dirs: Sequence[Path],
    max_age_seconds: int = 3600,
    preserved_names: Sequence[str] = ("models", "transcripts"),
    active_session_ids: Optional[Collection[str]] = None,
) -> int:
    """
    Remove session directories, symlinks, and files older than max_age_seconds.

    Preserves:
    - Specified directories (like 'models' downloaded ASR weights).
    - Active session IDs provided in active_session_ids.
    - Files outside the designated workspace roots (strict containment).

    Safely unlinks symlinks without traversing into target directories.
    Returns count of removed items.
    """
    now = time.time()
    removed_count = 0
    preserved = set(preserved_names)
    active = set(active_session_ids) if active_session_ids else set()

    for base_dir in base_dirs:
        if not base_dir.is_dir():
            continue

        base_dir_resolved = base_dir.resolve()

        for item in base_dir.iterdir():
            if item.name in preserved or item.name in active:
                continue

            try:
                # Handle symlinks safely: do not follow to target!
                if item.is_symlink():
                    try:
                        link_mtime = os.lstat(item).st_mtime
                    except Exception:
                        link_mtime = 0
                    if now - link_mtime > max_age_seconds:
                        item.unlink(missing_ok=True)
                        removed_count += 1
                    continue

                # Ensure item is strictly inside base_dir
                if not item.resolve().is_relative_to(base_dir_resolved):
                    continue

                mtime = item.stat().st_mtime
                if now - mtime > max_age_seconds:
                    if item.is_dir():
                        shutil.rmtree(item, ignore_errors=True)
                        removed_count += 1
                    elif item.is_file():
                        item.unlink(missing_ok=True)
                        removed_count += 1
            except Exception:
                pass

    return removed_count


def calculate_dir_size_str(directory: Path) -> str:
    """Return formatted size string of all regular files in a directory."""
    if not directory.is_dir():
        return "0 B"
    try:
        total = sum(f.stat().st_size for f in directory.iterdir() if f.is_file())
        return format_file_size(total)
    except Exception:
        return "0 B"
