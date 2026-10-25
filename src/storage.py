"""Session storage and TTL lifecycle management for ClipForge."""

from pathlib import Path
import shutil
import time
from typing import Sequence, Tuple

from src.video_info import format_file_size


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
) -> int:
    """
    Remove session directories and files older than max_age_seconds.

    Preserves specified directories (like 'models' downloaded ASR weights).
    Returns count of removed items.
    """
    now = time.time()
    removed_count = 0
    preserved = set(preserved_names)

    for base_dir in base_dirs:
        if not base_dir.is_dir():
            continue

        for item in base_dir.iterdir():
            if item.name in preserved:
                continue

            try:
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
