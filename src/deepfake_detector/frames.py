"""Frame sources: images, image directories, and videos (via ffmpeg).

A "frame" is one sampled still from the input, always returned as a
Pillow RGB image together with its index and timestamp. Video decoding
is delegated to the system ffmpeg/ffprobe binaries so the package has
no heavy native dependency (no OpenCV); image inputs need no ffmpeg
at all.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".mpg", ".mpeg"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".ppm", ".pgm"}

MAX_FRAMES_DEFAULT = 16


@dataclass
class Frame:
    """One sampled still."""

    index: int
    timestamp: float  # seconds into the source (0 for still images)
    image: Image.Image  # RGB


def _require_tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(
            f"{name} is required to read video input but was not found on PATH. "
            "Install ffmpeg, or analyse an image / image directory instead."
        )
    return path


def probe_video(path: Path) -> dict:
    """Return ffprobe stream info (frame count, fps, duration) for a video."""
    _require_tool("ffprobe")
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_frames,r_frame_rate,duration",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    info = json.loads(result.stdout or "{}")
    streams = info.get("streams", [])
    stream = streams[0] if streams else {}
    fps = 0.0
    rate = stream.get("r_frame_rate", "0/1")
    try:
        num, den = rate.split("/")
        fps = float(num) / float(den) if float(den) else 0.0
    except (ValueError, ZeroDivisionError):
        fps = 0.0
    duration = float(stream.get("duration") or info.get("format", {}).get("duration") or 0.0)
    nb_frames = stream.get("nb_frames")
    frame_count = int(nb_frames) if nb_frames and str(nb_frames).isdigit() else 0
    if frame_count == 0 and fps > 0 and duration > 0:
        frame_count = max(1, int(round(fps * duration)))
    return {"frame_count": frame_count, "fps": fps, "duration": duration}


def _even_indices(total: int, count: int) -> list[int]:
    """Evenly spaced frame indices covering first..last (inclusive)."""
    if total <= 0:
        return []
    if count <= 1 or total == 1:
        return [0]
    count = min(count, total)
    return sorted({round(i * (total - 1) / (count - 1)) for i in range(count)})


def extract_video_frames(video: Path, max_frames: int = MAX_FRAMES_DEFAULT) -> list[Frame]:
    """Sample up to ``max_frames`` evenly spaced frames from a video file."""
    _require_tool("ffmpeg")
    info = probe_video(video)
    indices = _even_indices(info["frame_count"], max_frames)
    if not indices:
        # Frame count unknown: fall back to 1 fps sampling, then thin out.
        with tempfile.TemporaryDirectory(prefix="dfd_frames_") as tmp:
            subprocess.run(
                ["ffmpeg", "-v", "error", "-i", str(video), "-vf", "fps=1",
                 str(Path(tmp) / "frame_%04d.png")],
                check=True,
            )
            files = sorted(Path(tmp).glob("frame_*.png"))
            picked = _even_indices(len(files), max_frames)
            return [
                Frame(index=i, timestamp=float(i), image=Image.open(f).convert("RGB"))
                for i, f in zip(picked, [files[j] for j in picked])
            ]
    frames: list[Frame] = []
    with tempfile.TemporaryDirectory(prefix="dfd_frames_") as tmp:
        for k, frame_idx in enumerate(indices):
            out = Path(tmp) / f"frame_{k:04d}.png"
            subprocess.run(
                [
                    "ffmpeg", "-v", "error", "-i", str(video),
                    "-vf", f"select='eq(n\\,{frame_idx})'",
                    "-vframes", "1", str(out),
                ],
                check=True,
            )
            timestamp = frame_idx / info["fps"] if info["fps"] > 0 else float(k)
            frames.append(
                Frame(index=frame_idx, timestamp=timestamp,
                      image=Image.open(out).convert("RGB"))
            )
    return frames


def load_image_frames(path: Path) -> list[Frame]:
    """A single image is a one-frame source."""
    return [Frame(index=0, timestamp=0.0, image=Image.open(path).convert("RGB"))]


def load_directory_frames(directory: Path, max_frames: int = MAX_FRAMES_DEFAULT) -> list[Frame]:
    """Sorted image files in a directory, evenly thinned to ``max_frames``."""
    files = sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not files:
        raise ValueError(f"No image files found in directory: {directory}")
    picked = _even_indices(len(files), max_frames)
    return [
        Frame(index=i, timestamp=float(i), image=Image.open(files[i]).convert("RGB"))
        for i in picked
    ]


def iter_frames(source: str | Path, max_frames: int = MAX_FRAMES_DEFAULT) -> list[Frame]:
    """Dispatch on path type: video file, image file, or image directory."""
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")
    if path.is_dir():
        return load_directory_frames(path, max_frames)
    suffix = path.suffix.lower()
    if suffix in VIDEO_EXTENSIONS:
        return extract_video_frames(path, max_frames)
    if suffix in IMAGE_EXTENSIONS:
        return load_image_frames(path)
    raise ValueError(f"Unsupported input type (extension {suffix!r}): {path}")
