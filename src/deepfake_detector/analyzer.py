"""Pipeline: frames -> face box -> per-frame features -> report.

This layer makes no verdict. It returns the measured features (and,
for video, the flicker between frames) so a later classifier layer can
score them with calibration, and so a human can read the evidence
directly. The CLI prints exactly this report.
"""

from __future__ import annotations

from pathlib import Path

from .faces import find_face
from .features import (
    AnalysisReport,
    FrameFeatures,
    blending_features,
    flicker_features,
    frequency_features,
)
from .frames import Frame, iter_frames


def analyze_frames(frames: list[Frame]) -> AnalysisReport:
    report = AnalysisReport()
    crops = []
    for frame in frames:
        face = find_face(frame.image)
        crop = face.crop(frame.image)
        crops.append(crop)
        report.frames.append(
            FrameFeatures(
                frame_index=frame.index,
                timestamp=frame.timestamp,
                face=face,
                frequency=frequency_features(crop),
                blending=blending_features(frame.image, face),
            )
        )
    report.flicker = flicker_features(crops)
    return report


def analyze(source: str | Path, max_frames: int = 16) -> AnalysisReport:
    """Analyse an image, an image directory, or a video file."""
    return analyze_frames(iter_frames(source, max_frames=max_frames))
