"""Explainable artifact features for one face region.

Three families, each cheap to compute and each tied to a known
deepfake tell:

* **Frequency** — GAN/upsampling pipelines leave excess high-frequency
  energy and a shifted spectral centroid that natural camera images
  rarely show (periodic artifacts also *lower* spectral flatness by
  concentrating energy at a few frequencies, so flatness is reported
  without a one-way claim). Computed from the 2D FFT of the face
  crop's luminance.
* **Blending boundary** — a swapped face is composited into a frame, so
  the gradient and colour step across the face border is sharper than
  the texture just inside it.
* **Temporal flicker** — frame-to-frame jitter of the face region in
  video; real footage moves smoothly, a swapped face shimmers.

Every feature is a plain float in a dataclass, so the report is
auditable: no hidden model, no opaque score at this layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from .faces import FaceBox

_CROP_SIZE = (128, 128)


@dataclass
class FrequencyFeatures:
    high_freq_ratio: float  # energy above 50% Nyquist / total energy
    spectral_flatness: float  # geometric / arithmetic mean of power spectrum
    spectral_centroid: float  # energy-weighted mean radius, 0..1 of Nyquist


@dataclass
class BlendingFeatures:
    border_gradient: float  # mean gradient magnitude on the border ring
    interior_gradient: float  # mean gradient magnitude inside the face
    seam_ratio: float  # border / (interior + eps); >1 means a visible seam
    border_color_step: float  # mean colour step across the border, 0..255


@dataclass
class FlickerFeatures:
    mean_frame_diff: float  # mean |crop_t - crop_{t-1}|, 0..255
    max_frame_diff: float


@dataclass
class FrameFeatures:
    frame_index: int
    timestamp: float
    face: FaceBox
    frequency: FrequencyFeatures
    blending: BlendingFeatures


@dataclass
class AnalysisReport:
    frames: list[FrameFeatures] = field(default_factory=list)
    flicker: FlickerFeatures | None = None

    def averages(self) -> dict[str, float]:
        """Mean of each per-frame feature across analysed frames."""
        if not self.frames:
            return {}
        keys = ("high_freq_ratio", "spectral_flatness", "spectral_centroid",
                "border_gradient", "interior_gradient", "seam_ratio",
                "border_color_step")
        out: dict[str, float] = {}
        for key in keys:
            values = []
            for f in self.frames:
                if hasattr(f.frequency, key):
                    values.append(getattr(f.frequency, key))
                else:
                    values.append(getattr(f.blending, key))
            out[key] = float(np.mean(values))
        if self.flicker is not None:
            out["mean_frame_diff"] = self.flicker.mean_frame_diff
            out["max_frame_diff"] = self.flicker.max_frame_diff
        return out

    def to_dict(self) -> dict:
        return {
            "frame_count": len(self.frames),
            "averages": self.averages(),
            "frames": [
                {
                    "frame_index": f.frame_index,
                    "timestamp": f.timestamp,
                    "face": {
                        "x": f.face.x, "y": f.face.y,
                        "width": f.face.width, "height": f.face.height,
                        "confidence": f.face.confidence, "method": f.face.method,
                    },
                    "frequency": vars(f.frequency),
                    "blending": vars(f.blending),
                }
                for f in self.frames
            ],
            "flicker": vars(self.flicker) if self.flicker else None,
        }


def _gray_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("L"), dtype=np.float64)


def frequency_features(face_crop: Image.Image) -> FrequencyFeatures:
    """FFT statistics of the face crop's luminance."""
    gray = _gray_array(face_crop.resize(_CROP_SIZE, Image.LANCZOS))
    gray = gray - gray.mean()
    spectrum = np.fft.fftshift(np.fft.fft2(gray))
    power = np.abs(spectrum) ** 2
    total = power.sum()
    if total <= 0:
        return FrequencyFeatures(0.0, 0.0, 0.0)
    h, w = power.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / (min(h, w) / 2)
    high = power[radius > 0.5].sum()
    flat_power = power + 1e-12
    flatness = float(np.exp(np.log(flat_power).mean()) / flat_power.mean())
    centroid = float((power * radius).sum() / total)
    return FrequencyFeatures(
        high_freq_ratio=float(high / total),
        spectral_flatness=flatness,
        spectral_centroid=centroid,
    )


def _gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(gray)
    return np.sqrt(gx * gx + gy * gy)


def blending_features(image: Image.Image, face: FaceBox) -> BlendingFeatures:
    """Gradient / colour step across the face border vs the interior."""
    arr = np.asarray(image.convert("RGB"), dtype=np.float64)
    gray = arr.mean(axis=2)
    grad = _gradient_magnitude(gray)
    x, y, w, h = face.x, face.y, face.width, face.height
    ring = 3  # pixels of border band, inside and outside
    x0, y0 = max(0, x - ring), max(0, y - ring)
    x1, y1 = min(arr.shape[1], face.right + ring), min(arr.shape[0], face.bottom + ring)
    border_mask = np.zeros(grad.shape, dtype=bool)
    border_mask[y0:y1, x0:x1] = True
    inner = np.zeros(grad.shape, dtype=bool)
    inner[y:y + h, x:x + w] = True
    border_mask &= ~inner
    # Interior band just inside the border (not the whole face, so the
    # comparison is local to the seam, not to eyes/mouth texture).
    inner_band = np.zeros(grad.shape, dtype=bool)
    ix0, iy0 = x + ring, y + ring
    ix1, iy1 = face.right - ring, face.bottom - ring
    if ix1 > ix0 and iy1 > iy0:
        inner_band[iy0:iy1, ix0:ix1] = True
        interior = grad[inner_band & inner]
    else:
        interior = grad[inner]
    border = grad[border_mask]
    border_mean = float(border.mean()) if border.size else 0.0
    interior_mean = float(interior.mean()) if interior.size else 0.0
    # Colour step: mean |inside pixel - outside pixel| along the border.
    steps: list[float] = []
    for yy in range(y, min(face.bottom, arr.shape[0])):
        if x > 0:
            steps.append(float(np.abs(arr[yy, x] - arr[yy, x - 1]).mean()))
        if face.right < arr.shape[1]:
            steps.append(float(np.abs(arr[yy, face.right] - arr[yy, face.right - 1]).mean()))
    for xx in range(x, min(face.right, arr.shape[1])):
        if y > 0:
            steps.append(float(np.abs(arr[y, xx] - arr[y - 1, xx]).mean()))
        if face.bottom < arr.shape[0]:
            steps.append(float(np.abs(arr[face.bottom, xx] - arr[face.bottom - 1, xx]).mean()))
    return BlendingFeatures(
        border_gradient=border_mean,
        interior_gradient=interior_mean,
        seam_ratio=border_mean / (interior_mean + 1e-6),
        border_color_step=float(np.mean(steps)) if steps else 0.0,
    )


def flicker_features(crops: list[Image.Image]) -> FlickerFeatures | None:
    """Frame-to-frame change of the (resized) face crops."""
    if len(crops) < 2:
        return None
    arrays = [np.asarray(c.resize(_CROP_SIZE, Image.LANCZOS), dtype=np.float64)
              for c in crops]
    diffs = [float(np.abs(b - a).mean()) for a, b in zip(arrays, arrays[1:])]
    return FlickerFeatures(mean_frame_diff=float(np.mean(diffs)),
                           max_frame_diff=float(np.max(diffs)))
