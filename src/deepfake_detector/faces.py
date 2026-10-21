"""Face-region location without a heavyweight detector.

The default finder segments skin-coloured pixels in YCbCr space (the
classic Cb 77-127 / Cr 133-173 band) combined with the RGB ordering
rule R > G > B with a clear red-blue gap, which keeps warm-toned
backgrounds and cool (blue/grey) backgrounds out of the mask. It takes
the largest plausible connected component and returns its bounding
box. This is deliberately simple and explainable: it finds *a
face-sized skin region* to analyse, it is not a biometric face
recogniser, and it never claims an identity.

When no skin region is found (line art, a masked face, a non-portrait
frame) the finder falls back to a centre crop so downstream feature
extraction still runs — the fallback is labelled as such in the box's
``method`` field, and callers can see its lower confidence.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from PIL import Image

# Largest size the skin mask is computed at: keeps the BFS cheap and
# makes results resolution-independent.
_MASK_MAX_SIDE = 160


@dataclass
class FaceBox:
    """A face region in pixel coordinates of the original image."""

    x: int
    y: int
    width: int
    height: int
    confidence: float  # 0..1
    method: str  # "skin" or "center-fallback"

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    def expanded(self, image_size: tuple[int, int], margin: float = 0.15) -> "FaceBox":
        """Grow the box by ``margin`` on each side, clamped to the image."""
        img_w, img_h = image_size
        pad_x = int(self.width * margin)
        pad_y = int(self.height * margin)
        x = max(0, self.x - pad_x)
        y = max(0, self.y - pad_y)
        right = min(img_w, self.right + pad_x)
        bottom = min(img_h, self.bottom + pad_y)
        return FaceBox(x, y, right - x, bottom - y, self.confidence, self.method)

    def crop(self, image: Image.Image) -> Image.Image:
        return image.crop((self.x, self.y, self.right, self.bottom))


def _skin_mask(image: Image.Image) -> np.ndarray:
    """Boolean skin mask at reduced resolution."""
    small = image.convert("RGB")
    if max(small.size) > _MASK_MAX_SIDE:
        small.thumbnail((_MASK_MAX_SIDE, _MASK_MAX_SIDE), Image.LANCZOS)
    rgb = np.asarray(small, dtype=np.int16)
    ycbcr = np.asarray(small.convert("YCbCr"), dtype=np.int16)
    y, cb, cr = ycbcr[:, :, 0], ycbcr[:, :, 1], ycbcr[:, :, 2]
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    return (
        (y > 40)
        & (cb >= 77) & (cb <= 127)
        & (cr >= 133) & (cr <= 173)
        & (r > g) & (g > b) & ((r - b) > 20)
    )


def _close_mask(mask: np.ndarray) -> np.ndarray:
    """3x3 morphological closing: fills 1-pixel gaps (facial features,
    compression speckles, patterned artifacts) without growing the
    region, so a fragmented skin area still forms one component."""
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    dilated = np.zeros_like(mask)
    eroded = np.zeros_like(mask)
    for dy in range(3):
        for dx in range(3):
            dilated |= padded[dy:dy + mask.shape[0], dx:dx + mask.shape[1]]
    padded_d = np.pad(dilated, 1, mode="constant", constant_values=True)
    eroded = np.ones_like(mask)
    for dy in range(3):
        for dx in range(3):
            eroded &= padded_d[dy:dy + mask.shape[0], dx:dx + mask.shape[1]]
    return eroded


def _largest_component(mask: np.ndarray) -> tuple[int, int, int, int, int] | None:
    """Largest 4-connected component as (x, y, w, h, pixel_count)."""
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    best: tuple[int, int, int, int, int] | None = None
    for sy in range(height):
        for sx in range(width):
            if not mask[sy, sx] or seen[sy, sx]:
                continue
            queue: deque[tuple[int, int]] = deque([(sx, sy)])
            seen[sy, sx] = True
            min_x = max_x = sx
            min_y = max_y = sy
            count = 0
            while queue:
                x, y = queue.popleft()
                count += 1
                min_x, max_x = min(min_x, x), max(max_x, x)
                min_y, max_y = min(min_y, y), max(max_y, y)
                for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                    if 0 <= nx < width and 0 <= ny < height and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        queue.append((nx, ny))
            if best is None or count > best[4]:
                best = (min_x, min_y, max_x - min_x + 1, max_y - min_y + 1, count)
    return best


def _center_fallback(image: Image.Image) -> FaceBox:
    w, h = image.size
    box_w, box_h = int(w * 0.5), int(h * 0.6)
    return FaceBox(
        x=(w - box_w) // 2,
        y=(h - box_h) // 2,
        width=box_w,
        height=box_h,
        confidence=0.3,
        method="center-fallback",
    )


def find_face(image: Image.Image) -> FaceBox:
    """Locate the dominant face region in ``image`` (RGB Pillow image)."""
    mask = _close_mask(_skin_mask(image))
    comp = _largest_component(mask)
    img_w, img_h = image.size
    scale_x = img_w / mask.shape[1]
    scale_y = img_h / mask.shape[0]
    if comp is not None:
        mx, my, mw, mh, count = comp
        area_fraction = count / mask.size
        aspect = mw / mh if mh else 0.0
        # Plausible face: at least 2% of the frame, not the whole frame,
        # roughly upright proportions (allow wide/narrow faces).
        if 0.02 <= area_fraction <= 0.90 and 0.4 <= aspect <= 2.5:
            box = FaceBox(
                x=int(mx * scale_x),
                y=int(my * scale_y),
                width=max(1, int(mw * scale_x)),
                height=max(1, int(mh * scale_y)),
                confidence=float(min(0.95, 0.5 + area_fraction)),
                method="skin",
            )
            # Clamp to the image (rounding at the edges).
            box.width = min(box.width, img_w - box.x)
            box.height = min(box.height, img_h - box.y)
            return box
    return _center_fallback(image)
