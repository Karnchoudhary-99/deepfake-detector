"""Generate deterministic synthetic samples for tests and the demo.

No real deepfakes are bundled (licensing + consent), so the "fake"
sample is built to exhibit the exact tells this detector measures:

* ``real.png`` — a smooth, skin-toned face region on a soft background
  (blurred noise: natural images are low-pass dominated).
* ``fake.png`` — the same base, but the face region is composited with
  a high-frequency checkerboard (upsampling-style artifact) and a
  hard 1-pixel seam around the face border (blending-boundary tell).

Run: ``python examples/make_samples.py [out_dir]`` — writes real.png,
fake.png, and a 2-second demo video pair (real.mp4 / fake.mp4) whose
fake frames additionally jitter by 1px per frame (flicker tell).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

SIZE = (256, 256)
FACE = (64, 48, 128, 160)  # x, y, w, h — skin-toned face region


def _base(rng: np.random.Generator) -> np.ndarray:
    """Smooth photo-like base: gradients + blurred noise."""
    yy, xx = np.mgrid[0:SIZE[1], 0:SIZE[0]]
    img = np.zeros((SIZE[1], SIZE[0], 3), dtype=np.float64)
    # Cool, clearly non-skin background (blue-grey) so the warm face
    # region is the only skin-coloured area in the frame.
    img[:, :, 0] = 50 + 0.15 * xx + 0.05 * yy
    img[:, :, 1] = 85 + 0.15 * xx
    img[:, :, 2] = 150 + 0.10 * yy
    noise = rng.normal(0, 18, (SIZE[1], SIZE[0], 1))
    img += noise
    x, y, w, h = FACE
    img[y:y + h, x:x + w] = [205, 150, 110]  # skin tone
    skin_noise = rng.normal(0, 6, (h, w, 3))
    img[y:y + h, x:x + w] += skin_noise
    blurred = np.asarray(
        Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(1.2))
    )
    return blurred


def make_real(rng: np.random.Generator) -> Image.Image:
    return Image.fromarray(_base(rng).astype(np.uint8))


def make_fake(rng: np.random.Generator) -> Image.Image:
    img = _base(rng).astype(np.float64)
    x, y, w, h = FACE
    # High-frequency checkerboard inside the face region.
    yy, xx = np.mgrid[0:h, 0:w]
    checker = ((xx + yy) % 2) * 46.0
    img[y:y + h, x:x + w, 0] += checker
    img[y:y + h, x:x + w, 1] -= checker * 0.5
    # Hard seam: brighten the exact border pixels.
    img[y, x:x + w] = [255, 255, 255]
    img[y + h - 1, x:x + w] = [255, 255, 255]
    img[y:y + h, x] = [255, 255, 255]
    img[y:y + h, x + w - 1] = [255, 255, 255]
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def _write_video(frames: list[Image.Image], out: Path) -> None:
    tmp = out.parent / f".{out.stem}_frames"
    tmp.mkdir(exist_ok=True)
    for i, frame in enumerate(frames):
        frame.save(tmp / f"f{i:03d}.png")
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-framerate", "10", "-i",
         str(tmp / "f%03d.png"), "-pix_fmt", "yuv420p", str(out)],
        check=True,
    )
    for f in tmp.iterdir():
        f.unlink()
    tmp.rmdir()


def main() -> int:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("samples")
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    real = make_real(rng)
    fake = make_fake(rng)
    real.save(out_dir / "real.png")
    fake.save(out_dir / "fake.png")
    # Videos: real frames drift smoothly, fake frames jitter 1px/frame.
    real_frames = [make_real(np.random.default_rng(100 + i)) for i in range(20)]
    fake_frames = []
    for i in range(20):
        frame = make_fake(np.random.default_rng(200 + i))
        fake_frames.append(frame.transform(
            SIZE, Image.AFFINE, (1, 0, (i % 2), 0, 1, 0)))
    _write_video(real_frames, out_dir / "real.mp4")
    _write_video(fake_frames, out_dir / "fake.mp4")
    print(f"wrote {out_dir}/real.png, fake.png, real.mp4, fake.mp4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
