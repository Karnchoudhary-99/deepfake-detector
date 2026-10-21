import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# Synthetic sample generator lives in examples/ (shared with the demo).
sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))

import make_samples  # noqa: E402


@pytest.fixture()
def sample_images(tmp_path):
    """Deterministic real/fake synthetic images (see make_samples docstring)."""
    rng = np.random.default_rng(42)
    real = make_samples.make_real(rng)
    fake = make_samples.make_fake(rng)
    real_path = tmp_path / "real.png"
    fake_path = tmp_path / "fake.png"
    real.save(real_path)
    fake.save(fake_path)
    return {"real": real_path, "fake": fake_path,
            "real_img": real, "fake_img": fake}


@pytest.fixture()
def blank_image():
    return Image.new("RGB", (200, 200), (20, 40, 200))
