import shutil

import pytest

from deepfake_detector.frames import _even_indices, iter_frames


def test_even_indices_cover_endpoints():
    assert _even_indices(100, 5) == [0, 25, 50, 74, 99]
    assert _even_indices(1, 5) == [0]
    assert _even_indices(0, 5) == []
    assert _even_indices(3, 10) == [0, 1, 2]


def test_single_image_source(sample_images):
    frames = iter_frames(sample_images["real"])
    assert len(frames) == 1
    assert frames[0].image.size == (256, 256)
    assert frames[0].timestamp == 0.0


def test_directory_source_thins_evenly(tmp_path, sample_images):
    # Isolated directory: tmp_path also holds the fixture images, so
    # the sequence lives in its own subdir.
    seq_dir = tmp_path / "seq"
    seq_dir.mkdir()
    for i in range(10):
        sample_images["real_img"].save(seq_dir / f"img_{i:02d}.png")
    frames = iter_frames(seq_dir, max_frames=4)
    assert len(frames) == 4
    assert [f.index for f in frames] == [0, 3, 6, 9]


def test_missing_input_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        iter_frames(tmp_path / "nope.png")


def test_empty_directory_raises(tmp_path):
    with pytest.raises(ValueError, match="No image files"):
        iter_frames(tmp_path)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_video_source_samples_frames(tmp_path):
    import make_samples  # from conftest's sys.path insert
    import numpy as np

    frames_in = [make_samples.make_real(np.random.default_rng(7 + i))
                 for i in range(20)]
    make_samples._write_video(frames_in, tmp_path / "clip.mp4")
    frames = iter_frames(tmp_path / "clip.mp4", max_frames=5)
    assert len(frames) == 5
    assert frames[0].timestamp <= frames[-1].timestamp
    assert all(f.image.size == (256, 256) for f in frames)
