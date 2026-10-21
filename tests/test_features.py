from PIL import Image

from deepfake_detector.analyzer import analyze_frames
from deepfake_detector.faces import find_face
from deepfake_detector.features import (
    blending_features,
    flicker_features,
    frequency_features,
)
from deepfake_detector.frames import Frame


def _face_crop(img):
    box = find_face(img)
    return box, box.crop(img)


def test_fake_has_higher_high_freq_energy(sample_images):
    _, real_crop = _face_crop(sample_images["real_img"])
    _, fake_crop = _face_crop(sample_images["fake_img"])
    real = frequency_features(real_crop)
    fake = frequency_features(fake_crop)
    assert fake.high_freq_ratio > real.high_freq_ratio
    assert fake.spectral_centroid > real.spectral_centroid
    assert 0.0 <= fake.spectral_flatness <= 1.0


def test_fake_has_stronger_blending_seam(sample_images):
    real_box, _ = _face_crop(sample_images["real_img"])
    fake_box, _ = _face_crop(sample_images["fake_img"])
    real = blending_features(sample_images["real_img"], real_box)
    fake = blending_features(sample_images["fake_img"], fake_box)
    assert fake.seam_ratio > real.seam_ratio
    assert fake.border_color_step > real.border_color_step


def test_flicker_grows_with_jitter(sample_images):
    steady = [sample_images["real_img"]] * 4
    jittered = [
        sample_images["fake_img"].transform(
            (256, 256), Image.AFFINE, (1, 0, i % 2, 0, 1, 0))
        for i in range(4)
    ]
    steady_f = flicker_features(steady)
    jitter_f = flicker_features(jittered)
    assert jitter_f is not None and steady_f is not None
    assert jitter_f.mean_frame_diff > steady_f.mean_frame_diff


def test_analyze_frames_report_shape(sample_images):
    frames = [
        Frame(index=0, timestamp=0.0, image=sample_images["real_img"]),
        Frame(index=1, timestamp=1.0, image=sample_images["fake_img"]),
    ]
    report = analyze_frames(frames)
    assert len(report.frames) == 2
    assert report.flicker is not None
    avgs = report.averages()
    assert 0.0 <= avgs["high_freq_ratio"] <= 1.0
    assert avgs["seam_ratio"] > 0
    d = report.to_dict()
    assert d["frame_count"] == 2 and d["flicker"] is not None


def test_analyze_single_frame_has_no_flicker(sample_images):
    report = analyze_frames(
        [Frame(index=0, timestamp=0.0, image=sample_images["real_img"])]
    )
    assert report.flicker is None
    assert report.averages()["high_freq_ratio"] >= 0.0
