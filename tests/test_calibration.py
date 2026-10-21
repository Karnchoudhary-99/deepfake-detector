import json

import pytest

from deepfake_detector.analyzer import analyze
from deepfake_detector.calibration import (
    fit_specs,
    load_profile,
    profile_to_specs,
    save_profile,
    specs_to_profile,
)
from deepfake_detector.classifier import DEFAULT_SPECS, Classifier


@pytest.fixture()
def labelled_rows(sample_images):
    real = analyze(sample_images["real"]).averages()
    fake = analyze(sample_images["fake"]).averages()
    return [real], [fake]


def test_fit_places_midpoints_between_class_means(labelled_rows):
    real_rows, fake_rows = labelled_rows
    fitted = {s.key: s for s in fit_specs(real_rows, fake_rows)}
    for key in ("high_freq_ratio", "seam_ratio", "border_color_step"):
        lo = min(real_rows[0][key], fake_rows[0][key])
        hi = max(real_rows[0][key], fake_rows[0][key])
        assert lo < fitted[key].midpoint < hi


def test_fitted_specs_still_separate(labelled_rows, sample_images):
    real_rows, fake_rows = labelled_rows
    clf = Classifier(fit_specs(real_rows, fake_rows))
    assert clf.classify(analyze(sample_images["real"])).label == "likely real"
    assert clf.classify(analyze(sample_images["fake"])).label == "likely fake"


def test_profile_round_trip(tmp_path):
    profile = specs_to_profile(DEFAULT_SPECS)
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))
    restored = load_profile(path)
    assert {s.key for s in restored} == {s.key for s in DEFAULT_SPECS}
    for orig, back in zip(
            sorted(DEFAULT_SPECS, key=lambda s: s.key),
            sorted(restored, key=lambda s: s.key)):
        assert back.midpoint == orig.midpoint
        assert back.scale == orig.scale
        assert back.domain == orig.domain
        assert back.direction == orig.direction


def test_profile_rejects_bad_version_and_unknown_key():
    with pytest.raises(ValueError):
        profile_to_specs({"version": 99, "specs": []})
    with pytest.raises(ValueError):
        profile_to_specs({"version": 1,
                          "specs": [{"key": "nope", "midpoint": 1, "scale": 1}]})


def test_save_and_load_profile(tmp_path):
    path = tmp_path / "p.json"
    save_profile(DEFAULT_SPECS, path)
    assert json.loads(path.read_text())["version"] == 1
    assert len(load_profile(path)) == len(DEFAULT_SPECS)
