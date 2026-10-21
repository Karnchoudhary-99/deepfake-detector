import json

import pytest

from deepfake_detector.analyzer import analyze
from deepfake_detector.classifier import (
    DEFAULT_SPECS,
    Classifier,
    evidence_for,
    label_for,
)
from deepfake_detector.features import FlickerFeatures


def _spec(key):
    return next(s for s in DEFAULT_SPECS if s.key == key)


def test_synthetic_real_scores_likely_real(sample_images):
    verdict = Classifier().classify(analyze(sample_images["real"]))
    assert verdict.label == "likely real"
    assert verdict.score < 0.35


def test_synthetic_fake_scores_likely_fake(sample_images):
    verdict = Classifier().classify(analyze(sample_images["fake"]))
    assert verdict.label == "likely fake"
    assert verdict.score > 0.65


def test_explanations_name_the_evidence(sample_images):
    verdict = Classifier().classify(analyze(sample_images["fake"]))
    lines = verdict.explain()
    assert lines
    assert any("high-frequency energy" in line for line in lines)
    assert all("points to fake" in line for line in lines)
    assert verdict.to_dict()["score"] == verdict.score
    json.dumps(verdict.to_dict())  # serialisable


def test_evidence_is_monotone_and_midpoint_is_half():
    spec = _spec("high_freq_ratio")
    assert evidence_for(spec, 0.05) < evidence_for(spec, 0.15) < evidence_for(spec, 0.30)
    assert evidence_for(spec, spec.midpoint) == pytest.approx(0.5)
    # Inverted direction: lower flatness -> more fake evidence.
    flat = _spec("spectral_flatness")
    assert evidence_for(flat, 0.004) > evidence_for(flat, 0.021)
    # Log domain: seam evidence grows with the ratio, multiplicatively.
    seam = _spec("seam_ratio")
    assert evidence_for(seam, 11.0) < evidence_for(seam, 41.0)


def test_label_bands():
    assert label_for(0.10) == "likely real"
    assert label_for(0.35) == "likely real"
    assert label_for(0.50) == "uncertain"
    assert label_for(0.65) == "likely fake"
    assert label_for(0.95) == "likely fake"


def test_flicker_feature_counts_only_for_video(sample_images):
    image_report = analyze(sample_images["fake"])
    assert image_report.flicker is None
    verdict = Classifier().classify(image_report)
    assert "mean_frame_diff" not in [e.spec.key for e in verdict.evidence]
    # Same image report with a shimmering video's flicker attached.
    image_report.flicker = FlickerFeatures(mean_frame_diff=2.2, max_frame_diff=2.9)
    verdict = Classifier().classify(image_report)
    flicker = next(e for e in verdict.evidence if e.spec.key == "mean_frame_diff")
    assert flicker.evidence > 0.5
    assert flicker.value == 2.2


def test_weights_renormalise_without_video(sample_images):
    # Score is a weighted mean over present features only, so it stays
    # in [0, 1] whether or not the video-only feature is present.
    verdict = Classifier().classify(analyze(sample_images["real"]))
    assert 0.0 <= verdict.score <= 1.0
