"""Calibrated classifier: feature measurements -> score + explanation.

The pipeline layer (``analyzer``) reports measurements only. This layer
turns them into a verdict, and — deliberately — keeps every step
auditable:

* Each feature is mapped to a 0..1 *evidence* value with a logistic
  curve centred on a calibrated midpoint: ``evidence = sigmoid((x - mid)
  / scale)`` for features where higher means "more fake", inverted for
  spectral flatness (periodic artifacts *lower* flatness). Two features
  live on a multiplicative scale (seam ratio, colour step, frame diff),
  so their curves are evaluated in log space.
* The score is the weighted mean of the per-feature evidence values.
  Weights sum to 1 for images; video adds the flicker feature and the
  weights are renormalised, so a score always means the same thing.
* The verdict is a band, not a binary claim: ``likely real`` below
  0.35, ``uncertain`` between 0.35 and 0.65, ``likely fake`` above
  0.65. The band edges are part of the profile, so a recalibration can
  move them.
* Every verdict carries its top contributing features as plain
  language, so a human can check the evidence instead of trusting a
  number.

Default midpoints/scales were fitted from the synthetic real/fake
samples in ``examples/`` (see ``calibration.fit_profile`` and
``docs/calibration.md``); they are a starting point, not ground truth
— rerun ``deepfake-detector calibrate`` on labelled samples from your
own domain before trusting the score on real footage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .features import AnalysisReport

# Band edges on the 0..1 score.
LIKELY_REAL_MAX = 0.35
LIKELY_FAKE_MIN = 0.65


@dataclass(frozen=True)
class FeatureSpec:
    """How one measured feature maps to fake-evidence."""

    key: str  # key in AnalysisReport.averages()
    label: str  # human-readable name
    weight: float  # relative weight; renormalised over present features
    midpoint: float  # centre of the logistic curve, in raw units
    scale: float  # curve steepness, in the same domain as midpoint
    domain: str = "linear"  # "linear" or "log" (multiplicative features)
    direction: str = "high"  # "high" = higher value -> more fake
    explains_fake: str = ""  # why a high evidence value points to fake
    explains_real: str = ""  # why a low evidence value points to real


# Defaults fitted on the synthetic samples (real vs fake means):
#   high_freq_ratio  0.078 vs 0.238   flatness 0.021 vs 0.0043
#   centroid         0.236 vs 0.340   seam     11.3  vs 40.9
#   colour step      24.4  vs 126.9   frame diff (video) 0.94 vs 2.24
DEFAULT_SPECS: tuple[FeatureSpec, ...] = (
    FeatureSpec("high_freq_ratio", "high-frequency energy", 0.25,
                midpoint=0.15, scale=0.03,
                explains_fake="excess high-frequency energy, typical of "
                "upsampled or GAN-generated texture",
                explains_real="frequency spectrum dominated by low "
                "frequencies, as in natural camera images"),
    FeatureSpec("spectral_centroid", "spectral centroid", 0.15,
                midpoint=0.29, scale=0.025,
                explains_fake="spectral energy shifted toward high "
                "frequencies",
                explains_real="spectral energy centred at low frequencies"),
    FeatureSpec("spectral_flatness", "spectral flatness", 0.10,
                midpoint=0.012, scale=0.004, direction="low",
                explains_fake="energy concentrated at a few periodic "
                "frequencies instead of spread naturally",
                explains_real="energy spread smoothly across frequencies"),
    FeatureSpec("seam_ratio", "blending seam ratio", 0.25,
                midpoint=21.5, scale=0.30, domain="log",
                explains_fake="a sharp gradient step at the face border, "
                "as if the face was composited in",
                explains_real="no sharp step at the face border"),
    FeatureSpec("border_color_step", "border colour step", 0.25,
                midpoint=54.6, scale=0.40, domain="log",
                explains_fake="an abrupt colour jump across the face "
                "border",
                explains_real="colour changes gradually across the face "
                "border"),
    FeatureSpec("mean_frame_diff", "temporal flicker", 0.20,
                midpoint=1.45, scale=0.25, domain="log",
                explains_fake="the face region shimmers frame to frame "
                "instead of moving smoothly",
                explains_real="the face region moves smoothly between "
                "frames"),
)


@dataclass
class FeatureEvidence:
    spec: FeatureSpec
    value: float
    evidence: float  # 0..1, higher = more fake-like
    contribution: float  # weight share * evidence, before renormalisation

    @property
    def points_to_fake(self) -> bool:
        return self.evidence >= 0.5


@dataclass
class Verdict:
    score: float  # 0..1 weighted mean evidence
    label: str  # "likely real" | "uncertain" | "likely fake"
    frame_count: int
    evidence: list[FeatureEvidence] = field(default_factory=list)

    def top_reasons(self, n: int = 3) -> list[FeatureEvidence]:
        """Features pulling hardest toward the verdict's side."""
        if self.label == "likely fake":
            ranked = sorted(self.evidence, key=lambda e: -e.evidence)
        elif self.label == "likely real":
            ranked = sorted(self.evidence, key=lambda e: e.evidence)
        else:
            ranked = sorted(self.evidence,
                            key=lambda e: -abs(e.evidence - 0.5))
        return ranked[:n]

    def explain(self) -> list[str]:
        """One plain-language line per top reason."""
        lines = []
        for ev in self.top_reasons():
            if ev.points_to_fake:
                why = ev.spec.explains_fake
                side = "points to fake"
            else:
                why = ev.spec.explains_real
                side = "points to real"
            lines.append(
                f"{ev.spec.label} = {ev.value:.4g} ({side}: {why}; "
                f"evidence {ev.evidence:.2f})"
            )
        return lines

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "label": self.label,
            "frame_count": self.frame_count,
            "evidence": [
                {
                    "feature": ev.spec.key,
                    "label": ev.spec.label,
                    "value": ev.value,
                    "evidence": ev.evidence,
                    "weight": ev.spec.weight,
                }
                for ev in self.evidence
            ],
            "explanation": self.explain(),
        }


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def evidence_for(spec: FeatureSpec, value: float) -> float:
    """Map one raw measurement to 0..1 fake-evidence."""
    if spec.domain == "log":
        if value <= 0:
            return 0.0 if spec.direction == "high" else 1.0
        x = math.log(value)
        mid = math.log(spec.midpoint)
    else:
        x, mid = value, spec.midpoint
    z = (x - mid) / spec.scale
    ev = _sigmoid(z)
    return 1.0 - ev if spec.direction == "low" else ev


def label_for(score: float) -> str:
    if score >= LIKELY_FAKE_MIN:
        return "likely fake"
    if score <= LIKELY_REAL_MAX:
        return "likely real"
    return "uncertain"


class Classifier:
    """Score an AnalysisReport with a fixed set of feature specs."""

    def __init__(self, specs: tuple[FeatureSpec, ...] = DEFAULT_SPECS):
        self.specs = specs

    def classify(self, report: AnalysisReport) -> Verdict:
        averages = report.averages()
        present = [s for s in self.specs if s.key in averages]
        total_weight = sum(s.weight for s in present)
        evidence: list[FeatureEvidence] = []
        score = 0.0
        for spec in present:
            value = averages[spec.key]
            ev = evidence_for(spec, value)
            evidence.append(FeatureEvidence(
                spec=spec, value=value, evidence=ev,
                contribution=spec.weight * ev))
            score += spec.weight * ev
        if total_weight > 0:
            score /= total_weight
        return Verdict(score=score, label=label_for(score),
                       frame_count=len(report.frames), evidence=evidence)


def classify(report: AnalysisReport,
            specs: tuple[FeatureSpec, ...] = DEFAULT_SPECS) -> Verdict:
    return Classifier(specs).classify(report)
