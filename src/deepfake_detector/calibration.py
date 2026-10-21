"""Fit classifier midpoints/scales from labelled sample measurements.

Calibration is deliberately boring: for each feature, the midpoint is
placed halfway between the real-sample mean and the fake-sample mean
(in log space for multiplicative features), and the scale is a quarter
of the class separation, so each class mean sits ~2 scale units from
the midpoint — evidence near 0.12 for the real mean and near 0.88 for
the fake mean on a well-separated feature. A feature that does not
separate the classes gets a wide scale and therefore contributes
almost nothing, which is the honest outcome for it.

A fitted profile is plain JSON (see ``specs_to_profile``), so it can
be inspected, versioned, and passed to ``classify --profile`` without
trusting any binary blob.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from .classifier import DEFAULT_SPECS, FeatureSpec

PROFILE_VERSION = 1


def _class_mean(rows: list[dict[str, float]], key: str) -> float | None:
    values = [r[key] for r in rows if key in r and r[key] is not None]
    if not values:
        return None
    return sum(values) / len(values)


def fit_specs(real_rows: list[dict[str, float]],
              fake_rows: list[dict[str, float]],
              base: tuple[FeatureSpec, ...] = DEFAULT_SPECS,
              ) -> tuple[FeatureSpec, ...]:
    """Return new specs with midpoints/scales fitted to the two classes.

    ``real_rows`` / ``fake_rows`` are ``AnalysisReport.averages()``
    dicts. A spec is left untouched when either class lacks the feature
    or the classes do not separate on it.
    """
    fitted: list[FeatureSpec] = []
    for spec in base:
        real_mean = _class_mean(real_rows, spec.key)
        fake_mean = _class_mean(fake_rows, spec.key)
        if real_mean is None or fake_mean is None:
            fitted.append(spec)
            continue
        if spec.domain == "log":
            if real_mean <= 0 or fake_mean <= 0:
                fitted.append(spec)
                continue
            real_d, fake_d = math.log(real_mean), math.log(fake_mean)
            midpoint = math.exp((real_d + fake_d) / 2)
            scale = max(abs(fake_d - real_d) / 4, 0.05)
        else:
            midpoint = (real_mean + fake_mean) / 2
            scale = max(abs(fake_mean - real_mean) / 4, 1e-3)
        # Direction "low" features (flatness) separate in the opposite
        # sense; the midpoint/scale math is symmetric, so nothing
        # special is needed — evidence inversion lives in the spec.
        fitted.append(FeatureSpec(
            key=spec.key, label=spec.label, weight=spec.weight,
            midpoint=midpoint, scale=scale, domain=spec.domain,
            direction=spec.direction, explains_fake=spec.explains_fake,
            explains_real=spec.explains_real))
    return tuple(fitted)


def specs_to_profile(specs: tuple[FeatureSpec, ...]) -> dict:
    return {
        "version": PROFILE_VERSION,
        "specs": [
            {
                "key": s.key, "label": s.label, "weight": s.weight,
                "midpoint": s.midpoint, "scale": s.scale,
                "domain": s.domain, "direction": s.direction,
                "explains_fake": s.explains_fake,
                "explains_real": s.explains_real,
            }
            for s in specs
        ],
    }


def profile_to_specs(profile: dict) -> tuple[FeatureSpec, ...]:
    if profile.get("version") != PROFILE_VERSION:
        raise ValueError(
            f"unsupported calibration profile version: "
            f"{profile.get('version')!r} (expected {PROFILE_VERSION})")
    by_key = {s.key: s for s in DEFAULT_SPECS}
    specs: list[FeatureSpec] = []
    for entry in profile.get("specs", []):
        key = entry["key"]
        if key not in by_key:
            raise ValueError(f"unknown feature in profile: {key!r}")
        base = by_key[key]
        specs.append(FeatureSpec(
            key=key,
            label=entry.get("label", base.label),
            weight=float(entry.get("weight", base.weight)),
            midpoint=float(entry["midpoint"]),
            scale=float(entry["scale"]),
            domain=entry.get("domain", base.domain),
            direction=entry.get("direction", base.direction),
            explains_fake=entry.get("explains_fake", base.explains_fake),
            explains_real=entry.get("explains_real", base.explains_real)))
    # Any default feature missing from the profile keeps its default.
    present = {s.key for s in specs}
    specs.extend(s for s in DEFAULT_SPECS if s.key not in present)
    return tuple(specs)


def save_profile(specs: tuple[FeatureSpec, ...], path: str | Path) -> None:
    Path(path).write_text(json.dumps(specs_to_profile(specs), indent=2) + "\n")


def load_profile(path: str | Path) -> tuple[FeatureSpec, ...]:
    return profile_to_specs(json.loads(Path(path).read_text()))
