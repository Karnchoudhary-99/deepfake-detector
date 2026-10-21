# Calibration

The classifier's only learned parameters are, per feature, a
**midpoint** and a **scale** for a logistic evidence curve:

```
evidence = sigmoid((x - midpoint) / scale)          # higher -> more fake
evidence = 1 - sigmoid((x - midpoint) / scale)     # direction "low"
```

For multiplicative features (`seam_ratio`, `border_color_step`,
`mean_frame_diff`) `x` and `midpoint` are log-transformed first, so a
doubling of the measurement moves the evidence by a fixed amount.

## How fitting works

`deepfake-detector calibrate <real_dir> <fake_dir> --out profile.json`
analyses every analysable file in both directories and, per feature:

* midpoint = halfway between the real-sample mean and the fake-sample
  mean (geometric mean for log-domain features);
* scale = a quarter of the class separation, so each class mean sits
  roughly 2 scale units from the midpoint — evidence near 0.12 for the
  real mean and near 0.88 for the fake mean on a well-separated
  feature.

A feature that does not separate the classes gets a wide scale and
therefore contributes almost nothing. That is the honest outcome for
it, not a bug to tune away.

## Profile format

A profile is plain JSON, versioned, and human-readable:

```json
{
  "version": 1,
  "specs": [
    {"key": "high_freq_ratio", "label": "high-frequency energy",
     "weight": 0.25, "midpoint": 0.15, "scale": 0.03,
     "domain": "linear", "direction": "high",
     "explains_fake": "...", "explains_real": "..."}
  ]
}
```

Features absent from a profile keep their built-in defaults, and
unknown feature keys or versions are rejected. Verdict bands
(`likely real` ≤ 0.35, `uncertain` 0.35–0.65, `likely fake` ≥ 0.65)
are fixed in code, not in the profile, so a recalibration cannot
silently redefine what "likely fake" means.

## Built-in defaults

Fitted on the synthetic samples in `examples/` (single-image means):

| feature | real mean | fake mean | midpoint | scale |
| --- | --- | --- | --- | --- |
| high_freq_ratio | 0.078 | 0.238 | 0.15 | 0.03 |
| spectral_centroid | 0.236 | 0.340 | 0.29 | 0.025 |
| spectral_flatness | 0.021 | 0.0043 | 0.012 | 0.004 |
| seam_ratio | 11.3 | 40.9 | 21.5 (log) | 0.30 (log) |
| border_color_step | 24.4 | 126.9 | 54.6 (log) | 0.40 (log) |
| mean_frame_diff (video) | 0.94 | 2.24 | 1.45 (log) | 0.25 (log) |

These numbers describe synthetic tells, not real deepfakes. Refit
before any real-world use, and re-check the score table in the README
after refitting.
