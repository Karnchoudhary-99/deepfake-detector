# Deepfake Detector

Detect AI-generated face videos and images with a lightweight,
explainable detector: frame sampling, face-region analysis, frequency
artifacts, and a calibrated score with the evidence behind it — no
black-box verdicts.

## Roadmap

- [x] Project scaffold (Python layout, README, CI)
- [x] Frame extraction + face detection pipeline
- [x] Artifact features: frequency spectra, blending boundaries, temporal flicker
- [x] CLI + demo (image and short video)
- [x] Classifier + calibrated scoring with explanations
- [x] Tests on synthetic samples (real deepfakes deliberately not bundled: licensing + consent), README, then public

## Usage

```bash
pip install -e .
deepfake-detector analyze samples/fake.png
deepfake-detector analyze clip.mp4 --max-frames 16 --json
```

The report lists, per frame, the face box found (skin-segmentation
finder, with a labelled centre-crop fallback), the frequency features
(high-frequency energy ratio, spectral flatness, centroid), the
blending-boundary features (border vs interior gradient, seam ratio,
colour step), and — for video — temporal flicker between face crops.
It reports measurements, not a verdict: the calibrated classifier
lands in the next step.

Generate the synthetic demo samples (no real deepfakes are bundled):

```bash
python examples/make_samples.py samples
deepfake-detector analyze samples/fake.png
deepfake-detector analyze samples/fake.mp4
```

Video input is decoded with the system `ffmpeg`/`ffprobe` (preinstalled
on GitHub's Ubuntu runners); image inputs need no ffmpeg. Face finding
uses YCbCr skin segmentation + the largest plausible component — no
OpenCV, no model weights, no identity recognition.

## Classifier: a score you can audit

`analyze` reports measurements only. `classify` turns them into a
verdict band — `likely real` (score ≤ 0.35), `uncertain` (0.35–0.65),
`likely fake` (≥ 0.65) — with the evidence behind it:

```bash
deepfake-detector classify samples/fake.png
# verdict: likely fake (score 0.904, 1 frames)
# evidence:
#   - high-frequency energy = 0.2383 (points to fake: excess high-frequency
#     energy, typical of upsampled or GAN-generated texture; evidence 0.95)
#   - blending seam ratio = 40.94 (points to fake: a sharp gradient step at
#     the face border, as if the face was composited in; evidence 0.90)
#   - border colour step = 126.9 (points to fake: an abrupt colour jump
#     across the face border; evidence 0.89)
```

Each feature maps to 0–1 evidence through a logistic curve centred on
a calibrated midpoint (log-scale for seam ratio, colour step, and
flicker, which are multiplicative); the score is the weighted mean of
the evidence values, renormalised over the features present (video
adds temporal flicker). Baselines on the bundled synthetic samples:

| sample | score | verdict |
| --- | --- | --- |
| `real.png` | 0.102 | likely real |
| `fake.png` | 0.904 | likely fake |
| `real.mp4` (16 frames) | 0.165 | likely real |
| `fake.mp4` (16 frames) | 0.916 | likely fake |

The built-in midpoints were fitted on synthetic samples — they are a
starting point, not ground truth. Refit them on labelled samples from
your own domain before trusting the score on real footage:

```bash
deepfake-detector calibrate data/real data/fake --out my-profile.json
deepfake-detector classify clip.mp4 --profile my-profile.json
```

See `docs/calibration.md` for how fitting works and what a profile
contains. The detector is an evidence tool, not a judge: treat
`uncertain` as "needs a human look", and never use a score alone to
accuse a real person.

## License

MIT
