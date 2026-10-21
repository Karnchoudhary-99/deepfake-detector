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
- [ ] Classifier + calibrated scoring with explanations
- [ ] Tests on real + synthetic samples, README, then public

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

## License

MIT
