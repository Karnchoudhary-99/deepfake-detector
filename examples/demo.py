"""End-to-end demo: generate synthetic samples, classify all four.

Run: ``python examples/demo.py [out_dir]`` — generates real/fake
images (and videos when ffmpeg is available) via
``examples/make_samples.py``, then prints each verdict with its top
evidence lines. No real deepfakes are involved; the "fake" samples
exhibit the tells this detector measures (see make_samples.py).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from deepfake_detector.analyzer import analyze  # noqa: E402
from deepfake_detector.classifier import Classifier  # noqa: E402


def main() -> int:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("samples")
    subprocess.run([sys.executable, str(Path(__file__).parent / "make_samples.py"),
                    str(out_dir)], check=True)
    targets = [out_dir / "real.png", out_dir / "fake.png"]
    for video in (out_dir / "real.mp4", out_dir / "fake.mp4"):
        if video.exists():
            targets.append(video)
    clf = Classifier()
    for target in targets:
        verdict = clf.classify(analyze(target))
        print(f"\n{target.name}: {verdict.label} (score {verdict.score:.3f})")
        for line in verdict.explain():
            print(f"  - {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
