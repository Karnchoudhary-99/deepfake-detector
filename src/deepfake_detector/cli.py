"""Command line.

* ``deepfake-detector analyze <image|video|directory>`` — prints a
  human-readable feature report, or ``--json`` for machines.
* ``deepfake-detector classify <image|video|directory>`` — the same
  analysis plus a calibrated verdict (score, band, and the evidence
  behind it). ``--profile`` loads a fitted calibration profile.
* ``deepfake-detector calibrate <real_dir> <fake_dir> --out profile.json``
  — fits midpoints/scales from labelled samples and writes a profile.

Exit code is 0 whenever the command completes; ``classify`` reports
``likely real`` / ``uncertain`` / ``likely fake`` bands, never a bare
binary claim.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .analyzer import analyze
from .calibration import fit_specs, load_profile, save_profile
from .classifier import Classifier


def _print_human(report) -> None:
    avgs = report.averages()
    print(f"frames analysed: {len(report.frames)}")
    if not avgs:
        print("no frames found")
        return
    print("average features across frames:")
    for key, value in avgs.items():
        print(f"  {key:<22} {value:.4f}")
    print("per-frame face boxes:")
    for frame in report.frames:
        face = frame.face
        print(
            f"  frame {frame.frame_index} @ {frame.timestamp:.2f}s: "
            f"box=({face.x},{face.y},{face.width}x{face.height}) "
            f"method={face.method} confidence={face.confidence:.2f} "
            f"hf_ratio={frame.frequency.high_freq_ratio:.4f} "
            f"seam_ratio={frame.blending.seam_ratio:.3f}"
        )
    if report.flicker is not None:
        print(
            f"temporal flicker: mean_frame_diff={report.flicker.mean_frame_diff:.3f} "
            f"max_frame_diff={report.flicker.max_frame_diff:.3f}"
        )


def _print_verdict(verdict) -> None:
    print(f"verdict: {verdict.label} (score {verdict.score:.3f}, "
          f"{verdict.frame_count} frames)")
    print("evidence:")
    for line in verdict.explain():
        print(f"  - {line}")


def _collect_averages(directory: str, max_frames: int) -> list[dict]:
    """Analyse every analysable file in a directory; warn, skip failures."""
    rows = []
    for path in sorted(Path(directory).iterdir()):
        if not path.is_file():
            continue
        try:
            report = analyze(path, max_frames=max_frames)
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            print(f"warning: skipping {path.name}: {exc}", file=sys.stderr)
            continue
        if report.frames:
            rows.append(report.averages())
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="deepfake-detector")
    sub = parser.add_subparsers(dest="command", required=True)
    analyse = sub.add_parser("analyze", help="analyse an image, video, or image directory")
    analyse.add_argument("source", help="path to image, video file, or image directory")
    analyse.add_argument("--max-frames", type=int, default=16,
                         help="frames to sample from video/directory input (default 16)")
    analyse.add_argument("--json", action="store_true", help="emit JSON instead of text")
    classify_p = sub.add_parser("classify", help="analyse and score an image, video, or image directory")
    classify_p.add_argument("source", help="path to image, video file, or image directory")
    classify_p.add_argument("--max-frames", type=int, default=16,
                            help="frames to sample from video/directory input (default 16)")
    classify_p.add_argument("--json", action="store_true", help="emit JSON instead of text")
    classify_p.add_argument("--profile", default=None,
                            help="calibration profile JSON (default: built-in)")
    calibrate_p = sub.add_parser("calibrate", help="fit a calibration profile from labelled sample directories")
    calibrate_p.add_argument("real_dir", help="directory of real samples")
    calibrate_p.add_argument("fake_dir", help="directory of fake samples")
    calibrate_p.add_argument("--out", required=True, help="path to write the profile JSON")
    calibrate_p.add_argument("--max-frames", type=int, default=16,
                             help="frames to sample from video inputs (default 16)")
    args = parser.parse_args(argv)
    if args.command == "calibrate":
        real_rows = _collect_averages(args.real_dir, args.max_frames)
        fake_rows = _collect_averages(args.fake_dir, args.max_frames)
        if not real_rows or not fake_rows:
            print("error: need at least one analysable sample in each directory",
                  file=sys.stderr)
            return 1
        specs = fit_specs(real_rows, fake_rows)
        save_profile(specs, args.out)
        print(f"fitted profile from {len(real_rows)} real + {len(fake_rows)} fake samples -> {args.out}")
        for spec in specs:
            print(f"  {spec.key:<20} midpoint={spec.midpoint:.4g} scale={spec.scale:.4g}")
        return 0
    try:
        report = analyze(args.source, max_frames=args.max_frames)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.command == "analyze":
        if args.json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            _print_human(report)
        return 0
    # classify
    try:
        specs = load_profile(args.profile) if args.profile else None
    except (ValueError, OSError, KeyError) as exc:
        print(f"error: bad profile {args.profile}: {exc}", file=sys.stderr)
        return 1
    verdict = Classifier(specs).classify(report) if specs else Classifier().classify(report)
    if args.json:
        print(json.dumps({"verdict": verdict.to_dict(),
                          "report": report.to_dict()}, indent=2))
    else:
        _print_verdict(verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
