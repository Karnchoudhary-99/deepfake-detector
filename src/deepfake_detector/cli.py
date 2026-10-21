"""Command line: ``deepfake-detector analyze <image|video|directory>``.

Prints a human-readable feature report, or ``--json`` for machines.
Exit code is 0 whenever analysis completes; the report carries
measurements, not a verdict (the calibrated classifier lands next).
"""

from __future__ import annotations

import argparse
import json
import sys

from .analyzer import analyze


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="deepfake-detector")
    sub = parser.add_subparsers(dest="command", required=True)
    analyse = sub.add_parser("analyze", help="analyse an image, video, or image directory")
    analyse.add_argument("source", help="path to image, video file, or image directory")
    analyse.add_argument("--max-frames", type=int, default=16,
                         help="frames to sample from video/directory input (default 16)")
    analyse.add_argument("--json", action="store_true", help="emit JSON instead of text")
    args = parser.parse_args(argv)
    try:
        report = analyze(args.source, max_frames=args.max_frames)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        _print_human(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
