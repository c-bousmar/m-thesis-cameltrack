#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import cv2


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert MOTChallenge sequence frames from <seq>/img1 into an .mp4 video."
    )
    parser.add_argument(
        "sequence_dir",
        type=Path,
        help="Path to the sequence directory containing img1/ and optionally seqinfo.ini",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output .mp4 path. Default: <sequence_dir>/<sequence_name>.mp4",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="FPS to use. If omitted, tries to read frameRate from seqinfo.ini, else defaults to 30.",
    )
    parser.add_argument(
        "--codec",
        type=str,
        default="mp4v",
        help="FourCC codec for mp4 writing. Default: mp4v",
    )
    return parser.parse_args()


def read_fps_from_seqinfo(seqinfo_path: Path) -> float | None:
    if not seqinfo_path.exists():
        return None

    for line in seqinfo_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("frameRate="):
            try:
                return float(line.split("=", 1)[1])
            except ValueError:
                return None
    return None


def main() -> None:
    args = parse_args()

    sequence_dir = args.sequence_dir
    img_dir = sequence_dir / "img1"

    if not sequence_dir.exists():
        raise FileNotFoundError(f"Sequence directory not found: {sequence_dir}")
    if not img_dir.exists():
        raise FileNotFoundError(f"img1 directory not found: {img_dir}")

    frame_paths = sorted(
        [p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS]
    )

    if not frame_paths:
        raise RuntimeError(f"No image frames found in {img_dir}")

    fps = args.fps
    if fps is None:
        fps = read_fps_from_seqinfo(sequence_dir / "seqinfo.ini")
    if fps is None:
        fps = 30.0

    first_frame = cv2.imread(str(frame_paths[0]))
    if first_frame is None:
        raise RuntimeError(f"Could not read first frame: {frame_paths[0]}")

    height, width = first_frame.shape[:2]

    output_path = args.output
    if output_path is None:
        output_path = sequence_dir / f"{sequence_dir.name}.mp4"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*args.codec)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for: {output_path}")

    for frame_path in frame_paths:
        frame = cv2.imread(str(frame_path))
        if frame is None:
            print(f"Warning: skipping unreadable frame {frame_path}")
            continue

        if frame.shape[1] != width or frame.shape[0] != height:
            raise RuntimeError(
                f"Frame size mismatch in {frame_path}: "
                f"expected {(width, height)}, got {(frame.shape[1], frame.shape[0])}"
            )

        writer.write(frame)

    writer.release()
    print(f"Saved video to: {output_path}")


if __name__ == "__main__":
    main()