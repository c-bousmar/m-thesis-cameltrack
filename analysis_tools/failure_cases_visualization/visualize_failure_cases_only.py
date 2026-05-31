#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Dict, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


VALID_DRAW_LABELS = {"TP", "FP", "ID_S"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Render a SportsMOT sequence video from a failure-case CSV. "
            "Tracks involved in an ID switch within the selected window are colored "
            "with matplotlib tab20; other tracks stay white with a thinner bbox."
        )
    )
    ap.add_argument("--csv", required=True, help="Path to the failure-case CSV.")
    ap.add_argument(
        "--sequence-dir",
        default="/globalsc/ucl/elen/cbousmar/datasets/SportsMOT/test",
        help="SportsMOT test directory containing sequence folders.",
    )
    ap.add_argument("--start", type=int, required=True, help="First frame to render (inclusive).")
    ap.add_argument("--end", type=int, required=True, help="Last frame to render (inclusive).")
    ap.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override FPS. If omitted, tries seqinfo.ini, else defaults to 25.",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output mp4 path. Default: same dir as CSV, named <sequence>_<start>_<end>.mp4",
    )
    ap.add_argument(
        "--codec",
        default="mp4v",
        help="FourCC codec for OpenCV VideoWriter (default: mp4v).",
    )
    return ap.parse_args()


def parse_bbox(text: str) -> Tuple[float, float, float, float]:
    vals = ast.literal_eval(text)
    if len(vals) != 4:
        raise ValueError(f"Expected 4 bbox values, got: {text}")
    x, y, w, h = map(float, vals)
    return x, y, w, h


def build_tab20_bgr_map(switched_track_ids: set[int]) -> Dict[int, Tuple[int, int, int]]:
    """
    Build a stable mapping:
        pred_id -> OpenCV BGR color
    using matplotlib's tab20 colormap.

    Only switched track IDs are assigned a color.
    """
    cmap = plt.get_cmap("tab10")
    switched_ids_sorted = sorted(int(tid) for tid in switched_track_ids)

    color_map: Dict[int, Tuple[int, int, int]] = {}
    for i, tid in enumerate(switched_ids_sorted):
        r, g, b, _ = cmap(i % 10)  # matplotlib returns RGBA in [0, 1]
        color_map[tid] = (
            int(b * 255),  # OpenCV uses BGR
            int(g * 255),
            int(r * 255),
        )
    return color_map


def read_seq_fps(seq_dir: Path) -> float | None:
    seqinfo = seq_dir / "seqinfo.ini"
    if not seqinfo.exists():
        return None
    for line in seqinfo.read_text().splitlines():
        if line.startswith("frameRate="):
            try:
                return float(line.split("=", 1)[1].strip())
            except ValueError:
                return None
    return None


def collect_frame_paths(img_dir: Path, start: int, end: int) -> Dict[int, Path]:
    paths: Dict[int, Path] = {}
    for frame in range(start, end + 1):
        p = img_dir / f"{frame:06d}.jpg"
        if p.exists():
            paths[frame] = p
    return paths


def draw_labeled_box(
    image: np.ndarray,
    bbox: Tuple[float, float, float, float],
    text: str,
    color: Tuple[int, int, int],
    thickness: int,
) -> None:
    x, y, w, h = bbox
    x1, y1 = int(round(x)), int(round(y))
    x2, y2 = int(round(x + w)), int(round(y + h))

    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)

    if not text:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    text_thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, text_thickness)

    rx1 = x1
    ry2 = max(th + baseline + 4, y1)
    ry1 = max(0, ry2 - th - baseline - 4)
    rx2 = x1 + tw + 6

    cv2.rectangle(image, (rx1, ry1), (rx2, ry2), color, -1)

    text_color = (0, 0, 0) if color == (255, 255, 255) else (255, 255, 255)

    cv2.putText(
        image,
        text,
        (x1 + 3, ry2 - baseline - 2),
        font,
        font_scale,
        text_color,
        text_thickness,
        lineType=cv2.LINE_AA,
    )


def main() -> None:
    args = parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    sequence_dir = Path(args.sequence_dir).expanduser().resolve()

    if args.start > args.end:
        raise ValueError("--start must be <= --end")
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    seq_name = csv_path.stem
    seq_dir = sequence_dir / seq_name
    img_dir = seq_dir / "img1"

    if not img_dir.exists():
        raise FileNotFoundError(f"Could not find image directory: {img_dir}")

    df = pd.read_csv(csv_path)
    needed_cols = {"frame", "track", "label", "bbox"}
    missing = needed_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")

    df = df[(df["frame"] >= args.start) & (df["frame"] <= args.end)].copy()
    if df.empty:
        raise ValueError("No rows in selected frame window.")

    df["bbox_tuple"] = df["bbox"].map(parse_bbox)

    switched_tracks = {
        int(t)
        for t in df.loc[df["label"].eq("ID_S") & df["track"].notna(), "track"].tolist()
    }
    track_colors = build_tab20_bgr_map(switched_tracks)

    frame_paths = collect_frame_paths(img_dir, args.start, args.end)
    if not frame_paths:
        raise FileNotFoundError(
            f"No frame images found in {img_dir} for range [{args.start}, {args.end}]"
        )

    first_frame = cv2.imread(str(frame_paths[min(frame_paths)]))
    if first_frame is None:
        raise RuntimeError(f"Could not read first frame image: {frame_paths[min(frame_paths)]}")
    height, width = first_frame.shape[:2]

    fps = args.fps or read_seq_fps(seq_dir) or 25.0
    out_path = (
        Path(args.out).expanduser().resolve()
        if args.out
        else csv_path.with_name(f"{seq_name}_{args.start}_{args.end}.mp4")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*args.codec),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for: {out_path}")

    frame_groups = {int(k): v for k, v in df.groupby("frame", sort=True)}

    try:
        for frame_idx in range(args.start, args.end + 1):
            frame_path = frame_paths.get(frame_idx)
            if frame_path is None:
                continue

            image = cv2.imread(str(frame_path))
            if image is None:
                continue

            frame_rows = frame_groups.get(frame_idx)
            if frame_rows is not None:
                # draw white first, then switched tracks on top
                draw_order = frame_rows.copy()
                draw_order["is_switched"] = draw_order["track"].apply(
                    lambda t: pd.notna(t) and int(t) in switched_tracks
                )
                draw_order = draw_order[draw_order["label"].isin(VALID_DRAW_LABELS)]
                draw_order = draw_order.sort_values(["is_switched", "track"], ascending=[True, True])

                for _, row in draw_order.iterrows():
                    track = row["track"]
                    bbox = row["bbox_tuple"]

                    if pd.isna(track):
                        continue
                    track_id = int(track)

                    if track_id in switched_tracks:
                        color = track_colors[track_id]
                        thickness = 2
                    else:
                        color = (255, 255, 255)
                        thickness = 1

                    draw_labeled_box(
                        image=image,
                        bbox=bbox,
                        text=str(track_id),
                        color=color,
                        thickness=thickness,
                    )

            hud = f"{seq_name} | frame {frame_idx} | window [{args.start}, {args.end}]"
            cv2.putText(
                image,
                hud,
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 0),
                2,
                lineType=cv2.LINE_AA,
            )
            writer.write(image)
    finally:
        writer.release()

    print(f"Saved: {out_path}")
    print(f"Sequence: {seq_name}")
    print(f"Switched track ids in window: {sorted(switched_tracks)}")


if __name__ == "__main__":
    main()