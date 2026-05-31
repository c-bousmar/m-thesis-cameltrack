#!/usr/bin/env python3
"""
GT->Pred correspondence timeline plot (batch mode).

Input:
  --dir PATH
    Directory containing various files.
    We process only *.csv files, EXCEPT "main.csv" which is ignored.

For each CSV:
- Generates <stem>_timeline.png next to the CSV
- Title = CSV filename

Row handling rules:
- If track is empty/NA => skip the row entirely.
- If gt is empty/NA (but track exists) => treat as a FP:
    create a new "row" appended at the bottom, named FP_X
    where X increments for each distinct FP track id (based on track).

Timeline:
- One horizontal row per gt plus FP_X rows
- Per frame: colored bar for track, blank if none
- Contiguous runs of same track are merged
- track written centered in each segment

Extras:
- FRAME_SCALE = 0.5 collapses the frame scale by half (visual compression only).
- If column label exists, frames where label contains "ID_S" are:
    (1) labeled once on the x-axis with exact frame number
    (2) drawn as vertical dotted markers going up to the impacted GT row(s)
- For each GT track row (NOT FP_X): draw a thin black baseline centered in the track.
- Default background: white
- Bar thickness: reduced by 2 (bar_height = 0.40)

Usage:
  python plot_gt_pred_timeline.py --dir /path/to/folder
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ---- Frame scale (0.5 = collapse by half) ----
FRAME_SCALE = 0.5


def _luminance(rgba) -> float:
    r, g, b = rgba[:3]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def load_and_normalize(
    csv_path: Path,
) -> Tuple[pd.DataFrame, Dict[int, str], List[int], Dict[int, Set[str]]]:
    """
    Returns:
      df_norm: columns FRAME(int), ROW_LABEL(str), track(float)
      fp_map: pred_id_int -> "FP_X"
      id_s_frames: sorted unique frames where label contains "ID_S" (if label exists)
      id_s_impacted: frame -> set(row_label) for impacted GT rows (excludes FP rows)
    """
    df = pd.read_csv(csv_path)

    required = {"frame", "gt", "track"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"[{csv_path.name}] Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["frame"] = pd.to_numeric(df["frame"], errors="raise").astype(int)
    df["gt"] = pd.to_numeric(df["gt"], errors="coerce")      # blanks -> NaN
    df["track"] = pd.to_numeric(df["track"], errors="coerce")  # blanks -> NaN

    # ---- ID_S impacted GT rows (from raw df) ----
    id_s_frames: List[int] = []
    id_s_impacted: Dict[int, Set[str]] = {}

    if "label" in df.columns:
        issue = df["label"].astype(str)
        mask = issue.str.contains("ID_S", case=False, na=False)
        if mask.any():
            id_s_frames = sorted(set(df.loc[mask, "frame"].astype(int).tolist()))
            impacted = df.loc[mask, ["frame", "gt"]].copy()
            impacted = impacted[~impacted["gt"].isna()]
            for f, gt in zip(impacted["frame"].astype(int).tolist(), impacted["gt"].tolist()):
                id_s_impacted.setdefault(int(f), set()).add(str(int(gt)))

    # Rule 1: if track is NA => skip row
    df = df[~df["track"].isna()].copy()

    # Rule 2: if gt is NA => FP rows appended at bottom, named FP_X (per distinct track)
    fp_map: Dict[int, str] = {}
    fp_counter = 0

    row_labels: List[str] = []
    for gt, pid in zip(df["gt"].tolist(), df["track"].tolist()):
        if pd.isna(gt):
            pid_int = int(pid)
            if pid_int not in fp_map:
                fp_counter += 1
                fp_map[pid_int] = f"FP_{fp_counter}"
            row_labels.append(fp_map[pid_int])
        else:
            row_labels.append(str(int(gt)))

    df["ROW_LABEL"] = row_labels
    df_norm = df[["frame", "ROW_LABEL", "track"]].copy()
    return df_norm, fp_map, id_s_frames, id_s_impacted


def _pick_one_pred(values: pd.Series) -> float:
    non_null = values.dropna()
    if non_null.empty:
        return float("nan")
    return float(non_null.iloc[0])


def build_row_frame_matrix(df_norm: pd.DataFrame) -> pd.DataFrame:
    grouped = df_norm.groupby(["ROW_LABEL", "frame"], sort=True)["track"].apply(_pick_one_pred)
    pivot = grouped.unstack("frame")

    min_f = int(df_norm["frame"].min())
    max_f = int(df_norm["frame"].max())
    all_frames = list(range(min_f, max_f + 1))
    pivot = pivot.reindex(columns=all_frames)

    def row_sort_key(label: str):
        if label.startswith("FP_"):
            try:
                return (1, int(label.split("_", 1)[1]))
            except Exception:
                return (1, 10**9)
        try:
            return (0, int(label))
        except Exception:
            return (0, 10**9)

    pivot = pivot.reindex(index=sorted(pivot.index.tolist(), key=row_sort_key))
    return pivot


def make_color_map(unique_pred_ids: List[int]):
    if len(unique_pred_ids) <= 20:
        cmap = plt.get_cmap("tab20", len(unique_pred_ids))
    else:
        cmap = plt.get_cmap("turbo", len(unique_pred_ids))
    return {pid: cmap(i) for i, pid in enumerate(unique_pred_ids)}


def iter_segments(frames: np.ndarray, preds: np.ndarray):
    cur_pid: Optional[int] = None
    start_idx: Optional[int] = None

    for i, pid in enumerate(preds):
        if np.isnan(pid):
            if cur_pid is not None:
                yield (cur_pid, int(frames[start_idx]), int(frames[i - 1]))
                cur_pid, start_idx = None, None
            continue

        pid_int = int(pid)
        if cur_pid is None:
            cur_pid, start_idx = pid_int, i
            continue

        if pid_int != cur_pid:
            yield (cur_pid, int(frames[start_idx]), int(frames[i - 1]))
            cur_pid, start_idx = pid_int, i

    if cur_pid is not None:
        yield (cur_pid, int(frames[start_idx]), int(frames[len(frames) - 1]))


def add_id_s_axis_labels(ax, id_s_frames: List[int], frames: np.ndarray):
    """Annotate unique ID_S frames on x-axis with exact frame numbers (once per frame)."""
    if not id_s_frames:
        return

    fmin, fmax = int(frames[0]), int(frames[-1])
    usable = [f for f in id_s_frames if fmin <= f <= fmax]
    if not usable:
        return

    xaxis_trans = ax.get_xaxis_transform()
    for f in usable:
        x = f * FRAME_SCALE
        ax.plot([x, x], [0.0, -0.03], transform=xaxis_trans, clip_on=False, linewidth=0.8, color="black")
        ax.text(
            x,
            -0.06,
            str(f),
            transform=xaxis_trans,
            ha="center",
            va="top",
            fontsize=8,
            rotation=90,
            clip_on=False,
            color="black",
        )


def add_id_s_vertical_markers(
    ax,
    frames: np.ndarray,
    row_labels: List[str],
    id_s_frames: List[int],
    id_s_impacted: Dict[int, Set[str]],
    y_bottom: float,
):
    """
    Draw a vertical dotted marker at each ID_S frame, extending up to the impacted GT row(s).
    If no impacted GT info for a frame, draw a short marker near the bottom only.
    """
    if not id_s_frames:
        return

    label_to_y = {lab: i for i, lab in enumerate(row_labels)}
    fmin, fmax = int(frames[0]), int(frames[-1])

    for f in id_s_frames:
        if not (fmin <= f <= fmax):
            continue
        x = f * FRAME_SCALE

        impacted_labels = sorted(id_s_impacted.get(int(f), set()))
        impacted_labels = [lab for lab in impacted_labels if lab in label_to_y and not lab.startswith("FP_")]

        if not impacted_labels:
            ax.vlines(x, y_bottom - 0.25, y_bottom, colors="black", linewidth=0.9, linestyles=":", zorder=4)
            continue

        for lab in impacted_labels:
            y_target = label_to_y[lab]
            ax.vlines(x, y_target, y_bottom, colors="black", linewidth=0.9, linestyles=":", zorder=4)


def plot_timeline(
    pivot: pd.DataFrame,
    out_path: Path,
    title: str,
    id_s_frames: List[int],
    id_s_impacted: Dict[int, Set[str]],
    dpi: int = 200,
    annotate: bool = True,
    bg: str = "white",
    bar_height: float = 0.40,
    grid: bool = True,
):
    row_labels = pivot.index.to_list()
    frames = np.array(pivot.columns.to_list(), dtype=int)

    vals = pivot.to_numpy()
    flat = vals.reshape(-1)
    flat = flat[~np.isnan(flat)]
    unique_pred_ids = sorted({int(x) for x in flat.tolist()})
    pid_to_color = make_color_map(unique_pred_ids)

    n_rows = len(row_labels)
    n_frames = len(frames)

    effective_frames = max(1, int(round(n_frames * FRAME_SCALE)))
    fig_w = max(12.0, effective_frames / 8.0)
    fig_h = max(4.0, n_rows * 0.35)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    x_left = (frames[0] - 0.5) * FRAME_SCALE
    x_right = (frames[-1] + 0.5) * FRAME_SCALE
    y_bottom = n_rows - 0.5

    for row_idx, label in enumerate(row_labels):
        y_center = row_idx
        preds = pivot.loc[label].to_numpy(dtype=float)

        # GT baseline centered (skip FP rows)
        if not label.startswith("FP_"):
            ax.hlines(y=y_center, xmin=x_left, xmax=x_right, colors="black", linewidth=0.8, zorder=1)

        for pid, f0, f1 in iter_segments(frames, preds):
            x0 = (f0 - 0.5) * FRAME_SCALE
            width = (f1 - f0 + 1) * FRAME_SCALE

            color = pid_to_color.get(pid, (0.2, 0.2, 0.2, 1.0))
            ax.add_patch(
                Rectangle(
                    (x0, y_center - bar_height / 2),
                    width,
                    bar_height,
                    facecolor=color,
                    edgecolor="none",
                    zorder=2,
                )
            )

            if annotate:
                xc = x0 + width / 2.0
                txt_color = "white" if _luminance(color) < 0.55 else "black"
                ax.text(
                    xc,
                    y_center,
                    str(pid),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=txt_color,
                    clip_on=True,
                    zorder=3,
                )

    # Vertical ID_S markers (to impacted GT rows)
    add_id_s_vertical_markers(
        ax=ax,
        frames=frames,
        row_labels=row_labels,
        id_s_frames=id_s_frames,
        id_s_impacted=id_s_impacted,
        y_bottom=y_bottom,
    )

    ax.set_xlim(x_left, x_right)
    ax.set_ylim(-0.5, n_rows - 0.5)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels)
    ax.invert_yaxis()

    ax.set_xlabel("Frame")
    ax.set_ylabel("GT ID / FP")
    ax.set_title(title)

    if grid:
        ax.grid(True, axis="x", linestyle=":", linewidth=0.6, alpha=0.7)

    # Regular x ticks
    if n_frames > 250:
        step = max(10, int(round(n_frames / 20)))
    else:
        step = max(1, int(round(n_frames / 25)))
    tick_labels = list(range(int(frames[0]), int(frames[-1]) + 1, step))
    tick_positions = [t * FRAME_SCALE for t in tick_labels]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(t) for t in tick_labels])

    # ID_S frame numbers on x-axis
    add_id_s_axis_labels(ax, id_s_frames=id_s_frames, frames=frames)

    if id_s_frames:
        fig.subplots_adjust(bottom=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def iter_csv_files(root_dir: Path) -> List[Path]:
    """
    Return CSV files under root_dir (non-recursive), excluding main.csv.
    """
    csvs = sorted([p for p in root_dir.iterdir() if p.is_file() and p.suffix.lower() == ".csv"])
    csvs = [p for p in csvs if p.name.lower() != "main.csv"]
    return csvs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="Base directory.")
    ap.add_argument("--dpi", type=int, default=200, help="Output DPI.")
    ap.add_argument("--no-annotate", action="store_true", help="Disable pred-id text labels.")
    ap.add_argument("--bg", default="white", help="Background color (default: white).")
    args = ap.parse_args()

    base_dir = Path(args.dir).expanduser().resolve()
    in_dir = base_dir / "failure_cases"
    out_dir = base_dir / "results" / "failure_cases"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_dir.exists() or not in_dir.is_dir():
        raise NotADirectoryError(in_dir)

    csv_files = iter_csv_files(in_dir)
    if not csv_files:
        print(f"No CSV files found in {in_dir} (or only main.csv).")
        return

    for csv_path in csv_files:
        try:
            df_norm, _fp_map, id_s_frames, id_s_impacted = load_and_normalize(csv_path)
            if df_norm.empty:
                print(f"[SKIP] {csv_path.name}: empty after filtering missing track.")
                continue

            pivot = build_row_frame_matrix(df_norm)

            title = csv_path.name
            out_path = out_dir / f"{csv_path.stem}_timeline.png"

            plot_timeline(
                pivot=pivot,
                out_path=out_path,
                title=title,
                id_s_frames=id_s_frames,
                id_s_impacted=id_s_impacted,
                dpi=args.dpi,
                annotate=not args.no_annotate,
                bg=args.bg,
            )

            print(f"[OK] {csv_path.name} -> {out_path}")

        except Exception as e:
            print(f"[ERR] {csv_path.name}: {e}")

if __name__ == "__main__":
    main()