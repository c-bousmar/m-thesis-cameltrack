"""
Standalone script to match tracker predictions against GT using the exact same
cost function as HOTA (global_alignment_score * IoU + Hungarian algorithm).

Usage:
    python match_tracks.py --dir /path/to/root --experiment SportsMOT-test

Input layout:
    <dir>/eval/gt/<experiment>/<seq>.txt
    <dir>/eval/pred/<experiment>/tracklab/<seq>.txt

Output:
    <dir>/eval/failure_cases/<experiment>/<seq>.csv

MOT .txt format (both GT and pred):
    frame, id, x, y, w, h, conf, -1, -1, -1
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

# IoU threshold (mirrors HOTA export_alpha = 0.5)
IOU_THRESHOLD = 0.5


# ============================================================
# I/O helpers
# ============================================================

def load_mot_txt(path: Path) -> dict[int, list[dict]]:
    """
    Parse a MOT-format .txt file.
    Returns dict: frame_id -> list of {id, bbox=[x,y,w,h]}
    """
    data = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            frame = int(parts[0])
            tid   = int(parts[1])
            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            data[frame].append({"id": tid, "bbox": [x, y, w, h]})
    return data


# ============================================================
# IoU
# ============================================================

def iou_matrix(gt_bboxes: list, pred_bboxes: list) -> np.ndarray:
    """
    Compute IoU between all pairs of GT and pred bboxes [x, y, w, h].
    Returns array of shape (num_gt, num_pred).
    """
    def to_xyxy(b):
        return b[0], b[1], b[0] + b[2], b[1] + b[3]

    iou = np.zeros((len(gt_bboxes), len(pred_bboxes)), dtype=float)
    for i, gb in enumerate(gt_bboxes):
        gx1, gy1, gx2, gy2 = to_xyxy(gb)
        g_area = (gx2 - gx1) * (gy2 - gy1)
        for j, pb in enumerate(pred_bboxes):
            px1, py1, px2, py2 = to_xyxy(pb)
            p_area = (px2 - px1) * (py2 - py1)
            ix1, iy1 = max(gx1, px1), max(gy1, py1)
            ix2, iy2 = min(gx2, px2), min(gy2, py2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            union = g_area + p_area - inter
            iou[i, j] = inter / union if union > 0 else 0.0
    return iou


# ============================================================
# Per-sequence matching  (mirrors HOTA eval_sequence exactly)
# ============================================================

def match_sequence(
    gt_data:   dict[int, list[dict]],
    pred_data: dict[int, list[dict]],
) -> list[dict]:
    """
    Two-pass approach identical to HOTA:

    Pass 1 — accumulate global association scores across all frames.
      For each frame, compute a sim-IoU weighted potential match count between
      every (gt_id, tracker_id) pair. Then derive global_alignment_score as
      a Jaccard-style score over the full sequence.

    Pass 2 — per-frame matching using global_alignment_score * IoU as cost.
      Same Hungarian assignment as HOTA. Detections are labelled
      TP, FP, FN, or ID_S.

    Returns a flat list of row dicts: {frame, gt, track, label, bbox}
    """
    all_frames = sorted(set(gt_data) | set(pred_data))

    # Collect all unique GT and tracker IDs to build index mappings
    all_gt_ids   = sorted({g["id"] for dets in gt_data.values()   for g in dets})
    all_pred_ids = sorted({p["id"] for dets in pred_data.values() for p in dets})

    if not all_gt_ids or not all_pred_ids:
        # Degenerate case: no GT or no preds in the whole sequence
        rows = []
        for frame in all_frames:
            for g in gt_data.get(frame, []):
                rows.append({"frame": frame, "gt": g["id"],   "track": None,    "label": "FN", "bbox": g["bbox"]})
            for p in pred_data.get(frame, []):
                rows.append({"frame": frame, "gt": None,      "track": p["id"], "label": "FP", "bbox": p["bbox"]})
        return rows

    gt_id_to_idx   = {gid: i for i, gid in enumerate(all_gt_ids)}
    pred_id_to_idx = {pid: i for i, pid in enumerate(all_pred_ids)}
    num_gt_ids   = len(all_gt_ids)
    num_pred_ids = len(all_pred_ids)

    # ── Pass 1: accumulate global association counts ─────────────────────────
    # Mirrors HOTA's first timestep loop exactly.
    #
    # potential_matches_count[i, j] accumulates a similarity-weighted count of
    # how often gt_id i and tracker_id j were co-present and IoU-compatible.
    #
    # gt_id_count[i]   = total frames where gt_id i appeared
    # pred_id_count[j] = total frames where tracker_id j appeared

    potential_matches_count = np.zeros((num_gt_ids, num_pred_ids), dtype=float)
    gt_id_count             = np.zeros((num_gt_ids, 1),            dtype=float)
    pred_id_count           = np.zeros((1, num_pred_ids),          dtype=float)

    for frame in all_frames:
        gt_dets   = gt_data.get(frame, [])
        pred_dets = pred_data.get(frame, [])
        if not gt_dets or not pred_dets:
            # still count appearances even when one side is empty
            for g in gt_dets:
                gt_id_count[gt_id_to_idx[g["id"]], 0] += 1
            for p in pred_dets:
                pred_id_count[0, pred_id_to_idx[p["id"]]] += 1
            continue

        gt_idxs   = np.array([gt_id_to_idx[g["id"]]   for g in gt_dets],   dtype=int)
        pred_idxs = np.array([pred_id_to_idx[p["id"]] for p in pred_dets], dtype=int)

        similarity = iou_matrix([g["bbox"] for g in gt_dets],
                                [p["bbox"] for p in pred_dets])  # (num_gt_t, num_pred_t)

        # sim_iou: normalised similarity (same formula as HOTA)
        sim_iou_denom = (similarity.sum(0)[np.newaxis, :]
                         + similarity.sum(1)[:, np.newaxis]
                         - similarity)
        sim_iou = np.zeros_like(similarity)
        mask = sim_iou_denom > np.finfo(float).eps
        sim_iou[mask] = similarity[mask] / sim_iou_denom[mask]

        potential_matches_count[gt_idxs[:, np.newaxis],
                                pred_idxs[np.newaxis, :]] += sim_iou

        gt_id_count[gt_idxs, 0]     += 1
        pred_id_count[0, pred_idxs] += 1

    # global_alignment_score: Jaccard over the full sequence (HOTA formula)
    global_alignment_score = potential_matches_count / np.maximum(
        1e-10,
        gt_id_count + pred_id_count - potential_matches_count,
    )

    # ── Pass 2: per-frame matching ────────────────────────────────────────────
    prev_tracker_for_gt: dict[int, int] = {}   # gt_id -> last matched tracker id
    rows = []

    for frame in all_frames:
        gt_dets   = gt_data.get(frame, [])
        pred_dets = pred_data.get(frame, [])

        # trivial cases
        if not gt_dets:
            for p in pred_dets:
                rows.append({"frame": frame, "gt": None, "track": p["id"],
                             "label": "FP", "bbox": p["bbox"]})
            continue
        if not pred_dets:
            for g in gt_dets:
                rows.append({"frame": frame, "gt": g["id"], "track": None,
                             "label": "FN", "bbox": g["bbox"]})
            continue

        gt_idxs   = np.array([gt_id_to_idx[g["id"]]   for g in gt_dets],   dtype=int)
        pred_idxs = np.array([pred_id_to_idx[p["id"]] for p in pred_dets], dtype=int)

        similarity = iou_matrix([g["bbox"] for g in gt_dets],
                                [p["bbox"] for p in pred_dets])

        # HOTA cost matrix: global_alignment_score * IoU
        score_mat = (global_alignment_score[gt_idxs[:, np.newaxis],
                                            pred_idxs[np.newaxis, :]]
                     * similarity)

        match_rows, match_cols = linear_sum_assignment(-score_mat)

        # Keep only pairs above the IoU threshold
        valid = similarity[match_rows, match_cols] >= IOU_THRESHOLD - np.finfo(float).eps
        match_rows = match_rows[valid]
        match_cols = match_cols[valid]

        matched_gt_idx   = set(match_rows.tolist())
        matched_pred_idx = set(match_cols.tolist())

        # TP / ID_S
        for r, c in zip(match_rows, match_cols):
            gt_id   = gt_dets[r]["id"]
            pred_id = pred_dets[c]["id"]
            prev    = prev_tracker_for_gt.get(gt_id)
            label   = "ID_S" if (prev is not None and prev != pred_id) else "TP"
            prev_tracker_for_gt[gt_id] = pred_id
            rows.append({"frame": frame, "gt": gt_id, "track": pred_id,
                         "label": label, "bbox": gt_dets[r]["bbox"]})

        # FN
        for i, g in enumerate(gt_dets):
            if i not in matched_gt_idx:
                rows.append({"frame": frame, "gt": g["id"], "track": None,
                             "label": "FN", "bbox": g["bbox"]})

        # FP
        for j, p in enumerate(pred_dets):
            if j not in matched_pred_idx:
                rows.append({"frame": frame, "gt": None, "track": p["id"],
                             "label": "FP", "bbox": p["bbox"]})

    return rows


# ============================================================
# CSV writer
# ============================================================

def write_csv(rows: list[dict], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "gt", "track", "label", "bbox"])
        for r in rows:
            writer.writerow([
                r["frame"],
                r["gt"]    if r["gt"]    is not None else "",
                r["track"] if r["track"] is not None else "",
                r["label"],
                r["bbox"],
            ])


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Match tracker predictions against GT (exact HOTA-style matching)."
    )
    parser.add_argument("--dir",        required=True, help="Root directory")
    parser.add_argument("--experiment", required=True, help="Experiment name (e.g. SportsMOT-test)")
    args = parser.parse_args()

    root       = Path(args.dir)
    experiment = args.experiment

    gt_dir   = root / "eval" / "gt"   / experiment
    pred_dir = root / "eval" / "pred" / experiment / "tracklab"
    out_dir  = root / "eval" / "failure_cases"

    if not gt_dir.exists():
        raise FileNotFoundError(f"GT directory not found: {gt_dir}")
    if not pred_dir.exists():
        raise FileNotFoundError(f"Pred directory not found: {pred_dir}")

    gt_files = sorted(gt_dir.glob("*.txt"))
    if not gt_files:
        raise FileNotFoundError(f"No GT .txt files found in {gt_dir}")

    for gt_path in gt_files:
        seq       = gt_path.stem
        pred_path = pred_dir / gt_path.name

        if not pred_path.exists():
            print(f"[SKIP] No pred file for {seq}")
            continue

        print(f"[{seq}] Loading ...")
        gt_data   = load_mot_txt(gt_path)
        pred_data = load_mot_txt(pred_path)

        print(f"[{seq}] Matching (pass 1: global alignment, pass 2: per-frame) ...")
        rows = match_sequence(gt_data, pred_data)

        out_path = out_dir / f"{seq}.csv"
        write_csv(rows, out_path)
        print(f"[{seq}] {len(rows)} rows → {out_path}")


if __name__ == "__main__":
    main()