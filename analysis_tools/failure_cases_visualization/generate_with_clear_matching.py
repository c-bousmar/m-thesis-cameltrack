import argparse
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
from scipy.optimize import linear_sum_assignment


# ============================================================
# IO
# ============================================================
def load_mot_txt(path, is_gt=False):
    data = defaultdict(list)

    with open(path, "r") as f:
        for line in f:
            if line.strip() == "":
                continue

            parts = line.strip().split(",")

            # Skip GT rows where class != 1
            if is_gt and len(parts) > 7:
                if int(float(parts[7])) != 1:
                    continue

            frame = int(float(parts[0]))
            obj_id = int(float(parts[1]))
            bbox = list(map(float, parts[2:6]))

            data[frame].append({"id": obj_id, "bbox": bbox})

    return data


# ============================================================
# Math
# ============================================================
def iou_matrix(gt, pred):
    if len(gt) == 0 or len(pred) == 0:
        return np.zeros((len(gt), len(pred)))

    M = np.zeros((len(gt), len(pred)))
    for i, g in enumerate(gt):
        for j, p in enumerate(pred):
            M[i, j] = iou(g["bbox"], p["bbox"])
    return M


def iou(a, b):
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[0] + a[2], b[0] + b[2])
    yB = min(a[1] + a[3], b[1] + b[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0

    areaA = a[2] * a[3]
    areaB = b[2] * b[3]
    return inter / (areaA + areaB - inter)


# ============================================================
# Core analysis
# ============================================================
def analyze_sequence(gt_file, pred_file, out_dir, iou_threshold=0.5):
    counts = Counter()
    seq_name = gt_file.stem

    gt = load_mot_txt(gt_file, is_gt=True)
    pred = load_mot_txt(pred_file, is_gt=False)

    frames = sorted(set(gt.keys()) | set(pred.keys()))

    last_pred = {}                    # last non-missing pred per GT
    pred_history = defaultdict(set)   # unique PRED_ID per GT

    total_frames = len(frames)
    total_matched = 0

    out_path = out_dir / f"{seq_name}.txt"

    with open(out_path, "w") as f:
        f.write("FRAME,GT_ID,PRED_ID,ISSUE,BBOX\n")

        for frame in frames:
            gt_objs = gt.get(frame, [])
            pred_objs = pred.get(frame, [])

            if len(gt_objs) and len(pred_objs):
                M = iou_matrix(gt_objs, pred_objs)
                cost = 1 - M
                rows, cols = linear_sum_assignment(cost)
            else:
                rows, cols = np.array([]), np.array([])

            matched_gt = set()
            matched_pred = set()

            # ---------------- MATCHED ----------------
            for r, c in zip(rows, cols):
                if M[r, c] < iou_threshold:
                    continue

                matched_gt.add(r)
                matched_pred.add(c)

                total_matched += 1

                gt_id = gt_objs[r]["id"]
                pred_id = pred_objs[c]["id"]
                bbox = gt_objs[r]["bbox"]
                bbox_str = " ".join(map(str, bbox))

                pred_history[gt_id].add(pred_id)

                prev = last_pred.get(gt_id)

                if prev is not None and pred_id != prev:
                    counts["ID_S"] += 1
                    issue = "ID_S"
                else:
                    issue = "N/A"

                f.write(f"{frame},{gt_id},{pred_id},{issue},{bbox_str}\n")

                last_pred[gt_id] = pred_id

            # ---------------- FN ----------------
            for i, g in enumerate(gt_objs):
                if i not in matched_gt:
                    bbox_str = " ".join(map(str, g["bbox"]))
                    counts["FN"] += 1
                    f.write(f"{frame},{g['id']},-1,FN,{bbox_str}\n")

            # ---------------- FP ----------------
            for j, p in enumerate(pred_objs):
                if j not in matched_pred:
                    bbox_str = " ".join(map(str, p["bbox"]))
                    counts["FP"] += 1
                    f.write(f"{frame},-1,{p['id']},FP,{bbox_str}\n")

    # ---------------- Compute ID_F from history ----------------
    total_id_f = 0
    fragmented_gt = 0
    total_pred_per_gt = 0

    for gt_id, pred_set in pred_history.items():
        n = len(pred_set)
        total_pred_per_gt += n
        if n > 1:
            fragmented_gt += 1
            total_id_f += n - 1

    counts["ID_F"] = total_id_f

    extra_metrics = {
        "total_frames": total_frames,
        "total_matched": total_matched,
        "fragmented_gt": fragmented_gt,
        "total_gt": len(pred_history),
        "total_pred_per_gt": total_pred_per_gt,
    }

    return counts, extra_metrics


# ============================================================
# Summary generation
# ============================================================
def generate_summary(gt_dir, pred_dir, out_dir, iou_threshold):
    sequences = sorted(gt_dir.glob("*.txt"))

    print(f"Analyzing {len(sequences)} sequences\n")

    global_counts = Counter()
    global_metrics = {
        "total_frames": 0,
        "total_matched": 0,
        "fragmented_gt": 0,
        "total_gt": 0,
        "total_pred_per_gt": 0,
    }

    main_file = out_dir / "main.csv"

    with open(main_file, "w") as summary:
        summary.write(
            "VIDEO,FN,FP,ID_S,ID_F,ASSOC_RATIO,"
            "PCT_FRAGMENTED,AVG_PRED_PER_GT,"
            "SWITCH_PER_1000F,INSTABILITY\n"
        )

        for gt_file in sequences:
            name = gt_file.name
            seq_name = gt_file.stem
            pred_file = pred_dir / name

            if not pred_file.exists():
                print(f"Missing prediction for {name}")
                continue

            print(f"→ {name}")

            counts, metrics = analyze_sequence(gt_file, pred_file, out_dir, iou_threshold)

            FN = counts["FN"]
            FP = counts["FP"]
            ID_S = counts["ID_S"]
            ID_F = counts["ID_F"]

            total_errors = FN + FP + ID_S
            assoc_ratio = ID_S / total_errors if total_errors > 0 else 0

            total_frames = metrics["total_frames"]
            total_matched = metrics["total_matched"]
            fragmented_gt = metrics["fragmented_gt"]
            total_gt = metrics["total_gt"]
            total_pred_per_gt = metrics["total_pred_per_gt"]

            pct_fragmented = fragmented_gt / total_gt if total_gt > 0 else 0
            avg_pred_per_gt = total_pred_per_gt / total_gt if total_gt > 0 else 0
            switch_rate_1000 = (ID_S / total_frames) * 1000 if total_frames > 0 else 0
            instability_score = ID_S / total_matched if total_matched > 0 else 0

            summary.write(
                f"{seq_name},{FN},{FP},{ID_S},{ID_F},{assoc_ratio:.4f},"
                f"{pct_fragmented:.4f},{avg_pred_per_gt:.4f},"
                f"{switch_rate_1000:.2f},{instability_score:.4f}\n"
            )

            global_counts.update(counts)
            for k in global_metrics:
                global_metrics[k] += metrics[k]

        # -------- TOTAL --------
        FN = global_counts["FN"]
        FP = global_counts["FP"]
        ID_S = global_counts["ID_S"]
        ID_F = global_counts["ID_F"]

        total_errors = FN + FP + ID_S
        assoc_ratio = ID_S / total_errors if total_errors > 0 else 0

        total_frames = global_metrics["total_frames"]
        total_matched = global_metrics["total_matched"]
        fragmented_gt = global_metrics["fragmented_gt"]
        total_gt = global_metrics["total_gt"]
        total_pred_per_gt = global_metrics["total_pred_per_gt"]

        pct_fragmented = fragmented_gt / total_gt if total_gt > 0 else 0
        avg_pred_per_gt = total_pred_per_gt / total_gt if total_gt > 0 else 0
        switch_rate_1000 = (ID_S / total_frames) * 1000 if total_frames > 0 else 0
        instability_score = ID_S / total_matched if total_matched > 0 else 0

        summary.write(
            f"TOTAL,{FN},{FP},{ID_S},{ID_F},{assoc_ratio:.4f},"
            f"{pct_fragmented:.4f},{avg_pred_per_gt:.4f},"
            f"{switch_rate_1000:.2f},{instability_score:.4f}\n"
        )

    print(f"\nSummary saved to {main_file}")


# ============================================================
# Runner
# ============================================================
def main(experiment, day, time, iou_threshold):
    base = Path("../outputs") / experiment / day / time / "eval"

    gt_dir = base / "gt" / "SportsMOT-val"
    pred_dir = base / "pred" / "SportsMOT-val" / "tracklab"
    out_dir = base / "failure_cases"

    out_dir.mkdir(parents=True, exist_ok=True)

    generate_summary(gt_dir, pred_dir, out_dir, iou_threshold)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="CAMELTrack_SportsMOT")
    parser.add_argument("--day", required=True)
    parser.add_argument("--time", required=True)
    parser.add_argument("--iou", type=float, default=0.5)

    args = parser.parse_args()

    main(args.experiment, args.day, args.time, args.iou)
