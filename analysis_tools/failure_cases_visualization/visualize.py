import argparse
from pathlib import Path
from collections import defaultdict
import configparser

import cv2
import numpy as np


# ============================================================
# Load failure file
# ============================================================
def load_failures(path):
    data = defaultdict(list)

    with open(path, "r") as f:
        next(f)  # skip header
        for line in f:
            frame, gt_id, pred_id, issue, bbox = line.strip().split(",", 4)
            x, y, w, h = map(float, bbox.split())

            data[int(frame)].append({
                "gt_id": int(gt_id),
                "pred_id": int(pred_id),
                "issue": issue,
                "bbox": [x, y, w, h]
            })

    return data


# ============================================================
# Heatmap creation
# ============================================================
def create_heatmap(seq, failures, img_dir, out_path, error_types):
    first_img = cv2.imread(str(sorted(img_dir.glob("*.jpg"))[0]))
    H, W = first_img.shape[:2]

    heat = np.zeros((H, W), dtype=np.float32)

    for frame, objs in failures.items():
        for obj in objs:
            if obj["issue"] in error_types:
                x, y, w, h = obj["bbox"]
                cx = int(x + w / 2)
                cy = int(y + h / 2)

                if 0 <= cx < W and 0 <= cy < H:
                    heat[cy, cx] += 1

    if heat.max() > 0:
        heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=25)
        heat = heat / heat.max()
        heat = (heat * 255).astype(np.uint8)
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    else:
        heat = np.zeros((H, W, 3), dtype=np.uint8)

    cv2.imwrite(str(out_path), heat)


# ============================================================
# Video creation
# ============================================================
def create_video(seq, failures, img_dir, out_path):
    frames = sorted(img_dir.glob("*.jpg"))
    if not frames:
        return

    first_img = cv2.imread(str(frames[0]))
    H, W = first_img.shape[:2]

    frameRate = 30
    seqinfo_path = img_dir.parent / "seqinfo.ini"
    if seqinfo_path.exists():
        config = configparser.ConfigParser()
        config.read(seqinfo_path)
        if "Sequence" in config and "frameRate" in config["Sequence"]:
            try:
                frameRate = int(config["Sequence"]["frameRate"])
            except ValueError:
                print(f"Invalid frameRate in {seqinfo_path}, using default 30.")
    else:
        print(f"seqinfo.ini not found for {seq}, using default 30 FPS.")

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        frameRate,
        (W, H)
    )

    cumulative_id_s = 0

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.4
    thickness = 1
    pad = 2  # internal padding (between text and rectangle)

    for frame_path in frames:
        frame_num = int(frame_path.stem)
        img = cv2.imread(str(frame_path))

        issues = failures.get(frame_num, [])

        frame_id_s = sum(1 for obj in issues if obj["issue"] == "ID_S")
        cumulative_id_s += frame_id_s

        for obj in issues:
            x, y, w, h = map(int, obj["bbox"])
            issue = obj["issue"]
            pred_id = obj["pred_id"]

            # ------------------------------------------------
            # Determine color + thickness (MATCHING previous version)
            # ------------------------------------------------
            if issue == "FP":
                color = (255, 0, 0)        # Blue
                thickness_box = 1
            elif issue == "FN":
                color = (0, 255, 0)        # Green
                thickness_box = 2
            elif issue == "ID_S":
                color = (0, 0, 255)        # Red
                thickness_box = 2
            else:
                color = (255, 255, 255)    # White
                thickness_box = 1

            # Draw bbox
            cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness_box)

            # ------------------------------------------------
            # Draw ID label (NOT for FN)
            # ------------------------------------------------
            if issue == "FN" or pred_id is None:
                continue

            label = str(pred_id)
            (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)

            # Rectangle touching bbox (no external gap)
            rect_x1 = x
            rect_y1 = y - (th + baseline + pad * 2)
            rect_x2 = x + tw + pad * 2
            rect_y2 = y

            # If rectangle goes outside top image border,
            # move it inside the bbox instead
            if rect_y1 < 0:
                rect_y1 = y
                rect_y2 = y + th + baseline + pad * 2

            # Draw filled rectangle
            cv2.rectangle(
                img,
                (rect_x1, rect_y1),
                (rect_x2, rect_y2),
                color,
                -1
            )

            # Draw text (black text, same as previous implementation)
            text_x = rect_x1 + pad
            text_y = rect_y2 - baseline - pad

            cv2.putText(
                img,
                label,
                (text_x, text_y),
                font,
                scale,
                (0, 0, 0),
                thickness,
                cv2.LINE_AA
            )

        # ----------------------------------------------------
        # Frame number + cumulative ID_S (top-left)
        # ----------------------------------------------------
        label = f"{frame_num} | ID_S total: {cumulative_id_s}"
        scale_global = 0.45
        pad_global = 4

        (tw, th), _ = cv2.getTextSize(label, font, scale_global, 1)
        cv2.rectangle(
            img,
            (0, 0),
            (tw + pad_global * 2, th + pad_global * 2),
            (0, 0, 0),
            -1
        )
        cv2.putText(
            img,
            label,
            (pad_global, th + pad_global),
            font,
            scale_global,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        writer.write(img)
    writer.release()


# ============================================================
# Process one sequence
# ============================================================

def process_sequence(seq, base, dataset_root):
    print(f"→ Processing {seq}")

    failure_file = base / "failure_cases" / f"{seq}.csv"
    if not failure_file.exists():
        print(f"Missing failure file for {seq}")
        return

    failures = load_failures(failure_file)

    img_dir = Path(dataset_root) / seq / "img1"
    if not img_dir.exists():
        print(f"Missing images for {seq}")
        return

    # Heatmaps
    create_heatmap(
        seq,
        failures,
        img_dir,
        base / "results" / "failure_cases" / f"{seq}_heatmap_det.png",
        error_types=["FP", "FN"]
    )

    create_heatmap(
        seq,
        failures,
        img_dir,
        base / "results" / "failure_cases" / f"{seq}_heatmap_assoc.png",
        error_types=["ID_S"]
    )

    # Video
    create_video(
        seq,
        failures,
        img_dir,
        base / "results" / "failure_cases" / f"{seq}_errors.mp4"
    )


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--day", required=True)
    parser.add_argument("--time", required=True)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--seq", default=None)

    args = parser.parse_args()

    base = Path("../outputs") / args.experiment / args.day / args.time / "eval"
    failure_dir = base / "failure_cases"

    if args.seq:
        sequences = [args.seq]
    else:
        sequences = [p.stem for p in failure_dir.glob("*.txt")]

    print(f"Found {len(sequences)} sequences")

    for seq in sequences:
        process_sequence(seq, base, args.dataset_root)

    print("\nDone.")


if __name__ == "__main__":
    main()
