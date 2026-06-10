import io
import logging
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import trackeval
from tabulate import tabulate
from tracklab.pipeline import Evaluator as EvaluatorBase

import configparser
from collections import defaultdict
import cv2
import csv

log = logging.getLogger(__name__)


class TrackEvalEvaluator(EvaluatorBase):
    """
    Evaluator using the TrackEval library (https://github.com/JonathonLuiten/TrackEval).
    Save on disk the tracking predictions and ground truth in MOT Challenge format and run the evaluation by calling TrackEval.
    """
    def __init__(self, cfg, eval_set, show_progressbar, dataset_path, tracking_dataset, *args, **kwargs):
        self.cfg = cfg
        self.tracking_dataset = tracking_dataset
        self.eval_set = eval_set
        self.trackeval_dataset_name = type(self.tracking_dataset).__name__
        self.trackeval_dataset_class = getattr(trackeval.datasets, cfg.dataset.dataset_class)
        self.show_progressbar = show_progressbar
        self.dataset_path = dataset_path

    def run(self, tracker_state):
        log.info("Starting evaluation using TrackEval library (https://github.com/JonathonLuiten/TrackEval)")

        tracker_name = 'tracklab'
        save_classes = self.trackeval_dataset_class.__name__ != 'MotChallenge2DBox'

        # Save predictions
        pred_save_path = Path(self.cfg.dataset.TRACKERS_FOLDER) / f"{self.trackeval_dataset_name}-{self.eval_set}" / tracker_name
        self.tracking_dataset.save_for_eval(
            tracker_state.detections_pred,
            tracker_state.image_metadatas,
            tracker_state.video_metadatas,
            pred_save_path,
            self.cfg.bbox_column_for_eval,
            save_classes,  # do not use classes for MOTChallenge2DBox
            is_ground_truth=False,
        )

        log.info(
            f"Tracking predictions saved in {self.trackeval_dataset_name} format in {pred_save_path}")

        if tracker_state.detections_gt is None or len(tracker_state.detections_gt) == 0:
            log.warning(
                f"Stopping evaluation because the current split ({self.eval_set}) has no ground truth detections.")
            return

        # Save ground truth
        gt_save_path = Path(self.cfg.dataset.GT_FOLDER) / f"{self.trackeval_dataset_name}-{self.eval_set}"
        if self.cfg.save_gt:
            self.tracking_dataset.save_for_eval(
                tracker_state.detections_gt,
                tracker_state.image_metadatas,
                tracker_state.video_metadatas,
                gt_save_path,
                self.cfg.bbox_column_for_eval,
                True,
                is_ground_truth=True
            )

        log.info(
            f"Tracking ground truth saved in {self.trackeval_dataset_name} format in {gt_save_path}")

        # Build TrackEval dataset
        dataset_config = self.trackeval_dataset_class.get_default_dataset_config()
        dataset_config['SEQ_INFO'] = tracker_state.video_metadatas.set_index('name')['nframes'].to_dict()
        dataset_config['BENCHMARK'] = self.trackeval_dataset_name  # required for trackeval.datasets.MotChallenge2DBox
        for key, value in self.cfg.dataset.items():
            dataset_config[key] = value

        if not self.cfg.save_gt:
            dataset_config['GT_FOLDER'] = self.dataset_path  # Location of GT data
            dataset_config['GT_LOC_FORMAT'] = '{gt_folder}/{seq}/Labels-GameState.json'  # '{gt_folder}/{seq}/gt/gt.txt'
        dataset = self.trackeval_dataset_class(dataset_config)

        # Build metrics
        metrics_config = {'METRICS': set(self.cfg.metrics), 'PRINT_CONFIG': False, 'THRESHOLD': 0.5}
        metrics_list = []
        for metric_name in self.cfg.metrics:
            try:
                metric = getattr(trackeval.metrics, metric_name)
                metrics_list.append(metric(metrics_config))
            except AttributeError:
                log.warning(f'Skipping evaluation for unknown metric: {metric_name}')

        # Build evaluator
        eval_config = trackeval.Evaluator.get_default_eval_config()
        for key, value in self.cfg.eval.items():
            if key == "NUM_PARALLEL_CORES":
                value = max(1, int(value))
            eval_config[key] = value
        evaluator = trackeval.Evaluator(eval_config)

        # Run evaluation
        with redirect_stdout(io.StringIO()) as stream:
            output_res, output_msg = evaluator.evaluate(
                [dataset],
                metrics_list,
                show_progressbar=self.show_progressbar
            )
        printed_results = stream.getvalue()
        log.info(printed_results)

        # Find CLEAR metric object from metrics_list
        dataset_name = dataset.get_name()
        clear_metric = None
        for m in metrics_list:
            if m.__class__.__name__ == "CLEAR":
                clear_metric = m
                break

        if clear_metric is None:
            log.warning("CLEAR metric not found — cannot extract failures.")
            return

        # Log results
        results = output_res[dataset.get_name()][tracker_name]
        dataset_name = dataset.get_name()
        all_sequences = extract_per_sequence_failures(
            output_res,
            dataset,
            dataset_name,
            tracker_name
        )
        output_dir = Path(self.cfg.dataset.FAILURE_CASE_FOLDER)
        write_sequence_csvs(all_sequences, output_dir)
        log.info("Per-sequence failure cases extracted and saved as CSVs in " + str(output_dir))

        output_dir = Path(self.cfg.dataset.OUTPUT_FOLDER) / "failure_cases"
        write_main_csv(all_sequences, output_dir)
        log.info("Main failure cases evaluation results saved as CSV in " + str(output_dir / "main.csv"))
        
        generate_failure_artifacts_from_csv_dir(
            failure_csv_dir=Path(self.cfg.dataset.FAILURE_CASE_FOLDER),
            dataset_root=Path(self.dataset_path) / self.cfg.dataset.SPLIT_TO_EVAL,
            out_dir=output_dir,
            seq=None,                  # or a specific seq string
        )
        log.info("Failure case's artifacts (heatmaps + videos) generated and saved in " + str(output_dir))

        # if the dataset has the process_trackeval_results method, use it to process the results
        if hasattr(self.tracking_dataset, 'process_trackeval_results'):
            self.tracking_dataset.process_trackeval_results(results, dataset_config, eval_config)

def _print_results(
    res_combined,
    res_by_video=None,
    scale_factor=1.0,
    title="",
    print_by_video=False,
):
    headers = res_combined.keys()
    data = [
        format_metric(name, res_combined[name], scale_factor)
        for name in headers
    ]
    log.info(f"{title}\n" + tabulate([data], headers=headers, tablefmt="plain"))
    if print_by_video and res_by_video:
        data = []
        for video_name, res in res_by_video.items():
            video_data = [video_name] + [
                format_metric(name, res[name], scale_factor)
                for name in headers
            ]
            data.append(video_data)
        headers = ["video"] + list(headers)
        log.info(
            f"{title} by videos\n"
            + tabulate(data, headers=headers, tablefmt="plain")
        )


def format_metric(metric_name, metric_value, scale_factor):
    if (
        "TP" in metric_name
        or "FN" in metric_name
        or "FP" in metric_name
        or "TN" in metric_name
    ):
        if metric_name == "MOTP":
            return np.around(metric_value * scale_factor, 3)
        return int(metric_value)
    else:
        return np.around(metric_value * scale_factor, 3)


# ============================================================
# Failure case per sequence per frame extraction (CSV) from TrackEval results
# ============================================================
def extract_per_sequence_failures(output_res, dataset, dataset_name, tracker_name):
    results = output_res[dataset_name][tracker_name]
    all_sequences = {}

    global_pred_offset = 1

    for seq, seq_res in results.items():

        if seq in ("COMBINED_SEQ", "SUMMARIES"):
            continue

        for cls, cls_res in seq_res.items():

            if "CLEAR" not in cls_res:
                continue

            clear_res = cls_res["CLEAR"]
            frame_results = clear_res.get("FRAME_RESULTS", None)
            if frame_results is None:
                continue

            rows = []
            last_pred_by_gt = {}
            seq_pred_ids = set()

            for t, fr in enumerate(frame_results):
                frame_id = fr["FrameId"]
                gt_ids = fr["GT_IDs"]
                trk_ids = fr["Tracker_IDs"]
                gt_bboxes = fr["gt_bboxes"]
                trk_bboxes = fr["tracker_bboxes"]
                match_rows = fr["match_rows"]
                match_cols = fr["match_cols"]
                is_idsw = fr["is_idsw"]

                # ---- TP / ID_S (Using TrackEval flags) ----
                for k, (r, c) in enumerate(zip(match_rows, match_cols)):
                    issue = "ID_S" if is_idsw[k] else "TP"
                    
                    gt_id = gt_ids[r]
                    local_pred = trk_ids[c]
                    global_pred = (local_pred + global_pred_offset if local_pred is not None else None)
                    bbox = gt_bboxes[r]

                    rows.append([frame_id, gt_id, global_pred, issue, bbox])

                    # Maintain the set of IDs seen in this sequence
                    if local_pred is not None:
                        seq_pred_ids.add(local_pred)

                # ---- FN ----
                for r in fr["unmatched_gt_idx"]:
                    bbox = gt_bboxes[r]
                    rows.append([frame_id, gt_ids[r], None, "FN", bbox])

                # ---- FP ----
                for c in fr["unmatched_tracker_idx"]:
                    bbox = trk_bboxes[c]
                    local_pred = trk_ids[c]
                    global_pred = (local_pred + global_pred_offset if local_pred is not None else None)
                    
                    rows.append([frame_id, None, global_pred, "FP", bbox])
                    
                    if local_pred is not None:
                        seq_pred_ids.add(local_pred)

            all_sequences[seq] = rows

            if len(seq_pred_ids) > 0:
                global_pred_offset += max(seq_pred_ids) + 1

            break

    return all_sequences


def write_sequence_csvs(all_sequences, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for seq, rows in all_sequences.items():

        out_path = output_dir / f"{seq}.csv"

        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["FRAME", "GT_ID", "PRED_ID", "ISSUE", "BBOX"])
            writer.writerows(rows)

def write_main_csv(all_sequences, output_dir):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / "main.csv"

    header = [
        "VIDEO",
        "FN",
        "FP",
        "ID_S",
        "ID_F",
        "AVG_ID_F_PER_TRACK",
        "ASSOC_RATIO",
    ]

    total_FN = 0
    total_FP = 0
    total_IDS = 0
    total_IDF = 0
    total_GT_tracks = 0

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for seq, rows in all_sequences.items():

            FN = 0
            FP = 0
            ID_S = 0
            gt_to_preds = defaultdict(set)

            for frame, gt_id, pred_id, issue, _ in rows:

                if issue == "FN":
                    FN += 1
                elif issue == "FP":
                    FP += 1
                elif issue == "ID_S":
                    ID_S += 1

                # Build GT → predicted ID mapping
                if gt_id is not None and pred_id is not None:
                    if issue in ("TP", "ID_S"):
                        gt_to_preds[gt_id].add(pred_id)

            # ---- Identity Fragmentation ----
            ID_F = 0
            for pred_set in gt_to_preds.values():
                if len(pred_set) > 1:
                    ID_F += (len(pred_set) - 1)

            n_tracks = len(gt_to_preds)

            AVG_ID_F_PER_TRACK = (
                ID_F / n_tracks if n_tracks > 0 else 0
            )

            total_problems = FN + FP + ID_S
            ASSOC_RATIO = (
                ID_S / total_problems if total_problems > 0 else 0
            )

            writer.writerow([
                seq,
                FN,
                FP,
                ID_S,
                ID_F,
                round(AVG_ID_F_PER_TRACK, 4),
                round(ASSOC_RATIO, 4),
            ])

            # ---- accumulate totals ----
            total_FN += FN
            total_FP += FP
            total_IDS += ID_S
            total_IDF += ID_F
            total_GT_tracks += n_tracks

        # ---- TOTAL ROW ----

        total_problems = total_FN + total_FP + total_IDS

        total_ASSOC_RATIO = (
            total_IDS / total_problems if total_problems > 0 else 0
        )

        total_AVG_IDF = (
            total_IDF / total_GT_tracks if total_GT_tracks > 0 else 0
        )

        writer.writerow([
            "TOTAL",
            total_FN,
            total_FP,
            total_IDS,
            total_IDF,
            round(total_AVG_IDF, 4),
            round(total_ASSOC_RATIO, 4),
        ])


# ============================================================
# Failure cases -> artifacts (video + heatmaps)
# ============================================================
def _parse_bbox_field(bbox_str: str):
    """
    Robust bbox parsing for CSV field.
    Accepts formats like:
      - "[x, y, w, h]"
      - "x y w h"
      - "x,y,w,h"
    Returns [x,y,w,h] as floats.
    """
    s = bbox_str.strip()

    # Strip brackets if list-like
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1].strip()

    # Try comma-separated first, else whitespace
    if "," in s:
        parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    else:
        parts = [p for p in s.split() if p != ""]

    if len(parts) != 4:
        raise ValueError(f"Invalid BBOX field: {bbox_str!r}")

    return list(map(float, parts))


def load_failures_from_csv(path: Path):
    """
    Reads per-sequence CSV with header:
      FRAME,GT_ID,PRED_ID,ISSUE,BBOX

    Returns:
      failures: dict[int frame] -> list[dict{gt_id,pred_id,issue,bbox}]
    """
    data = defaultdict(list)

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        # Expect columns: FRAME, GT_ID, PRED_ID, ISSUE, BBOX
        for row in reader:
            frame = int(row["FRAME"])
            issue = row["ISSUE"].strip()

            gt_raw = row.get("GT_ID", "").strip()
            pr_raw = row.get("PRED_ID", "").strip()

            gt_id = None if gt_raw in ("", "None", "nan") else int(float(gt_raw))
            pred_id = None if pr_raw in ("", "None", "nan") else int(float(pr_raw))

            bbox = _parse_bbox_field(row["BBOX"])

            data[frame].append(
                {"gt_id": gt_id, "pred_id": pred_id, "issue": issue, "bbox": bbox}
            )
    return data


def create_heatmap(seq, failures, img_dir: Path, out_path: Path, error_types):
    frames = sorted(img_dir.glob("*.jpg"))
    if not frames:
        return

    first_img = cv2.imread(str(frames[0]))
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

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), heat)


def create_video(seq, failures, img_dir: Path, out_path: Path):
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
                pass

    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        frameRate,
        (W, H),
    )

    cumulative_id_s = 0

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.4
    thickness = 1
    pad = 2  # small internal padding between text and rectangle

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

            # ----- Choose color -----
            if issue == "FP":          # Blue
                color = (255, 0, 0)
                thickness_box = 1
            elif issue == "FN":        # Green
                color = (0, 255, 0)
                thickness_box = 2
            elif issue == "ID_S":      # Red
                color = (0, 0, 255)
                thickness_box = 2
            else:                      # TP / other
                color = (255, 255, 255)
                thickness_box = 1

            # Draw bbox
            cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness_box)

            # ----- Draw ID label (NOT for FN) -----
            if issue != "FN" and pred_id is not None:

                label = str(pred_id)
                (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)

                # Rectangle touches bbox (no external padding)
                rect_x1 = x
                rect_y1 = y - (th + baseline + pad * 2)
                rect_x2 = x + tw + pad * 2
                rect_y2 = y

                # If rectangle goes above image, move it inside bbox
                if rect_y1 < 0:
                    rect_y1 = y
                    rect_y2 = y + th + baseline + pad * 2

                # Draw filled rectangle
                cv2.rectangle(
                    img,
                    (rect_x1, rect_y1),
                    (rect_x2, rect_y2),
                    color,
                    -1,
                )

                # Draw text inside rectangle (small internal padding)
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
                    cv2.LINE_AA,
                )

        # ----- Frame label (global counter top-left) -----
        global_label = f"{frame_num} | ID_S total: {cumulative_id_s}"
        scale_global = 0.45
        pad_global = 4

        (tw, th), _ = cv2.getTextSize(global_label, font, scale_global, 1)
        cv2.rectangle(img, (0, 0), (tw + pad_global * 2, th + pad_global * 2), (0, 0, 0), -1)
        cv2.putText(
            img,
            global_label,
            (pad_global, th + pad_global),
            font,
            scale_global,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        writer.write(img)
    writer.release()


def generate_failure_artifacts_from_csv_dir(
    failure_csv_dir: Path,
    dataset_root: Path,
    out_dir: Path = None,
    seq: str = None,
):
    """
    Reads per-sequence CSVs from failure_csv_dir and creates:
      - {seq}_heatmap_det.png (FP+FN)
      - {seq}_heatmap_assoc.png (ID_S)
      - {seq}_errors.mp4

    Assumes image frames are in:
      {dataset_root}/{seq}/img1/*.jpg
    """
    failure_csv_dir = Path(failure_csv_dir)
    dataset_root = Path(dataset_root)
    out_dir = Path(out_dir) if out_dir is not None else failure_csv_dir

    if seq:
        sequences = [seq]
    else:
        sequences = [p.stem for p in failure_csv_dir.glob("*.csv") if p.name != "main.csv"]

    for s in sequences:
        csv_path = failure_csv_dir / f"{s}.csv"
        if not csv_path.exists():
            continue

        img_dir = dataset_root / s / "img1"
        if not img_dir.exists():
            log.warning(f"Missing images for {s}: {img_dir}")
            continue

        failures = load_failures_from_csv(csv_path)

        create_heatmap(s, failures, img_dir, out_dir / f"{s}_heatmap_det.png", error_types=["FP", "FN"])
        create_heatmap(s, failures, img_dir, out_dir / f"{s}_heatmap_assoc.png", error_types=["ID_S"])
        create_video(s, failures, img_dir, out_dir / f"{s}_errors.mp4")
