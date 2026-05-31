#!/usr/bin/env python3
from __future__ import annotations

import argparse
import configparser
import os
import shutil
from pathlib import Path
from typing import Dict, Tuple, List, Set

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# HELPERS
# ============================================================

def build_frame_start_end_dict(dataset_folder: str) -> Dict[str, Tuple[int, int]]:
    frame_start_end_dict: Dict[str, Tuple[int, int]] = {}
    start_frame = 1

    seq_dirs = sorted([
        d for d in os.listdir(dataset_folder)
        if os.path.isdir(os.path.join(dataset_folder, d))
    ])

    for seq_name in seq_dirs:
        seq_dir = os.path.join(dataset_folder, seq_name)
        seqinfo_file = os.path.join(seq_dir, "seqinfo.ini")

        if not os.path.exists(seqinfo_file):
            continue

        config = configparser.ConfigParser()
        config.optionxform = str
        config.read(seqinfo_file)

        seq_length = int(config["Sequence"]["seqLength"])
        end_frame = start_frame + seq_length - 2

        frame_start_end_dict[seq_name] = (start_frame, end_frame)
        start_frame = end_frame + 1

    return frame_start_end_dict


def build_id_offset_dict(
    ids_file: str,
    embeddings_file: str,
    frame_start_end_dict: Dict[str, Tuple[int, int]],
) -> Dict[str, int]:
    """
    For each video, compute the ID offset = max ID seen in all preceding videos.
    The local ID for an entry in the current video is: global_id - offset.
    """
    sorted_seqs = sorted(
        [(s, e, name) for name, (s, e) in frame_start_end_dict.items()],
        key=lambda x: x[0],
    )

    def frame_to_video(frame_id: int) -> str | None:
        for s, e, name in sorted_seqs:
            if s <= frame_id <= e:
                return name
        return None

    max_id_per_video: Dict[str, int] = {}

    with open(embeddings_file, "r") as f_emb, open(ids_file, "r") as f_ids:
        for line_emb, line_id in zip(f_emb, f_ids):
            line_emb = line_emb.strip()
            line_id = line_id.strip()
            if not line_emb or not line_id:
                continue

            parts = line_emb.split()
            frame_id = int(parts[0])
            video = frame_to_video(frame_id)
            if video is None:
                continue

            obj_id = int(line_id)
            if video not in max_id_per_video or obj_id > max_id_per_video[video]:
                max_id_per_video[video] = obj_id

    id_offset_dict: Dict[str, int] = {}
    running_max = 0

    for _, _, name in sorted_seqs:
        id_offset_dict[name] = running_max
        if name in max_id_per_video:
            running_max = max(running_max, max_id_per_video[name] + 1)

    return id_offset_dict


def load_embeddings_for_frames(
    embeddings_file: str,
    ids_file: str,
    video_name: str,
    frame_start_end_dict: Dict[str, Tuple[int, int]],
    id_offset_dict: Dict[str, int],
    frames_of_interest: List[int],
    obj_types: Tuple[str, ...] = ("T", "D"),
) -> Dict[int, Dict[Tuple[str, int], np.ndarray]]:
    """
    Returns:
        data[local_frame][(obj_type, local_id)] = embedding
    """
    if video_name not in frame_start_end_dict:
        raise ValueError(f"Video '{video_name}' not found in dataset folder.")

    seq_frame_start, seq_frame_end = frame_start_end_dict[video_name]
    id_offset = id_offset_dict[video_name]
    frames_set = set(frames_of_interest)

    data: Dict[int, Dict[Tuple[str, int], np.ndarray]] = {f: {} for f in frames_of_interest}

    with open(embeddings_file, "r") as f_emb, open(ids_file, "r") as f_ids:
        for line_emb, line_id in zip(f_emb, f_ids):
            line_emb = line_emb.strip()
            line_id = line_id.strip()

            if not line_emb or not line_id:
                continue

            parts = line_emb.split()
            frame_id = int(parts[0])
            obj_type = parts[1]

            if obj_type not in obj_types:
                continue

            if not (seq_frame_start <= frame_id <= seq_frame_end):
                continue

            local_frame = frame_id - seq_frame_start + 2
            if local_frame not in frames_set:
                continue

            values = [float(x) for x in parts[2:] if x.lower() != "nan"]
            if not values:
                continue

            emb = np.array(values, dtype=np.float32)
            local_id = int(line_id) - id_offset

            data[local_frame][(obj_type, local_id)] = emb

    return data


def load_track_and_detection_presence(
    embeddings_file: str,
    ids_file: str,
    video_name: str,
    frame_start_end_dict: Dict[str, Tuple[int, int]],
    id_offset_dict: Dict[str, int],
    frames_of_interest: List[int],
    track_ids: List[int],
) -> Tuple[Dict[int, Set[int]], Dict[int, Set[int]]]:
    """
    Returns:
      - track_present_frames[tid] = set of local frames where T<tid> exists
      - det_present_frames[tid]   = set of local frames where D<tid> exists
    """
    if video_name not in frame_start_end_dict:
        raise ValueError(f"Video '{video_name}' not found in dataset folder.")

    seq_frame_start, seq_frame_end = frame_start_end_dict[video_name]
    id_offset = id_offset_dict[video_name]
    frames_set = set(frames_of_interest)
    track_ids_set = set(track_ids)

    track_present_frames: Dict[int, Set[int]] = {tid: set() for tid in track_ids}
    det_present_frames: Dict[int, Set[int]] = {tid: set() for tid in track_ids}

    with open(embeddings_file, "r") as f_emb, open(ids_file, "r") as f_ids:
        for line_emb, line_id in zip(f_emb, f_ids):
            line_emb = line_emb.strip()
            line_id = line_id.strip()

            if not line_emb or not line_id:
                continue

            parts = line_emb.split()
            frame_id = int(parts[0])
            obj_type = parts[1]

            if obj_type not in ("T", "D"):
                continue

            if not (seq_frame_start <= frame_id <= seq_frame_end):
                continue

            local_frame = frame_id - seq_frame_start + 2
            if local_frame not in frames_set:
                continue

            local_id = int(line_id) - id_offset
            if local_id not in track_ids_set:
                continue

            if obj_type == "T":
                track_present_frames[local_id].add(local_frame)
            else:
                det_present_frames[local_id].add(local_frame)

    return track_present_frames, det_present_frames


def build_occlusion_mask(
    frames: List[int],
    track_present: Set[int],
    det_present: Set[int],
) -> Dict[int, bool]:
    return {
        f: ((f in track_present) and (f not in det_present))
        for f in frames
    }


def find_first_reappearance_frame(
    frames: List[int],
    track_present: Set[int],
    det_present: Set[int],
) -> int | None:
    """
    Returns the first frame where an occlusion ends, i.e. the first frame with
    T+D association immediately after an occluded frame.
    """
    prev_was_occluded = False

    for f in frames:
        is_occluded = (f in track_present) and (f not in det_present)
        has_association = (f in track_present) and (f in det_present)

        if prev_was_occluded and has_association:
            return f

        prev_was_occluded = is_occluded

    return None


def mask_to_regions(
    frames: List[int],
    mask: Dict[int, bool],
) -> List[Tuple[int, int]]:
    regions: List[Tuple[int, int]] = []
    current_start = None
    prev_frame = None

    for f in frames:
        if mask.get(f, False):
            if current_start is None:
                current_start = f
            prev_frame = f
        else:
            if current_start is not None and prev_frame is not None:
                regions.append((current_start, prev_frame))
                current_start = None
                prev_frame = None

    if current_start is not None and prev_frame is not None:
        regions.append((current_start, prev_frame))

    return regions


def compute_identity_aware_occlusion_regions(
    frames: List[int],
    gt1_track_present: Set[int],
    gt1_det_present: Set[int],
    gt2_track_present: Set[int],
    gt2_det_present: Set[int],
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], int | None]:
    """
    Identity-aware occlusion display.

    Raw IDs:
      - before switch: GT1 -> raw gt1, GT2 -> raw gt2
      - from switch frame onward: GT1 -> raw gt2, GT2 -> raw gt1

    The switch frame is the earliest reappearance frame of either raw track.
    """
    raw_gt1_mask = build_occlusion_mask(frames, gt1_track_present, gt1_det_present)
    raw_gt2_mask = build_occlusion_mask(frames, gt2_track_present, gt2_det_present)

    gt1_reappearance = find_first_reappearance_frame(frames, gt1_track_present, gt1_det_present)
    gt2_reappearance = find_first_reappearance_frame(frames, gt2_track_present, gt2_det_present)

    candidates = [f for f in (gt1_reappearance, gt2_reappearance) if f is not None]
    switch_frame = min(candidates) if candidates else None

    gt1_mask: Dict[int, bool] = {}
    gt2_mask: Dict[int, bool] = {}

    for f in frames:
        if switch_frame is not None and f >= switch_frame:
            gt1_mask[f] = raw_gt2_mask[f]
            gt2_mask[f] = raw_gt1_mask[f]
        else:
            gt1_mask[f] = raw_gt1_mask[f]
            gt2_mask[f] = raw_gt2_mask[f]

    gt1_regions = mask_to_regions(frames, gt1_mask)
    gt2_regions = mask_to_regions(frames, gt2_mask)

    return gt1_regions, gt2_regions, switch_frame


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0, 0])

def norm_euclidean_sim(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    """
    Normalized Euclidean similarity.

    Equivalent to the pairwise scalar version of:

        a = F.normalize(a, p=2, dim=-1)
        b = F.normalize(b, p=2, dim=-1)
        sim = 1 - torch.cdist(a, b, p=2) / 2

    Returns:
        similarity in approximately [0, 1] for non-zero normalized embeddings.
        Higher means more similar.
    """
    a_norm = a / max(float(np.linalg.norm(a, ord=2)), eps)
    b_norm = b / max(float(np.linalg.norm(b, ord=2)), eps)

    dist = np.linalg.norm(a_norm - b_norm, ord=2)
    sim = 1.0 - dist / 2.0

    return float(sim)


def copy_timeline_if_available(info_folder: str, video_name: str, output_dir: Path) -> None:
    source = Path(info_folder) / "eval" / "failure_cases" / f"{video_name}_timeline.png"
    destination = output_dir / f"{video_name}_timeline.png"

    if source.exists():
        shutil.copy2(source, destination)
        print(f"Copied timeline: {destination}")
    else:
        print(f"Timeline not found, skipping copy: {source}")


# ============================================================
# PIPELINE STEPS
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot normalized Euclidean similarity between the two end-frame detections "
            "and the history of Object 1/Object 2 over the explicit window [--start, --end]. "
            "Object IDs are the pred_ids assigned at frame 1. "
            "At the last frame (--end), detections are interpreted as swapped "
            "because of the ID switch."
        )
    )
    parser.add_argument("--info-folder", required=True,
                        help="Folder containing tracks_dets_embs.txt and tracks_dets_ids.txt")
    parser.add_argument("--dataset-folder", required=True,
                        help="MOT dataset folder containing sequence directories")
    parser.add_argument("--video-name", required=True,
                        help="Sequence name, e.g. v_0kUtTtmLaJA_c008")
    parser.add_argument("--start", type=int, required=True,
                        help="First local frame of the history window (inclusive)")
    parser.add_argument("--end", type=int, required=True,
                        help="Last local frame taken into account; used as the query frame")
    parser.add_argument("--gt1", type=int, required=True,
                        help="Predicted ID assigned to GT1 at frame 1")
    parser.add_argument("--gt2", type=int, required=True,
                        help="Predicted ID assigned to GT2 at frame 1")
    parser.add_argument("--gt1-color", required=True,
                        help="Matplotlib color for GT1 (e.g. green, tab:blue, #2ca02c)")
    parser.add_argument("--gt2-color", required=True,
                        help="Matplotlib color for GT2 (e.g. brown, tab:orange, #8c564b)")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory where <video-name>_memory.png is saved")
    return parser.parse_args()


def resolve_window(start: int, end: int) -> Tuple[List[int], int]:
    FIRST_LOCAL_FRAME = 2

    if start < FIRST_LOCAL_FRAME:
        print(f"Warning: start clamped to {FIRST_LOCAL_FRAME} (local frames start at {FIRST_LOCAL_FRAME}).")
        start = FIRST_LOCAL_FRAME

    if start >= end:
        raise ValueError("--start must be strictly smaller than --end.")

    query_frame = end
    memory_frames = list(range(start, end))
    return memory_frames, query_frame


def get_detection_embedding(
    data: Dict[int, Dict[Tuple[str, int], np.ndarray]],
    query_frame: int,
    detection_id: int,
) -> np.ndarray:
    det_key = ("D", detection_id)
    if det_key not in data[query_frame]:
        raise ValueError(
            f"Detection ID {detection_id} of type 'D' not found at frame {query_frame}."
        )
    return data[query_frame][det_key]


def compute_similarity_series(
    data: Dict[int, Dict[Tuple[str, int], np.ndarray]],
    det_emb: np.ndarray,
    memory_frames: List[int],
    track_id: int,
) -> Tuple[List[int], List[float]]:
    """
    For each frame in memory_frames, compute normalized Euclidean similarity between det_emb
    and the tracklet embedding of track_id. Frames where the track is absent are skipped.
    """
    frames, sims = [], []
    track_key = ("T", track_id)

    for f in memory_frames:
        if track_key in data[f]:
            frames.append(f)
            # sims.append(cosine_sim(det_emb, data[f][track_key]))
            sims.append(norm_euclidean_sim(det_emb, data[f][track_key]))

    return frames, sims


def plot_similarity(
    series: List[Tuple[List[int], List[float], str, Dict[str, object]]],
    query_frame: int,
    start: int,
    end: int,
    video_name: str,
    gt1_occlusion_regions: List[Tuple[int, int]],
    gt2_occlusion_regions: List[Tuple[int, int]],
    switch_frame: int | None,
    gt1_color: str,
    gt2_color: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))

    gt1_label_added = False
    for s, e in gt1_occlusion_regions:
        ax.axvspan(
            s - 0.5, e + 0.5,
            color=gt1_color,
            alpha=0.18,
            zorder=0,
            label="Object 1 occlusion" if not gt1_label_added else None,
        )
        gt1_label_added = True

    gt2_label_added = False
    for s, e in gt2_occlusion_regions:
        ax.axvspan(
            s - 0.5, e + 0.5,
            color=gt2_color,
            alpha=0.18,
            zorder=0,
            label="Object 2 occlusion" if not gt2_label_added else None,
        )
        gt2_label_added = True

    for frames, sims, label, style in series:
        if not frames:
            print(f"Warning: no data found for '{label}' in the window.")
            continue

        ax.plot(
            frames,
            sims,
            marker=style["marker"],
            color=style["color"],
            markersize=5,
            linewidth=1.7,
            label=label,
            zorder=2,
        )

    if switch_frame is not None:
        ax.axvline(
            x=switch_frame,
            color="black",
            linestyle=":",
            linewidth=1.0,
            label=f"ID switch ({switch_frame})",
        )

    ax.axvline(
        x=query_frame,
        color="red",
        linestyle="--",
        linewidth=1.0,
    )

    ax.set_title(
        "Embedding Similarity Between Object Tracklets and Post-Occlusion Reference Detections"
    )
    ax.set_xlabel("Frame")
    ax.set_ylabel("Euclidean Similarity")
    ax.set_xlim(start, end)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved memory plot: {output_path}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    args = parse_args()

    embeddings_file = os.path.join(args.info_folder, "tracks_dets_embs.txt")
    ids_file = os.path.join(args.info_folder, "tracks_dets_ids.txt")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.video_name}_memory.png"

    frame_start_end_dict = build_frame_start_end_dict(args.dataset_folder)

    print("Computing ID offsets per video...")
    id_offset_dict = build_id_offset_dict(ids_file, embeddings_file, frame_start_end_dict)
    print(f"ID offset for '{args.video_name}': {id_offset_dict[args.video_name]}")

    memory_frames, query_frame = resolve_window(args.start, args.end)
    all_frames = memory_frames + [query_frame]

    print(f"Loading embeddings for {len(all_frames)} frames...")
    data = load_embeddings_for_frames(
        embeddings_file=embeddings_file,
        ids_file=ids_file,
        video_name=args.video_name,
        frame_start_end_dict=frame_start_end_dict,
        id_offset_dict=id_offset_dict,
        frames_of_interest=all_frames,
    )

    # Because of the post-occlusion ID switch at the end frame:
    # - actual GT1 detection is carried by raw pred_id gt2
    # - actual GT2 detection is carried by raw pred_id gt1
    gt1_det_emb = get_detection_embedding(data, query_frame, args.gt2)
    gt2_det_emb = get_detection_embedding(data, query_frame, args.gt1)

    series = [
        (*compute_similarity_series(data, gt1_det_emb, memory_frames, args.gt2),
        "Obj.1 ref. det. vs Obj.2 tracklet",
        {"color": args.gt1_color, "marker": "x"}),

        (*compute_similarity_series(data, gt1_det_emb, memory_frames, args.gt1),
        "Obj.1 ref. det. vs Obj.1 tracklet",
        {"color": args.gt1_color, "marker": "o"}),

        (*compute_similarity_series(data, gt2_det_emb, memory_frames, args.gt2),
        "Obj.2 ref. det. vs Obj.2 tracklet",
        {"color": args.gt2_color, "marker": "o"}),

        (*compute_similarity_series(data, gt2_det_emb, memory_frames, args.gt1),
        "Obj.2 ref. det. vs Obj.1 tracklet",
        {"color": args.gt2_color, "marker": "x"}),
    ]

    if all(len(frames) == 0 for frames, _, _, _ in series):
        raise ValueError(
            "No data found for any GT/track combination in the specified window. "
            "Check your IDs and frame range."
        )

    track_present_frames, det_present_frames = load_track_and_detection_presence(
        embeddings_file=embeddings_file,
        ids_file=ids_file,
        video_name=args.video_name,
        frame_start_end_dict=frame_start_end_dict,
        id_offset_dict=id_offset_dict,
        frames_of_interest=memory_frames,
        track_ids=[args.gt1, args.gt2],
    )

    gt1_occlusion_regions, gt2_occlusion_regions, switch_frame = compute_identity_aware_occlusion_regions(
        frames=memory_frames,
        gt1_track_present=track_present_frames[args.gt1],
        gt1_det_present=det_present_frames[args.gt1],
        gt2_track_present=track_present_frames[args.gt2],
        gt2_det_present=det_present_frames[args.gt2],
    )

    plot_similarity(
        series=series,
        query_frame=query_frame,
        start=args.start,
        end=args.end,
        video_name=args.video_name,
        gt1_occlusion_regions=gt1_occlusion_regions,
        gt2_occlusion_regions=gt2_occlusion_regions,
        switch_frame=switch_frame,
        gt1_color=args.gt1_color,
        gt2_color=args.gt2_color,
        output_path=output_path,
    )

    copy_timeline_if_available(
        info_folder=args.info_folder,
        video_name=args.video_name,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()