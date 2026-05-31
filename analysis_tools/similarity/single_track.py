from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Tuple, List

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import configparser


# ============================================================
# HELPERS
# ============================================================
def make_color_map(unique_track_ids: List[int]) -> Dict[int, tuple]:
    if len(unique_track_ids) <= 20:
        cmap = plt.get_cmap("tab20", len(unique_track_ids))
    else:
        cmap = plt.get_cmap("turbo", len(unique_track_ids))
    return {tid: cmap(i) for i, tid in enumerate(unique_track_ids)}

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
    obj_types: Tuple[str, ...] = ("T",),
) -> Dict[int, Dict[Tuple[str, int], np.ndarray]]:
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

            if not line_emb:
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


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0, 0])

def norm_euclidean_sim(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    """
    Normalized Euclidean similarity.
    Returns a similarity score in [0, 1], where 1 means identical.
    """
    dist = np.linalg.norm(a - b)
    norm = np.linalg.norm(a) + np.linalg.norm(b) + eps
    return 1.0 - (dist / norm)

def load_ref_history_with_detection_status(
    embeddings_file: str,
    ids_file: str,
    video_name: str,
    frame_start_end_dict: Dict[str, Tuple[int, int]],
    id_offset_dict: Dict[str, int],
    frames_of_interest: List[int],
    ref_id: int,
) -> Tuple[Dict[int, np.ndarray], Dict[int, bool]]:
    """
    Returns:
      - ref_emb_per_frame: local_frame -> embedding of T<ref_id>
      - ref_has_detection_per_frame: local_frame -> whether D<ref_id> also exists at that frame

    Assumption:
      a tracklet T<ref_id> is associated with a detection at a frame if a D entry
      with the same local ID exists at the same frame.
    """
    if video_name not in frame_start_end_dict:
        raise ValueError(f"Video '{video_name}' not found in dataset folder.")

    seq_frame_start, seq_frame_end = frame_start_end_dict[video_name]
    id_offset = id_offset_dict[video_name]
    frames_set = set(frames_of_interest)

    ref_emb_per_frame: Dict[int, np.ndarray] = {}
    ref_has_detection_per_frame: Dict[int, bool] = {}
    det_present_frames = set()

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
            if local_id != ref_id:
                continue

            values = [float(x) for x in parts[2:] if x.lower() != "nan"]
            if not values:
                continue

            emb = np.array(values, dtype=np.float32)

            if obj_type == "T":
                ref_emb_per_frame[local_frame] = emb
            elif obj_type == "D":
                det_present_frames.add(local_frame)

    for f in ref_emb_per_frame:
        ref_has_detection_per_frame[f] = (f in det_present_frames)

    return ref_emb_per_frame, ref_has_detection_per_frame


def compute_t_consecutive_similarity_series(
    ref_emb_per_frame: Dict[int, np.ndarray],
    ref_has_detection_per_frame: Dict[int, bool],
) -> Tuple[List[int], List[float], List[bool]]:
    """
    For each frame where T<ref> exists, compute the mean cosine similarity between
    the current embedding and all previous embeddings of the same tracklet.

    The first frame is skipped because it has no previous frame to compare to.
    """
    frames_sorted = sorted(ref_emb_per_frame.keys())

    out_frames: List[int] = []
    out_means: List[float] = []
    out_has_det: List[bool] = []

    previous_embs: List[np.ndarray] = []

    for f in frames_sorted:
        cur_emb = ref_emb_per_frame[f]

        if previous_embs:
            # sims = [cosine_sim(cur_emb, prev_emb) for prev_emb in previous_embs]
            sims = [norm_euclidean_sim(cur_emb, prev_emb) for prev_emb in previous_embs]
            out_frames.append(f)
            out_means.append(float(np.mean(sims)))
            out_has_det.append(ref_has_detection_per_frame.get(f, False))

        previous_embs.append(cur_emb)

    return out_frames, out_means, out_has_det

def compute_t_tminus1_similarity_series(
    ref_emb_per_frame: Dict[int, np.ndarray],
    ref_has_detection_per_frame: Dict[int, bool],
) -> Tuple[List[int], List[float], List[bool]]:
    """
    For each frame f where T<ref> exists, compute cosine similarity with frame f-1
    only if the same tracklet also exists at f-1.
    """
    frames_sorted = sorted(ref_emb_per_frame.keys())
    frame_set = set(frames_sorted)

    out_frames: List[int] = []
    out_sims: List[float] = []
    out_has_det: List[bool] = []

    for f in frames_sorted:
        if (f - 1) not in frame_set:
            continue

        # sim = cosine_sim(ref_emb_per_frame[f], ref_emb_per_frame[f - 1])
        sim = norm_euclidean_sim(ref_emb_per_frame[f], ref_emb_per_frame[f - 1])

        out_frames.append(f)
        out_sims.append(sim)
        out_has_det.append(ref_has_detection_per_frame.get(f, False))

    return out_frames, out_sims, out_has_det

def compute_t_tlastframe_similarity_series(
    ref_emb_per_frame: Dict[int, np.ndarray],
    ref_has_detection_per_frame: Dict[int, bool],
) -> Tuple[List[int], List[float], List[bool]]:
    """
    For each frame where T<ref> exists, compute cosine similarity with the
    last available frame of the same tracklet.

    The last frame itself is included and has similarity 1.0 with itself.
    """
    frames_sorted = sorted(ref_emb_per_frame.keys())

    out_frames: List[int] = []
    out_sims: List[float] = []
    out_has_det: List[bool] = []

    if not frames_sorted:
        return out_frames, out_sims, out_has_det

    last_frame = frames_sorted[-1]
    last_emb = ref_emb_per_frame[last_frame]

    for f in frames_sorted:
        # sim = cosine_sim(ref_emb_per_frame[f], last_emb)
        sim = norm_euclidean_sim(ref_emb_per_frame[f], last_emb)
        out_frames.append(f)
        out_sims.append(sim)
        out_has_det.append(ref_has_detection_per_frame.get(f, False))

    return out_frames, out_sims, out_has_det

# ============================================================
# PIPELINE STEPS
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot cosine similarity between a reference tracklet and all other "
            "tracklets over an explicit frame window [--start, --end]."
        )
    )
    parser.add_argument("--info-folder", required=True,
                        help="Folder containing tracks_dets_embs.txt and tracks_dets_ids.txt")
    parser.add_argument("--dataset-folder", required=True,
                        help="MOT dataset folder containing sequence directories")
    parser.add_argument("--video-name", required=True,
                        help="Sequence name, e.g. v_0kUtTtmLaJA_c008")
    parser.add_argument("--ref", type=int, required=True,
                        help="Local ID of the reference tracklet (type T)")
    parser.add_argument("--start", type=int, required=True,
                        help="First local frame of the window (inclusive)")
    parser.add_argument("--end", type=int, required=True,
                        help="Last local frame of the window (inclusive)")
    parser.add_argument("--line1", type=int, required=True,
                        help="First vertical line frame")
    parser.add_argument("--line2", type=int, required=True,
                        help="Second vertical line frame")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for the plot")
    return parser.parse_args()


def collect_all_track_ids(
    data: Dict[int, Dict[Tuple[str, int], np.ndarray]],
) -> List[int]:
    """Return sorted list of all unique tracklet IDs found across all frames."""
    ids = set()
    for frame_data in data.values():
        for (obj_type, obj_id) in frame_data:
            if obj_type == "T":
                ids.add(obj_id)
    return sorted(ids)


def compute_similarity_series(
    data: Dict[int, Dict[Tuple[str, int], np.ndarray]],
    ref_emb_per_frame: Dict[int, np.ndarray],
    window_frames: List[int],
    track_id: int,
) -> Tuple[List[int], List[float]]:
    """
    For each frame in window_frames where both the reference tracklet and
    track_id are present, compute their cosine similarity.
    """
    frames, sims = [], []
    track_key = ("T", track_id)

    for f in window_frames:
        if f not in ref_emb_per_frame:
            continue
        if track_key in data[f]:
            frames.append(f)
            # sims.append(cosine_sim(ref_emb_per_frame[f], data[f][track_key]))
            sims.append(norm_euclidean_sim(ref_emb_per_frame[f], data[f][track_key]))

    return frames, sims


def plot_similarity(
    series: List[Tuple[List[int], List[float], int]],
    ref_id: int,
    video_name: str,
    start: int,
    end: int,
    line1: int,
    line2: int,
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))

    unique_track_ids = sorted({tid for _, _, tid in series})
    tid_to_color = make_color_map(unique_track_ids)

    frame_to_values: Dict[int, List[float]] = {}
    all_plotted_values: List[float] = []

    for _, (frames, sims, tid) in enumerate(series):
        if not frames:
            print(f"Warning: tracklet {tid} has no overlap with ref {ref_id} in [{start}, {end}].")
            continue

        for f, s in zip(frames, sims):
            frame_to_values.setdefault(f, []).append(s)
            all_plotted_values.append(s)

        ax.plot(
            frames,
            sims,
            # marker="o",
            # markersize=3,
            linewidth=1.2,
            color=tid_to_color[tid],
            label=f"T{tid} vs T{ref_id}",
            zorder=2,
        )

    mean_frames = sorted(frame_to_values.keys())
    mean_values = [float(np.mean(frame_to_values[f])) for f in mean_frames]

    if mean_frames:
        all_plotted_values.extend(mean_values)
        ax.plot(
            mean_frames,
            mean_values,
            # marker="o",
            # markersize=3,
            linewidth=2.8,
            color="black",
            label=f"Mean (all vs T{ref_id})",
            zorder=15,
        )

    ax.axvline(line1, linestyle="--", linewidth=1.5, color="black", zorder=20, label="Occlusion start")
    ax.axvline(line2, linestyle="--", linewidth=1.5, color="gray", zorder=20, label="Occlusion end")

    mean_lookup = {f: v for f, v in zip(mean_frames, mean_values)}

    if line1 in mean_lookup:
        y1 = mean_lookup[line1]
        ax.scatter([line1], [y1], color="black", s=22, zorder=25)
        ax.annotate(
            f"{y1:.3f}",
            xy=(line1, y1),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
            color="white",
            zorder=30,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="black", edgecolor="black", linewidth=0.5),
        )

    if line2 in mean_lookup:
        y2 = mean_lookup[line2]
        ax.scatter([line2], [y2], color="black", s=22, zorder=25)
        ax.annotate(
            f"{y2:.3f}",
            xy=(line2, y2),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
            color="white",
            zorder=30,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="black", edgecolor="black", linewidth=0.5),
        )

    ax.set_title(
        f"{video_name} — cosine similarity of ref tracklet T{ref_id} "
        f"vs all other tracklets  |  window [{start}, {end}]"
    )
    ax.set_xlabel("Frame")
    ax.set_ylabel("Similarity")
    ax.set_xlim(start, end)

    if all_plotted_values:
        y_min = min(all_plotted_values)
        y_max = max(all_plotted_values)
        margin = max(0.02, 0.05 * (y_max - y_min if y_max > y_min else 1.0))
        lower = max(-1.0, y_min - margin)
        upper = min(1.0, y_max + margin)
        ax.set_ylim(lower, upper)
    else:
        ax.set_ylim(-1.0, 1.0)
    # ax.set_ylim(0.0, 1.0)

    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(
        loc="upper right",
        fontsize=7,
        ncol=max(1, len(series) // 40 + 1),
    )
    fig.tight_layout()
    fig.savefig(f"{output_dir}/tracklets_similarity.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {output_dir}")


def plot_tracklet_history(
    consecutive_frames: List[int],
    consecutive_sims: List[float],
    consecutive_has_detection: List[bool],
    tminus1_frames: List[int],
    tminus1_sims: List[float],
    tminus1_has_detection: List[bool],
    last_frames: List[int],
    last_sims: List[float],
    last_has_detection: List[bool],
    tid: int,
    video_name: str,
    start: int,
    end: int,
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))

    if not consecutive_frames and not tminus1_frames and not last_frames:
        print(f"Warning: no self-history values available for T{tid}.")
        plt.close(fig)
        return

    all_values: List[float] = []

    def plot_one_series(frames, sims, has_det, line_color):
        all_values.extend(sims)
        ax.plot(
            frames,
            sims,
            color=line_color,
            zorder=2,
        )
    plot_one_series(consecutive_frames, consecutive_sims, consecutive_has_detection, "tab:blue")
    plot_one_series(tminus1_frames, tminus1_sims, tminus1_has_detection, "tab:green")
    plot_one_series(last_frames, last_sims, last_has_detection, "tab:red")

    # Shade regions without detection association
    no_det_regions = []
    current_start = None
    for f, det in zip(consecutive_frames, consecutive_has_detection):
        if not det:
            if current_start is None:
                current_start = f
        else:
            if current_start is not None:
                no_det_regions.append((current_start, f))
                current_start = None
    if current_start is not None:
        no_det_regions.append((current_start, consecutive_frames[-1]))
    for s, e in no_det_regions:
        ax.axvspan(s, e, color="lightgrey", alpha=0.35, zorder=0)
    
    if not all_values:
        plt.close(fig)
        return

    y_min = min(all_values)
    y_max = max(all_values)
    margin = max(0.02, 0.05 * (y_max - y_min if y_max > y_min else 1.0))
    lower = max(-1.0, y_min - margin)
    upper = min(1.0, y_max + margin)
    ax.set_xlim(start, end)
    ax.set_ylim(lower, upper)
    # ax.set_xlim(start, end)
    # ax.set_ylim(0.0, 1.0)

    ax.set_title(
        f"Tracklet {tid} similarity historic"
        f"| window [{start}, {end}] | {video_name} "
    )
    ax.set_xlabel("Frame")
    ax.set_ylabel("Similarity")
    ax.grid(True, linestyle="--", alpha=0.4)

    legend_elements = [
        Line2D([0], [0], color='tab:blue', label='t vs mean(1,...,t-1)'),
        Line2D([0], [0], color='tab:green', label='t vs t-1'),
        Line2D([0], [0], color='tab:red', label='t vs last'),
        Patch(facecolor='lightgrey', alpha=0.35, label='No detection association'),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(f"{output_dir}/tracklet_{tid}_history.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved self-history plot: {output_dir}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    args = parse_args()

    if args.start > args.end:
        raise ValueError(f"--start ({args.start}) must be <= --end ({args.end}).")

    embeddings_file = os.path.join(args.info_folder, "tracks_dets_embs.txt")
    ids_file = os.path.join(args.info_folder, "tracks_dets_ids.txt")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_start_end_dict = build_frame_start_end_dict(args.dataset_folder)

    print("Computing ID offsets per video...")
    id_offset_dict = build_id_offset_dict(ids_file, embeddings_file, frame_start_end_dict)
    print(f"ID offset for '{args.video_name}': {id_offset_dict[args.video_name]}")

    window_frames = list(range(args.start, args.end + 1))

    print(f"Loading embeddings for {len(window_frames)} frames (T only)...")
    data = load_embeddings_for_frames(
        embeddings_file=embeddings_file,
        ids_file=ids_file,
        video_name=args.video_name,
        frame_start_end_dict=frame_start_end_dict,
        id_offset_dict=id_offset_dict,
        frames_of_interest=window_frames,
        obj_types=("T",),
    )

    ref_key = ("T", args.ref)
    ref_emb_per_frame: Dict[int, np.ndarray] = {}
    for f in window_frames:
        if ref_key in data[f]:
            ref_emb_per_frame[f] = data[f][ref_key]

    if not ref_emb_per_frame:
        raise ValueError(
            f"Reference tracklet T{args.ref} not found in any frame of [{args.start}, {args.end}]. "
            "Check --ref and the window bounds."
        )

    print(f"Reference tracklet T{args.ref} present in {len(ref_emb_per_frame)} / {len(window_frames)} frames.")

    all_track_ids = collect_all_track_ids(data)
    print(f"Found {len(all_track_ids)} unique tracklet(s) in the window.")

    series = []
    for tid in all_track_ids:
        if tid == args.ref:
            continue
        frames, sims = compute_similarity_series(data, ref_emb_per_frame, window_frames, tid)
        series.append((frames, sims, tid))

    plot_similarity(
        series=series,
        ref_id=args.ref,
        video_name=args.video_name,
        start=args.start,
        end=args.end,
        line1=args.line1,
        line2=args.line2,
        output_dir=output_dir,
    )

    for tid in all_track_ids:
        ref_history_embs, ref_has_detection_per_frame = load_ref_history_with_detection_status(
            embeddings_file=embeddings_file,
            ids_file=ids_file,
            video_name=args.video_name,
            frame_start_end_dict=frame_start_end_dict,
            id_offset_dict=id_offset_dict,
            frames_of_interest=window_frames,
            ref_id=tid,
        )

        consecutive_frames, consecutive_sims, consecutive_has_det = compute_t_consecutive_similarity_series(
            ref_emb_per_frame=ref_history_embs,
            ref_has_detection_per_frame=ref_has_detection_per_frame,
        )

        tminus1_frames, tminus1_sims, tminus1_has_det = compute_t_tminus1_similarity_series(
            ref_emb_per_frame=ref_history_embs,
            ref_has_detection_per_frame=ref_has_detection_per_frame,
        )

        last_frames, last_sims, last_has_det = compute_t_tlastframe_similarity_series(
            ref_emb_per_frame=ref_history_embs,
            ref_has_detection_per_frame=ref_has_detection_per_frame,
        )

        if not consecutive_frames and not tminus1_frames and not last_frames:
            continue

        plot_tracklet_history(
            consecutive_frames=consecutive_frames,
            consecutive_sims=consecutive_sims,
            consecutive_has_detection=consecutive_has_det,
            tminus1_frames=tminus1_frames,
            tminus1_sims=tminus1_sims,
            tminus1_has_detection=tminus1_has_det,
            last_frames=last_frames,
            last_sims=last_sims,
            last_has_detection=last_has_det,
            tid=tid,
            video_name=args.video_name,
            start=args.start,
            end=args.end,
            output_dir=output_dir,
        )

if __name__ == "__main__":
    main()