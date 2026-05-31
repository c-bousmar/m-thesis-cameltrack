
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import configparser
import os
import shutil
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Set

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONSTANTS
# ============================================================

CURVE_COLORS = {
    "GAFFE": "navy",
    "TE app.": "green",
    "TE bbox": "mediumorchid",
    "TE kp.": "orange",
    "Sum": "red",
}


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
    reference_embeddings_file: str,
    frame_start_end_dict: Dict[str, Tuple[int, int]],
) -> Dict[str, int]:
    sorted_seqs = sorted(
        [(s, e, name) for name, (s, e) in frame_start_end_dict.items()],
        key=lambda x: x[0],
    )

    def frame_to_video(frame_id: int) -> Optional[str]:
        for s, e, name in sorted_seqs:
            if s <= frame_id <= e:
                return name
        return None

    max_id_per_video: Dict[str, int] = {}

    with open(reference_embeddings_file, "r") as f_emb, open(ids_file, "r") as f_ids:
        for line_emb, line_id in zip(f_emb, f_ids):
            line_emb = line_emb.strip()
            line_id = line_id.strip()
            if not line_emb or not line_id:
                continue

            parts = line_emb.split()
            if len(parts) < 2:
                continue

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
            if len(parts) < 3:
                continue

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
            if len(parts) < 2:
                continue

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


def find_occlusion_and_reappearance(
    frames: List[int],
    track_present: Set[int],
    det_present: Set[int],
) -> Tuple[Optional[int], Optional[int]]:
    """
    Occlusion begins when T is present and D is absent.
    Reappearance is the first later frame where both T and D are present again.
    """
    occlusion_start: Optional[int] = None
    was_occluded = False

    for f in frames:
        is_occluded = (f in track_present) and (f not in det_present)
        has_reappeared = (f in track_present) and (f in det_present)

        if occlusion_start is None and is_occluded:
            occlusion_start = f
            was_occluded = True
            continue

        if was_occluded and has_reappeared:
            return occlusion_start, f

    return occlusion_start, None


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


def copy_timeline_if_available(info_folder: str, video_name: str, output_dir: Path) -> None:
    source = Path(info_folder) / "eval" / "failure_cases" / f"{video_name}_timeline.png"
    destination = output_dir / f"{video_name}_timeline.png"

    if source.exists():
        shutil.copy2(source, destination)
        print(f"Copied timeline: {destination}")
    else:
        print(f"Timeline not found, skipping copy: {source}")


def resolve_window(window_start: int, window_end: int) -> List[int]:
    FIRST_LOCAL_FRAME = 2

    if window_start < FIRST_LOCAL_FRAME:
        print(
            f"Warning: window_start clamped to {FIRST_LOCAL_FRAME} "
            f"(local frames start at {FIRST_LOCAL_FRAME})."
        )
        window_start = FIRST_LOCAL_FRAME

    if window_start > window_end:
        raise ValueError("--window-start must be <= --window-end.")

    return list(range(window_start, window_end + 1))


def get_embedding_if_present(
    data: Dict[int, Dict[Tuple[str, int], np.ndarray]],
    frame: int,
    obj_type: str,
    obj_id: int,
) -> Optional[np.ndarray]:
    return data.get(frame, {}).get((obj_type, obj_id), None)


def compute_similarity_series_for_tracklet(
    data: Dict[int, Dict[Tuple[str, int], np.ndarray]],
    frames: List[int],
    tracklet_id: int,
    reference_frame: int,
) -> Tuple[List[int], List[float]]:
    """
    Reference:
      detection D<tracklet_id> at frame reference_frame

    Series:
      for each frame f in frames, compare reference detection with T<tracklet_id> at frame f
    """
    ref_det = get_embedding_if_present(data, reference_frame, "D", tracklet_id)
    if ref_det is None:
        raise ValueError(
            f"Reference detection D{tracklet_id} not found at frame {reference_frame}."
        )

    out_frames: List[int] = []
    out_sims: List[float] = []

    for f in frames:
        track_emb = get_embedding_if_present(data, f, "T", tracklet_id)
        if track_emb is None:
            continue

        out_frames.append(f)
        # out_sims.append(cosine_sim(ref_det, track_emb))
        out_sims.append(norm_euclidean_sim(ref_det, track_emb))


    return out_frames, out_sims


def plot_similarity(
    series: List[Tuple[List[int], List[float], str, str, float]],
    window_start: int,
    plot_end: int,
    video_name: str,
    tracklet_id: int,
    occlusion_start: int,
    reappearance_frame: int,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.axvspan(
        occlusion_start - 0.5,
        reappearance_frame - 0.5,
        color="lightgrey",
        alpha=0.20,
        zorder=0,
    )

    for frames, sims, label, color, linewidth in series:
        if not frames:
            print(f"Warning: no data found for '{label}' in the window.")
            continue

        pre_occ_frames = [f for f in frames if f < occlusion_start]
        pre_occ_sims = [s for f, s in zip(frames, sims) if f < occlusion_start]

        occ_frames = [f for f in frames if occlusion_start <= f < reappearance_frame]
        occ_sims = [s for f, s in zip(frames, sims) if occlusion_start <= f < reappearance_frame]

        post_occ_frames = [f for f in frames if f >= reappearance_frame]
        post_occ_sims = [s for f, s in zip(frames, sims) if f >= reappearance_frame]

        ax.plot(
            frames,
            sims,
            linewidth=linewidth,
            color=color,
            zorder=2,
        )

        if pre_occ_frames:
            ax.plot(
                pre_occ_frames,
                pre_occ_sims,
                linestyle="None",
                marker="o",
                markersize=3.2,
                color=color,
                zorder=3,
            )

        if occ_frames:
            ax.plot(
                occ_frames,
                occ_sims,
                linestyle="None",
                marker="x",
                markersize=3.6,
                color=color,
                zorder=3,
            )

        if post_occ_frames:
            ax.plot(
                post_occ_frames,
                post_occ_sims,
                linestyle="None",
                marker="o",
                markersize=3.6,
                color=color,
                zorder=3,
            )

    ax.set_title(
        f"Embeddings Similarity Evolution Across CAMEL Stages: Tracklet vs. Detection at Reappearance (track {tracklet_id})"
    )
    ax.set_xlabel("Frame")
    ax.set_ylabel("Cosine similarity")
    ax.set_xlim(window_start, plot_end)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.4)

    legend_handles = [
        Patch(facecolor="lightgrey", edgecolor="lightgrey", alpha=0.20, label="Occlusion"),
        Line2D([0], [0], marker="o", color="black", linestyle="None",
               markersize=5, label="w/ new detection"),
        Line2D([0], [0], marker="x", color="black", linestyle="None",
               markersize=5, label="w/o new detection"),
        Line2D([0], [0], color=series[0][3], linewidth=series[0][4], label="Bbox TE output"),
        Line2D([0], [0], color=series[1][3], linewidth=series[1][4], label="Keypoints TE output"),
        Line2D([0], [0], color=series[2][3], linewidth=series[2][4], label="Appearence TE output"),
        Line2D([0], [0], color=series[3][3], linewidth=series[3][4], label="Sum of TEs output"),
        Line2D([0], [0], color=series[4][3], linewidth=series[4][4], label="GAFFE output"),
    ]

    legend = ax.legend(handles=legend_handles)
    legend._legend_box.align = "left"

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {output_path}")


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "For one or more tracklet IDs, automatically detect the first occlusion "
            "and the first reappearance inside the provided window. Then compare the "
            "detection D<ID> at reappearance with the tracklet T<ID> at every frame "
            "from --window-start to that reappearance frame, and plot one curve per "
            "CAMEL embedding stage."
        )
    )
    parser.add_argument(
        "--info-folder",
        required=True,
        help="Folder containing tracks_dets_*.txt and tracks_dets_ids.txt",
    )
    parser.add_argument(
        "--dataset-folder",
        required=True,
        help="MOT dataset folder containing sequence directories",
    )
    parser.add_argument(
        "--video-name",
        required=True,
        help="Sequence name, e.g. v_0kUtTtmLaJA_c008",
    )
    parser.add_argument(
        "--window-start",
        type=int,
        required=True,
        help="First local frame of the observation window (inclusive)",
    )
    parser.add_argument(
        "--window-end",
        type=int,
        required=True,
        help="Last local frame of the candidate observation window (inclusive)",
    )
    parser.add_argument(
        "--tracklet-ids",
        type=int,
        nargs="+",
        required=True,
        help="One or more local tracklet IDs to analyze",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory where the figures are saved",
    )
    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    args = parse_args()

    ids_file = os.path.join(args.info_folder, "tracks_dets_ids.txt")

    embedding_files = {
        "GAFFE": os.path.join(args.info_folder, "tracks_dets_embs.txt"),
        "TE app.": os.path.join(args.info_folder, "tracks_dets_tokens_app.txt"),
        "TE bbox": os.path.join(args.info_folder, "tracks_dets_tokens_bbox.txt"),
        "TE kp.": os.path.join(args.info_folder, "tracks_dets_tokens_kp.txt"),
        "Sum": os.path.join(args.info_folder, "tracks_dets_tokens.txt"),
    }

    curve_colors = CURVE_COLORS.copy()

    curve_linewidths = {
        "TE bbox": 1.4,
        "TE kp.": 1.4,
        "TE app.": 1.4,
        "Sum": 1.4,
        "GAFFE": 1.8,
    }

    for label, path in embedding_files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing embedding file for '{label}': {path}")

    if not os.path.exists(ids_file):
        raise FileNotFoundError(f"Missing ids file: {ids_file}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_start_end_dict = build_frame_start_end_dict(args.dataset_folder)
    frames = resolve_window(args.window_start, args.window_end)

    print("Computing ID offsets per video...")
    id_offset_dict = build_id_offset_dict(
        ids_file=ids_file,
        reference_embeddings_file=embedding_files["GAFFE"],
        frame_start_end_dict=frame_start_end_dict,
    )
    print(f"ID offset for '{args.video_name}': {id_offset_dict[args.video_name]}")

    print(f"Loading CAMEL step embeddings for {len(frames)} frames...")
    loaded_data: Dict[str, Dict[int, Dict[Tuple[str, int], np.ndarray]]] = {}

    for step_name, emb_file in embedding_files.items():
        print(f"  - Loading {step_name}: {emb_file}")
        loaded_data[step_name] = load_embeddings_for_frames(
            embeddings_file=emb_file,
            ids_file=ids_file,
            video_name=args.video_name,
            frame_start_end_dict=frame_start_end_dict,
            id_offset_dict=id_offset_dict,
            frames_of_interest=frames,
            obj_types=("T", "D"),
        )

    track_present_frames, det_present_frames = load_track_and_detection_presence(
        embeddings_file=embedding_files["GAFFE"],
        ids_file=ids_file,
        video_name=args.video_name,
        frame_start_end_dict=frame_start_end_dict,
        id_offset_dict=id_offset_dict,
        frames_of_interest=frames,
        track_ids=args.tracklet_ids,
    )

    for tracklet_id in args.tracklet_ids:
        print(f"\nProcessing tracklet {tracklet_id}...")

        occlusion_start, reappearance_frame = find_occlusion_and_reappearance(
            frames=frames,
            track_present=track_present_frames[tracklet_id],
            det_present=det_present_frames[tracklet_id],
        )

        if occlusion_start is None:
            print(
                f"Skipping tracklet {tracklet_id}: no occlusion found in "
                f"[{args.window_start}, {args.window_end}]."
            )
            continue

        if reappearance_frame is None:
            print(
                f"Skipping tracklet {tracklet_id}: occlusion found at frame {occlusion_start}, "
                f"but no new detection found afterward in [{args.window_start}, {args.window_end}]."
            )
            continue

        effective_frames = [f for f in frames if f <= reappearance_frame]

        series: List[Tuple[List[int], List[float], str, str, float]] = []

        for step_name in [
            "TE bbox",
            "TE kp.",
            "TE app.",
            "Sum",
            "GAFFE",
        ]:
            step_data = loaded_data[step_name]

            step_frames, step_sims = compute_similarity_series_for_tracklet(
                data=step_data,
                frames=effective_frames,
                tracklet_id=tracklet_id,
                reference_frame=reappearance_frame,
            )

            series.append((
                step_frames,
                step_sims,
                step_name,
                curve_colors[step_name],
                curve_linewidths[step_name],
            ))

        if all(len(s_frames) == 0 for s_frames, _, _, _, _ in series):
            print(
                f"Skipping tracklet {tracklet_id}: no similarity data found up to "
                f"reappearance frame {reappearance_frame}."
            )
            continue

        output_path = (
            output_dir
            / f"{args.video_name}_tracklet_{tracklet_id}_camel_steps_until_reappearance.png"
        )

        plot_similarity(
            series=series,
            window_start=args.window_start,
            plot_end=reappearance_frame,
            video_name=args.video_name,
            tracklet_id=tracklet_id,
            occlusion_start=occlusion_start,
            reappearance_frame=reappearance_frame,
            output_path=output_path,
        )

    copy_timeline_if_available(
        info_folder=args.info_folder,
        video_name=args.video_name,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()