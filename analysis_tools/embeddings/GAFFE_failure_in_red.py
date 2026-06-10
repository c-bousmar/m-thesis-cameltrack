import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from PIL import Image
import io
import configparser
import os
import csv


# ============================================================
# FAILURE-AWARE EMBEDDING VISUALIZATION (RED HIGHLIGHTING)
# ============================================================

"""
This script visualizes the evolution of learned embeddings
for a multi-object tracking (MOT) model using dimensionality
reduction (t-SNE or PCA), while explicitly highlighting tracking errors.

Compared to the standard embedding visualization, this version
adds supervision from failure annotations:

- ID switches (ID_S) are highlighted in red
- All other embeddings are shown in gray
- This allows direct visual inspection of where identity
  inconsistencies occur in the embedding space

The goal is to analyze whether identity errors correspond to:
- poorly separated clusters
- overlapping embeddings between identities
- unstable temporal regions in feature space

This provides a qualitative diagnostic tool for tracking failures.

Input files:
- tracks_dets_embs.txt: embedding vectors (tracklets + detections)
- tracks_dets_ids.txt: identity labels
- failure_cases/<video>.csv: annotated tracking errors (ID switches)

Output:
- GIF showing embedding evolution over time
- with ID switch events highlighted in red

============================================================
"""

# ============================================================
# PARAMETERS
# ============================================================

info_folder = "/globalsc/ucl/elen/athieltg/Master_Thesis_MOT/outputs/CAMELTrackMOT17/2026-02-20/19-38-56 mot17 split 50-25-25"
dataset_folder = "/globalsc/ucl/elen/athieltg/Master_Thesis_MOT/data/MOT17/test"

frame_start = 70
frame_end   = 130
video_name  = "MOT17-09-FRCNN"

embeddings_file = os.path.join(info_folder, "tracks_dets_embs.txt")
ids_file        = os.path.join(info_folder, "tracks_dets_ids.txt")
failure_file    = os.path.join(info_folder, "eval/failure_cases", f"{video_name}.csv")

use_tsne = True
method_name = "tsne" if use_tsne else "pca"
SHOW = "both"  # "tracklets", "detections", "both"

output_gif = os.path.join(
    info_folder,
    f"{method_name}_{video_name}_GAFFE_{frame_start}_{frame_end}_failure_in_red.gif"
)


# ============================================================
# RECONSTRUCT GLOBAL FRAME INDEXING
# ============================================================

"""
The dataset consists of multiple sequential video folders.
Each sequence defines a local frame range stored in seqinfo.ini.

This step reconstructs the global frame indexing across sequences
to correctly align embeddings with frame IDs.
"""

frame_start_end_dict = {}
start_frame = 1
seq_dirs = sorted([d for d in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, d))])

for seq_name in seq_dirs:
    seq_dir = os.path.join(dataset_folder, seq_name)
    seqinfo_file = os.path.join(seq_dir, "seqinfo.ini")
    if os.path.exists(seqinfo_file):
        config = configparser.ConfigParser()
        config.optionxform = str
        config.read(seqinfo_file)
        seq_length = int(config["Sequence"]["seqLength"])
        end_frame = start_frame + seq_length - 2
        frame_start_end_dict[seq_name] = (start_frame, end_frame)
        start_frame = end_frame + 1

seq_frame_start, seq_frame_end = frame_start_end_dict[video_name]


# ============================================================
# LOAD ID SWITCH ANNOTATIONS
# ============================================================

"""
Load failure cases from CSV and extract ID switch events.

Each event is stored as a (frame, track_id) pair.
These will be highlighted in red in the embedding space.
"""

id_switch_events = set()

if os.path.exists(failure_file):
    with open(failure_file, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            issue = row["ISSUE"].strip()
            frame_raw = row["FRAME"].strip()
            pred_raw  = row.get("PRED_ID", "").strip()

            if pred_raw in ("", "None", "nan"):
                continue

            if issue == "ID_S":
                local_frame = int(frame_raw)
                pred_id = int(float(pred_raw))

                if frame_start <= local_frame <= frame_end:
                    id_switch_events.add((local_frame, pred_id))


print("ID switch events found:", len(id_switch_events))


# ============================================================
# LOAD AND FILTER EMBEDDINGS
# ============================================================

"""
Load embeddings and associated metadata:
- frame index
- track identity
- object type (tracklet or detection)

Only embeddings within the selected temporal window are kept.
"""

embeddings_list = []
frame_ids = []
track_ids = []
types = []

with open(embeddings_file, "r") as f_emb, open(ids_file, "r") as f_ids:
    for line_emb, line_id in zip(f_emb, f_ids):

        line_emb = line_emb.strip()
        if not line_emb:
            continue

        parts = line_emb.split()
        frame_id = int(parts[0])
        obj_type = parts[1]

        if seq_frame_start <= frame_id <= seq_frame_end:

            local_frame_id = frame_id - seq_frame_start + 2

            if frame_start <= local_frame_id <= frame_end:

                values = [float(x) for x in parts[2:] if x.lower() != "nan"]

                embeddings_list.append(values)
                frame_ids.append(local_frame_id)
                track_ids.append(int(line_id.strip()))
                types.append(1 if obj_type == "T" else 0)


embeddings_array = np.array(embeddings_list, dtype=np.float32)
frame_ids = np.array(frame_ids)
track_ids = np.array(track_ids)
types = np.array(types)

if len(track_ids) != len(embeddings_array):
    raise ValueError("IDs and embeddings length mismatch after filtering.")


# ============================================================
# DIMENSIONALITY REDUCTION
# ============================================================

"""
t-SNE emphasizes local neighborhood structure.
PCA emphasizes global variance structure.
"""

if use_tsne:
    reducer = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
else:
    reducer = PCA(n_components=2)

embeddings_2d = reducer.fit_transform(embeddings_array)


# ============================================================
# COLOR ASSIGNMENT
# ============================================================

"""
Default visualization:
- gray = normal embeddings
- red  = ID switch events
"""

x_min, x_max = embeddings_2d[:, 0].min(), embeddings_2d[:, 0].max()
y_min, y_max = embeddings_2d[:, 1].min(), embeddings_2d[:, 1].max()


# ============================================================
# GENERATE TEMPORAL GIF (FAILURES IN RED)
# ============================================================

"""
Each frame shows embeddings accumulated up to time t.

ID switch points are highlighted in red to visually
correlate failures with embedding structure.
"""

frames = []

for f in range(frame_start, frame_end + 1):

    fig, ax = plt.subplots(figsize=(8, 6))

    mask_frame = frame_ids <= f

    # Default color (gray)
    base_colors = np.full((len(track_ids), 3), 0.7)

    # Highlight ID switch events
    highlight_mask = np.array([
        (frame_ids[i], track_ids[i]) in id_switch_events
        for i in range(len(track_ids))
    ])

    colors_current = base_colors.copy()
    colors_current[highlight_mask] = [1.0, 0.0, 0.0]  # red

    track_mask = mask_frame & (types == 1)
    det_mask   = mask_frame & (types == 0)

    if SHOW in ["tracklets", "both"]:
        ax.scatter(
            embeddings_2d[track_mask, 0],
            embeddings_2d[track_mask, 1],
            c=colors_current[track_mask],
            marker="o",
            s=30,
            alpha=0.9,
            label="Tracklets"
        )

    if SHOW in ["detections", "both"]:
        ax.scatter(
            embeddings_2d[det_mask, 0],
            embeddings_2d[det_mask, 1],
            c=colors_current[det_mask],
            marker="^",
            s=30,
            alpha=0.9,
            label="Detections"
        )

    ax.set_xlim(x_min - 1, x_max + 1)
    ax.set_ylim(y_min - 1, y_max + 1)

    ax.set_title(
        f"{method_name.upper()} embeddings — {video_name}\nframes ≤ {f}"
    )

    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    plt.close(fig)
    buf.seek(0)

    frames.append(Image.open(buf).copy())
    buf.close()


# ============================================================
# SAVE OUTPUT GIF
# ============================================================

frames[0].save(
    output_gif,
    save_all=True,
    append_images=frames[1:],
    duration=100,
    loop=0
)

print("✅ GIF saved:", output_gif)