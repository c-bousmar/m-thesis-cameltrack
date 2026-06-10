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
# ID SWITCH-FOCUSED EMBEDDING VISUALIZATION
# ============================================================

"""
This script visualizes embedding trajectories in a multi-object
tracking (MOT) system, but only for identities involved in ID switches.

Unlike full embedding visualizations, this version isolates
the problematic identities to better understand failure modes.

The goal is to answer:
- Do ID switch tracks form ambiguous clusters?
- Are these embeddings overlapping with other identities?
- Can representation space explain tracking failures?

Only embeddings belonging to identities that experienced
at least one ID switch (ID_S) are displayed.

Input files:
- tracks_dets_embs.txt: embedding vectors
- tracks_dets_ids.txt: identity labels
- failure_cases/<video>.csv: ID switch annotations

Output:
- GIF showing embedding space evolution
- restricted to ID-switching identities only

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

embeddings_file = info_folder + "/tracks_dets_embs.txt"
ids_file        = info_folder + "/tracks_dets_ids.txt"
colors_file     = info_folder + "/tracklab_cmap.npy"
failure_file    = info_folder + "/eval/failure_cases/" + video_name + ".csv"

use_tsne = True
method_name = "tsne" if use_tsne else "pca"
SHOW = "both"  # "tracklets", "detections", "both"

output_gif = info_folder + f"/{method_name}_{video_name}_GAFFE_{frame_start}_{frame_end}_IDsw_only.gif"


# ============================================================
# RECONSTRUCT SEQUENCE FRAME INDEXING
# ============================================================

"""
The dataset is composed of multiple sequential video folders.
Each sequence defines a local frame range stored in seqinfo.ini.

This step reconstructs global frame indexing so that embeddings
can be correctly aligned with temporal positions.
"""

frame_start_end_dict = {}
start_frame = 1

seq_dirs = sorted([
    d for d in os.listdir(dataset_folder)
    if os.path.isdir(os.path.join(dataset_folder, d))
])

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
# EXTRACT ID SWITCHING IDENTITIES
# ============================================================

"""
From failure logs, extract all track IDs that experienced
at least one ID switch (ID_S) within the selected interval.

These identities define the subset of embeddings to visualize.
"""

id_switch_ids = set()

if os.path.exists(failure_file):
    with open(failure_file, "r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            issue = row["ISSUE"].strip()
            pred_raw = row.get("PRED_ID", "").strip()
            frame_raw = row.get("FRAME", "").strip()

            if issue != "ID_S":
                continue

            if pred_raw in ("", "None", "nan"):
                continue

            if frame_raw in ("", "None", "nan"):
                continue

            global_frame = int(float(frame_raw))

            # Keep only events within the selected time window
            if frame_start <= global_frame <= frame_end:
                id_switch_ids.add(int(float(pred_raw)))

print("ID switch IDs in selected interval:", id_switch_ids)


# ============================================================
# LOAD AND FILTER EMBEDDINGS
# ============================================================

"""
Load embeddings and keep only those:
- within the selected sequence
- within the selected temporal window

Unlike previous scripts, we do NOT filter by ID switch here.
Filtering is applied later at visualization time.
"""

embeddings_list = []
frame_ids = []
types = []
track_ids = []

with open(embeddings_file, "r") as f_emb, open(ids_file, "r") as f_ids:

    for line_emb, line_id in zip(f_emb, f_ids):

        line_emb = line_emb.strip()
        line_id  = line_id.strip()

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
                types.append(1 if obj_type == "T" else 0)
                track_ids.append(int(line_id))


embeddings_array = np.array(embeddings_list, dtype=np.float32)
frame_ids = np.array(frame_ids)
types = np.array(types)
track_ids = np.array(track_ids)


# ============================================================
# DIMENSIONALITY REDUCTION
# ============================================================

"""
Apply PCA or t-SNE to project embeddings into 2D space.

- t-SNE: emphasizes local structure and cluster separation
- PCA: preserves global variance structure
"""

if use_tsne:
    reducer = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate=200,
        random_state=42
    )
else:
    reducer = PCA(n_components=2)

embeddings_2d = reducer.fit_transform(embeddings_array)


# ============================================================
# COLOR MAPPING
# ============================================================

"""
TrackLab colormap ensures consistent coloring per identity.
"""

BASE_COLORS = np.load(colors_file)
cmap_size = len(BASE_COLORS)

colors = []
for tid in track_ids:
    if tid == -1:
        colors.append((0.6, 0.6, 0.6))
    else:
        idx = (tid - 1) % cmap_size
        colors.append(BASE_COLORS[idx])

colors = np.array(colors)


# ============================================================
# GLOBAL AXIS LIMITS
# ============================================================

x_min, x_max = embeddings_2d[:, 0].min(), embeddings_2d[:, 0].max()
y_min, y_max = embeddings_2d[:, 1].min(), embeddings_2d[:, 1].max()


# ============================================================
# GENERATE GIF (ID SWITCH ONLY VISUALIZATION)
# ============================================================

"""
At each timestep, only embeddings belonging to identities
that experienced at least one ID switch are displayed.

This isolates failure-prone tracks to analyze:
- cluster ambiguity
- embedding overlap
- temporal instability
"""

frames = []

for f in range(frame_start, frame_end + 1):

    fig, ax = plt.subplots(figsize=(8, 6))

    mask_frame = frame_ids <= f

    # Select only identities that have ID switches
    mask_idsw = np.isin(track_ids, list(id_switch_ids))

    track_mask = mask_frame & mask_idsw & (types == 1)
    det_mask   = mask_frame & mask_idsw & (types == 0)

    if SHOW in ["tracklets", "both"]:
        ax.scatter(
            embeddings_2d[track_mask, 0],
            embeddings_2d[track_mask, 1],
            c=colors[track_mask],
            marker="o",
            s=30,
            alpha=0.9,
            label="Tracklets"
        )

    if SHOW in ["detections", "both"]:
        ax.scatter(
            embeddings_2d[det_mask, 0],
            embeddings_2d[det_mask, 1],
            c=colors[det_mask],
            marker="^",
            s=30,
            alpha=0.9,
            label="Detections"
        )

    ax.set_xlim(x_min - 1, x_max + 1)
    ax.set_ylim(y_min - 1, y_max + 1)

    ax.set_title(f"{method_name.upper()} embeddings — ID Switch tracks only")
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