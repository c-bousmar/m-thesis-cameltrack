import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from PIL import Image
import io
import configparser
import os


# ============================================================
# TEMPORAL ENCODER FUSED EMBEDDING VISUALIZATION (t-SNE / PCA)
# ============================================================

"""
This script visualizes high-dimensional fused token representations
produced by the Temporal Encoders (TEs) using dimensionality
reduction (t-SNE or PCA).

IMPORTANT:
- Unlike GAFFE output embeddings, this version uses aggregated
  outputs from the Temporal Encoders (TEs).
- Each point corresponds to a tracklet or detection token.
- The evolution is visualized over time as frames are progressively revealed.
"""

# ============================================================
# PARAMETERS
# ============================================================

info_folder = "/globalsc/ucl/elen/athieltg/Master_Thesis_MOT/outputs/CAMELTrackMOT17/2026-02-17/16-44-52 mot17 split 50-25-25"
dataset_folder = "/globalsc/ucl/elen/athieltg/Master_Thesis_MOT/data/MOT17/test_50_25_25"

tokens_file = info_folder + "/tracks_dets_tokens.txt"
ids_file    = info_folder + "/tracks_dets_ids.txt"
colors_file = info_folder + "/tracklab_cmap.npy"

frame_start = 1
frame_end   = 60
video_name = "MOT17-09-FRCNN"

use_tsne = True
method_name = "tsne" if use_tsne else "pca"
SHOW = "both"
output_gif = info_folder + f"/{method_name}_{video_name}_tokens_{SHOW}_{frame_start}_{frame_end}.gif"


# ============================================================
# SEQUENCE FRAME INDEX RECONSTRUCTION
# ============================================================

"""
Reconstructs global frame indexing from dataset structure.

Each sequence in MOT dataset has its own local frame indexing,
stored in seqinfo.ini. This block maps them into a continuous
global timeline.
"""

frame_start_end_dict = {}
start_frame = 1

seq_dirs = [d for d in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, d))]
seq_dirs = sorted(seq_dirs)

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

print("Sequence frames:")
for seq_name, seq_len in frame_start_end_dict.items():
    print(seq_name, seq_len)


# ============================================================
# LOAD TRACK IDS
# ============================================================

"""
Loads track IDs aligned with embeddings/tokens.

This is used to color points consistently across identities.
"""

track_ids_full = []
with open(ids_file, "r") as f:
    for line in f:
        line = line.strip()
        if line:
            track_ids_full.append(int(line))
track_ids_full = np.array(track_ids_full)


# ============================================================
# LOAD TOKENS (TE OUTPUTS) + FILTER BY FRAME RANGE
# ============================================================

"""
Loads token embeddings produced by Temporal Encoders (TEs).

Each line contains:
- frame index
- object type (T = tracklet, D = detection)
- token embedding vector

Only tokens within the selected frame range are retained.
"""

tokens_list = []
frame_ids = []
types = []
track_ids = []

with open(tokens_file, "r") as f_tok, open(ids_file, "r") as f_ids:
    for line_idx, (line_tok, line_id) in enumerate(zip(f_tok, f_ids)):

        line_tok = line_tok.strip()
        line_id  = line_id.strip()

        if not line_tok:
            continue

        parts = line_tok.split()
        frame_id = int(parts[0])
        obj_type = parts[1]

        seq_frame_start, seq_frame_end = frame_start_end_dict[video_name]

        if seq_frame_start <= frame_id <= seq_frame_end:
            local_frame_id = frame_id - seq_frame_start + 1

            if frame_start <= local_frame_id <= frame_end:

                values = [float(x) for x in parts[2:] if x.lower() != "nan"]

                tokens_list.append(values)
                frame_ids.append(local_frame_id)
                types.append(1 if obj_type == "T" else 0)
                track_ids.append(int(line_id))


tokens_array = np.array(tokens_list, dtype=np.float32)
frame_ids = np.array(frame_ids)
types = np.array(types)
track_ids = np.array(track_ids)

if len(track_ids) != len(tokens_array):
    raise ValueError("IDs and tokens length mismatch after filtering.")


# ============================================================
# DIMENSIONALITY REDUCTION (t-SNE / PCA)
# ============================================================

"""
Projects high-dimensional TE token embeddings into 2D space
for visualization.

- t-SNE: preserves local neighborhood structure
- PCA: preserves global variance structure
"""

if use_tsne:
    reducer = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
else:
    reducer = PCA(n_components=2)

tokens_2d = reducer.fit_transform(tokens_array)


# ============================================================
# COLOR ASSIGNMENT (TRACKLAB COLORMAP)
# ============================================================

"""
Assigns a consistent color per track identity using TrackLab palette.
Unassigned identities are shown in gray.
"""

BASE_COLORS = np.load(colors_file)
cmap_size = len(BASE_COLORS)

colors = []
for tid in track_ids:
    if tid == -1:
        colors.append((0.6, 0.6, 0.6))
    else:
        idx = (int(tid) - 1) % cmap_size
        colors.append(BASE_COLORS[idx])
colors = np.array(colors)


# ============================================================
# GLOBAL VISUALIZATION BOUNDS
# ============================================================

x_min, x_max = tokens_2d[:, 0].min(), tokens_2d[:, 1].max()
y_min, y_max = tokens_2d[:, 1].min(), tokens_2d[:, 1].max()


# ============================================================
# GIF GENERATION OVER TIME
# ============================================================

"""
Creates an animation showing how token embeddings evolve over time.

At each frame:
- Only tokens up to time t are displayed
- Tracklets and detections are optionally separated
"""

frames = []

for f in range(frame_start, frame_end + 1):

    fig, ax = plt.subplots(figsize=(8, 6))

    mask_frame = frame_ids <= f
    track_mask = mask_frame & (types == 1)
    det_mask   = mask_frame & (types == 0)

    if SHOW in ["tracklets", "both"]:
        ax.scatter(
            tokens_2d[track_mask, 0],
            tokens_2d[track_mask, 1],
            c=colors[track_mask],
            marker="o",
            s=30,
            alpha=0.9,
            label="Tracklets"
        )

    if SHOW in ["detections", "both"]:
        ax.scatter(
            tokens_2d[det_mask, 0],
            tokens_2d[det_mask, 1],
            c=colors[det_mask],
            marker="^",
            s=30,
            alpha=0.9,
            label="Detections"
        )

    ax.set_xlim(x_min - 1, x_max + 1)
    ax.set_ylim(y_min - 1, y_max + 1)

    ax.set_title(f"{method_name.upper()} tokens (frames ≤ {f})")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")

    ax.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    plt.close(fig)
    buf.seek(0)

    frame_img = Image.open(buf)
    frames.append(frame_img.copy())
    buf.close()


# ============================================================
# SAVE GIF
# ============================================================

frames[0].save(output_gif, save_all=True, append_images=frames[1:], duration=100, loop=0)

print("✅ GIF saved:", output_gif)