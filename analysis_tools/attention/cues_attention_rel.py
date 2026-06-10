import numpy as np
import matplotlib.pyplot as plt
import os
import configparser


# ============================================================
# CONFIGURATION
# ============================================================

info_folder = "/globalsc/ucl/elen/athieltg/Master_Thesis_MOT/outputs/CAMELTrackSportsMOT/2026-04-12/GAFFE3-2B sim_th=0.1"
dataset_folder = "/globalsc/ucl/elen/athieltg/Master_Thesis_MOT/data/SportsMOT/test"

video_name = "v_cC2mHWqMcjk_c007"
target_pred_id = 150

# Optional frame markers for temporal annotation
frame_marker = []

weights_file = os.path.join(info_folder, "cross_attention_weights.txt")
ids_file = os.path.join(info_folder, "tracks_dets_ids.txt")

cue_names = ["Appearance", "Keypoints", "BBox"]
cue_colors = ["#3498db", "#e74c3c", "#2ecc71"]

output_plot = os.path.join(
    info_folder,
    f"attention_{video_name}_ID_{target_pred_id}.png"
)


# ============================================================
# 1. VIDEO FRAME RANGE EXTRACTION
# ============================================================

"""
The dataset is composed of multiple sequential video folders.
Each sequence defines a local frame index range stored in seqinfo.ini.

This step reconstructs the global frame indexing across sequences.
"""

frame_start_end_dict = {}
start_frame = 1

seq_dirs = sorted([
    d for d in os.listdir(dataset_folder)
    if os.path.isdir(os.path.join(dataset_folder, d))
])

for seq_name in seq_dirs:
    seqinfo_file = os.path.join(dataset_folder, seq_name, "seqinfo.ini")

    if os.path.exists(seqinfo_file):
        config = configparser.ConfigParser()
        config.read(seqinfo_file)

        seq_length = int(config["Sequence"]["seqLength"])
        end_frame = start_frame + seq_length - 2

        frame_start_end_dict[seq_name] = (start_frame, end_frame)
        start_frame = end_frame + 1

seq_min, seq_max = frame_start_end_dict[video_name]


# ============================================================
# 2. SYNCHRONIZED LOG PARSING
# ============================================================

"""
This step reads two synchronized logs:

- cross_attention_weights.txt:
    stores per-frame cue fusion weights

- tracks_dets_ids.txt:
    stores association between predictions and track IDs

Only valid track-level entries corresponding to the target ID
are retained and normalized into probability distributions.
"""

data_dict = {}

with open(weights_file, "r") as f_w, open(ids_file, "r") as f_ids:
    for line_w, line_id in zip(f_w, f_ids):

        parts_w = line_w.strip().split()
        if not parts_w:
            continue

        frame_id = int(parts_w[0])

        if seq_min <= frame_id <= seq_max:

            current_id = int(line_id.strip())
            obj_type = parts_w[1]

            # Keep only track-level entries for the target identity
            if current_id == target_pred_id and obj_type == "T":

                weights = [float(x) for x in parts_w[3:]]
                total = sum(weights) if sum(weights) > 0 else 1

                rel_frame = frame_id - seq_min + 1

                # Normalize cue contributions
                data_dict[rel_frame] = [
                    w / total for w in weights[:3]
                ]


# ============================================================
# 3. TEMPORAL VISUALIZATION OF CUE IMPORTANCE
# ============================================================

"""
This step visualizes the evolution of cue importance over time
for a given tracked identity.

Each curve corresponds to a modality:
- Appearance
- Keypoints
- Bounding Box geometry
"""

sorted_frames = sorted(data_dict.keys())

if not sorted_frames:
    raise ValueError(f"No data found for track ID {target_pred_id}")

weights_array = np.array([data_dict[f] for f in sorted_frames])

fig, ax = plt.subplots(figsize=(14, 7))

# Plot modality-specific importance curves
for i, cue in enumerate(cue_names):
    ax.plot(
        sorted_frames,
        weights_array[:, i],
        label=cue,
        color=cue_colors[i],
        linewidth=3.0,
        alpha=0.8
    )

# Add temporal reference markers
for marker in frame_marker:
    if min(sorted_frames) <= marker <= max(sorted_frames):
        ax.axvline(
            x=marker,
            color='black',
            linestyle='--',
            linewidth=1.2,
            alpha=0.5
        )

        ax.text(
            marker,
            -0.015,
            f"{marker}",
            color='#2c3e50',
            fontsize=9,
            fontweight='bold',
            ha='center',
            va='top',
            rotation=90,
            transform=ax.get_xaxis_transform()
        )


# ============================================================
# 4. STYLING
# ============================================================

ax.set_title(
    f"{video_name} - Modality importance evolution - ID {target_pred_id}",
    fontsize=16,
    pad=25
)

ax.set_xlabel("Frame", fontsize=12, labelpad=15)
ax.set_ylabel("Relative attention weight", fontsize=12)

ax.set_ylim(-0.02, 1.05)
ax.set_xlim(min(sorted_frames), max(sorted_frames))

ax.grid(True, linestyle=':', alpha=0.5)
ax.legend(loc='best', frameon=True, shadow=True, fontsize=10)

# Frame styling for publication-quality rendering
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color('#333333')
    spine.set_linewidth(1.0)

plt.tight_layout()
plt.savefig(output_plot, dpi=230, bbox_inches='tight')
plt.show()