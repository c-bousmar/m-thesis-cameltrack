import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

"""
============================================================
RELATIVE OBJECT-LEVEL ATTENTION VISUALIZATION SCRIPT
============================================================
This script visualizes the evolution of object-level attention
in a multi-object tracking (MOT) context.

It focuses on how a target identity distributes its attention
towards other interacting entities over time.

For each timestamp, the script extracts self-attention weights
from logged model outputs and tracks how much importance is
assigned to different surrounding identities.

The visualization shows:
- Temporal evolution of attention from a target ID
- Relative contribution of interacting IDs
- Optional temporal smoothing and windowing
- Optional frame-level event markers

Input files:
- self_attention_weights.txt: attention matrices per frame
- tracks_dets_ids.txt: mapping between detections and IDs
- tracks_dets_embs.txt: frame-wise embeddings (alignment)

Output:
A line plot showing how the target identity
adjusts its attention distribution over time depending on
interactions with other objects.

This helps interpret:
- identity ambiguity in crowded scenes
- interaction-driven attention shifts
- robustness of tracking associations
============================================================
"""

# ============================================================
# CONFIGURATION
# ============================================================

info_folder = "/globalsc/ucl/elen/athieltg/Master_Thesis_MOT/outputs/CAMELTrackDanceTrack/2026-04-09/GAFFE3-2B sim_th=0.1"

video_name = "dancetrack0065"
target_pred_id = 111
interactors_to_plot = [112, 114, 115, 116]

# Optional temporal visualization window (local frame indices)
USE_WINDOW = True
FRAME_WINDOW = (450, 550)

# Optional temporal reference markers
frame_marker = []

SMOOTHING_SIGMA = 1

# Input files
self_weights_file = os.path.join(info_folder, "self_attention_weights.txt")
ids_file          = os.path.join(info_folder, "tracks_dets_ids.txt")
embs_file         = os.path.join(info_folder, "tracks_dets_embs.txt")
colors_file       = os.path.join(info_folder, "tracklab_cmap.npy")

# Output
if USE_WINDOW:
    output_plot = os.path.join(
        info_folder,
        f"relative_IDs_attention_{video_name}_target_{target_pred_id}_window_{FRAME_WINDOW[0]}-{FRAME_WINDOW[1]}.png"
    )
else:
    output_plot = os.path.join(
        info_folder,
        f"relative_IDs_attention_{video_name}_target_{target_pred_id}_full.png"
    )

# ============================================================
# 1. COLOR MAPPING
# ============================================================

"""
Load TrackLab colormap if available.
Used to ensure consistent identity coloring across visualizations.
"""

if os.path.exists(colors_file):
    BASE_COLORS = np.load(colors_file)
    cmap_size = len(BASE_COLORS)
else:
    BASE_COLORS = None


def get_id_color(id_val):
    """Return deterministic color for a given identity."""
    if BASE_COLORS is not None:
        return BASE_COLORS[(id_val - 1) % cmap_size]
    return None


# ============================================================
# 2. FRAME → ID MAPPING
# ============================================================

"""
Reconstruct identity assignments per frame using synchronized logs.
This allows consistent indexing between embeddings and attention logs.
"""

frame_to_ids = {}

with open(embs_file, "r") as f_e, open(ids_file, "r") as f_i:
    for line_e, line_i in zip(f_e, f_i):
        parts_e = line_e.strip().split()
        if not parts_e:
            continue

        frame_id = int(parts_e[0])
        real_id = int(line_i.strip())

        frame_to_ids.setdefault(frame_id, []).append(real_id)


# ============================================================
# 3. ATTENTION EXTRACTION
# ============================================================

"""
Extract normalized self-attention weights for:
- target identity
- selected interacting identities
"""

selected_ids = [target_pred_id] + interactors_to_plot
attention_results = {oid: {} for oid in selected_ids}

with open(self_weights_file, "r") as f_s:
    for line in f_s:
        parts = line.strip().split()
        if not parts:
            continue

        frame_id = int(parts[0])
        obj_type = parts[1]
        obj_idx  = int(parts[2])
        weights  = np.array([float(x) for x in parts[3:]])

        ids_in_frame = frame_to_ids.get(frame_id, [])

        # Keep only target track-level entries
        if obj_type == "T" and obj_idx < len(ids_in_frame):
            if ids_in_frame[obj_idx] == target_pred_id:

                local_weights = {
                    ids_in_frame[c]: w
                    for c, w in enumerate(weights)
                    if c < len(ids_in_frame) and ids_in_frame[c] in selected_ids
                }

                norm = sum(local_weights.values())
                if norm > 0:
                    for oid, w in local_weights.items():
                        attention_results[oid][frame_id] = w / norm


# ============================================================
# 4. TEMPORAL VISUALIZATION
# ============================================================

"""
Visualize identity-level attention evolution over time.
Each curve represents how much the target attends to:
- itself
- surrounding interacting identities
"""

fig, ax = plt.subplots(figsize=(14, 7))

global_frames = sorted(attention_results[target_pred_id].keys())

if not global_frames:
    raise ValueError(f"No data found for target ID {target_pred_id}")

seq_offset = global_frames[0] - 1
local_frames = [f - seq_offset for f in global_frames]


for oid in selected_ids:
    if not attention_results[oid]:
        continue

    y_vals = [attention_results[oid].get(f, 0) for f in global_frames]

    if SMOOTHING_SIGMA > 0 and len(y_vals) > SMOOTHING_SIGMA:
        y_vals = gaussian_filter1d(y_vals, sigma=SMOOTHING_SIGMA)

    color = get_id_color(oid)

    if oid == target_pred_id:
        label = f"Target (ID {oid})"
        lw, alpha, z = 3.0, 1.0, 10
    else:
        label = f"ID {oid}"
        lw, alpha, z = 1.5, 0.7, 5

    ax.plot(local_frames, y_vals,
            label=label,
            color=color,
            linewidth=lw,
            alpha=alpha,
            zorder=z)


# ============================================================
# 5. TEMPORAL MARKERS
# ============================================================

for marker in frame_marker:
    if min(local_frames) <= marker <= max(local_frames):
        ax.axvline(x=marker, color='black', linestyle='--', linewidth=1.2, alpha=0.5)

        ax.text(
            marker, -0.015, str(marker),
            color='#2c3e50',
            fontsize=9,
            fontweight='bold',
            ha='center',
            va='top',
            rotation=90,
            transform=ax.get_xaxis_transform()
        )


# ============================================================
# 6. STYLING
# ============================================================

if USE_WINDOW:
    ax.set_xlim(FRAME_WINDOW[0], FRAME_WINDOW[1])
else:
    ax.set_xlim(min(local_frames), max(local_frames))

ax.set_title(
    f"{video_name} - Identity-level attention dynamics - Target ID {target_pred_id}",
    fontsize=16,
    pad=25
)

ax.set_xlabel("Frame", fontsize=12, labelpad=15)
ax.set_ylabel("Relative attention weight", fontsize=12)

ax.set_ylim(-0.02, 1.05)

ax.grid(True, linestyle=':', alpha=0.5)
ax.legend(loc='best', frameon=True, shadow=True, fontsize=10, title="Identity IDs")

for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color('#333333')
    spine.set_linewidth(1.0)

plt.tight_layout()
plt.savefig(output_plot, dpi=230, bbox_inches='tight')
plt.show()