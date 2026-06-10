import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from collections import defaultdict
from pathlib import Path


# ============================================================
# IDENTITY PERSISTENCE TIMELINE VISUALIZATION
# ============================================================

"""
This script visualizes the temporal evolution of tracking
associations between ground-truth identities and predicted IDs
in a multi-object tracking (MOT) system.

Each horizontal row corresponds to a ground-truth identity.
The timeline shows how predicted IDs are assigned over time,
allowing direct inspection of association errors.

Represented events:
- Correct association segments (colored bars with predicted IDs)
- Identity switches (changes in predicted ID along time)
- False negatives (thin black segments where no prediction exists)
- Missing or unannotated frames (empty intervals)

This visualization is designed for qualitative error analysis,
in particular to detect:
- ID switches (IDSW)
- track fragmentation
- missed detections (FN)

It provides a compact temporal view of tracking stability.
============================================================
"""

# ============================================================
# CONFIGURATION
# ============================================================

info_folder = "/globalsc/ucl/elen/athieltg/Master_Thesis_MOT/outputs/CAMELTrackDanceTrack/2026-05-06/GAFFE (3)"
video_name  = "dancetrack0030"

failure_file = os.path.join(
    info_folder, "eval", "failure_cases", f"{video_name}.csv"
)

colors_file = os.path.join(info_folder, "tracklab_cmap.npy")

output_file = os.path.join(
    info_folder,
    f"IDs_correspondence_{video_name}.png"
)

FRAME_SCALE = 0.5
BAR_HEIGHT = 0.4
FN_BAR_THICKNESS = 0.20
EMPTY_BAR_THICKNESS = 0.02

# Optional frames to highlight on the timeline
ID_S_FRAMES = []


# ============================================================
# LOAD COLOR MAP
# ============================================================

"""
Load TrackLab colormap used for consistent identity visualization.
"""

BASE_COLORS = np.load(colors_file)
cmap_size = len(BASE_COLORS)


# ============================================================
# LOAD FAILURE DATA + FRAME RANGE
# ============================================================

"""
Parse failure_cases CSV and reconstruct per-GT temporal annotations:
- FN (false negatives)
- mismatches (FP / ID switches)

Also extracts global frame range of the sequence.
"""

gt_frame_info = defaultdict(dict)
all_frames = []

with open(failure_file, "r") as f:
    reader = csv.DictReader(f)

    for row in reader:
        try:
            frame = int(row["FRAME"])
        except:
            continue

        all_frames.append(frame)
        issue = row["ISSUE"]

        if row["GT_ID"] == "" or row["GT_ID"] is None:
            continue

        gt_id = int(row["GT_ID"])

        if issue == "FN":
            gt_frame_info[gt_id][frame] = ("FN", None)
        else:
            try:
                pred_id = int(row["PRED_ID"])
            except:
                continue

            gt_frame_info[gt_id][frame] = (issue, pred_id)


frame_start = min(all_frames)
frame_end   = max(all_frames)

all_gt_ids = sorted(gt_frame_info.keys())
n_gt = len(all_gt_ids)
n_frames = frame_end - frame_start + 1


# ============================================================
# BUILD TEMPORAL PIVOT MATRIX
# ============================================================

"""
Convert sparse annotations into a dense matrix:
shape = [num_GT_ids, num_frames]

Values:
- NaN  -> no annotation
- -1   -> FN
- >0   -> predicted track ID
"""

pivot = np.full((n_gt, n_frames), np.nan)

for i, gt_id in enumerate(all_gt_ids):
    for frame in range(frame_start, frame_end + 1):

        col = frame - frame_start

        if frame not in gt_frame_info[gt_id]:
            continue

        issue, pred_id = gt_frame_info[gt_id][frame]

        if issue == "FN":
            pivot[i, col] = -1
        elif pred_id is not None:
            pivot[i, col] = pred_id


# ============================================================
# COLOR ASSIGNMENT
# ============================================================

"""
Assign a unique color to each predicted track ID.
"""

unique_pred_ids = sorted({
    int(x)
    for x in pivot.flatten()
    if not np.isnan(x) and x > 0
})

pid_to_color = {
    pid: BASE_COLORS[(pid - 1) % cmap_size]
    for pid in unique_pred_ids
}

FN_COLOR = (0, 0, 0)
EMPTY_COLOR = (0, 0, 0)


# ============================================================
# HELPER FUNCTION
# ============================================================

def draw_id_text(ax, x, y, pid, color):
    """
    Draw identity label with automatic contrast adjustment.
    """
    r, g, b = color[:3]
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    txt_color = "white" if luminance < 0.5 else "black"

    ax.text(
        x, y, str(int(pid)),
        ha='center',
        va='center',
        fontsize=12,
        fontweight='bold',
        color=txt_color,
        zorder=3
    )


# ============================================================
# VISUALIZATION
# ============================================================

fig_w = max(12, n_frames * FRAME_SCALE / 8)
fig_h = max(4, n_gt * 0.35)

fig, ax = plt.subplots(figsize=(fig_w, fig_h))

ax.set_facecolor("white")

ax.set_xlim((frame_start - 0.5) * FRAME_SCALE,
            (frame_end + 0.5) * FRAME_SCALE)

ax.set_ylim(-0.5, n_gt - 0.5)

ax.set_yticks(range(n_gt))
ax.set_yticklabels(all_gt_ids)

ax.invert_yaxis()

ax.set_xlabel("Frame")
ax.set_ylabel("GT ID")

ax.set_title(f"Ground truth & prediction timeline ({video_name})")


# ============================================================
# DRAW TEMPORAL SEGMENTS
# ============================================================

for row_idx, gt_id in enumerate(all_gt_ids):

    preds = pivot[row_idx]

    cur_pid = None
    start_idx = None

    for i, pid in enumerate(preds):

        # -------------------------
        # Missing frame
        # -------------------------
        if np.isnan(pid):

            if cur_pid is not None:
                f0 = start_idx + frame_start
                f1 = i - 1 + frame_start
                width = (f1 - f0 + 1) * FRAME_SCALE

                if cur_pid == -1:
                    ax.add_patch(Rectangle(
                        ((f0 - 0.5) * FRAME_SCALE, row_idx - FN_BAR_THICKNESS / 2),
                        width,
                        FN_BAR_THICKNESS,
                        facecolor=FN_COLOR,
                        edgecolor='none',
                        zorder=2
                    ))
                else:
                    color = pid_to_color.get(cur_pid, (0.2, 0.2, 0.2))
                    ax.add_patch(Rectangle(
                        ((f0 - 0.5) * FRAME_SCALE, row_idx - BAR_HEIGHT / 2),
                        width,
                        BAR_HEIGHT,
                        facecolor=color,
                        edgecolor='none',
                        zorder=2
                    ))
                    draw_id_text(ax, (f0 + f1) / 2 * FRAME_SCALE, row_idx, cur_pid, color)

                cur_pid, start_idx = None, None

            # empty frame
            ax.add_patch(Rectangle(
                ((i + frame_start - 0.5) * FRAME_SCALE, row_idx - EMPTY_BAR_THICKNESS / 2),
                FRAME_SCALE,
                EMPTY_BAR_THICKNESS,
                facecolor=EMPTY_COLOR,
                edgecolor='none',
                zorder=1
            ))
            continue

        # -------------------------
        # Identity segment update
        # -------------------------
        pid_int = int(pid)

        if cur_pid is None:
            cur_pid, start_idx = pid_int, i
            continue

        if pid_int != cur_pid:

            f0 = start_idx + frame_start
            f1 = i - 1 + frame_start
            width = (f1 - f0 + 1) * FRAME_SCALE

            if cur_pid == -1:
                ax.add_patch(Rectangle(
                    ((f0 - 0.5) * FRAME_SCALE, row_idx - FN_BAR_THICKNESS / 2),
                    width,
                    FN_BAR_THICKNESS,
                    facecolor=FN_COLOR,
                    edgecolor='none',
                    zorder=2
                ))
            else:
                color = pid_to_color.get(cur_pid, (0.2, 0.2, 0.2))
                ax.add_patch(Rectangle(
                    ((f0 - 0.5) * FRAME_SCALE, row_idx - BAR_HEIGHT / 2),
                    width,
                    BAR_HEIGHT,
                    facecolor=color,
                    edgecolor='none',
                    zorder=2
                ))
                draw_id_text(ax, (f0 + f1) / 2 * FRAME_SCALE, row_idx, cur_pid, color)

            cur_pid, start_idx = pid_int, i

    # -------------------------
    # Close last segment
    # -------------------------
    if cur_pid is not None:

        f0 = start_idx + frame_start
        f1 = frame_end
        width = (f1 - f0 + 1) * FRAME_SCALE

        if cur_pid == -1:
            ax.add_patch(Rectangle(
                ((f0 - 0.5) * FRAME_SCALE, row_idx - FN_BAR_THICKNESS / 2),
                width,
                FN_BAR_THICKNESS,
                facecolor=FN_COLOR,
                edgecolor='none',
                zorder=2
            ))
        else:
            color = pid_to_color.get(cur_pid, (0.2, 0.2, 0.2))
            ax.add_patch(Rectangle(
                ((f0 - 0.5) * FRAME_SCALE, row_idx - BAR_HEIGHT / 2),
                width,
                BAR_HEIGHT,
                facecolor=color,
                edgecolor='none',
                zorder=2
            ))
            draw_id_text(ax, (f0 + f1) / 2 * FRAME_SCALE, row_idx, cur_pid, color)


# ============================================================
# HIGHLIGHT FRAMES
# ============================================================

y_bottom = n_gt - 0.5 + 0.05

for f in ID_S_FRAMES:
    x = f * FRAME_SCALE
    ax.axvline(x=x, color='red', linestyle=':', linewidth=0.8, zorder=1)
    ax.text(
        x, y_bottom, str(f),
        ha='center',
        va='top',
        fontsize=10,
        rotation=90,
        color='red',
        zorder=5
    )


# ============================================================
# GRID + AXIS TICKS
# ============================================================

ax.grid(True, axis='x', linestyle=':', linewidth=0.6, alpha=0.7)

step = max(1, int(round(n_frames / 25)))
tick_labels = list(range(frame_start, frame_end + 1, step))
tick_positions = [t * FRAME_SCALE for t in tick_labels]

ax.set_xticks(tick_positions)
ax.set_xticklabels([str(t) for t in tick_labels])


# ============================================================
# SAVE FIGURE
# ============================================================

plt.tight_layout()
plt.savefig(output_file, dpi=200)
plt.close()

print("✅ Graph saved:", output_file)