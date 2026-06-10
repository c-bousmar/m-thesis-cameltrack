import cv2
from PIL import Image
import numpy as np
import os
import configparser


# ============================================================
# SIDE-BY-SIDE VISUALIZATION: RAW VIDEO + EMBEDDING GIF
# ============================================================

"""
This script aligns a raw tracking video with a precomputed
embedding visualization (stored as a GIF) and generates a
side-by-side comparison video.

The objective is to provide a synchronized qualitative tool
to analyze how embedding space evolution relates to actual
video content.

Left panel:
- Original video frames

Right panel:
- t-SNE / PCA embedding visualization (stored as GIF)

This is particularly useful to:
- visually correlate ID switches with embedding behavior
- observe cluster formation over time
- validate whether representation space aligns with motion/identity

The final output is an MP4 video combining both sources.
"""


# ============================================================
# PARAMETERS
# ============================================================

folder = "/CECI/home/ucl/elen/cbousmar/CAMELTrack/outputs/CAMELTrack_SportsMOT/2026-02-19/19-11-41"
dataset_folder = "/globalsc/ucl/elen/cbousmar/datasets/SportsMOT/val"

frame_start = 650
frame_end   = 720
video_name  = "v_00HRwkvvjtQ_c003"

speed_factor = 0.5

video_path = os.path.join(folder, "visualization/videos", f"{video_name}.mp4")
gif_path   = os.path.join(folder, f"tsne_{video_name}_GAFFE_{frame_start}_{frame_end}.gif")
output_path = os.path.join(folder, f"tsne_{video_name}_GAFFE_{frame_start}_{frame_end}_side_by_side.mp4")


# ============================================================
# RECONSTRUCT GLOBAL FRAME INDEXING (DATASET STRUCTURE)
# ============================================================

"""
The dataset is composed of multiple sequential video folders.
Each sequence has its own frame indexing stored in seqinfo.ini.

This step reconstructs global frame ranges to align:
- video frames
- embedding frames
"""

frame_start_end_dict = {}
start_frame_global = 1

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
        end_frame_global = start_frame_global + seq_length - 2

        frame_start_end_dict[seq_name] = (start_frame_global, end_frame_global)
        start_frame_global = end_frame_global + 1

seq_frame_start, seq_frame_end = frame_start_end_dict[video_name]


# ============================================================
# LOAD GIF FRAMES (EMBEDDING VISUALIZATION)
# ============================================================

"""
The embedding visualization is stored as a GIF.
We extract each frame so it can be synchronized with the video.
"""

gif = Image.open(gif_path)
gif_frames = []

try:
    while True:
        gif_frames.append(gif.convert("RGB"))
        gif.seek(gif.tell() + 1)
except EOFError:
    pass


# ============================================================
# OPEN ORIGINAL VIDEO STREAM
# ============================================================

cap = cv2.VideoCapture(video_path)

orig_fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Adjust playback speed for synchronization
out_fps = orig_fps * speed_factor

# Output video has doubled width (side-by-side layout)
out_width = width * 2
out_height = height

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, out_fps, (out_width, out_height))


# ============================================================
# FRAME SYNCHRONIZATION LOOP
# ============================================================

"""
For each video frame:
- match corresponding embedding frame (GIF frame)
- concatenate both views horizontally
- optionally slow down playback for better readability
"""

frame_idx = 0
gif_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    local_frame_id = frame_idx

    # Only process frames within selected interval
    if frame_start <= local_frame_id <= frame_end:

        # Resize embedding frame to match video resolution
        gif_frame = np.array(gif_frames[gif_idx].resize((width, height)))
        gif_idx += 1

        # Convert RGB (PIL) → BGR (OpenCV)
        gif_frame = cv2.cvtColor(gif_frame, cv2.COLOR_RGB2BGR)

        # Side-by-side concatenation
        combined = np.concatenate((frame, gif_frame), axis=1)

        # Repeat frames to slow down playback
        repeat_count = int(round(1 / speed_factor))
        for _ in range(repeat_count):
            out.write(combined)

    elif local_frame_id > frame_end:
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()
out.release()

print(f"✅ Saved side-by-side video: {output_path}")