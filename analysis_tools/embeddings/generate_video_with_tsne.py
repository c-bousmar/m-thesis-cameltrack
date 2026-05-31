import cv2
from PIL import Image
import numpy as np
import os
import configparser

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
# IDENTIFY GLOBAL FRAME RANGES FOR EACH VIDEO
# ============================================================
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
# LOAD GIF FRAMES
# ============================================================
gif = Image.open(gif_path)
gif_frames = []
try:
    while True:
        gif_frames.append(gif.convert("RGB"))
        gif.seek(gif.tell() + 1)
except EOFError:
    pass

# ============================================================
# OPEN VIDEO
# ============================================================
cap = cv2.VideoCapture(video_path)
orig_fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out_fps = orig_fps * speed_factor

out_width = width * 2
out_height = height
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, out_fps, (out_width, out_height))

# ============================================================
# PROCESS FRAMES
# ============================================================
frame_idx = 0
gif_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    local_frame_id = frame_idx

    if frame_start <= local_frame_id <= frame_end:
        gif_frame = np.array(gif_frames[gif_idx].resize((width, height)))
        gif_idx += 1
        gif_frame = cv2.cvtColor(gif_frame, cv2.COLOR_RGB2BGR)
        combined = np.concatenate((frame, gif_frame), axis=1)
        repeat_count = int(round(1 / speed_factor))
        for _ in range(repeat_count):
            out.write(combined)
    elif local_frame_id > frame_end:
        break

# ============================================================
# RELEASE RESOURCES
# ============================================================
cap.release()
out.release()
print(f"✅ Saved side-by-side video: {output_path}")