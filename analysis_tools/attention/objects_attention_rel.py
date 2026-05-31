import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter1d

# ============================================================
# PARAMETRES
# ============================================================
info_folder = "/globalsc/ucl/elen/athieltg/Master_Thesis_MOT/outputs/CAMELTrackDanceTrack/2026-04-09/GAFFE3-2B sim_th=0.1"
video_name = "dancetrack0065"
target_pred_id = 111
interactors_to_plot = [112, 114, 115, 116]

# --- REGLAGE DE LA FENÊTRE D'AFFICHAGE ---
USE_WINDOW = True    # Mettre à False pour afficher toute la vidéo
FRAME_WINDOW = (450, 550)  # Bornes en frames LOCALES (ex: de la frame 10 à 80)


# AJOUTE TES MARKERS ICI (ex: frames clés de crossover)
frame_marker = [] 

SMOOTHING_SIGMA = 1 

# Fichiers
self_weights_file = os.path.join(info_folder, "self_attention_weights.txt")
ids_file          = os.path.join(info_folder, "tracks_dets_ids.txt")
embs_file         = os.path.join(info_folder, "tracks_dets_embs.txt")
colors_file       = os.path.join(info_folder, "tracklab_cmap.npy")
if USE_WINDOW:
    output_plot = os.path.join(info_folder, f"relative_IDs_attention_{video_name}_target_{target_pred_id}_window_{FRAME_WINDOW[0]}-{FRAME_WINDOW[1]}.png")
else:    output_plot = os.path.join(info_folder, f"relative_IDs_attention_{video_name}_target_{target_pred_id}_full.png")

# ============================================================
# 1. CHARGEMENT DE LA COLORMAP ET MAPPING
# ============================================================
if os.path.exists(colors_file):
    BASE_COLORS = np.load(colors_file)
    cmap_size = len(BASE_COLORS)
else:
    BASE_COLORS = None

def get_id_color(id_val):
    if BASE_COLORS is not None:
        return BASE_COLORS[(id_val - 1) % cmap_size]
    return None

frame_to_ids = {}
with open(embs_file, "r") as f_e, open(ids_file, "r") as f_i:
    for line_e, line_i in zip(f_e, f_i):
        parts_e = line_e.strip().split()
        if not parts_e: continue
        f_id = int(parts_e[0])
        real_id = int(line_i.strip())
        if f_id not in frame_to_ids:
            frame_to_ids[f_id] = []
        frame_to_ids[f_id].append(real_id)

# ============================================================
# 2. LECTURE ET NORMALISATION
# ============================================================
selected_ids = [target_pred_id] + interactors_to_plot
attention_results = {oid: {} for oid in selected_ids}

with open(self_weights_file, "r") as f_s:
    for line in f_s:
        parts = line.strip().split()
        if not parts: continue
        f_id, obj_type, obj_idx = int(parts[0]), parts[1], int(parts[2])
        weights = [float(x) for x in parts[3:]]
        ids_in_frame = frame_to_ids.get(f_id, [])
        
        if obj_type == "T" and obj_idx < len(ids_in_frame) and ids_in_frame[obj_idx] == target_pred_id:
            local_weights = {ids_in_frame[c]: w for c, w in enumerate(weights) 
                             if c < len(ids_in_frame) and ids_in_frame[c] in selected_ids}
            s = sum(local_weights.values())
            if s > 0:
                for oid, w in local_weights.items():
                    attention_results[oid][f_id] = w / s

# ============================================================
# 3. PLOT (HARMONIZED, LOCALIZED & CLOSED BOX)
# ============================================================
fig, ax = plt.subplots(figsize=(14, 7))

# --- Localisation des frames ---
global_frames = sorted(attention_results[target_pred_id].keys())
if not global_frames:
    print(f"⚠️ Aucun ID {target_pred_id} trouvé.")
    exit()

seq_offset = global_frames[0] - 1 
local_frames = [f - seq_offset for f in global_frames]

for oid in selected_ids:
    if not attention_results[oid]: continue
    
    y_vals = [attention_results[oid].get(f, 0) for f in global_frames]
    if SMOOTHING_SIGMA > 0 and len(y_vals) > SMOOTHING_SIGMA:
        y_vals = gaussian_filter1d(y_vals, sigma=SMOOTHING_SIGMA)
    
    id_color = get_id_color(oid)
    label = f"Target (ID {oid})" if oid == target_pred_id else f"ID {oid}"
    lw = 3.0 if oid == target_pred_id else 1.5
    alpha = 1.0 if oid == target_pred_id else 0.7
    z = 10 if oid == target_pred_id else 5
    
    ax.plot(local_frames, y_vals, label=label, color=id_color, 
            linewidth=lw, alpha=alpha, zorder=z)

# Ajout des lignes verticales (Markers)
for marker in frame_marker:
    if min(local_frames) <= marker <= max(local_frames):
        ax.axvline(x=marker, color='black', linestyle='--', linewidth=1.2, alpha=0.5)
        ax.text(marker, -0.015, f"{marker}", color='#2c3e50', fontsize=9, 
                fontweight='bold', ha='center', va='top', rotation=90,
                transform=ax.get_xaxis_transform())

# --- GESTION DE LA FENÊTRE (ZOOM) ---
if USE_WINDOW:
    ax.set_xlim(FRAME_WINDOW[0], FRAME_WINDOW[1])
else:
    ax.set_xlim(min(local_frames), max(local_frames))

# --- ESTHÉTIQUE ---
ax.set_title(f"{video_name} - Identity importance evolution - Target ID {target_pred_id}", 
             fontsize=16, pad=25)

ax.set_xlabel("Frame", fontsize=12, labelpad=15) 
ax.set_ylabel("Relative attention weight", fontsize=12)

ax.set_ylim(-0.02, 1.05)

ax.grid(True, linestyle=':', alpha=0.5)
ax.legend(loc='best', frameon=True, shadow=True, fontsize=10, title="People IDs")

# --- FERMETURE DU RECTANGLE ---
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color('#333333')
    spine.set_linewidth(1.0)

plt.tight_layout()
plt.savefig(output_plot, dpi=230, bbox_inches='tight')
print(f"✅ Plot saved to {output_plot}")
plt.show()