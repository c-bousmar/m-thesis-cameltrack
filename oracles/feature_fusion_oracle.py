import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from scipy.optimize import linear_sum_assignment

from cameltrack.cameltrack import CAMELTrack, Tracklet
from tracklab.datastruct import TrackingDataset

log = logging.getLogger(__name__)

INFTY_COST = 1e5


def compute_iou_matrix(boxes1, boxes2):
    """ Calcule la matrice d'intersection sur union entre deux listes de bboxes [x, y, w, h] """
    boxes1_ltrb = np.concatenate([boxes1[:, :2], boxes1[:, :2] + boxes1[:, 2:4]], axis=1)
    boxes2_ltrb = np.concatenate([boxes2[:, :2], boxes2[:, :2] + boxes2[:, 2:4]], axis=1)
    boxes1_ltrb = np.expand_dims(boxes1_ltrb, axis=1)
    boxes2_ltrb = np.expand_dims(boxes2_ltrb, axis=0)
    left_top = np.maximum(boxes1_ltrb[..., :2], boxes2_ltrb[..., :2])
    right_bottom = np.minimum(boxes1_ltrb[..., 2:], boxes2_ltrb[..., 2:])
    intersection_dims = np.clip(right_bottom - left_top, a_min=0, a_max=None)
    intersection_area = intersection_dims[..., 0] * intersection_dims[..., 1]
    area_boxes1 = (boxes1[:, 2] * boxes1[:, 3]).reshape(-1, 1)
    area_boxes2 = (boxes2[:, 2] * boxes2[:, 3]).reshape(1, -1)
    union_area = area_boxes1 + area_boxes2 - intersection_area
    return intersection_area / union_area


def norm_euclidean_sim_matrix(track_embs, track_masks, det_embs, det_masks):
    """
    track_embs: Tensor [B, T, E]
    track_masks: Tensor [B, T]
    det_embs: Tensor [B, D, E]
    det_masks: Tensor [B, D]

    returns:
    td_sim_matrix: Tensor [B, T, D]
        padded pairs are set to -inf
    """
    track_embs = F.normalize(track_embs, p=2, dim=-1)
    det_embs = F.normalize(det_embs, p=2, dim=-1)
    td_sim_matrix = torch.cdist(track_embs, det_embs, p=2)
    td_sim_matrix = 1 - td_sim_matrix / 2
    
    # Masquage des éléments paddés
    mask = track_masks.unsqueeze(2) * det_masks.unsqueeze(1)
    td_sim_matrix[~mask] = -float("inf")
    return td_sim_matrix


class FEATURE_FUSION_ORACLE(CAMELTrack):

    level = "image"
    input_columns = ["bbox_conf"]
    output_columns = ["track_id"]

    def __init__(
        self,
        CAMEL,
        device,
        tracking_dataset: TrackingDataset,
        cfg,
        min_det_conf: float = 0.4,
        min_init_det_conf: float = 0.9,
        min_num_hits: int = 0,
        max_wo_hits: int = 150,
        max_track_gallery_size: int = 50,
        override_camel_cfg={},
        checkpoint_path=None,
        training_enabled=False,
        **kwargs,
    ):
        super().__init__(
            CAMEL=CAMEL,
            device=device,
            tracking_dataset=tracking_dataset,
            min_det_conf=min_det_conf,
            min_init_det_conf=min_init_det_conf,
            min_num_hits=min_num_hits,
            max_wo_hits=max_wo_hits,
            max_track_gallery_size=max_track_gallery_size,
            override_camel_cfg=override_camel_cfg,
            checkpoint_path=cfg.get("checkpoint_path", checkpoint_path),
            training_enabled=training_enabled,
            **kwargs,
        )

        self.level = "image"
        self.cfg = cfg
        self.device = device
        self.tracking_dataset = tracking_dataset
        
        # Liste des cues extraites par CAMEL
        self.cue_names = list(self.CAMEL.temp_encs.keys())

        print("\n" + "="*60)
        print("FEATURE_FUSION_ORACLE INITIALIZED (EXP 11 - 3 CUES)")
        print(f"Optimizing weights linearly for cues: {self.cue_names}")
        print("="*60 + "\n")

    def _generate_3_cue_weights(self, steps=21):
        """ Génère des triplets de poids valides dont la somme vaut 1.0 """
        weights = []
        for w1 in np.linspace(0.0, 1.0, steps):
            for w2 in np.linspace(0.0, 1.0 - w1, steps):
                w3 = 1.0 - w1 - w2
                if w3 >= -1e-5:  # Tolérance flottante
                    weights.append([w1, w2, max(0.0, w3)])
        return weights

    @torch.no_grad()
    def associate_dets_to_trks(self, tracklets, detections):
        """
        Surcharge conforme à l'Expérience 11 : Combinaison linéaire à 3 composantes
        et monitoring informatif.
        """
        image_id = detections[0].image_id
        print(f"\n--- Processing Image ID: {image_id} | Active Tracklets: {len(tracklets)} | Detections: {len(detections)} ---")

        # Cas limites de base
        if not tracklets:
            print("[INFO] Aucun tracklet actif. Saut de l'association.")
            return np.empty((0, 2)), [], list(range(len(detections))), np.empty((0,))
        if not detections:
            print("[INFO] Aucune détection sur cette frame. Saut de l'association.")
            return np.empty((0, 2)), list(range(len(tracklets))), [], np.empty((0,))

        # ------------------------------------------------------------
        # 1. Pipeline natif CAMEL (Preprocessing + Tokenization)
        # ------------------------------------------------------------
        batch = self.build_camel_batch(tracklets, detections)
        tracks, dets = self.CAMEL.predict_preprocess(batch)
        tracks, dets = self.CAMEL.tokenize(tracks, dets)

        # ------------------------------------------------------------
        # 2. Calcul des matrices de coût par cue
        # ------------------------------------------------------------
        cue_cost_matrices = {}
        print(f"[STEP 1] Extractions des jetons (Tokens) pour les cues : {self.cue_names}")
        
        for cue in self.cue_names:
            if cue not in tracks.tokens or cue not in dets.tokens:
                print(f"  -> [WARN] Cue '{cue}' manquante dans les tokens extraits.")
                continue

            track_tokens = tracks.tokens[cue]  
            det_tokens = dets.tokens[cue]      

            # Pooling temporel adaptatif
            if len(track_tokens.shape) == 4:
                track_embs = torch.nanmean(track_tokens[0], dim=1)
            else:
                track_embs = track_tokens[0]

            if len(det_tokens.shape) == 4:
                det_embs = torch.nanmean(det_tokens[0], dim=1)
            else:
                det_embs = det_tokens[0]

            track_masks = torch.ones(track_embs.shape[0], dtype=torch.bool, device=self.device)
            det_masks = torch.ones(det_embs.shape[0], dtype=torch.bool, device=self.device)

            # Calcul de la similarité via la fonction requise
            sim_matrix = norm_euclidean_sim_matrix(
                track_embs.unsqueeze(0), 
                track_masks.unsqueeze(0), 
                det_embs.unsqueeze(0), 
                det_masks.unsqueeze(0)
            )[0]

            cost_matrix = -sim_matrix
            cost_matrix[torch.isinf(cost_matrix)] = INFTY_COST
            
            cue_cost_matrices[cue] = cost_matrix.cpu().numpy()
            print(f"  -> Matrice de coût générée pour '{cue}' | Shape: {cue_cost_matrices[cue].shape}")

        if not cue_cost_matrices:
            raise RuntimeError("Aucune matrice de coût n'a pu être générée à partir des cues.")

# ------------------------------------------------------------
        # 3. Construction de la Matrice d'association cible (Ground Truth)
        # ------------------------------------------------------------
        print("[STEP 2] Alignement géométrique (IoU) des détections avec la Ground Truth...")
        
        # Récupération de l'image_id de manière conforme à l'oracle qui marche
        # Dans CAMELTrack, les objets 'Detection' ont souvent un attribut de métadonnée ou stockent l'id brut
        first_det = detections[0]
        if hasattr(first_det, "image_id"):
            image_id = first_det.image_id
            if torch.is_tensor(image_id):
                image_id = image_id.item()
        else:
            # Fallback de secours si stocké sous un autre nom
            image_id = getattr(first_det, "id", image_id)
            if torch.is_tensor(image_id):
                image_id = image_id.item()

        eval_set = self.cfg.get("eval_set", "val")
        all_detections_gt = self.tracking_dataset.sets[eval_set].detections_gt
        
        # Filtrage identique à l'oracle fonctionnel
        detections_gt = all_detections_gt[all_detections_gt.image_id == image_id]

        # Si c'est toujours vide, c'est que l'ID extrait n'est pas le bon index. 
        # On extrait alors la valeur native depuis la première détection (cas des objets Pandas déballés)
        if len(detections_gt) == 0 and hasattr(first_det, "metadata"):
            image_id = first_det.metadata.id
            detections_gt = all_detections_gt[all_detections_gt.image_id == image_id]

        print(f"--- [DEBUG ORACLE] Image ID extrait : {image_id} ---")
        print(f"Nombre de lignes chargées depuis la GT (detections_gt): {len(detections_gt)}")

        gt_matches = {}  
        if len(detections_gt) > 0:
            bbox_ltwh_gt = np.vstack(detections_gt.bbox_ltwh.values)
            bbox_ltwh_pred = np.vstack([det.bbox_ltwh.cpu().numpy() for det in detections])
            
            iou_matrix = compute_iou_matrix(bbox_ltwh_gt, bbox_ltwh_pred)
            iou_cost = 1.0 - iou_matrix
            iou_cost[iou_cost > 0.5] = INFTY_COST
            
            # Match identique à ton oracle fonctionnel (col_ind = pred, row_ind = gt)
            row_ind, col_ind = linear_sum_assignment(iou_cost)
            valid_matches = iou_cost[row_ind, col_ind] < 1.0
            row_ind = row_ind[valid_matches]
            col_ind = col_ind[valid_matches]

            for r, c in zip(row_ind, col_ind):
                raw_val = detections_gt.iloc[r]['track_id']
                gt_matches[c] = int(raw_val.item()) if torch.is_tensor(raw_val) else int(raw_val)

        # Assigner les ID réels aux objets détections courants en type natif
        for d_idx, det in enumerate(detections):
            det.oracle_track_id = int(gt_matches.get(d_idx, -1))

        # Matrice cible binaire pour l'optimisation
        num_tracks = len(tracklets)
        num_dets = len(detections)
        target_gt_matrix = np.zeros((num_tracks, num_dets), dtype=bool)

        for t_idx, trk in enumerate(tracklets):
            trk_oracle_id = getattr(trk, "oracle_track_id", None)
            if trk_oracle_id is None and hasattr(trk, "last_detection"):
                trk_oracle_id = getattr(trk.last_detection, "oracle_track_id", -1)
            
            if torch.is_tensor(trk_oracle_id): trk_oracle_id = int(trk_oracle_id.item())
            else: trk_oracle_id = int(trk_oracle_id) if trk_oracle_id is not None else -1
            
            for d_idx, det in enumerate(detections):
                det_oracle_id = det.oracle_track_id
                if trk_oracle_id == det_oracle_id and trk_oracle_id != -1:
                    target_gt_matrix[t_idx, d_idx] = True

        print(f"Nombre de liens valides trouvés dans target_gt_matrix : {np.sum(target_gt_matrix)}")

        # ------------------------------------------------------------
        # 4. Recherche de la combinaison linéaire optimale (3 Cues)
        # ------------------------------------------------------------
        cues_list = list(cue_cost_matrices.keys())
        best_accuracy = -1.0
        best_cost_matrix = None
        best_weights = None

        print(f"[STEP 3] Lancement de la Grid Search linéaire (Cues actives à fusionner : {cues_list})")

        if len(cues_list) == 1:
            best_cost_matrix = cue_cost_matrices[cues_list[0]]
            best_weights = [1.0]
        elif len(cues_list) == 2:
            for alpha in np.linspace(0.0, 1.0, 21):
                current_cost_matrix = alpha * cue_cost_matrices[cues_list[0]] + (1 - alpha) * cue_cost_matrices[cues_list[1]]
                r_ind, c_ind = linear_sum_assignment(current_cost_matrix)
                correct_matches = sum([1 for r, c in zip(r_ind, c_ind) if target_gt_matrix[r, c]])
                accuracy = correct_matches / max(num_tracks, num_dets)
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_cost_matrix = current_cost_matrix
                    best_weights = [alpha, 1 - alpha]
        else:
            # Cas à 3 cues ou plus (on utilise les 3 premières de la liste)
            weight_triplets = self._generate_3_cue_weights(steps=100)
            for w in weight_triplets:
                current_cost_matrix = (w[0] * cue_cost_matrices[cues_list[0]] + 
                                       w[1] * cue_cost_matrices[cues_list[1]] + 
                                       w[2] * cue_cost_matrices[cues_list[2]])
                
                r_ind, c_ind = linear_sum_assignment(current_cost_matrix)
                correct_matches = sum([1 for r, c in zip(r_ind, c_ind) if target_gt_matrix[r, c]])
                accuracy = correct_matches / max(num_tracks, num_dets)
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_cost_matrix = current_cost_matrix
                    best_weights = w

        # Log des paramètres de fusion optimaux trouvés par l'Oracle pour cette frame
        weight_str = ", ".join([f"{cues_list[i]}: {best_weights[i]:.2f}" for i in range(len(cues_list))])
        print(f"  -> [ORACLE SOLUTION] Poids optimaux trouvés : ({weight_str}) | Association Accuracy max: {best_accuracy * 100:.2f}%")

        # ------------------------------------------------------------
        # 5. Appariement final & seuillage
        # ------------------------------------------------------------
        print("[STEP 4] Résolution finale de l'affectation et filtrage par seuil de confiance...")
        row_ind, col_ind = linear_sum_assignment(best_cost_matrix)

        matched = []
        used_tracks = set()
        used_dets = set()
        sim_threshold = self.CAMEL.sim_threshold

        for r, c in zip(row_ind, col_ind):
            sim = -best_cost_matrix[r, c]
            if sim >= sim_threshold:
                matched.append([r, c])
                used_tracks.add(r)
                used_dets.add(c)
            else:
                print(f"  -> [REJET] Match trk_{r} <-> det_{c} rejeté : Similarité de fusion ({sim:.3f}) < seuil ({sim_threshold})")

        unmatched_trks = [i for i in range(num_tracks) if i not in used_tracks]
        unmatched_dets = [j for j in range(num_dets) if j not in used_dets]
        matched = np.array(matched) if len(matched) > 0 else np.empty((0, 2), dtype=int)

        print(f"[RESULTAT] Finalisé avec {len(matched)} Éléments associés, {len(unmatched_trks)} Perdus (unmatched trks), {len(unmatched_dets)} Nouveaux (unmatched dets).")

        # Sauvegarde des ID Oracle pour propager la mémoire à l'étape suivante
        for m in matched:
            tracklets[m[0]].oracle_track_id = detections[m[1]].oracle_track_id

        # Reconstruction du tenseur de similarités attendu en sortie
        td_sim_matrix = torch.from_numpy(-best_cost_matrix).to(self.device)

        return (
            matched,
            unmatched_trks,
            unmatched_dets,
            td_sim_matrix
        )