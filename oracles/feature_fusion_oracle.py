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
    """
    Compute the IoU matrix between two arrays of bounding boxes in the format "ltwh".

    Args:
        boxes1 (np.ndarray): Array of bounding boxes with shape [N, 4], each row is [left, top, width, height].
        boxes2 (np.ndarray): Array of bounding boxes with shape [M, 4], each row is [left, top, width, height].

    Returns:
        iou_matrix (np.ndarray): IoU matrix of shape [N, M] where each element [i, j] is the IoU between boxes1[i] and boxes2[j].
    """

    boxes1_ltrb = np.concatenate([boxes1[:, :2], boxes1[:, :2] + boxes1[:, 2:4]], axis=1)  # [N, 4]
    boxes2_ltrb = np.concatenate([boxes2[:, :2], boxes2[:, :2] + boxes2[:, 2:4]], axis=1)  # [M, 4]

    boxes1_ltrb = np.expand_dims(boxes1_ltrb, axis=1)  # [N, 1, 4]
    boxes2_ltrb = np.expand_dims(boxes2_ltrb, axis=0)  # [1, M, 4]

    left_top = np.maximum(boxes1_ltrb[..., :2], boxes2_ltrb[..., :2])  # [N, M, 2]
    right_bottom = np.minimum(boxes1_ltrb[..., 2:], boxes2_ltrb[..., 2:])  # [N, M, 2]

    intersection_dims = np.clip(right_bottom - left_top, a_min=0, a_max=None)  # [N, M, 2]
    intersection_area = intersection_dims[..., 0] * intersection_dims[..., 1]  # [N, M]

    area_boxes1 = (boxes1[:, 2] * boxes1[:, 3]).reshape(-1, 1)  # [N, 1]
    area_boxes2 = (boxes2[:, 2] * boxes2[:, 3]).reshape(1, -1)  # [1, M]

    union_area = area_boxes1 + area_boxes2 - intersection_area  # [N, M]

    iou_matrix = intersection_area / union_area  # [N, M]

    return iou_matrix


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
    
    mask = track_masks.unsqueeze(2) * det_masks.unsqueeze(1)
    td_sim_matrix[~mask] = -float("inf")
    return td_sim_matrix


class FEATURE_FUSION_ORACLE(CAMELTrack):
    """
    Oracle for evaluating the upper bound of multi-cue fusion in CAMELTrack.

    This module assumes privileged access to ground-truth annotations in order to:
    1. Evaluate the discriminative power of individual tracking cues.
    2. Compute an oracle-optimal linear combination of cue-specific cost matrices.
    3. Estimate the best achievable association accuracy given the current feature space.

    The oracle does NOT reflect a real-world tracking system:
    it performs grid search over fusion weights using ground-truth annotations.
    """
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
        self.cue_names = list(self.CAMEL.temp_encs.keys())

    def _generate_3_cue_weights(self, steps=21):
        """
        Generate a simplex grid of valid convex combinations for 3 cues.

        Each weight vector w = (w1, w2, w3) satisfies:
            w1 + w2 + w3 = 1, w_i ≥ 0

        Used for brute-force oracle optimization of cue fusion.
        """
        weights = []
        for w1 in np.linspace(0.0, 1.0, steps):
            for w2 in np.linspace(0.0, 1.0 - w1, steps):
                w3 = 1.0 - w1 - w2
                if w3 >= -1e-5:
                    weights.append([w1, w2, max(0.0, w3)])
        return weights

    @torch.no_grad()
    def associate_dets_to_trks(self, tracklets, detections):
        """
        Oracle association between tracklets and detections.

        The method:
        1. Extracts cue embeddings for tracks and detections
        2. Builds per-cue cost matrices
        3. Aligns detections with ground-truth identities (oracle supervision)
        4. Constructs oracle target assignment matrix
        5. Searches for optimal linear fusion of cues
        6. Runs final Hungarian matching using optimal costs
        """
        image_id = detections[0].image_id

        if not tracklets:
            return np.empty((0, 2)), [], list(range(len(detections))), np.empty((0,))
        if not detections:
            return np.empty((0, 2)), list(range(len(tracklets))), [], np.empty((0,))

        batch = self.build_camel_batch(tracklets, detections)
        tracks, dets = self.CAMEL.predict_preprocess(batch)
        tracks, dets = self.CAMEL.tokenize(tracks, dets)

        cue_cost_matrices = {}
        
        for cue in self.cue_names:
            if cue not in tracks.tokens or cue not in dets.tokens:
                continue

            track_tokens = tracks.tokens[cue]  
            det_tokens = dets.tokens[cue]      

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

            sim_matrix = norm_euclidean_sim_matrix(
                track_embs.unsqueeze(0), 
                track_masks.unsqueeze(0), 
                det_embs.unsqueeze(0), 
                det_masks.unsqueeze(0)
            )[0]

            cost_matrix = -sim_matrix
            cost_matrix[torch.isinf(cost_matrix)] = INFTY_COST
            
            cue_cost_matrices[cue] = cost_matrix.cpu().numpy()

        if not cue_cost_matrices:
            raise RuntimeError("No cost matrices generated from cues.")
        
        first_det = detections[0]
        if hasattr(first_det, "image_id"):
            image_id = first_det.image_id
            if torch.is_tensor(image_id):
                image_id = image_id.item()
        else:
            image_id = getattr(first_det, "id", image_id)
            if torch.is_tensor(image_id):
                image_id = image_id.item()

        eval_set = self.cfg.get("eval_set", "val")
        all_detections_gt = self.tracking_dataset.sets[eval_set].detections_gt
        
        detections_gt = all_detections_gt[all_detections_gt.image_id == image_id]

        if len(detections_gt) == 0 and hasattr(first_det, "metadata"):
            image_id = first_det.metadata.id
            detections_gt = all_detections_gt[all_detections_gt.image_id == image_id]

        gt_matches = {}  
        if len(detections_gt) > 0:
            bbox_ltwh_gt = np.vstack(detections_gt.bbox_ltwh.values)
            bbox_ltwh_pred = np.vstack([det.bbox_ltwh.cpu().numpy() for det in detections])
            
            iou_matrix = compute_iou_matrix(bbox_ltwh_gt, bbox_ltwh_pred)
            iou_cost = 1.0 - iou_matrix
            iou_cost[iou_cost > 0.5] = INFTY_COST
            
            row_ind, col_ind = linear_sum_assignment(iou_cost)
            valid_matches = iou_cost[row_ind, col_ind] < 1.0
            row_ind = row_ind[valid_matches]
            col_ind = col_ind[valid_matches]

            for r, c in zip(row_ind, col_ind):
                raw_val = detections_gt.iloc[r]['track_id']
                gt_matches[c] = int(raw_val.item()) if torch.is_tensor(raw_val) else int(raw_val)

        for d_idx, det in enumerate(detections):
            det.oracle_track_id = int(gt_matches.get(d_idx, -1))

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

        cues_list = list(cue_cost_matrices.keys())
        best_accuracy = -1.0
        best_cost_matrix = None
        best_weights = None

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

        unmatched_trks = [i for i in range(num_tracks) if i not in used_tracks]
        unmatched_dets = [j for j in range(num_dets) if j not in used_dets]
        matched = np.array(matched) if len(matched) > 0 else np.empty((0, 2), dtype=int)

        for m in matched:
            tracklets[m[0]].oracle_track_id = detections[m[1]].oracle_track_id

        td_sim_matrix = torch.from_numpy(-best_cost_matrix).to(self.device)

        return (
            matched,
            unmatched_trks,
            unmatched_dets,
            td_sim_matrix
        )