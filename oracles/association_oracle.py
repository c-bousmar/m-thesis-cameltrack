import logging
import pandas as pd
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from tracklab.pipeline import ImageLevelModule

log = logging.getLogger(__name__)

INFTY_COST = 1e+5

class ASSOCIATION_ORACLE(ImageLevelModule):
    input_columns = ["bbox_ltwh"]
    output_columns = [
        "track_id",
    ]

    def __init__(self,
                 cfg,
                 device,
                 tracking_dataset,
                 **kwargs):
        super().__init__(batch_size=1)
        self.cfg = cfg
        self.device = device
        self.tracking_dataset = tracking_dataset

    @torch.no_grad()
    def preprocess(self, image, detections: pd.DataFrame, metadata: pd.Series):
        return []

    @torch.no_grad()
    def process(self, batch, detections: pd.DataFrame, metadatas: pd.DataFrame):
        if len(detections) == 0:
            return []

        image_id = metadatas.id.unique()[0]
        video_id = metadatas.video_id.unique()[0]

        all_detections_gt = self.tracking_dataset.sets[self.cfg['eval_set']].detections_gt
        detections_gt = all_detections_gt[all_detections_gt.image_id == image_id]

        assert detections_gt.video_id.unique()[0] == video_id

        if self.cfg['return_gt']:
            detections_gt.index += all_detections_gt.index.max() + 1
            return detections_gt

        if len(detections_gt) == 0:
            return []

        col_ind, row_ind = self.ground_truth_to_prediction_match(detections, detections_gt)

        matched_gt = detections_gt.iloc[row_ind]
        matched_detections = detections.iloc[col_ind]
        matched_detections['track_id'] = matched_gt['track_id'].values

        return matched_detections

    def ground_truth_to_prediction_match(self, detections, detections_gt):
        bbox_ltwh_gt = np.vstack(detections_gt.bbox_ltwh.values)
        bbox_ltwh = np.vstack(detections.bbox_ltwh.values)
        assert len(bbox_ltwh_gt) > 0 and len(bbox_ltwh) > 0

        cost_matrix = 1 - compute_iou_matrix(bbox_ltwh_gt, bbox_ltwh)

        cost_matrix[cost_matrix > 0.5] = INFTY_COST
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        valid_matches = cost_matrix[row_ind, col_ind] < 1.0
        row_ind = row_ind[valid_matches]
        col_ind = col_ind[valid_matches]
        return col_ind, row_ind


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