import math
import itertools as it
import numpy as np
from src.model_utils.config import config

octave_base_scale = 5
class GeneratDefaultGridCells:
    def __init__(self):
        self.center_priors = []
        anchor_size = np.array(config.anchor_size)
        for i, feature_size in enumerate(config.feature_size):
            w, h = anchor_size[i], anchor_size[i]
            for i, j in it.product(range(feature_size), repeat=2):
                cx, cy = (j + 0.5) * w, (i + 0.5) * w
                self.center_priors.append([cx, cy, h, w])

        def to_ltrb(cx, cy, h, w):
            h, w = h * octave_base_scale, w * octave_base_scale
            return cx - h / 2, cy - w / 2, cx + h / 2, cy + w / 2

        self.center_priors = np.array(self.center_priors, dtype='float32')
        self.center_priors_ltrb = np.array(tuple(to_ltrb(*i) for i in self.center_priors), dtype='float32')


center_priors = GeneratDefaultGridCells().center_priors
center_priors_ltrb = GeneratDefaultGridCells().center_priors_ltrb
x1, y1, x2, y2 = np.split(center_priors_ltrb[:, :4], 4, axis=-1)
num_level_cells_list = [1600, 400, 100]
# The area of Anchor
vol_anchors = (x2 - x1) * (y2 - y1)


def nanodet_bboxes_encode(boxes):
    def bbox_overlaps(bbox):
        xmin = np.maximum(x1, bbox[0])
        ymin = np.maximum(y1, bbox[1])
        xmax = np.minimum(x2, bbox[2])
        ymax = np.minimum(y2, bbox[3])
        # 并行化运算
        w = np.maximum(xmax - xmin, 0.)
        h = np.maximum(ymax - ymin, 0.)

        inter_vol = h * w
        union_vol = vol_anchors + (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) - inter_vol
        iou = inter_vol / union_vol
        return np.squeeze(iou)

    pre_scores = np.zeros((config.num_nanodet_boxes,), dtype=np.float32)
    res_boxes = np.zeros((config.num_nanodet_boxes, 4), dtype=np.float32)
    res_labels = np.zeros((config.num_nanodet_boxes,), dtype=np.int64)
    res_center_priors = np.zeros((config.num_nanodet_boxes, 4), dtype=np.float32)

    for bbox in boxes:
        label = int(bbox[4])
        scores = bbox_overlaps(bbox)

        gt_cx = (bbox[0] + bbox[2]) / 2.0
        gt_cy = (bbox[1] + bbox[3]) / 2.0
        gt_points = np.stack((gt_cx, gt_cy), axis=0)
        grid_priors_cx = (center_priors_ltrb[:, 0] + center_priors_ltrb[:, 2]) / 2.0
        grid_priors_cy = (center_priors_ltrb[:, 1] + center_priors_ltrb[:, 3]) / 2.0
        grid_cells_points = np.stack((grid_priors_cx, grid_priors_cy), axis=1)

        distances = grid_cells_points - gt_points
        distances = np.power(distances, 2)
        distances = np.sum(distances, axis=-1)
        distances = np.sqrt(distances)
        candidate_idxs = []
        start_idx = 0
        topk = 9
        for level, cells_per_level in enumerate(num_level_cells_list):
            end_idx = start_idx + cells_per_level
            distances_per_level = distances[start_idx:end_idx]
            selectable_k = min(config.topk, cells_per_level)
            topk_idxs_per_level = np.argsort(distances_per_level, axis=0)
            topk_idxs_per_level = topk_idxs_per_level[:selectable_k]
            candidate_idxs.append(topk_idxs_per_level + start_idx)
            start_idx = end_idx
        candidate_idxs = np.concatenate(candidate_idxs, axis=0)

        candidate_overlaps = scores[candidate_idxs]
        overlaps_mean = candidate_overlaps.mean(0)
        overlaps_std = candidate_overlaps.std(0)
        overlaps_thr = overlaps_mean + overlaps_std
        is_pos = (candidate_overlaps >= overlaps_thr)

        l_ = grid_priors_cx[candidate_idxs] - bbox[0]
        t_ = grid_priors_cy[candidate_idxs] - bbox[1]
        r_ = bbox[2] - grid_priors_cx[candidate_idxs]
        b_ = bbox[3] - grid_priors_cy[candidate_idxs]
        is_in_gts = np.min(np.stack([l_, t_, r_, b_], axis=0), axis=0) > 0.01

        is_pos = is_pos & is_in_gts
        # 正样本坐标
        idx = candidate_idxs[is_pos]
        mask = np.zeros((config.num_nanodet_boxes,), np.bool)
        mask[idx] = True
        mask = mask & (scores > pre_scores)
        pre_scores = np.maximum(pre_scores, scores * mask)
        res_labels = mask * label + (1 - mask) * res_labels

        for i in range(4):
            res_boxes[:, i] = mask * bbox[i] + (1 - mask) * res_boxes[:, i]

    index = np.nonzero(res_labels)
    a = center_priors_ltrb[index]
    b = res_boxes[index]
    res_center_priors[index] = center_priors_ltrb[index]
    num_match = np.array([len(np.nonzero(res_labels)[0])], dtype=np.int32)
    return res_boxes, res_labels, res_center_priors, num_match


def intersect(box_a, box_b):
    """Compute the intersect of two sets of boxes."""
    max_yx = np.minimum(box_a[:, 2:4], box_b[2:4])
    min_yx = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_yx - min_yx), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes."""
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1]))
    area_b = ((box_b[2] - box_b[0]) *
              (box_b[3] - box_b[1]))
    union = area_a + area_b - inter
    return inter / union

if __name__ == "__main__":
    bboxes = np.array([[23.6667, 23.8757, 238.6326, 151.8874, 2],[78,90,220,190,5]], np.float32)
    # bboxes = np.array([[1, 1, 1, 1, 5]], np.float32)
    nanodet_bboxes_encode(bboxes)
    print("!")