import math
from collections import OrderedDict
from typing import List, Tuple

import torch
from torch import Tensor, nn
from torchvision.ops.misc import FrozenBatchNorm2d

from monai.utils.module import look_up_option


# class BalancedPositiveNegativeSampler:
#     """
#     This class samples batches, ensuring that they contain a fixed proportion of positives
#     """

#     def __init__(self, batch_size_per_image: int, positive_fraction: float) -> None:
#         """
#         Args:
#             batch_size_per_image (int): number of elements to be selected per image
#             positive_fraction (float): percentage of positive elements per batch
#         """
#         self.batch_size_per_image = batch_size_per_image
#         self.positive_fraction = positive_fraction

#     def __call__(self, matched_idxs: List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
#         """
#         Args:
#             matched idxs: list of tensors containing -1, 0 or positive values.
#                 Each tensor corresponds to a specific image.
#                 -1 values are ignored, 0 are considered as negatives and > 0 as
#                 positives.

#         Returns:
#             pos_idx (list[tensor])
#             neg_idx (list[tensor])

#         Returns two lists of binary masks for each image.
#         The first list contains the positive elements that were selected,
#         and the second list the negative example.
#         """
#         pos_idx = []
#         neg_idx = []
#         for matched_idxs_per_image in matched_idxs:
#             positive = torch.where(matched_idxs_per_image >= 1)[0]
#             negative = torch.where(matched_idxs_per_image == 0)[0]

#             num_pos = int(self.batch_size_per_image * self.positive_fraction)
#             # protect against not enough positive examples
#             num_pos = min(positive.numel(), num_pos)
#             num_neg = self.batch_size_per_image - num_pos
#             # protect against not enough negative examples
#             num_neg = min(negative.numel(), num_neg)

#             # randomly select positive and negative examples
#             perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
#             perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

#             pos_idx_per_image = positive[perm1]
#             neg_idx_per_image = negative[perm2]

#             # create binary mask from indices
#             pos_idx_per_image_mask = torch.zeros_like(matched_idxs_per_image, dtype=torch.uint8)
#             neg_idx_per_image_mask = torch.zeros_like(matched_idxs_per_image, dtype=torch.uint8)

#             pos_idx_per_image_mask[pos_idx_per_image] = 1
#             neg_idx_per_image_mask[neg_idx_per_image] = 1

#             pos_idx.append(pos_idx_per_image_mask)
#             neg_idx.append(neg_idx_per_image_mask)

#         return pos_idx, neg_idx


@torch.jit._script_if_tracing
def encode_boxes(reference_boxes: Tensor, proposals: Tensor, weights: Tensor) -> Tensor:
    """
    Encode a set of proposals with respect to some
    reference boxes

    Args:
        reference_boxes (Tensor): reference boxes
        proposals (Tensor): boxes to be encoded
        weights (Tensor[4] or Tensor[6]): the weights for ``(x, y, w, h) or (x,y,z, w,h,d)``
    """

    spatial_dims = look_up_option(len(weights), [4,6])//2
    # implementation starts here
    targets_dxyz = []
    targets_dwhd = []
    for axis in range(spatial_dims):
        ex_whd_axis = proposals[:,2*axis+1] - proposals[:,2*axis]
        ex_ctr_xyz_axis =  proposals[:,2*axis] + 0.5 * ex_whd_axis
        
        gt_whd_axis = reference_boxes[:,2*axis+1] - reference_boxes[:,2*axis]
        gt_ctr_xyz_axis =reference_boxes[:,2*axis] + 0.5 * gt_whd_axis

        targets_dxyz.append( weights[axis] * (gt_ctr_xyz_axis - ex_ctr_xyz_axis) / ex_whd_axis )
        targets_dwhd.append( weights[axis+spatial_dims] * torch.log(gt_whd_axis / ex_whd_axis) )

    targets = torch.stack(targets_dxyz+targets_dwhd, dim=1)
    return targets


class BoxCoder:
    """
    This class encodes and decodes a set of bounding boxes into
    the representation used for training the regressors.
    """

    def __init__(
        self, weights: Tuple[float], bbox_xform_clip: float = math.log(1000.0 / 16)
    ) -> None:
        """
        Args:
            weights (4-element tuple or 6-element tuple)
            bbox_xform_clip (float)
        """
        look_up_option(len(weights), [4,6])
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def encode(self, reference_boxes: List[Tensor], proposals: List[Tensor]) -> List[Tensor]:
        boxes_per_image = [len(b) for b in reference_boxes]
        reference_boxes = torch.cat(reference_boxes, dim=0)
        proposals = torch.cat(proposals, dim=0)
        targets = self.encode_single(reference_boxes, proposals)
        return targets.split(boxes_per_image, 0)

    def encode_single(self, reference_boxes: Tensor, proposals: Tensor) -> Tensor:
        """
        Encode a set of proposals with respect to some
        reference boxes

        Args:
            reference_boxes (Tensor): reference boxes
            proposals (Tensor): boxes to be encoded
        """
        dtype = reference_boxes.dtype
        device = reference_boxes.device
        weights = torch.as_tensor(self.weights, dtype=dtype, device=device)
        targets = encode_boxes(reference_boxes, proposals, self.weights)

        return targets

    def decode(self, rel_codes: Tensor, boxes: List[Tensor]) -> Tensor:
        assert isinstance(boxes, (list, tuple))
        assert isinstance(rel_codes, torch.Tensor)
        boxes_per_image = [b.size(0) for b in boxes]
        concat_boxes = torch.cat(boxes, dim=0)
        box_sum = 0
        for val in boxes_per_image:
            box_sum += val
        if box_sum > 0:
            rel_codes = rel_codes.reshape(box_sum, -1)
        pred_boxes = self.decode_single(rel_codes, concat_boxes)
        if box_sum > 0:
            pred_boxes = pred_boxes.reshape(box_sum, -1, 4)
        return pred_boxes

    def decode_single(self, rel_codes: Tensor, boxes: Tensor) -> Tensor:
        """
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.

        Args:
            rel_codes (Tensor): encoded boxes
            boxes (Tensor): reference boxes.
        """

        spatial_dims = look_up_option(boxes.shape[1], [4,6])//2
        if len(self.weights) != 2*spatial_dims:
            raise ValueError(f"Dimension of weights should be twice of image dimension. Yet got weights={self.weights}")

        boxes = boxes.to(rel_codes.dtype)

        pred_boxes = []
        for axis in range(spatial_dims):
            whd_axis = boxes[:, 2*axis+1] - boxes[:, 2*axis]
            ctr_xyz_axis = boxes[:, 2*axis] + 0.5 * whd_axis
            dxyz_axis = rel_codes[:, axis] / self.weights[axis]
            dwhd_axis = rel_codes[:, spatial_dims+axis] / self.weights[axis+spatial_dims]

            # Prevent sending too large values into torch.exp()
            dwhd_axis = torch.clamp(dwhd_axis, max=self.bbox_xform_clip)

            pred_ctr_xyx_axis = dxyz_axis * whd_axis + ctr_xyz_axis
            pred_whd_axis = torch.exp(dwhd_axis) * whd_axis

            # Distance from center to box's corner.
            c_to_c_whd_axis = torch.tensor(0.5, dtype=pred_ctr_xyx_axis.dtype, device=pred_whd_axis.device) * pred_whd_axis

            pred_boxes.append(pred_ctr_xyx_axis - c_to_c_whd_axis)
            pred_boxes.append(pred_ctr_xyx_axis + c_to_c_whd_axis)
        
        pred_boxes = torch.stack(pred_boxes, dim=1)
        return pred_boxes


# class BoxLinearCoder:
#     """
#     The linear box-to-box transform defined in FCOS. The transformation is parameterized
#     by the distance from the center of (square) src box to 4 edges of the target box.
#     """

#     def __init__(self, normalize_by_size: bool = True) -> None:
#         """
#         Args:
#             normalize_by_size (bool): normalize deltas by the size of src (anchor) boxes.
#         """
#         self.normalize_by_size = normalize_by_size

#     def encode_single(self, reference_boxes: Tensor, proposals: Tensor) -> Tensor:
#         """
#         Encode a set of proposals with respect to some reference boxes

#         Args:
#             reference_boxes (Tensor): reference boxes
#             proposals (Tensor): boxes to be encoded

#         Returns:
#             Tensor: the encoded relative box offsets that can be used to
#             decode the boxes.
#         """
#         # get the center of reference_boxes
#         reference_boxes_ctr_x = 0.5 * (reference_boxes[:, 0] + reference_boxes[:, 2])
#         reference_boxes_ctr_y = 0.5 * (reference_boxes[:, 1] + reference_boxes[:, 3])

#         # get box regression transformation deltas
#         target_l = reference_boxes_ctr_x - proposals[:, 0]
#         target_t = reference_boxes_ctr_y - proposals[:, 1]
#         target_r = proposals[:, 2] - reference_boxes_ctr_x
#         target_b = proposals[:, 3] - reference_boxes_ctr_y

#         targets = torch.stack((target_l, target_t, target_r, target_b), dim=1)
#         if self.normalize_by_size:
#             reference_boxes_w = reference_boxes[:, 2] - reference_boxes[:, 0]
#             reference_boxes_h = reference_boxes[:, 3] - reference_boxes[:, 1]
#             reference_boxes_size = torch.stack(
#                 (reference_boxes_w, reference_boxes_h, reference_boxes_w, reference_boxes_h), dim=1
#             )
#             targets = targets / reference_boxes_size

#         return targets

#     def decode_single(self, rel_codes: Tensor, boxes: Tensor) -> Tensor:
#         """
#         From a set of original boxes and encoded relative box offsets,
#         get the decoded boxes.

#         Args:
#             rel_codes (Tensor): encoded boxes
#             boxes (Tensor): reference boxes.

#         Returns:
#             Tensor: the predicted boxes with the encoded relative box offsets.
#         """

#         boxes = boxes.to(rel_codes.dtype)

#         ctr_x = 0.5 * (boxes[:, 0] + boxes[:, 2])
#         ctr_y = 0.5 * (boxes[:, 1] + boxes[:, 3])
#         if self.normalize_by_size:
#             boxes_w = boxes[:, 2] - boxes[:, 0]
#             boxes_h = boxes[:, 3] - boxes[:, 1]
#             boxes_size = torch.stack((boxes_w, boxes_h, boxes_w, boxes_h), dim=1)
#             rel_codes = rel_codes * boxes_size

#         pred_boxes1 = ctr_x - rel_codes[:, 0]
#         pred_boxes2 = ctr_y - rel_codes[:, 1]
#         pred_boxes3 = ctr_x + rel_codes[:, 2]
#         pred_boxes4 = ctr_y + rel_codes[:, 3]
#         pred_boxes = torch.stack((pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4), dim=1)
#         return pred_boxes


# class Matcher:
#     """
#     This class assigns to each predicted "element" (e.g., a box) a ground-truth
#     element. Each predicted element will have exactly zero or one matches; each
#     ground-truth element may be assigned to zero or more predicted elements.

#     Matching is based on the MxN match_quality_matrix, that characterizes how well
#     each (ground-truth, predicted)-pair match. For example, if the elements are
#     boxes, the matrix may contain box IoU overlap values.

#     The matcher returns a tensor of size N containing the index of the ground-truth
#     element m that matches to prediction n. If there is no match, a negative value
#     is returned.
#     """

#     BELOW_LOW_THRESHOLD = -1
#     BETWEEN_THRESHOLDS = -2

#     __annotations__ = {
#         "BELOW_LOW_THRESHOLD": int,
#         "BETWEEN_THRESHOLDS": int,
#     }

#     def __init__(self, high_threshold: float, low_threshold: float, allow_low_quality_matches: bool = False) -> None:
#         """
#         Args:
#             high_threshold (float): quality values greater than or equal to
#                 this value are candidate matches.
#             low_threshold (float): a lower quality threshold used to stratify
#                 matches into three levels:
#                 1) matches >= high_threshold
#                 2) BETWEEN_THRESHOLDS matches in [low_threshold, high_threshold)
#                 3) BELOW_LOW_THRESHOLD matches in [0, low_threshold)
#             allow_low_quality_matches (bool): if True, produce additional matches
#                 for predictions that have only low-quality match candidates. See
#                 set_low_quality_matches_ for more details.
#         """
#         self.BELOW_LOW_THRESHOLD = -1
#         self.BETWEEN_THRESHOLDS = -2
#         assert low_threshold <= high_threshold
#         self.high_threshold = high_threshold
#         self.low_threshold = low_threshold
#         self.allow_low_quality_matches = allow_low_quality_matches

#     def __call__(self, match_quality_matrix: Tensor) -> Tensor:
#         """
#         Args:
#             match_quality_matrix (Tensor[float]): an MxN tensor, containing the
#             pairwise quality between M ground-truth elements and N predicted elements.

#         Returns:
#             matches (Tensor[int64]): an N tensor where N[i] is a matched gt in
#             [0, M - 1] or a negative value indicating that prediction i could not
#             be matched.
#         """
#         if match_quality_matrix.numel() == 0:
#             # empty targets or proposals not supported during training
#             if match_quality_matrix.shape[0] == 0:
#                 raise ValueError("No ground-truth boxes available for one of the images during training")
#             else:
#                 raise ValueError("No proposal boxes available for one of the images during training")

#         # match_quality_matrix is M (gt) x N (predicted)
#         # Max over gt elements (dim 0) to find best gt candidate for each prediction
#         matched_vals, matches = match_quality_matrix.max(dim=0)
#         if self.allow_low_quality_matches:
#             all_matches = matches.clone()
#         else:
#             all_matches = None  # type: ignore[assignment]

#         # Assign candidate matches with low quality to negative (unassigned) values
#         below_low_threshold = matched_vals < self.low_threshold
#         between_thresholds = (matched_vals >= self.low_threshold) & (matched_vals < self.high_threshold)
#         matches[below_low_threshold] = self.BELOW_LOW_THRESHOLD
#         matches[between_thresholds] = self.BETWEEN_THRESHOLDS

#         if self.allow_low_quality_matches:
#             assert all_matches is not None
#             self.set_low_quality_matches_(matches, all_matches, match_quality_matrix)

#         return matches

#     def set_low_quality_matches_(self, matches: Tensor, all_matches: Tensor, match_quality_matrix: Tensor) -> None:
#         """
#         Produce additional matches for predictions that have only low-quality matches.
#         Specifically, for each ground-truth find the set of predictions that have
#         maximum overlap with it (including ties); for each prediction in that set, if
#         it is unmatched, then match it to the ground-truth with which it has the highest
#         quality value.
#         """
#         # For each gt, find the prediction with which it has highest quality
#         highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)
#         # Find highest quality match available, even if it is low, including ties
#         gt_pred_pairs_of_highest_quality = torch.where(match_quality_matrix == highest_quality_foreach_gt[:, None])
#         # Example gt_pred_pairs_of_highest_quality:
#         #   tensor([[    0, 39796],
#         #           [    1, 32055],
#         #           [    1, 32070],
#         #           [    2, 39190],
#         #           [    2, 40255],
#         #           [    3, 40390],
#         #           [    3, 41455],
#         #           [    4, 45470],
#         #           [    5, 45325],
#         #           [    5, 46390]])
#         # Each row is a (gt index, prediction index)
#         # Note how gt items 1, 2, 3, and 5 each have two ties

#         pred_inds_to_update = gt_pred_pairs_of_highest_quality[1]
#         matches[pred_inds_to_update] = all_matches[pred_inds_to_update]


# class SSDMatcher(Matcher):
#     def __init__(self, threshold: float) -> None:
#         super().__init__(threshold, threshold, allow_low_quality_matches=False)

#     def __call__(self, match_quality_matrix: Tensor) -> Tensor:
#         matches = super().__call__(match_quality_matrix)

#         # For each gt, find the prediction with which it has the highest quality
#         _, highest_quality_pred_foreach_gt = match_quality_matrix.max(dim=1)
#         matches[highest_quality_pred_foreach_gt] = torch.arange(
#             highest_quality_pred_foreach_gt.size(0), dtype=torch.int64, device=highest_quality_pred_foreach_gt.device
#         )

#         return matches


def overwrite_eps(model: nn.Module, eps: float) -> None:
    """
    This method overwrites the default eps values of all the
    FrozenBatchNorm2d layers of the model with the provided value.
    This is necessary to address the BC-breaking change introduced
    by the bug-fix at pytorch/vision#2933. The overwrite is applied
    only when the pretrained weights are loaded to maintain compatibility
    with previous versions.

    Args:
        model (nn.Module): The model on which we perform the overwrite.
        eps (float): The new value of eps.
    """
    for module in model.modules():
        if isinstance(module, FrozenBatchNorm2d):
            module.eps = eps


def retrieve_out_channels(model: nn.Module, size: Tuple[int, int]) -> List[int]:
    """
    This method retrieves the number of output channels of a specific model.

    Args:
        model (nn.Module): The model for which we estimate the out_channels.
            It should return a single Tensor or an OrderedDict[Tensor].
        size (Tuple[int, int]): The size (wxh) of the input.

    Returns:
        out_channels (List[int]): A list of the output channels of the model.
    """
    in_training = model.training
    model.eval()

    with torch.no_grad():
        # Use dummy data to retrieve the feature map sizes to avoid hard-coding their values
        device = next(model.parameters()).device
        tmp_img = torch.zeros((1, 3, size[1], size[0]), device=device)
        features = model(tmp_img)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        out_channels = [x.size(1) for x in features.values()]

    if in_training:
        model.train()

    return out_channels