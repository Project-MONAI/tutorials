import math
from typing import List, Optional

import torch
from torch import nn, Tensor

from monai.utils.module import look_up_option


class AnchorGenerator(nn.Module):
    """
    Module that generates anchors for a set of feature maps and
    image sizes.

    The module support computing anchors at multiple sizes and aspect ratios
    per feature map. This module assumes aspect ratio = height / width for
    each anchor.

    sizes and aspect_ratios should have the same number of elements, and it should
    correspond to the number of feature maps.

    sizes[i] and aspect_ratios[i] can have an arbitrary number of elements.
    2D: for each element, anchor width and height w:h = 1:aspect_ratios[i,j]
    3D: for each element, anchor width, height, and depth w:h:d = 1:aspect_ratios[i,j,0]:aspect_ratios[i,j,1]
    
    AnchorGenerator will output a set of sizes[i] * aspect_ratios[i] anchors
    per spatial location for feature map i.

    Args:
        sizes (Tuple[Tuple[int]]):
        aspect_ratios (Tuple[Tuple[float]]):
    """

    __annotations__ = {
        "cell_anchors": List[torch.Tensor],
    }

    def __init__(
        self,
        spatial_dims: int,
        sizes=((128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),),

    ):
        super().__init__()

        if not isinstance(sizes[0], (list, tuple)):
            # TODO change this
            sizes = tuple((s,) for s in sizes)
        if not isinstance(aspect_ratios[0], (list, tuple)):
            aspect_ratios = (aspect_ratios,) * len(sizes)

        assert len(sizes) == len(aspect_ratios)

        spatial_dims = look_up_option(spatial_dims, [2, 3])
        self.spatial_dims = spatial_dims
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.cell_anchors = [
            self.generate_anchors(size, aspect_ratio) for size, aspect_ratio in zip(sizes, aspect_ratios)
        ]

    # TODO: https://github.com/pytorch/pytorch/issues/26792
    # For every (aspect_ratios, scales) combination, output a zero-centered anchor with those values.
    # (scales, aspect_ratios) are usually an element of zip(self.scales, self.aspect_ratios)
    # This method assumes aspect ratio = height / width for an anchor.
    def generate_anchors(
        self,
        scales: List[int],
        aspect_ratios: List,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ):
        scales = torch.as_tensor(scales, dtype=dtype, device=device)
        aspect_ratios = torch.as_tensor(aspect_ratios, dtype=dtype, device=device)
        if self.spatial_dims != len(aspect_ratios.shape)-1:
            ValueError(f"In {self.spatial_dims}-D image, aspect_ratios for each level should be {len(aspect_ratios.shape)-1}-D. But got aspect_ratios with shape {aspect_ratios.shape}.")

        if (self.spatial_dims == 3) and (aspect_ratios.shape[1] != 2):
            ValueError(f"In {self.spatial_dims}-D image, aspect_ratios for each level should has shape (_,2). But got aspect_ratios with shape {aspect_ratios.shape}.")

        # if 2d, w:h = 1:aspect_ratios
        if self.spatial_dims == 2:
            area_scale = torch.sqrt(aspect_ratios)
            w_ratios = 1 / area_scale
            h_ratios = area_scale
        # if 3d, w:h:d = 1:aspect_ratios[:,0]:aspect_ratios[:,1]
        elif self.spatial_dims == 3:
            area_scale = torch.pow(aspect_ratios[:,0]*aspect_ratios[:,1], 1/3.0)   
            w_ratios = 1 / area_scale
            h_ratios = aspect_ratios[:,0]/area_scale
            d_ratios = aspect_ratios[:,1]/area_scale

        ws = (w_ratios[:, None] * scales[None, :]).view(-1)
        hs = (h_ratios[:, None] * scales[None, :]).view(-1)        
        if self.spatial_dims == 2:
            base_anchors = torch.stack([-ws, ws, -hs, hs], dim=1) / 2        
        elif self.spatial_dims == 3:
            ds = (d_ratios[:, None] * scales[None, :]).view(-1)
            base_anchors = torch.stack([-ws, ws, -hs, hs, -ds, ds], dim=1) / 2

        return base_anchors.round()

    def set_cell_anchors(self, dtype: torch.dtype, device: torch.device):
        self.cell_anchors = [cell_anchor.to(dtype=dtype, device=device) for cell_anchor in self.cell_anchors]

    def num_anchors_per_location(self):
        return [len(s) * len(a) for s, a in zip(self.sizes, self.aspect_ratios)]

    # For every combination of (a, (g, s), i) in (self.cell_anchors, zip(grid_sizes, strides), 0:2),
    # output g[i] anchors that are s[i] distance apart in direction i, with the same dimensions as a.
    def grid_anchors(self, grid_sizes: List[List[int]], strides: List[List[Tensor]]) -> List[Tensor]:
        anchors = []
        cell_anchors = self.cell_anchors
        assert cell_anchors is not None

        if not (len(grid_sizes) == len(strides) == len(cell_anchors)):
            raise ValueError(
                "Anchors should be Tuple[Tuple[int]] because each feature "
                "map could potentially have different sizes and aspect ratios. "
                "There needs to be a match between the number of "
                "feature maps passed and the number of sizes / aspect ratios specified."
            )

        for size, stride, base_anchors in zip(grid_sizes, strides, cell_anchors):
            device = base_anchors.device

            # For output anchor, compute [x_center, x_center, y_center, y_center]
            shifts_xyz = []
            for axis in range(self.spatial_dims):
                shifts_xyz.append( torch.arange(0, size[axis], dtype=torch.int32, device=device) * stride[axis] )
            shifts_xyz = list( torch.meshgrid(*tuple(shifts_xyz), indexing="ij") )
            for axis in range(self.spatial_dims):
                shifts_xyz[axis] = shifts_xyz[axis].reshape(-1)
            
            if self.spatial_dims == 2:
                shifts = torch.stack((shifts_xyz[0], shifts_xyz[0], shifts_xyz[1], shifts_xyz[1]), dim=1)
            elif self.spatial_dims == 3:
                shifts = torch.stack((shifts_xyz[0], shifts_xyz[0], shifts_xyz[1], shifts_xyz[1], shifts_xyz[2], shifts_xyz[2]), dim=1)

            # For every (base anchor, output anchor) pair,
            # offset each zero-centered base anchor by the center of the output anchor.
            anchors.append((shifts.view(-1, 1, self.spatial_dims*2) + base_anchors.view(1, -1, self.spatial_dims*2)).reshape(-1, self.spatial_dims*2))

        return anchors

    def forward(self, images, orig_image_size_list, feature_maps: List[Tensor]) -> List[Tensor]:
        grid_sizes = [feature_map.shape[-self.spatial_dims:] for feature_map in feature_maps]
        image_size = images.shape[-self.spatial_dims:]
        dtype, device = feature_maps[0].dtype, feature_maps[0].device
        strides = [
            [
                torch.tensor(image_size[axis] // g[axis], dtype=torch.int64, device=device) for axis in range(self.spatial_dims)
            ]
            for g in grid_sizes
        ]
        self.set_cell_anchors(dtype, device)
        anchors_over_all_feature_maps = self.grid_anchors(grid_sizes, strides)
        anchors: List[List[torch.Tensor]] = []
        for _ in range(len(orig_image_size_list)):
            anchors_in_image = [anchors_per_feature_map for anchors_per_feature_map in anchors_over_all_feature_maps]
            anchors.append(anchors_in_image)
        anchors = [torch.cat(anchors_per_image) for anchors_per_image in anchors]
        return anchors


# class DefaultBoxGenerator(nn.Module):
#     """
#     This module generates the default boxes of SSD for a set of feature maps and image sizes.

#     Args:
#         aspect_ratios (List[List[int]]): A list with all the aspect ratios used in each feature map.
#         min_ratio (float): The minimum scale :math:`\text{s}_{\text{min}}` of the default boxes used in the estimation
#             of the scales of each feature map. It is used only if the ``scales`` parameter is not provided.
#         max_ratio (float): The maximum scale :math:`\text{s}_{\text{max}}`  of the default boxes used in the estimation
#             of the scales of each feature map. It is used only if the ``scales`` parameter is not provided.
#         scales (List[float]], optional): The scales of the default boxes. If not provided it will be estimated using
#             the ``min_ratio`` and ``max_ratio`` parameters.
#         steps (List[int]], optional): It's a hyper-parameter that affects the tiling of defalt boxes. If not provided
#             it will be estimated from the data.
#         clip (bool): Whether the standardized values of default boxes should be clipped between 0 and 1. The clipping
#             is applied while the boxes are encoded in format ``(cx, cy, w, h)``.
#     """

#     def __init__(
#         self,
#         aspect_ratios: List[List[int]],
#         min_ratio: float = 0.15,
#         max_ratio: float = 0.9,
#         scales: Optional[List[float]] = None,
#         steps: Optional[List[int]] = None,
#         clip: bool = True,
#     ):
#         super().__init__()
#         if steps is not None:
#             assert len(aspect_ratios) == len(steps)
#         self.aspect_ratios = aspect_ratios
#         self.steps = steps
#         self.clip = clip
#         num_outputs = len(aspect_ratios)

#         # Estimation of default boxes scales
#         if scales is None:
#             if num_outputs > 1:
#                 range_ratio = max_ratio - min_ratio
#                 self.scales = [min_ratio + range_ratio * k / (num_outputs - 1.0) for k in range(num_outputs)]
#                 self.scales.append(1.0)
#             else:
#                 self.scales = [min_ratio, max_ratio]
#         else:
#             self.scales = scales

#         self._wh_pairs = self._generate_wh_pairs(num_outputs)

#     def _generate_wh_pairs(
#         self, num_outputs: int, dtype: torch.dtype = torch.float32, device: torch.device = torch.device("cpu")
#     ) -> List[Tensor]:
#         _wh_pairs: List[Tensor] = []
#         for k in range(num_outputs):
#             # Adding the 2 default width-height pairs for aspect ratio 1 and scale s'k
#             s_k = self.scales[k]
#             s_prime_k = math.sqrt(self.scales[k] * self.scales[k + 1])
#             wh_pairs = [[s_k, s_k], [s_prime_k, s_prime_k]]

#             # Adding 2 pairs for each aspect ratio of the feature map k
#             for ar in self.aspect_ratios[k]:
#                 sq_ar = math.sqrt(ar)
#                 w = self.scales[k] * sq_ar
#                 h = self.scales[k] / sq_ar
#                 wh_pairs.extend([[w, h], [h, w]])

#             _wh_pairs.append(torch.as_tensor(wh_pairs, dtype=dtype, device=device))
#         return _wh_pairs

#     def num_anchors_per_location(self):
#         # Estimate num of anchors based on aspect ratios: 2 default boxes + 2 * ratios of feaure map.
#         return [2 + 2 * len(r) for r in self.aspect_ratios]

#     # Default Boxes calculation based on page 6 of SSD paper
#     def _grid_default_boxes(
#         self, grid_sizes: List[List[int]], image_size: List[int], dtype: torch.dtype = torch.float32
#     ) -> Tensor:
#         default_boxes = []
#         for k, f_k in enumerate(grid_sizes):
#             # Now add the default boxes for each width-height pair
#             if self.steps is not None:
#                 x_f_k = image_size[0] / self.steps[k]
#                 y_f_k = image_size[1] / self.steps[k]
#             else:
#                 y_f_k, x_f_k = f_k

#             shifts_x = ((torch.arange(0, f_k[1]) + 0.5) / x_f_k).to(dtype=dtype)
#             shifts_y = ((torch.arange(0, f_k[0]) + 0.5) / y_f_k).to(dtype=dtype)
#             shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
#             shift_x = shift_x.reshape(-1)
#             shift_y = shift_y.reshape(-1)

#             shifts = torch.stack((shift_x, shift_y) * len(self._wh_pairs[k]), dim=-1).reshape(-1, 2)
#             # Clipping the default boxes while the boxes are encoded in format (cx, cy, w, h)
#             _wh_pair = self._wh_pairs[k].clamp(min=0, max=1) if self.clip else self._wh_pairs[k]
#             wh_pairs = _wh_pair.repeat((f_k[0] * f_k[1]), 1)

#             default_box = torch.cat((shifts, wh_pairs), dim=1)

#             default_boxes.append(default_box)

#         return torch.cat(default_boxes, dim=0)

#     def __repr__(self) -> str:
#         s = self.__class__.__name__ + "("
#         s += "aspect_ratios={aspect_ratios}"
#         s += ", clip={clip}"
#         s += ", scales={scales}"
#         s += ", steps={steps}"
#         s += ")"
#         return s.format(**self.__dict__)

#     def forward(self, images, orig_image_size_list, feature_maps: List[Tensor]) -> List[Tensor]:
#         grid_sizes = [feature_map.shape[-2:] for feature_map in feature_maps]
#         image_size = images.shape[-2:]
#         dtype, device = feature_maps[0].dtype, feature_maps[0].device
#         default_boxes = self._grid_default_boxes(grid_sizes, image_size, dtype=dtype)
#         default_boxes = default_boxes.to(device)

#         dboxes = []
#         for _ in orig_image_size_list:
#             dboxes_in_image = default_boxes
#             dboxes_in_image = torch.cat(
#                 [
#                     dboxes_in_image[:, :2] - 0.5 * dboxes_in_image[:, 2:],
#                     dboxes_in_image[:, :2] + 0.5 * dboxes_in_image[:, 2:],
#                 ],
#                 -1,
#             )
#             dboxes_in_image[:, 0::2] *= image_size[1]
#             dboxes_in_image[:, 1::2] *= image_size[0]
#             dboxes.append(dboxes_in_image)
#         return dboxes