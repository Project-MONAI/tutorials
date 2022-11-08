# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Dict, Hashable, Mapping, Optional, List, Sequence, Tuple, Union, Any

import torch
import numpy as np

from monai.data.meta_obj import get_track_meta
from monai.config.type_definitions import DtypeLike, NdarrayOrTensor, KeysCollection
from monai.transforms.post.array import Activations, AsDiscrete, RemoveSmallObjects
from monai.transforms.transform import Transform, MapTransform, RandomizableTransform
from monai.transforms.utils_pytorch_numpy_unification import max, maximum, min
from monai.networks.layers import GaussianFilter, apply_filter, separable_filtering
from monai.utils import TransformBackends, convert_to_numpy, optional_import, min_version
from monai.utils.type_conversion import convert_to_dst_type, convert_to_tensor, convert_data_type
from monai.utils.misc import ensure_tuple_rep
from monai.utils.enums import PostFix
DEFAULT_POST_FIX = PostFix.meta()


label, _ = optional_import("scipy.ndimage.measurements", name="label")
disk, _ = optional_import("skimage.morphology", name="disk")
opening, _ = optional_import("skimage.morphology", name="opening")
watershed, _ = optional_import("skimage.segmentation", name="watershed")
find_contours, _ = optional_import("skimage.measure", name="find_contours")
centroid, _ = optional_import("skimage.measure", name="centroid")
skimage, _ = optional_import("skimage", "0.19.0", min_version)

__all__ = [
    "Watershed",
    "GenerateWatershedMask",
    "GenerateInstanceBorder",
    "GenerateDistanceMap",
    "GenerateWatershedMarkers",
    "WatershedD",
    "WatershedDict",
    "Watershedd",
    "GenerateWatershedMaskD",
    "GenerateWatershedMaskDict",
    "GenerateWatershedMaskd",
    "GenerateInstanceBorderD",
    "GenerateInstanceBorderDict",
    "GenerateInstanceBorderd",
    "GenerateDistanceMapD",
    "GenerateDistanceMapDict",
    "GenerateDistanceMapd",
    "GenerateWatershedMarkersD",
    "GenerateWatershedMarkersDict",
    "GenerateWatershedMarkersd",
    "GenerateSuccinctContour", 
    "GenerateInstanceContour", 
    "GenerateInstanceCentroid", 
    "GenerateInstanceType",
    "RandShiftIntensityd",
]


class SobelGradients(Transform):
    """Calculate Sobel gradients of a grayscale image with the shape of (CxH[xWxDx...]).

    Args:
        kernel_size: the size of the Sobel kernel. Defaults to 3.
        spatial_axes: the axes that define the direction of the gradient to be calculated. It calculate the gradient
            along each of the provide axis. By default it calculate the gradient for all spatial axes.
        normalize_kernels: if normalize the Sobel kernel to provide proper gradients. Defaults to True.
        normalize_gradients: if normalize the output gradient to 0 and 1. Defaults to False.
        padding_mode: the padding mode of the image when convolving with Sobel kernels. Defaults to `"reflect"`.
            Acceptable values are ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
            See ``torch.nn.Conv1d()`` for more information.
        dtype: kernel data type (torch.dtype). Defaults to `torch.float32`.

    """

    backend = [TransformBackends.TORCH]

    def __init__(
        self,
        kernel_size: int = 3,
        spatial_axes: Optional[Union[Sequence[int], int]] = None,
        normalize_kernels: bool = True,
        normalize_gradients: bool = False,
        padding_mode: str = "reflect",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.padding = padding_mode
        self.spatial_axes = spatial_axes
        self.normalize_kernels = normalize_kernels
        self.normalize_gradients = normalize_gradients
        self.kernel_diff, self.kernel_smooth = self._get_kernel(kernel_size, dtype)

    def _get_kernel(self, size, dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        if size < 3:
            raise ValueError(f"Sobel kernel size should be at least three. {size} was given.")
        if size % 2 == 0:
            raise ValueError(f"Sobel kernel size should be an odd number. {size} was given.")

        kernel_diff = torch.tensor([[[-1, 0, 1]]], dtype=dtype)
        kernel_smooth = torch.tensor([[[1, 2, 1]]], dtype=dtype)
        kernel_expansion = torch.tensor([[[1, 2, 1]]], dtype=dtype)

        if self.normalize_kernels:
            if not dtype.is_floating_point:
                raise ValueError(
                    f"`dtype` for Sobel kernel should be floating point when `normalize_kernel==True`. {dtype} was given."
                )
            kernel_diff /= 2.0
            kernel_smooth /= 4.0
            kernel_expansion /= 4.0

        # Expand the kernel to larger size than 3
        expand = (size - 3) // 2
        for _ in range(expand):
            kernel_diff = F.conv1d(kernel_diff, kernel_expansion, padding=2)
            kernel_smooth = F.conv1d(kernel_smooth, kernel_expansion, padding=2)

        return kernel_diff.squeeze(), kernel_smooth.squeeze()

    def __call__(self, image: NdarrayOrTensor) -> torch.Tensor:
        image_tensor = convert_to_tensor(image, track_meta=get_track_meta())

        # Check/set spatial axes
        n_spatial_dims = image_tensor.ndim - 1  # excluding the channel dimension
        valid_spatial_axes = list(range(n_spatial_dims)) + list(range(-n_spatial_dims, 0))

        # Check gradient axes to be valid
        if self.spatial_axes is None:
            spatial_axes = list(range(n_spatial_dims))
        else:
            invalid_axis = set(ensure_tuple(self.spatial_axes)) - set(valid_spatial_axes)
            if invalid_axis:
                raise ValueError(
                    f"The provide axes to calculate gradient is not valid: {invalid_axis}. "
                    f"The image has {n_spatial_dims} spatial dimensions so it should be: {valid_spatial_axes}."
                )
            spatial_axes = [ax % n_spatial_dims if ax < 0 else ax for ax in ensure_tuple(self.spatial_axes)]

        # Add batch dimension for separable_filtering
        image_tensor = image_tensor.unsqueeze(0)

        # Get the Sobel kernels
        kernel_diff = self.kernel_diff.to(image_tensor.device)
        kernel_smooth = self.kernel_smooth.to(image_tensor.device)

        # Calculate gradient
        grad_list = []
        for ax in spatial_axes:
            kernels = [kernel_smooth] * n_spatial_dims
            kernels[ax - 1] = kernel_diff
            grad = separable_filtering(image_tensor, kernels, mode=self.padding)
            if self.normalize_gradients:
                grad_min = grad.min()
                if grad_min != grad.max():
                    grad -= grad_min
                grad_max = grad.max()
                if grad_max > 0:
                    grad /= grad_max
            grad_list.append(grad)

        grads = torch.cat(grad_list, dim=1)

        # Remove batch dimension and convert the gradient type to be the same as input image
        grads = convert_to_dst_type(grads.squeeze(0), image_tensor)[0]

        return grads


class Watershed(Transform):
    """
    Use `skimage.segmentation.watershed` to get instance segmentation results from images.
    See: https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.watershed.

    Args:
        connectivity: An array with the same number of dimensions as image whose non-zero elements indicate
            neighbors for connection. Following the scipy convention, default is a one-connected array of
            the dimension of the image.
        dtype: target data content type to convert, default is np.uint8.

    """

    backend = [TransformBackends.NUMPY]

    def __init__(self, connectivity: Optional[int] = 1, dtype: DtypeLike = np.uint8) -> None:
        self.connectivity = connectivity
        self.dtype = dtype

    def __call__(  # type: ignore
        self, image: NdarrayOrTensor, mask: Optional[NdarrayOrTensor] = None, markers: Optional[NdarrayOrTensor] = None
    ) -> NdarrayOrTensor:
        """
        Args:
            image: image where the lowest value points are labeled first. Shape must be [1, H, W, [D]].
            mask: optional, the same shape as image. Only points at which mask == True will be labeled.
                If None (no mask given), it is a volume of all 1s.
            markers: optional, the same shape as image. The desired number of markers, or an array marking
                the basins with the values to be assigned in the label matrix. Zero means not a marker.
                If None (no markers given), the local minima of the image are used as markers.
        """

        image = convert_to_numpy(image)
        markers = convert_to_numpy(markers)
        mask = convert_to_numpy(mask)

        instance_seg = watershed(image, markers=markers, mask=mask, connectivity=self.connectivity)

        return convert_to_dst_type(instance_seg, image, dtype=self.dtype)[0]


class GenerateWatershedMask(Transform):
    """
    generate mask used in `watershed`. Only points at which mask == True will be labeled.

    Args:
        softmax: if True, apply a softmax function to the prediction.
        sigmoid: if True, apply a sigmoid function to the prediction.
        threshold: if not None, threshold the float values to int number 0 or 1 with specified theashold.
        remove_small_objects: whether need to remove some objects in the marker. Defaults to True.
        min_size: objects smaller than this size are removed if `remove_small_objects` is True. Defaults to 10.
        dtype: target data content type to convert, default is np.uint8.

    """

    backend = [TransformBackends.NUMPY]

    def __init__(
        self,
        softmax: bool = True,
        sigmoid: bool = False,
        threshold: Optional[float] = None,
        remove_small_objects: bool = True,
        min_size: int = 10,
        dtype: DtypeLike = np.uint8,
    ) -> None:
        if sigmoid and threshold is None:
            raise ValueError("Threshold is needed when using sigmoid activation.")

        self.dtype = dtype
        self.activations = Activations(sigmoid=sigmoid, softmax=softmax)
        self.asdiscrete = AsDiscrete(threshold=threshold, argmax=softmax)
        if remove_small_objects:
            self.remove_small_objects = RemoveSmallObjects(min_size=min_size)
        else:
            self.remove_small_objects = None  # type: ignore

    def __call__(self, prob_map: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Args:
            prob_map: probability map of segmentation, shape must be [C, H, W, [D]]
        """

        pred = self.activations(prob_map)
        pred = self.asdiscrete(pred)

        pred = convert_to_numpy(pred)

        pred = label(pred)[0]
        if self.remove_small_objects:
            pred = self.remove_small_objects(pred)
        pred[pred > 0] = 1  # type: ignore

        return convert_to_dst_type(pred, prob_map, dtype=self.dtype)[0]


class GenerateInstanceBorder(Transform):
    """
    Generate instance border by hover map. The more parts of the image that cannot be identified as foreground areas,
    the larger the grey scale value. The grey value of the instance's border will be larger.

    Args:
        kernel_size: the size of the Sobel kernel. Defaults to 21.
        min_size: objects smaller than this size are removed if `remove_small_objects` is True. Defaults to 10.
        remove_small_objects: whether need to remove some objects in segmentation results. Defaults to True.
        dtype: target data content type to convert, default is np.float32.


    Raises:
        ValueError: when the `mask` shape is not [1, H, W].
        ValueError: when the `hover_map` shape is not [2, H, W].

    """

    backend = [TransformBackends.NUMPY]

    def __init__(
        self,
        kernel_size: int = 21,
        min_size: int = 10,
        remove_small_objects: bool = True,
        dtype: DtypeLike = np.float32,
    ) -> None:

        self.dtype = dtype

        self.sobel_gradient = SobelGradients(kernel_size=kernel_size)
        if remove_small_objects:
            self.remove_small_objects = RemoveSmallObjects(min_size=min_size)
        else:
            self.remove_small_objects = None  # type: ignore

    def __call__(self, mask: NdarrayOrTensor, hover_map: NdarrayOrTensor) -> NdarrayOrTensor:  # type: ignore
        """
        Args:
            mask: binarized segmentation result.  Shape must be [1, H, W].
            hover_map:  horizontal and vertical distances of nuclear pixels to their centres of mass. Shape must be [2, H, W].
                The first and second channel represent the horizontal and vertical maps respectively. For more details refer
                to papers: https://arxiv.org/abs/1812.06499.

        Return:
            Instance border map.

        Raises:
            ValueError: when the `hover_map` has only one value.
            ValueError: when the `sobel gradient map` has only one value.

        """
        if len(mask.shape) != 3 or len(hover_map.shape) != 3:
            raise ValueError(
                f"Suppose the mask and hover map should be with shape of [C, H, W], but got {mask.shape}, {hover_map.shape}"
            )
        if mask.shape[0] != 1:
            raise ValueError(f"Suppose the mask only has one channel, but got {mask.shape[0]}")
        if hover_map.shape[0] != 2:
            raise ValueError(f"Suppose the hover map only has two channels, but got {hover_map.shape[0]}")

        hover_h = hover_map[0:1, ...]
        hover_v = hover_map[1:2, ...]

        hover_h_min, hover_h_max = min(hover_h), max(hover_h)
        hover_v_min, hover_v_max = min(hover_v), max(hover_v)
        if (hover_h_max - hover_h_min) == 0 or (hover_v_max - hover_v_min) == 0:
            raise ValueError("Not a valid hover map, please check your input")
        hover_h = (hover_h - hover_h_min) / (hover_h_max - hover_h_min)
        hover_v = (hover_v - hover_v_min) / (hover_v_max - hover_v_min)
        sobelh = self.sobel_gradient(hover_h)[0, ...]
        sobelv = self.sobel_gradient(hover_v)[1, ...]
        
        # cv
#         import cv2
#         sobelh = cv2.Sobel(np.array(hover_h.squeeze().detach().cpu()), cv2.CV_64F, 0, 1, ksize=21)
#         sobelv = cv2.Sobel(np.array(hover_v.squeeze().detach().cpu()), cv2.CV_64F, 1, 0, ksize=21)
        
        sobelh_min, sobelh_max = min(sobelh), max(sobelh)
        sobelv_min, sobelv_max = min(sobelv), max(sobelv)
        if (sobelh_max - sobelh_min) == 0 or (sobelv_max - sobelv_min) == 0:
            raise ValueError("Not a valid sobel gradient map")
        sobelh = 1 - (sobelh - sobelh_min) / (sobelh_max - sobelh_min)
        sobelv = 1 - (sobelv - sobelv_min) / (sobelv_max - sobelv_min)

        # combine the h & v values using max
        overall = maximum(sobelh, sobelv)
        overall = overall - (1 - mask)
        overall[overall < 0] = 0

        return convert_to_dst_type(overall, mask, dtype=self.dtype)[0]


class GenerateDistanceMap(Transform):
    """
    Generate distance map.
    In general, the instance map is calculated from the distance to the background.
    Here, we use 1 - "instance border map" to generate the distance map.
    Nuclei values form mountains so inverse to get basins.

    Args:
        smooth_fn: execute smooth function on distance map. Defaults to None. You can specify
            callable functions for smoothing.
            For example, if you want apply gaussian smooth, you can specify `smooth_fn = GaussianSmooth()`
        dtype: target data content type to convert, default is np.float32.
    """

    backend = [TransformBackends.NUMPY]

    def __init__(self, smooth_fn: Optional[Callable] = None, dtype: DtypeLike = np.float32) -> None:
        self.smooth_fn = smooth_fn
        self.dtype = dtype

    def __call__(self, mask: NdarrayOrTensor, instance_border: NdarrayOrTensor) -> NdarrayOrTensor:  # type: ignore
        """
        Args:
            mask: binarized segmentation result. Shape must be [1, H, W].
            instance_border: foreground probability map. Shape must be [1, H, W].
        """
        if mask.shape[0] != 1 or mask.ndim != 3:
            raise ValueError(f"Input mask should be with size of [1, H, W], but got {mask.shape}")
        if instance_border.shape[0] != 1 or instance_border.ndim != 3:
            raise ValueError(f"Input instance_border should be with size of [1, H, W], but got {instance_border.shape}")

        distance_map = (1.0 - instance_border) * mask

        if callable(self.smooth_fn):
            distance_map = self.smooth_fn(distance_map)

        return convert_to_dst_type(-distance_map, mask, dtype=self.dtype)[0]


class GenerateWatershedMarkers(Transform):
    """
    Generate markers to be used in `watershed`. The watershed algorithm treats pixels values as a local topography
    (elevation). The algorithm floods basins from the markers until basins attributed to different markers meet on
    watershed lines. Generally, markers are chosen as local minima of the image, from which basins are flooded.
    Here is the implementation from HoVerNet papar.
    For more details refer to papers: https://arxiv.org/abs/1812.06499.

    Args:
        threshold: threshold the float values of foreground probability map to int 0 or 1 with specified theashold.
            It turns uncertain area to 1 and other area to 0. Defaults to 0.4.
        radius: the radius of the disk-shaped footprint used in `opening`. Defaults to 2.
        min_size: objects smaller than this size are removed if `remove_small_objects` is True. Defaults to 10.
        remove_small_objects: whether need to remove some objects in the marker. Defaults to True.
        postprocess_fn: execute additional post transformation on marker. Defaults to None.
        dtype: target data content type to convert, default is np.uint8.

    """

    backend = [TransformBackends.NUMPY]

    def __init__(
        self,
        threshold: float = 0.4,
        radius: int = 2,
        min_size: int = 10,
        remove_small_objects: bool = True,
        postprocess_fn: Optional[Callable] = None,
        dtype: DtypeLike = np.uint8,
    ) -> None:
        self.threshold = threshold
        self.radius = radius
        self.postprocess_fn = postprocess_fn
        self.dtype = dtype

        if remove_small_objects:
            self.remove_small_objects = RemoveSmallObjects(min_size=min_size)

    def __call__(self, mask: NdarrayOrTensor, instance_border: NdarrayOrTensor) -> NdarrayOrTensor:  # type: ignore
        """
        Args:
            mask: binarized segmentation result. Shape must be [1, H, W].
            instance_border: instance border map. Shape must be [1, H, W].
        """
        if mask.shape[0] != 1 or mask.ndim != 3:
            raise ValueError(f"Input mask should be with size of [1, H, W], but got {mask.shape}")
        if instance_border.shape[0] != 1 or instance_border.ndim != 3:
            raise ValueError(f"Input instance_border should be with size of [1, H, W], but got {instance_border.shape}")

        instance_border = instance_border >= self.threshold  # uncertain area

        marker = mask - convert_to_dst_type(instance_border, mask, np.uint8)[0]  # certain foreground
        marker[marker < 0] = 0  # type: ignore
        if self.postprocess_fn:
            marker = self.postprocess_fn(marker)

        marker = convert_to_numpy(marker)

        marker = opening(marker.squeeze(), disk(self.radius))
        marker = label(marker)[0]
        if self.remove_small_objects:
            marker = self.remove_small_objects(marker[None])

        return convert_to_dst_type(marker, mask, dtype=self.dtype)[0]



class Watershedd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.apps.pathology.transforms.array.Watershed`.
    Use `skimage.segmentation.watershed` to get instance segmentation results from images.
    See: https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.watershed.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        mask_key: keys of mask used in watershed. Only points at which mask == True will be labeled.
        markers_key: keys of markers used in watershed. If None (no markers given), the local minima of the image are
            used as markers.
        connectivity: An array with the same number of dimensions as image whose non-zero elements indicate neighbors
            for connection. Following the scipy convention, default is a one-connected array of the dimension of the
            image.
        dtype: target data content type to convert. Defaults to np.uint8.
        allow_missing_keys: don't raise exception if key is missing.

    Raises:
        ValueError: when the `image` shape is not [1, H, W].
        ValueError: when the `mask` shape is not [1, H, W].

    """

    backend = Watershed.backend

    def __init__(
        self,
        keys: KeysCollection,
        mask_key: Optional[str] = "mask",
        markers_key: Optional[str] = None,
        connectivity: Optional[int] = 1,
        dtype: DtypeLike = np.uint8,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.mask_key = mask_key
        self.markers_key = markers_key
        self.transform = Watershed(connectivity=connectivity, dtype=dtype)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        markers = d[self.markers_key] if self.markers_key else None
        mask = d[self.mask_key] if self.mask_key else None

        for key in self.key_iterator(d):
            d[key] = self.transform(d[key], mask, markers)

        return d


class GenerateWatershedMaskd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.apps.pathology.transforms.array.GenerateWatershedMask`.

    Args:
        keys: keys of the corresponding items to be transformed.
        mask_key: the mask will be written to the value of `{mask_key}`.
        softmax: if True, apply a softmax function to the prediction.
        sigmoid: if True, apply a sigmoid function to the prediction.
        threshold: if not None, threshold the float values to int number 0 or 1 with specified theashold.
        remove_small_objects: whether need to remove some objects in the marker. Defaults to True.
        min_size: objects smaller than this size are removed if `remove_small_objects` is True. Defaults to 10.
        dtype: target data content type to convert. Defaults to np.uint8.
        allow_missing_keys: don't raise exception if key is missing.

    """

    backend = GenerateWatershedMask.backend

    def __init__(
        self,
        keys: KeysCollection,
        mask_key: str = "mask",
        softmax: bool = True,
        sigmoid: bool = False,
        threshold: Optional[float] = None,
        remove_small_objects: bool = True,
        min_size: int = 10,
        dtype: DtypeLike = np.uint8,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.mask_key = mask_key
        self.transform = GenerateWatershedMask(
            softmax=softmax,
            sigmoid=sigmoid,
            threshold=threshold,
            remove_small_objects=remove_small_objects,
            min_size=min_size,
            dtype=dtype,
        )

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            mask = self.transform(d[key])
            key_to_add = f"{self.mask_key}"
            if key_to_add in d:
                raise KeyError(f"Mask with key {key_to_add} already exists.")
            d[key_to_add] = mask
        return d


class GenerateInstanceBorderd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.apps.pathology.transforms.array.GenerateInstanceBorder`.

    Args:
        keys: keys of the corresponding items to be transformed.
        hover_map_key: keys of hover map used to generate probability map.
        border_key: the instance border map will be written to the value of `{border_key}`.
        kernel_size: the size of the Sobel kernel. Defaults to 21.
        min_size: objects smaller than this size are removed if `remove_small_objects` is True. Defaults to 10.
        remove_small_objects: whether need to remove some objects in segmentation results. Defaults to True.
        dtype: target data content type to convert, default is np.float32.
        allow_missing_keys: don't raise exception if key is missing.

    Raises:
        ValueError: when the `hover_map` has only one value.
        ValueError: when the `sobel gradient map` has only one value.

    """

    backend = GenerateInstanceBorder.backend

    def __init__(
        self,
        keys: KeysCollection,
        hover_map_key: str = "hover_map",
        border_key: str = "border",
        kernel_size: int = 21,
        min_size: int = 10,
        remove_small_objects: bool = True,
        dtype: DtypeLike = np.float32,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.hover_map_key = hover_map_key
        self.border_key = border_key
        self.transform = GenerateInstanceBorder(
            kernel_size=kernel_size, remove_small_objects=remove_small_objects, min_size=min_size, dtype=dtype
        )

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            instance_border = self.transform(d[key], d[self.hover_map_key])
            key_to_add = f"{self.border_key}"
            if key_to_add in d:
                raise KeyError(f"Instance border map with key {key_to_add} already exists.")
            d[key_to_add] = instance_border
        return d


class GenerateDistanceMapd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.apps.pathology.transforms.array.GenerateDistanceMap`.

    Args:
        keys: keys of the corresponding items to be transformed.
        border_key: keys of the instance border map used to generate distance map.
        dist_key: the distance map will be written to the value of `{dist_key}`.
        smooth_fn: execute smooth function on distance map. Defaults to None. You can specify
            callable functions for smoothing.
            For example, if you want apply gaussian smooth, you can specify `smooth_fn = GaussianSmooth()`
        dtype: target data content type to convert, default is np.float32.
        allow_missing_keys: don't raise exception if key is missing.
    """

    backend = GenerateDistanceMap.backend

    def __init__(
        self,
        keys: KeysCollection,
        border_key: str = "border",
        dist_key: str = "dist",
        smooth_fn: Optional[Callable] = None,
        dtype: DtypeLike = np.float32,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.border_key = border_key
        self.dist_key = dist_key
        self.transform = GenerateDistanceMap(smooth_fn=smooth_fn, dtype=dtype)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            distance_map = self.transform(d[key], d[self.border_key])
            key_to_add = f"{self.dist_key}"
            if key_to_add in d:
                raise KeyError(f"Distance map with key {key_to_add} already exists.")
            d[key_to_add] = distance_map
        return d


class GenerateWatershedMarkersd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.apps.pathology.transforms.array.GenerateWatershedMarkers`.

    Args:
        keys: keys of the corresponding items to be transformed.
        border_key: keys of the instance border map used to generate markers.
        markers_key: the markers will be written to the value of `{markers_key}`.
        threshold: threshold the float values of instance border map to int 0 or 1 with specified theashold.
            It turns uncertain area to 1 and other area to 0. Defaults to 0.4.
        radius: the radius of the disk-shaped footprint used in `opening`. Defaults to 2.
        min_size: objects smaller than this size are removed if `remove_small_objects` is True. Defaults to 10.
        remove_small_objects: whether need to remove some objects in the marker. Defaults to True.
        postprocess_fn: execute additional post transformation on marker. Defaults to None.
        dtype: target data content type to convert, default is np.uint8.
        allow_missing_keys: don't raise exception if key is missing.
    """

    backend = GenerateWatershedMarkers.backend

    def __init__(
        self,
        keys: KeysCollection,
        border_key: str = "border",
        markers_key: str = "markers",
        threshold: float = 0.4,
        radius: int = 2,
        min_size: int = 10,
        remove_small_objects: bool = True,
        postprocess_fn: Optional[Callable] = None,
        dtype: DtypeLike = np.uint8,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.border_key = border_key
        self.markers_key = markers_key
        self.transform = GenerateWatershedMarkers(
            threshold=threshold,
            radius=radius,
            min_size=min_size,
            remove_small_objects=remove_small_objects,
            postprocess_fn=postprocess_fn,
            dtype=dtype,
        )

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            markers = self.transform(d[key], d[self.border_key])
            key_to_add = f"{self.markers_key}"
            if key_to_add in d:
                raise KeyError(f"Markers with key {key_to_add} already exists.")
            d[key_to_add] = markers
        return d



class GenerateSuccinctContour(Transform):
    """
    Converts Scipy-style contours(generated by skimage.measure.find_contours) to a more succinct version which only includes
    the pixels to which lines need to be drawn (i.e. not the intervening pixels along each line).

    Args:
        height: height of bounding box, used to detect direction of line segment.
        width: width of bounding box, used to detect direction of line segment.

    Returns:
        the pixels that need to be joined by straight lines to describe the outmost pixels of the foreground similar to
            OpenCV's cv.CHAIN_APPROX_SIMPLE (anti-clockwise)
    """

    backend = [TransformBackends.NUMPY]

    def __init__(self, height: int, width: int) -> None:
        self.height = height
        self.width = width

    def _generate_contour_coord(self, current: np.ndarray, previous: np.ndarray) -> Tuple[int, int]:
        """
        Generate contour coordinates. Given the previous and current coordinates of border positions,
        returns the int pixel that marks the extremity of the segmented pixels.

        Args:
            current: coordinates of the current border position.
            previous: coordinates of the previous border position.
        """

        p_delta = (current[0] - previous[0], current[1] - previous[1])

        if p_delta == (0.0, 1.0) or p_delta == (0.5, 0.5) or p_delta == (1.0, 0.0):
            row = int(current[0] + 0.5)
            col = int(current[1])
        elif p_delta == (0.0, -1.0) or p_delta == (0.5, -0.5):
            row = int(current[0])
            col = int(current[1])
        elif p_delta == (-1, 0.0) or p_delta == (-0.5, -0.5):
            row = int(current[0])
            col = int(current[1] + 0.5)
        elif p_delta == (-0.5, 0.5):
            row = int(current[0] + 0.5)
            col = int(current[1] + 0.5)

        return row, col

    def _calculate_distance_from_topleft(self, sequence: Sequence[Tuple[int, int]]) -> int:
        """
        Each sequence of coordinates describes a boundary between foreground and background starting and ending at two sides
        of the bounding box. To order the sequences correctly, we compute the distance from the topleft of the bounding box
        around the perimeter in a clockwise direction.

        Args:
            sequence: list of border points coordinates.

        Returns:
            the distance round the perimeter of the bounding box from the top-left origin
        """
        distance: int
        first_coord = sequence[0]
        if first_coord[0] == 0:
            distance = first_coord[1]
        elif first_coord[1] == self.width - 1:
            distance = self.width + first_coord[0]
        elif first_coord[0] == self.height - 1:
            distance = 2 * self.width + self.height - first_coord[1]
        else:
            distance = 2 * (self.width + self.height) - first_coord[0]

        return distance

    def __call__(self, contours: List[np.ndarray]) -> np.ndarray:
        """
        Args:
            contours: list of (n, 2)-ndarrays, scipy-style clockwise line segments, with lines separating foreground/background.
                Each contour is an ndarray of shape (n, 2), consisting of n (row, column) coordinates along the contour.
        """
        pixels: List[Tuple[int, int]] = []
        sequences = []
        corners = [False, False, False, False]
        # print('raw', contours)
        for group in contours:
            sequence: List[Tuple[int, int]] = []
            last_added = None
            prev = None
            corner = -1

            for i, coord in enumerate(group):
                if i == 0:
                    # originating from the top, so must be heading south east
                    if coord[0] == 0.0:
                        corner = 1
                        pixel = (0, int(coord[1] - 0.5))
                        if pixel[1] == self.width - 1:
                            corners[1] = True
                        elif pixel[1] == 0.0:
                            corners[0] = True
                    # originating from the left, so must be heading north east
                    elif coord[1] == 0.0:
                        corner = 0
                        pixel = (int(coord[0] + 0.5), 0)
                    # originating from the bottom, so must be heading north west
                    elif coord[0] == self.height - 1:
                        corner = 3
                        pixel = (int(coord[0]), int(coord[1] + 0.5))
                        if pixel[1] == self.width - 1:
                            corners[2] = True
                    # originating from the right, so must be heading south west
                    elif coord[1] == self.width - 1:
                        corner = 2
                        pixel = (int(coord[0] - 0.5), int(coord[1]))
                    else:
                        raise ValueError(f"Incorrect coord!, {coord}")
                    # print('********', i, coord[0], coord[1], self.width, self.height)
                    sequence.append(pixel)
                    last_added = pixel
                elif i == len(group) - 1:
                    # add this point
                    pixel = self._generate_contour_coord(coord, prev)  # type: ignore
                    if pixel != last_added:
                        sequence.append(pixel)
                        last_added = pixel
                elif np.any(coord - prev != group[i + 1] - coord):
                    pixel = self._generate_contour_coord(coord, prev)  # type: ignore
                    if pixel != last_added:
                        sequence.append(pixel)
                        last_added = pixel

                # flag whether each corner has been crossed
                if i == len(group) - 1:
                    if corner == 0:
                        if coord[0] == 0:
                            corners[corner] = True
                    elif corner == 1:
                        if coord[1] == self.width - 1:
                            corners[corner] = True
                    elif corner == 2:
                        if coord[0] == self.height - 1:
                            corners[corner] = True
                    elif corner == 3:
                        if coord[1] == 0.0:
                            corners[corner] = True

                prev = coord
            dist = self._calculate_distance_from_topleft(sequence)

            sequences.append({"distance": dist, "sequence": sequence})

        # check whether we need to insert any missing corners
        if corners[0] is False:
            sequences.append({"distance": 0, "sequence": [(0, 0)]})
        if corners[1] is False:
            sequences.append({"distance": self.width, "sequence": [(0, self.width - 1)]})
        if corners[2] is False:
            sequences.append({"distance": self.width + self.height, "sequence": [(self.height - 1, self.width - 1)]})
        if corners[3] is False:
            sequences.append({"distance": 2 * self.width + self.height, "sequence": [(self.height - 1, 0)]})

        # join the sequences into a single contour
        # starting at top left and rotating clockwise
        sequences.sort(key=lambda x: x.get("distance"))  # type: ignore

        last = (-1, -1)
        for _sequence in sequences:
            if _sequence["sequence"][0] == last:  # type: ignore
                pixels.pop()
            if pixels:
                pixels = [*pixels, *_sequence["sequence"]]  # type: ignore
            else:
                pixels = _sequence["sequence"]  # type: ignore
            last = pixels[-1]

        if pixels[0] == last:
            pixels.pop(0)

        if pixels[0] == (0, 0):
            pixels.append(pixels.pop(0))

        return np.flip(convert_to_numpy(pixels, dtype=np.int32))  # type: ignore


class GenerateInstanceContour(Transform):
    """
    Generate contour for each instance in a 2D array. Use `GenerateSuccinctContour` to only include
    the pixels to which lines need to be drawn

    Args:
        points_num: assumed that the created contour does not form a contour if it does not contain more points
            than the specified value. Defaults to 3.
        level: optional. Value along which to find contours in the array. By default, the level is set
            to (max(image) + min(image)) / 2.

    """

    backend = [TransformBackends.NUMPY]

    def __init__(self, points_num: int = 3, level: Optional[float] = None) -> None:
        self.level = level
        self.points_num = points_num

    def __call__(self, image: NdarrayOrTensor, offset: Sequence[int] = (0, 0)) -> np.ndarray:
        """
        Args:
            image: instance-level segmentation result. Shape should be [C, H, W]
            offset: optional, offset of starting position of the instance in the array, default is (0, 0).
        """
        inst_contour_cv = find_contours(image, level=self.level)
        generate_contour = GenerateSuccinctContour(image.shape[0], image.shape[1])
        inst_contour = generate_contour(inst_contour_cv)

        # < `self.points_num` points don't make a contour, so skip, likely artifact too
        # as the contours obtained via approximation => too small or sthg
        if inst_contour.shape[0] < self.points_num:
            print(f"< {self.points_num} points don't make a contour, so skip")
            return None  # type: ignore
        # check for tricky shape
        elif len(inst_contour.shape) != 2:
            print(f"{len(inst_contour.shape)} != 2, check for tricky shape")
            return None  # type: ignore
        else:
            inst_contour[:, 0] += offset[0]  # type: ignore
            inst_contour[:, 1] += offset[1]  # type: ignore
            return inst_contour


class GenerateInstanceCentroid(Transform):
    """
    Generate instance centroid using `skimage.measure.centroid`.

    Args:
        dtype: the data type of output centroid.

    """

    backend = [TransformBackends.NUMPY]

    def __init__(self, dtype: Optional[DtypeLike] = None) -> None:
        self.dtype = dtype

    def __call__(self, image: NdarrayOrTensor, offset: Union[Sequence[int], int] = 0) -> np.ndarray:
        """
        Args:
            image: instance-level segmentation result. Shape should be [1, H, W, [D]]
            offset: optional, offset of starting position of the instance in the array, default is 0 for each dim.

        """
        image = convert_to_numpy(image)
        image = image.squeeze(0)  # squeeze channel dim
        ndim = len(image.shape)
        offset = ensure_tuple_rep(offset, ndim)

        inst_centroid = centroid(image)
        for i in range(ndim):
            inst_centroid[i] += offset[i]

        return convert_to_dst_type(inst_centroid, image, dtype=self.dtype)[0]  # type: ignore


class GenerateInstanceType(Transform):
    """
    Generate instance type and probability for each instance.
    """

    backend = [TransformBackends.NUMPY]

    def __init__(self) -> None:
        super().__init__()

    def __call__(  # type: ignore
        self, type_pred: NdarrayOrTensor, seg_pred: NdarrayOrTensor, bbox: np.ndarray, instance_id: int
    ) -> Tuple[int, float]:
        """
        Args:
            type_pred: pixel-level type prediction map after activation function.
            seg_pred: pixel-level segmentation prediction map after activation function.
            bbox: bounding box coordinates of the instance, shape is [channel, 2 * spatial dims].
            instance_id: get instance type from specified instance id.
        """

        type_pred = convert_to_numpy(type_pred)
        seg_pred = convert_to_numpy(seg_pred)
        rmin, rmax, cmin, cmax = bbox.flatten()
        seg_map_crop = seg_pred[0, rmin:rmax, cmin:cmax]
        type_map_crop = type_pred[0, rmin:rmax, cmin:cmax]

        seg_map_crop = seg_map_crop == instance_id
        # print(type_map_crop.shape, seg_map_crop.shape, type(seg_map_crop))
        inst_type = type_map_crop[seg_map_crop]  # type: ignore
        type_list, type_pixels = np.unique(inst_type, return_counts=True)
        type_list = list(zip(type_list, type_pixels))
        type_list = sorted(type_list, key=lambda x: x[1], reverse=True)  # type: ignore
        inst_type = type_list[0][0]
        if inst_type == 0:  # ! pick the 2nd most dominant if exist
            if len(type_list) > 1:
                inst_type = type_list[1][0]
        type_dict = {v[0]: v[1] for v in type_list}
        type_prob = type_dict[inst_type] / (np.sum(seg_map_crop) + 1.0e-6)
        return (int(inst_type), float(type_prob))


# +
from monai.transforms.utils_pytorch_numpy_unification import clip

def add_to_brightness(images, random_state, parents, hooks, range=None):
    """Perturbe the brightness of input images."""
    img = images[0]  # aleju input batch as default (always=1 in our case)
    value = random_state.uniform(*range)
    ret = np.clip(img + value, 0, 255)
    ret = ret.astype(np.uint8)
    return [ret]
 

# -

WatershedD = WatershedDict = Watershedd
GenerateWatershedMaskD = GenerateWatershedMaskDict = GenerateWatershedMaskd
GenerateInstanceBorderD = GenerateInstanceBorderDict = GenerateInstanceBorderd
GenerateDistanceMapD = GenerateDistanceMapDict = GenerateDistanceMapd
GenerateWatershedMarkersD = GenerateWatershedMarkersDict = GenerateWatershedMarkersd
