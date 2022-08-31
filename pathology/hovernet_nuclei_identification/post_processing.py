"""Post Processing of Hovernet for patch inference."""
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

import numpy as np
from scipy.ndimage import (binary_fill_holes, generate_binary_structure, label,
                           measurements, gaussian_filter)
from scipy.signal import correlate2d
from skimage.exposure import rescale_intensity
from skimage.measure import find_contours, moments
from skimage.morphology import disk, opening
from skimage.segmentation import watershed


def get_bounding_box(img):
    """Get bounding box coordinate information."""

    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmax += 1
    cmax += 1

    return [rmin, rmax, cmin, cmax]


def _remove_small_objects(pred, min_size=64, connectivity=1):
    """Remove connected components smaller than the specified size.

    This function is taken from skimage.morphology.remove_small_objects, but the warning
    is removed when a single label is provided.

    Args:
        pred: input labelled array
        min_size: minimum size of instance in output array
        connectivity: The connectivity defining the neighborhood of a pixel.

    Returns:
        out: output array with instances removed under min_size

    """
    out = pred

    if min_size == 0:  # shortcut for efficiency
        return out

    if out.dtype == bool:
        selem = generate_binary_structure(pred.ndim, connectivity)
        ccs = np.zeros_like(pred, dtype=np.int32)
        label(pred, selem, output=ccs)
    else:
        ccs = out

    try:
        component_sizes = np.bincount(ccs.ravel())
    except ValueError as exc:
        raise ValueError(
            "Negative value labels are not supported. Try "
            "relabeling the input with `scipy.ndimage.label` or "
            "`skimage.morphology.label`."
        ) from exc

    too_small = component_sizes < min_size
    too_small_mask = too_small[ccs]
    out[too_small_mask] = 0

    return out

def _get_sobel_kernel(size):
    """Get sobel kernel with a given (odd) size."""

    assert size % 2 == 1, "Kernel size must be odd. Requested size=%d" % size

    h_range = np.arange(
        -size // 2 + 1,
        size // 2 + 1,
        dtype=np.float32
    )
    v_range = np.arange(
        -size // 2 + 1,
        size // 2 + 1,
        dtype=np.float32
    )
    h, v = np.meshgrid(h_range, v_range)
    kernel_h = h / (h * h + v * v + 1.0e-15)
    kernel_v = v / (h * h + v * v + 1.0e-15)
    return kernel_h, kernel_v

def _proc_np_hv(pred, hover):
    """Process Nuclei Prediction with XY Coordinate Map.

    Args:
        pred: prediction output, assuming
              channel 0 contain probability map of nuclei
              channel 1 containing the regressed X-map
              channel 2 containing the regressed Y-map
      
        returns: instance labelled segmentation map

    """
    pred = np.array(pred, dtype=np.float32)
    hover = np.array(hover, dtype=np.float32)

    h_dir_raw = hover[0]
    v_dir_raw = hover[1]

    k_h, k_v = _get_sobel_kernel(17)

    # processing
    blb = np.array(pred >= 0.5, dtype=np.int32)

    blb = measurements.label(blb)[0]
    blb = _remove_small_objects(blb, min_size=10)
    blb[blb > 0] = 1  # background is 0 already

    h_dir = rescale_intensity(h_dir_raw,out_range=(0,1))
    v_dir = rescale_intensity(v_dir_raw,out_range=(0,1))

    sobelh = correlate2d(h_dir,k_h, mode='same',boundary = 'symm')
    sobelv = correlate2d(v_dir,k_v,mode='same',boundary = 'symm')
    sobelh = 1 - rescale_intensity(sobelh,out_range=(0,1))
    sobelv = 1 - rescale_intensity(sobelv,out_range=(0,1))

    # combine the h & v values using max
    overall = np.maximum(sobelh, sobelv)
    overall = overall - (1 - blb)
    overall[overall < 0] = 0

    dist = (1.0 - overall) * blb

    ## nuclei values form mountains so inverse to get basins
    dist = np.negative(gaussian_filter(dist, sigma=0.4))
    overall = np.array(overall >= 0.4, dtype=np.int32)

    marker = blb - overall
    marker[marker < 0] = 0
    marker = binary_fill_holes(marker).astype("uint8")
    marker = opening(marker, disk(2))
    marker = measurements.label(marker)[0]
    marker = _remove_small_objects(marker, min_size=10)

    proced_pred = watershed(dist, markers=marker, mask=blb)

    return proced_pred

def _coords_to_pixel(current,previous):
    """For contour coordinate generation.
       Given the previous and current border positions,
       returns the int pixel that marks the extremity
       of the segmented pixels
    """

    p_delta = (current[0] - previous[0], current[1] - previous[1])

    if p_delta==(0.,1.):
        row=int(current[0] + 0.5 )
        col=int(current[1])
    elif p_delta==(0.,-1.):
        row=int(current[0] )
        col=int(current[1])
    elif p_delta==(0.5,0.5):
        row=int(current[0] + 0.5)
        col=int(current[1])
    elif p_delta==(0.5,-0.5):
        row=int(current[0])
        col=int(current[1])
    elif p_delta==(1.,0.):
        row=int(current[0] + 0.5)
        col=int(current[1])    
    elif p_delta==(-1,0.):
        row=int(current[0])
        col=int(current[1] + 0.5)
    elif p_delta==(-0.5,0.5):
        row=int(current[0] + 0.5)
        col=int(current[1] + 0.5)
    elif p_delta==(-0.5,-0.5):
        row=int(current[0])
        col=int(current[1] + 0.5)

    return row, col

def _dist_from_topleft(sequence, h, w):
    """For contour coordinate generation.
       Each sequence of cordinates describes a boundary between
       foreground and background starting and ending at two sides
       of the bounding box. To order the sequences correctly,
       we compute the distannce from the topleft of the bounding box
       around the perimeter in a clock-wise direction.

        Args:
            sequence: list of border points
            h: height of the bounding box
            w: width of the bounding box

        Returns:
            distance: the distance round the perimeter of the bounding
                box from the top-left origin
    """

    first = sequence[0]
    if first[0]==0:
        distance = first[1]
    elif first[1] == w-1:
        distance = w + first[0]
    elif first[0] == h-1:
        distance = 2*w + h - first[1]
    else:
        distance = 2*(w + h) - first[0]

    return distance

def _sp_contours_to_cv(contours, h, w):
    """Converts Scipy-style contours to a more succinct version
       which only includes the pixels to which lines need to
       be drawn between (i.e. not the intervening pixels along each line).

    Args:
        contours: scipy-style clockwise line segments, with line separating foreground/background
        h: Height of bounding box - used to detect direction of line segment
        w: Width of bounding box - used to detect direction of line segment

    Returns:
        pixels: the pixels that need to be joined by straight lines to
                describe the outmost pixels of the foreground similar to
                OpenCV's cv.CHAIN_APPROX_SIMPLE (anti-clockwise)

    """
    pixels = None
    sequences = []
    corners = [False,False,False,False]

    for group in contours:
        sequence = []
        last_added = None
        prev = None
        corner = -1

        for i,coord in enumerate(group):
            if i==0:
                if coord[0] == 0.0:
                    # originating from the top, so must be heading south east
                    corner = 1
                    pixel =(0,int(coord[1] - 0.5))
                    if pixel[1] == w - 1:
                        corners[1]=True
                    elif pixel[1] == 0.0:
                        corners[0]=True
                elif coord[1] == 0.0:
                    corner = 0
                    # originating from the left, so must be heading north east
                    pixel =(int(coord[0] + 0.5), 0)
                elif coord[0] == h - 1:
                    corner = 3
                    # originating from the bottom, so must be heading north west
                    pixel = (int(coord[0]), int(coord[1] + 0.5))
                    if pixel[1] == w - 1:
                        corners[2]=True
                elif coord[1] == w - 1:
                    corner = 2
                    # originating from the right, so must be heading south west
                    pixel =(int(coord[0] - 0.5),int(coord[1]))

                sequence.append(pixel)
                last_added = pixel
            elif i==len(group)-1:
                # add this point
                pixel = _coords_to_pixel(coord,prev)
                if pixel!=last_added:
                    sequence.append(pixel)
                    last_added = pixel
            elif np.any(coord-prev != group[i+1]-coord):
                pixel = _coords_to_pixel(coord,prev)
                if pixel!=last_added:
                    sequence.append(pixel)
                    last_added = pixel

            # flag whether each corner has been crossed
            if i == len(group)-1:
                if corner == 0:
                    if coord[0] == 0:
                        corners[corner]=True
                elif corner == 1:
                    if coord[1] == w -1:
                        corners[corner]=True
                elif corner == 2:
                    if coord[0] == h -1:
                        corners[corner]=True
                elif corner == 3:
                    if coord[1] == 0.0:
                        corners[corner]=True

            prev=coord

        dist = _dist_from_topleft(sequence,h,w)

        sequences.append({"distance": dist , "sequence":sequence})

    # check whether we need to insert any missing corners
    if corners[0] is False:
        sequences.append({"distance": 0 , "sequence":[(0,0)]})
    if corners[1] is False:
        sequences.append({"distance": w , "sequence":[(0,w -1)]})
    if corners[2] is False:
        sequences.append({"distance": w + h , "sequence":[(h -1,w - 1)]})
    if corners[3] is False:
        sequences.append({"distance": 2*w + h , "sequence":[(h -1,0)]})


    # now, join the sequences into a single contour
    # starting at top left and rotating clockwise
    sequences.sort(key=lambda x:x.get("distance"))

    last = (-1,-1)
    for sequence in sequences:
        if sequence["sequence"][0] == last:
            pixels.pop()

        if pixels:
            pixels = [*pixels,*sequence["sequence"]]
        else:
            pixels = sequence["sequence"]

        last = pixels[-1]

    if pixels[0] == last:
        pixels.pop(0)

    if pixels[0] == (0,0):
        pixels.append(pixels.pop(0))

    pixels = np.array(pixels).astype("int32")
    pixels = np.flip(pixels)

    return pixels


def process(pred_map, return_centroids=False):
    """Post processing script for image tiles.

    Args:
        pred_map: commbined output of tp, np and hv branches, in the same order
        output_classes: number of types considered at output of nc branch
        return_centroids: whether to generate coords for each nucleus instance

    Returns:
        pred_inst:     pixel-wise nuclear instance segmentation prediction
        pred_type_out: pixel-wise nuclear type prediction

    """
    output_classes = ("type_prediction" in pred_map.keys())

    if output_classes:
        pred_type = np.squeeze(np.asarray(pred_map["type_prediction"]).astype(np.int32))

    pred_inst = np.squeeze(np.asarray(pred_map["nucleus_prediction"]))
    pred_inst = _proc_np_hv(pred_inst,np.squeeze(np.asarray(pred_map["horizontal_vertical"])))

    inst_info_dict = None
    if return_centroids or output_classes is not None:
        inst_id_list = np.unique(pred_inst)[1:]  # exlcude background
        inst_info_dict = {}
        for inst_id in inst_id_list:
            inst_map = pred_inst == inst_id
            rmin, rmax, cmin, cmax = get_bounding_box(inst_map)
            inst_bbox = np.array([[rmin, cmin], [rmax, cmax]])
            inst_map = inst_map[
                inst_bbox[0][0] : inst_bbox[1][0], inst_bbox[0][1] : inst_bbox[1][1]
            ]
            inst_map = inst_map.astype(np.uint8)
            inst_moment = moments(inst_map, order=3)

            inst_contour_cv = find_contours(inst_map, 0.5)
            inst_contour = _sp_contours_to_cv(inst_contour_cv,inst_map.shape[0],inst_map.shape[1])

            # < 3 points dont make a contour, so skip, likely artifact too
            # as the contours obtained via approximation => too small or sthg
            if inst_contour.shape[0] < 3:
                continue
            if len(inst_contour.shape) != 2:
                continue # ! check for tricky shape

            inst_centroid = [
                (inst_moment[0,1] / inst_moment[0,0]),
                (inst_moment[1,0] / inst_moment[0,0]),
            ]
            inst_centroid = np.array(inst_centroid)
            inst_contour[:, 0] += inst_bbox[0][1]  # X
            inst_contour[:, 1] += inst_bbox[0][0]  # Y
            inst_centroid[0] += inst_bbox[0][1]  # X
            inst_centroid[1] += inst_bbox[0][0]  # Y
            inst_info_dict[inst_id] = {  # inst_id should start at 1
                "bounding_box": inst_bbox,
                "centroid": inst_centroid,
                "contour": inst_contour,
                "type_probability": None,
                "type": None,
            }

    if output_classes is not None:
        for inst_id in list(inst_info_dict.keys()):
            rmin, cmin, rmax, cmax = (inst_info_dict[inst_id]["bounding_box"]).flatten()
            inst_map_crop = pred_inst[rmin:rmax, cmin:cmax]
            inst_type_crop = pred_type[rmin:rmax, cmin:cmax]
            inst_map_crop = (
                inst_map_crop == inst_id
            )
            inst_type = inst_type_crop[inst_map_crop]
            type_list, type_pixels = np.unique(inst_type, return_counts=True)
            type_list = list(zip(type_list, type_pixels))
            type_list = sorted(type_list, key=lambda x: x[1], reverse=True)
            inst_type = type_list[0][0]
            if inst_type == 0:  # ! pick the 2nd most dominant if exist
                if len(type_list) > 1:
                    inst_type = type_list[1][0]
            type_dict = {v[0]: v[1] for v in type_list}
            type_prob = type_dict[inst_type] / (np.sum(inst_map_crop) + 1.0e-6)
            inst_info_dict[inst_id]["type"] = int(inst_type)
            inst_info_dict[inst_id]["type_probability"] = float(type_prob)

    return pred_inst, inst_info_dict
