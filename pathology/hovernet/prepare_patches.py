import os
import math
import tqdm
import glob
import shutil
import pathlib

import numpy as np
import scipy.io as sio
from PIL import Image
from argparse import ArgumentParser


def load_img(path):
    return np.array(Image.open(path).convert("RGB"))


def load_ann(path):
    # assumes that ann is HxW
    ann_inst = sio.loadmat(path)["inst_map"]
    ann_type = sio.loadmat(path)["type_map"]

    # merge classes for CoNSeP (utilise 3 nuclei classes and background keep the same with paper)
    ann_type[(ann_type == 3) | (ann_type == 4)] = 3
    ann_type[(ann_type == 5) | (ann_type == 6) | (ann_type == 7)] = 4

    ann = np.dstack([ann_inst, ann_type])
    ann = ann.astype("int32")

    return ann


class PatchExtractor():
    """Extractor to generate patches with or without padding.
    Turn on debug mode to see how it is done.

    Args:
        x         : input image, should be of shape HWC
        patch_size  : a tuple of (h, w)
        step_size : a tuple of (h, w)
    Return:
        a list of sub patches, each patch has dtype same as x

    Examples:
        >>> xtractor = PatchExtractor((450, 450), (120, 120))
        >>> img = np.full([1200, 1200, 3], 255, np.uint8)
        >>> patches = xtractor.extract(img, 'mirror')

    """

    def __init__(self, patch_size, step_size):
        self.patch_type = "mirror"
        self.patch_size = patch_size
        self.step_size = step_size

    def __get_patch(self, x, ptx):
        pty = (ptx[0] + self.patch_size[0], ptx[1] + self.patch_size[1])
        win = x[ptx[0] : pty[0], ptx[1] : pty[1]]
        assert (
            win.shape[0] == self.patch_size[0] and win.shape[1] == self.patch_size[1]
        ), "[BUG] Incorrect Patch Size {0}".format(win.shape)
        return win

    def __extract_valid(self, x):
        """Extracted patches without padding, only work in case patch_size > step_size.

        Note: to deal with the remaining portions which are at the boundary a.k.a
        those which do not fit when slide left->right, top->bottom), we flip
        the sliding direction then extract 1 patch starting from right / bottom edge.
        There will be 1 additional patch extracted at the bottom-right corner.

        Args:
            x         : input image, should be of shape HWC
            patch_size  : a tuple of (h, w)
            step_size : a tuple of (h, w)
        Return:
            a list of sub patches, each patch is same dtype as x

        """
        im_h = x.shape[0]
        im_w = x.shape[1]

        def extract_infos(length, patch_size, step_size):
            flag = (length - patch_size) % step_size != 0
            last_step = math.floor((length - patch_size) / step_size)
            last_step = (last_step + 1) * step_size
            return flag, last_step

        h_flag, h_last = extract_infos(im_h, self.patch_size[0], self.step_size[0])
        w_flag, w_last = extract_infos(im_w, self.patch_size[1], self.step_size[1])

        sub_patches = []
        # Deal with valid block
        for row in range(0, h_last, self.step_size[0]):
            for col in range(0, w_last, self.step_size[1]):
                win = self.__get_patch(x, (row, col))
                sub_patches.append(win)
        # Deal with edge case
        if h_flag:
            row = im_h - self.patch_size[0]
            for col in range(0, w_last, self.step_size[1]):
                win = self.__get_patch(x, (row, col))
                sub_patches.append(win)
        if w_flag:
            col = im_w - self.patch_size[1]
            for row in range(0, h_last, self.step_size[0]):
                win = self.__get_patch(x, (row, col))
                sub_patches.append(win)
        if h_flag and w_flag:
            ptx = (im_h - self.patch_size[0], im_w - self.patch_size[1])
            win = self.__get_patch(x, ptx)
            sub_patches.append(win)
        return sub_patches

    def __extract_mirror(self, x):
        """Extracted patches with mirror padding the boundary such that the
        central region of each patch is always within the orginal (non-padded)
        image while all patches' central region cover the whole orginal image.

        Args:
            x         : input image, should be of shape HWC
            patch_size  : a tuple of (h, w)
            step_size : a tuple of (h, w)
        Return:
            a list of sub patches, each patch is same dtype as x

        """
        diff_h = self.patch_size[0] - self.step_size[0]
        padt = diff_h // 2
        padb = diff_h - padt

        diff_w = self.patch_size[1] - self.step_size[1]
        padl = diff_w // 2
        padr = diff_w - padl

        pad_type = "reflect"
        x = np.lib.pad(x, ((padt, padb), (padl, padr), (0, 0)), pad_type)
        sub_patches = self.__extract_valid(x)
        return sub_patches

    def extract(self, x, patch_type):
        patch_type = patch_type.lower()
        self.patch_type = patch_type
        if patch_type == "valid":
            return self.__extract_valid(x)
        elif patch_type == "mirror":
            return self.__extract_mirror(x)
        else:
            assert False, "Unknown Patch Type [%s]" % patch_type
        return


def main(cfg):
    xtractor = PatchExtractor(cfg["patch_size"], cfg["step_size"])

    for phase in ["Train", "Test"]:
        img_dir = os.path.join(cfg["root"], f"{phase}/Images")
        ann_dir = os.path.join(cfg["root"], f"{phase}/Labels")

        file_list = glob.glob(os.path.join(ann_dir, f"*mat"))
        file_list.sort()  # ensure same ordering across platform

        out_dir = f"{cfg['root']}/Prepared/{phase}"
        if os.path.isdir(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir)

        pbar_format = "Process File: |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]"
        pbarx = tqdm.tqdm(
            total=len(file_list), bar_format=pbar_format, ascii=True, position=0
        )

        for file_path in file_list:
            base_name = pathlib.Path(file_path).stem

            img = load_img(f"{img_dir}/{base_name}.png")
            ann = load_ann(f"{ann_dir}/{base_name}.mat")

            # *
            img = np.concatenate([img, ann], axis=-1)
            sub_patches = xtractor.extract(img, cfg["extract_type"])

            pbar_format = "Extracting  : |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]"
            pbar = tqdm.tqdm(
                total=len(sub_patches),
                leave=False,
                bar_format=pbar_format,
                ascii=True,
                position=1,
            )

            for idx, patch in enumerate(sub_patches):
                image_patch = patch[..., :3].transpose(2, 0, 1)  # make channel first
                inst_map_patch = patch[..., 3][None]  # add channel dim
                type_map_patch = patch[..., 4][None]  # add channel dim
                np.save("{0}/{1}_{2:03d}_image.npy".format(out_dir, base_name, idx), image_patch)
                np.save("{0}/{1}_{2:03d}_inst_map.npy".format(out_dir, base_name, idx), inst_map_patch)
                np.save("{0}/{1}_{2:03d}_type_map.npy".format(out_dir, base_name, idx), type_map_patch)
                pbar.update()
            pbar.close()
            # *

            pbarx.update()
        pbarx.close()


def parse_arguments():
    parser = ArgumentParser(description="Extract patches from the original images")

    parser.add_argument(
        "--root",
        type=str,
        default="/home/yunliu/Workspace/Data/CoNSeP",
        help="root path to image folder containing training/test",
    )
    parser.add_argument("--type", type=str, default="mirror", dest="extract_type", help="Choose 'mirror' or 'valid'")
    parser.add_argument("--ps", nargs='+', type=int, default=[540, 540], dest="patch_size", help="patch size")
    parser.add_argument("--ss", nargs='+', type=int, default=[164, 164], dest="step_size", help="patch size")
    args = parser.parse_args()
    config_dict = vars(args)

    return config_dict


if __name__ == "__main__":
    cfg = parse_arguments()

    main(cfg)
