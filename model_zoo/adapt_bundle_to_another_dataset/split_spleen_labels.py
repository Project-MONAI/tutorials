import os

import nibabel as nib
import numpy as np

root_dir = "./data"

src_label = os.path.join(root_dir, "RawData/Training/label")

dst_Spleen = os.path.join(root_dir, "btcv_spleen/labelsTr")

os.makedirs(os.path.join(root_dir, "btcv_spleen"), exist_ok=True)
os.makedirs(dst_Spleen, exist_ok=True)

for path, subdirs, files in os.walk(src_label):
    for file in files:
        label_file = os.path.join(src_label, file)
        label = nib.load(label_file)
        label = nib.funcs.as_closest_canonical(label, enforce_diag=False)
        affine = label.affine
        label = np.array(label.dataobj)
        label = label.astype(np.float32)
        label[label == 5] = 0
        label[label == 6] = 5
        label[label == 11] = 6
        label[label > 6] = 0
        print("file: ", file)
        # 1-spleen, 2-right kidney, 3-left kidney, 4-Gallbladder, 5-liver, 6-Pancreas
        print("label-0: ", (label == 0).sum())
        print("label-1: ", (label == 1).sum())
        print("label-2: ", (label == 2).sum())
        print("label-3: ", (label == 3).sum())
        print("label-4: ", (label == 4).sum())
        print("label-5: ", (label == 5).sum())
        print("label-6: ", (label == 6).sum())
        print("label-7: ", (label == 7).sum())
        print("label-8: ", (label == 8).sum())
        print("label-9: ", (label == 9).sum())

        mask = label.copy()
        mask[mask != 1] = 0
        if (mask == 1).sum() != 0:
            nib.save(
                nib.Nifti1Image(mask, affine),
                os.path.join(dst_Spleen, file),
            )
