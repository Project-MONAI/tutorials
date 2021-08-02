# Summary

Our solution is a simple average of efficientnet B7 and B8 featured UNets trained with different hyperparameters and initialized with different weights from imagenet pretraining which are available in the timm repository (https://github.com/rwightman/pytorch-image-models)
We give a rough indication of required pip packages and versions in `requirements.txt`.

in order to reproduce our solution it is sufficient to copy the training data to ./input and run the training script `train.py` with the following flags. The resulting pytorch model files are saved under `./ouput/` and then can be used in our inference kernel on kaggle

```
python train.py -C cfg_seg_40_1024_full -s 182732
python train.py -C cfg_seg_40_1024_full -s 822837
python train.py -C cfg_seg_40_1024_full -s 460642
python train.py -C cfg_seg_40_1024_full -s 457251

python train.py -C cfg_seg_40_1024d_full -s 657028
python train.py -C cfg_seg_40_1024d_full -s 770799
python train.py -C cfg_seg_40_1024d_full -s 825460
python train.py -C cfg_seg_40_1024d_full -s 962001

python train.py -C cfg_seg_philipp_16_ch_1024_full -s 407698
python train.py -C cfg_seg_philipp_16_ch_1024_full -s 96511

python train.py -C cfg_seg_philipp_16_ch_1024_16_full -s 86841
python train.py -C cfg_seg_philipp_16_ch_1024_16_full -s 828902

python train.py -C cfg_seg_philipp_16_ch_1024_ap_full -s 868472
python train.py -C cfg_seg_philipp_16_ch_1024_ap_full -s 183105

python train.py -C cfg_seg_philipp_16_ch_1024_nons_full -s 701922
python train.py -C cfg_seg_philipp_16_ch_1024_nons_full -s 7259
```