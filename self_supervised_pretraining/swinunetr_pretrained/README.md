# Pre-trained SwinUNETR Backbone on ~50,000 3D Volumes

This tutorial is an example of how to use the latest set of SwinUNETR pre-trained weights. The pre-training was done using multiple public datasets with over ~58,000 3D CT and MR volumes.

Benchmark on BTCV with using the pre-trained weights as compared to random initialization shows promising gains in terms of accuracy/performance.

### Steps to run the tutorial
1. Download the dataset [BTCV](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789) (More detail is provided in the Data section)
2. Download the pretrained SwinUNETR weights on the 50,000 3D volumes from the following link. More information about the dataset and training is provided below
3. Modify the paths for pretrained_checkpoint, data_root, json_path & logdir in ssl_train.ipynb
4. Run the 'swinunetr_finetune.ipynb'

### 1. Data
Pre-training Dataset: Datasets from multiple sources has been used. Below is a list of all the different datasets that were utilized for training the SwinUNETR network architecture:

| Dataset Name            | Source          | Body Region            |
|-------------------------|-----------------|------------------------|
| Aomic-1000              | Open Neuro      |                        |
| QTIM                    | Open Neuro      |                        |
| Brats 2021              | Brats Challenge |                        |
| LUNA 16                 | LUNA Challenge  |                        |
| NLST                    | TCIA            |                        |
| Duke Breast Cancer MR   | TCIA            |                        |
| Acrin Breast MR         | TCIA            |                        |
| Prostate MR             | TCIA            |                        |
| Stony Brook Chest CT    | TCIA            |                        |
| Upenn GBM               | TCIA            |                        |
| LIDC                    | TCIA            |                        |
| Covid-19                | TCIA            |                        |
| Colonography Trials     | TCIA            |                        |
| Head and Neck Carcinoma | TCIA            |                        |


Fine-tuning Dataset: The dataset from Beyond the Cranial Vault Challenge
[(BTCV)](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789)
2015 hosted at MICCAI, was used as a fully supervised fine-tuning task on the pre-trained weights. The dataset
consists of 30 3D Volumes with annotated labels of up to 13 different organs [2]. There are 3 JSON files provided in the
json_files directory for the dataset. They correspond to having a different number of training volumes ranging from
6, 12 and 24. All 3 JSON files have the same validation split.

### 2. Pretrained SwinUNETR Weights

The entire SwinUNETR model including encoder and decoder was trained end-to-end using self-supervised learning techniques as outlined in [1]. 

### 3. Finetuning SwinUNETR with BTCV Experiment Hyper-parameters

Training Hyper-parameters for Fine-tuning BTCV task : \
Number of Steps: 30000 \
Validation Frequency: 100 steps \
Batch Size: 1 3D Volume (4 samples are drawn per 3D volume) \
Learning Rate: 1e-4 \
Loss Function: DiceCELoss

### 5. Results of the Fine-tuning vs Random Initialization on BTCV

| Training Volumes      | Validation Volumes | Random Init Dice score | Pre-trained Dice Score | Relative Performance Improvement |
| ----------------      | ----------------   |------------------------|------------------------| ----------------        |
| 6      | 6 | ~                      | 71.50                  | ~ |
| 12      | 6 | ~                      | ~                      | ~ |
| 24      | 6 | ~                      | ~                      | ~ |

### Tutorial Usage Citation

If you found the tutorial to be helpful in your work please support us by citing the below reference:

1.) Tang, Yucheng, Dong Yang, Wenqi Li, Holger R. Roth, Bennett Landman, Daguang Xu, Vishwesh Nath, and Ali Hatamizadeh. "Self-supervised pre-training of swin transformers for 3d medical image analysis." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 20730-20740. 2022.

Bibtex: `@inproceedings{tang2022self,
  title={Self-supervised pre-training of swin transformers for 3d medical image analysis},
  author={Tang, Yucheng and Yang, Dong and Li, Wenqi and Roth, Holger R and Landman, Bennett and Xu, Daguang and Nath, Vishwesh and Hatamizadeh, Ali},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={20730--20740},
  year={2022}
}
`

2.) Tang, Yucheng, et al. "High-resolution 3D abdominal segmentation with random patch network fusion."
Medical Image Analysis 69 (2021): 101894.