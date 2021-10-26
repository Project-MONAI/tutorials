# Self-Supervised Pretraining Tutorial

Authored by Vishwesh Nath (a.k.a Vish)

This directory contains two scripts. The first script 'ssl_script_train.py' generates 
a good set of pre-trained weights using unlabeled data with self-supervised tasks that 
are based on augmentations of different types. The second script 'ssl_finetune_train.py' uses
the pre-trained weights generated from the first script and finetunes on a fully supervised task.

### 1.Data
Pretraining Dataset: The TCIA Covid-19 dataset was used for generating the pretrained weights.
The dataset contains a total of 771 3D CT Volumes. The volumes were split into training and validation sets
of 600 and 171 3D volumes correspondingly. PUT LINKS & CITATIONS HERE

Finetuning Dataset: The dataset from Beyond the Cranial Vault Challenge 2015 hosted at MICCAI was used as a
fully supervised Finetuning task on the pre-trained weights. The dataset contains of 30 3D Volumes with annotated labels
for 13 different organs. PUT LINKS & CITATIONS HERE

### 2. Network Architectures

Describe the ViT AE and the UNETR

### 3. Self-supervised Tasks

Describe the data augmentation flow and show a figure with the ground truth

### 4. Experiment Hyper-parameters

Describe all the hyper-parameters that were used for the SSL tasks the training 
hyper-parameters and the fine-tuning hyper-parameters etc

### 4. Show the Training & Validation Curves for pretraining SSL

Just a plot

### 5. Show the results of the Finetuning vs Random Initialization

Plot/Table works either way
