# Preprocessing Open-I Dataset

The Open-I dataset provides a collection of 3,996 radiology reports
with 8,121 associated images in PA, AP and lateral views. In this tutorial, we utilize the images from fronal view with their corresponding reports for training and
evaluation of the TransChex model. The chest x-ray images and reports are originally from the Indiana University hospital (see the licencing information below).
The 14 finding categories in this work include Atelectasis, Cardiomegaly, Consolidation, Edema, Enlarged-Cardiomediastinum, Fracture, Lung-Lesion, Lung-Opacity, No-Finding, Pleural-Effusion, Pleural-Other, Pneumonia, Pneumothorax and Support-Devices. More information can be found in the following link:
https://openi.nlm.nih.gov/faq

License: Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)

In this section, we provide the steps that are needed for preprocessing the Open-I dataset for
the multi-label disease classification tutorial using TransCheX model. As a result, once the following steps are
completed, the dataset can be readily used for the tutorial.

### Preprocessing Steps
1) Create a new folder named 'monai_data' for downloading the raw data and preprocessing.
2) Download the chest X-ray images in PNG format from this [link](https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz). Copy the downloaded file (NLMCXR_png.tgz) to 'monai_data' directory and extract it to 'monai_data/dataset_orig/NLMCXR_png/'.
3) Download the reports in XML format from this [link](https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz). Copy the downloaded file (NLMCXR_reports.tgz) to 'monai_data' directory and extract it to 'monai_data/dataset_orig/NLMCXR_reports/'.
4) Download the splits of train, validation and test datasets from this [link](https://drive.google.com/u/1/uc?id=1jvT0jVl9mgtWy4cS7LYbF43bQE4mrXAY&export=download). Copy the downloaded file (TransChex_openi.zip)
to 'monai_data' directory and extract it here.
5) Run 'preprocess_openi.py' to process the images and reports.
