## Tutorial: VISTA-2D Model Creation

This tutorial will guide the users to setting up all the datasets, running pre-processing, creation of organized json file lists which can be provided to VISTA-2D training pipeline.
Some dataset need to be manually downloaded, others will be downloaded by a provided script. Please do not manually unzip any of the downloaded files, it will be automatically handled in the final step.

### List of Datasets
1.) [Cellpose](https://www.cellpose.org/dataset)

2.) [TissueNet](https://datasets.deepcell.org/login)

3.) [Kaggle Nuclei Segmentation](https://www.kaggle.com/c/data-science-bowl-2018/data)

4.) [Omnipose - OSF repository](https://osf.io/xmury/)

5.) [NIPS Cell Segmentation Challenge](https://neurips22-cellseg.grand-challenge.org/)

6.) [LiveCell](https://sartorius-research.github.io/LIVECell/)

7.) [Deepbacs](https://github.com/HenriquesLab/DeepBacs/wiki/Segmentation)

Datasets 1-4 need to be manually downloaded, instructions to download them have been provided below.

### Manual Dataset Download Instructions
#### 1.) Cellpose:
The dataset can be downloaded from this [link](https://www.cellpose.org/dataset). Please see below screenshots to assist in downloading it
![cellpose_agreement.png](cellpose_agreement.png)
Please enter your email and accept terms and conditions to download the dataset.

![cellpose_links.png](cellpose_links.png)
Click on train.zip and test.zip to download both directories independently. They both need to be placed in a `cellpose_dataset` directory. The `cellpose_dataset` will have to be created by the user in the root data directory.

#### 2.) TissueNet
Login credentials have to be created at below provided link. Please see below screenshots for further assistance.

![tissuenet_login.png](tissuenet_login.png)
Please create an account at the provided [link](https://datasets.deepcell.org/login).

![tissuenet_download.png](tissuenet_download.png)
After logging in, the above page will be visible, please make sure that version 1.0 is selected for TissueNet before clicking on download button.
All the downloaded files need to be placed in a `tissuenet_dataset` directory, this directory has to be created by the user.

#### 3.) Kaggle Nuclei Segmentation
Kaggle credentials are required in order to access this dataset at this [link](https://www.kaggle.com/c/data-science-bowl-2018/data), the user will have to register for the challenge to access and download the dataset.
Please refer below screenshots for additional help.

![kaggle_download.png](kaggle_download.png)
The `Download All` button needs to be used so all files are downloaded, the files need to be placed in a directory created by the user `kaggle_dataset`.

#### 4.) Omnipose
The Omnipose dataset is hosted on an [OSF repository](https://osf.io/xmury/) and the dataset part needs to be downloaded from it. Please refer below screenshots for further assistance.

![omnipose_download.png](omnipose_download.png)
The `datasets` directory needs to be selected as highlighted in the screenshot, then `download as zip` needs to be pressed for downloading the dataset. The user will have to place all the files in
a user created directory named `omnipose_dataset`.

### The remaining datasets will be downloaded by a python script.
To run the script use the following example command `python all_file_downloader.py --download_path provide_the_same_root_data_path`

After completion of downloading of all datasets, below is how the data root directory should look:

![data_tree.png](data_tree.png)

### Process the downloaded data
To execute VISTA-2D training pipeline, some datasets require label conversion. Please use the `root_data_path` as the input to the script, example command to execute the script is given below:

`python generate_json.py --data_root provide_the_same_root_data_path`

### Generation of Json data lists (Optional)
If one desires to generate JSON files from scratch, `generate_json.py` script performs both processing and creation of JSON files.
To execute VISTA-2D training pipeline, some datasets require label conversion and then a json file list which the VISTA-2D training uses a format.
Creating the json lists from the raw dataset sources, please use the `root_data_path` as the input to the script, example command to execute the script is given below:

`python generate_json.py --data_root provide_the_same_root_data_path`
