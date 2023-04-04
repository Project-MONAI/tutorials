## Pipeline run
```bash
## [pipeline] option 1: one-click solution
python -m monai.apps.nnunet nnUNetV2Runner run --input "./input.yaml"


## [pipeline] option 2: one-click solution with dict input
DIR_BASE="/home/dongy/Projects/MONAI/nnunet/nnunet_runner/data"
DIR_RAW="${DIR_BASE}/nnUNet_raw_data_base"
DIR_PREPROCESSED="${DIR_BASE}/nnUNet_preprocessed"
DIR_RESULTS="${DIR_BASE}/nnUNet_trained_models"

python -m monai.apps.nnunet nnUNetV2Runner run --input "{'dataset_name_or_id': 996, 'nnunet_raw': '${DIR_RAW}', 'nnunet_preprocessed': '${DIR_PREPROCESSED}', 'nnunet_results': '${DIR_RESULTS}'}"
```

## Component run
```bash
## [component] convert dataset
python -m monai.apps.nnunet nnUNetV2Runner convert_dataset --input "./input.yaml"


## [component] converting msd datasets
python -m monai.apps.nnunet nnUNetV2Runner convert_msd_dataset --input "./input.yaml" --data_dir "/home/dongy/Data/MSD/NGC/Task05_Prostate"


## [component] experiment planning and data pre-processing
python -m monai.apps.nnunet nnUNetV2Runner plan_and_process --input "./input.yaml"


## [component] single-gpu training for all 20 models
python -m monai.apps.nnunet nnUNetV2Runner train --input "./input.yaml"


## [component] single-gpu training for a single model
python -m monai.apps.nnunet nnUNetV2Runner train_single_model --input "./input.yaml" \
    --config "3d_fullres" \
    --fold 0 \
    --trainer_class_name "nnUNetTrainer_5epochs" \
    --export_validation_probabilities true


## [component] multi-gpu training for all 20 models
export CUDA_VISIBLE_DEVICES=0,1 # optional
python -m monai.apps.nnunet nnUNetV2Runner train --input "./input.yaml" --num_gpus 2


## [component] multi-gpu training for a single model
export CUDA_VISIBLE_DEVICES=0,1 # optional
python -m monai.apps.nnunet nnUNetV2Runner train_single_model --input "./input.yaml" \
    --config "3d_fullres" \
    --fold 0 \
    --trainer_class_name "nnUNetTrainer_5epochs" \
    --export_validation_probabilities true \
    --num_gpus 2


## [component] find best configuration
python -m monai.apps.nnunet nnUNetV2Runner find_best_configuration --input "./input.yaml"


## [component] ensemble
python -m monai.apps.nnunet nnUNetV2Runner predict_ensemble --input "./input.yaml"
```
