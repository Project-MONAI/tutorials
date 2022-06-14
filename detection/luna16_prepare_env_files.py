import os
import json
import logging
import sys

def main():
    #  ------------- Modification starts -------------
    raw_data_base_dir = "/orig_datasets/"  # the directory of the raw images
    resampled_data_base_dir = "/datasets/"  # the directory of the resampled images
    downloaded_datasplit_dir = "LUNA16_datasplit"  # the directory of downloaded data split files

    out_trained_models_dir = "trained_models"  # the directory of trained model weights
    out_tensorboard_events_dir = "tfevent_train"  # the directory of tensorboard training curves
    out_inference_result_dir = "result"  # the directory of predicted boxes for inference
    #  ------------- Modification ends ---------------

    try:
        os.mkdir(out_trained_models_dir)
    except FileExistsError:
        pass

    try:
        os.mkdir(out_tensorboard_events_dir)
    except FileExistsError:
        pass

    try:
        os.mkdir(out_inference_result_dir)
    except FileExistsError:
        pass

    # generate env json file for image resampling
    out_file = "config/environment_luna16_prepare.json"
    env_dict = {}
    env_dict["orig_data_base_dir"] = raw_data_base_dir
    env_dict["data_base_dir"] = resampled_data_base_dir
    env_dict["data_list_file_path"] = os.path.join(downloaded_datasplit_dir,"original/dataset_fold0.json")
    with open(out_file, "w") as outfile:
        json.dump(env_dict, outfile, indent=4)


    # generate env json file for training and inference
    for fold in range(10):
        out_file = "config/environment_luna16_fold"+str(fold)+".json"
        env_dict = {}
        env_dict["model_path"] = os.path.join(out_trained_models_dir,"model_luna16_fold"+str(fold)+".pt")
        env_dict["data_base_dir"] = resampled_data_base_dir
        env_dict["data_list_file_path"] = os.path.join(downloaded_datasplit_dir,"dataset_fold"+str(fold)+".json")
        env_dict["tfevent_path"] = os.path.join(out_tensorboard_events_dir,"luna16_fold"+str(fold))
        env_dict["result_list_file_path"] = os.path.join(out_inference_result_dir,"result_luna16_fold"+str(fold)+".json")
        with open(out_file, "w") as outfile:
            json.dump(env_dict, outfile, indent=4)


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
