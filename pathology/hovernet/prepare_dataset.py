import os
import glob
import logging
import argparse
import numpy as np

def prepara_dataset(root_path, phase):
    data_dir = os.path.join(root_path, phase)

    files = sorted(
        glob.glob(os.path.join(data_dir, "*/*.npy")))

    logging.info(f'Train total data {len(files)}')
    for file in files:
        data = np.load(file)
        np.save(file.replace('.npy', '_image.npy'), data[..., :3].transpose(2, 0, 1))
        np.save(file.replace('.npy', '_inst_map.npy'), data[..., 3][None])
        np.save(file.replace('.npy', '_type_map.npy'), data[..., 4][None])

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset_root', default='/workspace/Data/CoNSeP/Prepared/consep')

    args = parser.parse_args()
    for phase in ["train", "valid"]:
        logging.info(f'Processing {phase} dataset...')
        prepara_dataset(args["dataset_root"], phase)

if __name__ == "__main__":
    main()
