import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from monai.apps.utils import download_and_extract


def download_spleen_dataset(root_dir: str):
    """
    This function is used to download Spleen dataset for this example.
    If you'd like to download other Decathlon datasets, please check
    ``monai.apps.datasets.DecathlonDataset`` for more details.
    """
    url = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar"
    md5 = "410d4a301da4e5b2f6f86ec3ddba524e"
    task = "Task09_Spleen"
    dataset_dir = os.path.join(root_dir, task)
    tarfile_name = f"{dataset_dir}.tar"
    download_and_extract(
        url=url, filepath=tarfile_name, output_dir=root_dir, hash_val=md5
    )


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-root_dir", type=str, help="the root path to put downloaded file."
    )
    args = parser.parse_args()
    download_spleen_dataset(root_dir=args.root_dir)
