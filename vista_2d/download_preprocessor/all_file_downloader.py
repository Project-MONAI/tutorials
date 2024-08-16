import argparse
import os

import requests
from tqdm import tqdm


def download_files(url_dict, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

    for key, url in url_dict.items():
        if key == "nips_train.zip" or key == "nips_test.zip":
            if not os.path.exists(os.path.join(directory, "nips_dataset")):
                os.mkdir(os.path.join(directory, "nips_dataset"))
            base_dir = os.path.join(directory, "nips_dataset")
        elif key == "deepbacs.zip":
            if not os.path.exists(os.path.join(directory, "deepbacs_dataset")):
                os.mkdir(os.path.join(directory, "deepbacs_dataset"))
            base_dir = os.path.join(directory, "deepbacs_dataset")
        elif key == "livecell":
            if not os.path.exists(os.path.join(directory, "livecell_dataset")):
                os.mkdir(os.path.join(directory, "livecell_dataset"))
            base_dir = os.path.join(directory, "livecell_dataset")
            print(f"Downloading from {key}: {url}")
            os.system(url + base_dir)
            continue

        try:
            print(f"Downloading from {key}: {url}")
            response = requests.get(url, stream=True, allow_redirects=True)
            total_size = int(response.headers.get("content-length", 0))

            # Extract the filename from the URL or use the key as the filename
            filename = os.path.basename(key)
            file_path = os.path.join(base_dir, filename)

            # Write the content to a file in the specified directory with progress
            with open(file_path, "wb") as file, tqdm(
                desc=filename,
                total=total_size,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(chunk_size=1024):
                    size = file.write(data)
                    bar.update(size)

            print(f"Saved to {file_path}")
        except Exception as e:
            print(f"Failed to download from {key} ({url}). Reason: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--dir", type=str, help="Directory to download files to", default="/set/the/path")

    args = parser.parse_args()
    directory = os.path.normpath(args.dir)

    url_dict = {
        "deepbacs.zip": "https://zenodo.org/records/5551009/files/DeepBacs_Data_Segmentation_StarDist_MIXED_dataset.zip?download=1",
        "nips_test.zip": "https://zenodo.org/records/10719375/files/Testing.zip?download=1",
        "nips_train.zip": "https://zenodo.org/records/10719375/files/Training-labeled.zip?download=1",
        "livecell": "wget --recursive --no-parent --cut-dirs=0 --timestamping -i urls.txt --directory-prefix="
        # Add URLs with keys here
    }
    download_files(url_dict, directory)


if __name__ == "__main__":
    main()
