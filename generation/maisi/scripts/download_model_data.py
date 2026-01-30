# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os, subprocess, shutil
import argparse
from tqdm.auto import tqdm
from monai.apps import download_url
from pathlib import Path
from typing import List, Dict, Optional
from huggingface_hub import hf_hub_download


def fetch_to_hf_path_cmd(
    items: List[Dict[str, str]],
    root_dir: str = "./",  # (kept for signature compatibility; not required)
    revision: str = "main",
    overwrite: bool = False,
    token: Optional[str] = None,  # or rely on env HF_TOKEN / HUGGINGFACE_HUB_TOKEN
) -> list[str]:
    """
    items: list of {"repo_id": "...", "filename": "path/in/repo.ext", "path": "local/target.ext"}
    Returns list of saved local paths (in the same order as items).

    Pure Python implementation (CI-safe): no `huggingface-cli` dependency.
    """
    saved = []

    for it in items:
        repo_id = it["repo_id"]
        repo_file = it["filename"]
        dst = Path(it["path"])
        dst.parent.mkdir(parents=True, exist_ok=True)

        if dst.exists() and not overwrite:
            saved.append(str(dst))
            continue

        # Download into HF cache, then copy to requested destination
        cached_path = hf_hub_download(
            repo_id=repo_id,
            filename=repo_file,
            revision=revision,
            token=token,  # if None, huggingface_hub will use env / cached auth if present
        )

        # Copy/move into place
        if dst.exists() and overwrite:
            dst.unlink()

        shutil.copy2(cached_path, dst)
        saved.append(str(dst))

    return saved


def download_model_data(generate_version, root_dir, model_only=False):
    # TODO: remove the `files` after the files are uploaded to the NGC
    if generate_version == "maisi3d-ddpm" or generate_version == "maisi3d-rflow":
        files = [
            {
                "path": "models/autoencoder_v1.pt",
                "repo_id": "nvidia/NV-Generate-CT",
                "filename": "models/autoencoder_v1.pt",
            },
            {
                "path": "models/mask_generation_autoencoder.pt",
                "repo_id": "nvidia/NV-Generate-CT",
                "filename": "models/mask_generation_autoencoder.pt",
            },
            {
                "path": "models/mask_generation_diffusion_unet.pt",
                "repo_id": "nvidia/NV-Generate-CT",
                "filename": "models/mask_generation_diffusion_unet.pt",
            },
        ]
        if not model_only:
            files += [
                {
                    "path": "datasets/all_anatomy_size_conditions.json",
                    "repo_id": "nvidia/NV-Generate-CT",
                    "filename": "datasets/all_anatomy_size_conditions.json",
                },
                {
                    "path": "datasets/all_masks_flexible_size_and_spacing_4000.zip",
                    "repo_id": "nvidia/NV-Generate-CT",
                    "filename": "datasets/all_masks_flexible_size_and_spacing_4000.zip",
                },
            ]
    else:
        raise ValueError(
            f"generate_version has to be chosen from ['maisi3d-ddpm', 'maisi3d-rflow'], yet got {generate_version}."
        )
    if generate_version == "maisi3d-ddpm":
        files += [
            {
                "path": "models/diff_unet_3d_ddpm-ct.pt",
                "repo_id": "nvidia/NV-Generate-CT",
                "filename": "models/diff_unet_3d_ddpm-ct.pt",
            },
            {
                "path": "models/controlnet_3d_ddpm-ct.pt",
                "repo_id": "nvidia/NV-Generate-CT",
                "filename": "models/controlnet_3d_ddpm-ct.pt",
            },
        ]
        if not model_only:
            files += [
                {
                    "path": "datasets/candidate_masks_flexible_size_and_spacing_3000.json",
                    "repo_id": "nvidia/NV-Generate-CT",
                    "filename": "datasets/candidate_masks_flexible_size_and_spacing_3000.json",
                },
            ]
    elif generate_version == "maisi3d-rflow":
        files += [
            {
                "path": "models/diff_unet_3d_rflow-ct.pt",
                "repo_id": "nvidia/NV-Generate-CT",
                "filename": "models/diff_unet_3d_rflow-ct.pt",
            },
            {
                "path": "models/controlnet_3d_rflow-ct.pt",
                "repo_id": "nvidia/NV-Generate-CT",
                "filename": "models/controlnet_3d_rflow-ct.pt",
            },
        ]
        if not model_only:
            files += [
                {
                    "path": "datasets/candidate_masks_flexible_size_and_spacing_4000.json",
                    "repo_id": "nvidia/NV-Generate-CT",
                    "filename": "datasets/candidate_masks_flexible_size_and_spacing_4000.json",
                },
            ]

    for file in files:
        file["path"] = file["path"] if "datasets/" not in file["path"] else os.path.join(root_dir, file["path"])
        if "repo_id" in file.keys():
            path = fetch_to_hf_path_cmd([file], root_dir=root_dir, revision="main")
            print("saved to:", path)
        else:
            download_url(url=file["url"], filepath=file["path"])
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model downloading")
    parser.add_argument(
        "--version",
        type=str,
        default="maisi3d-rflow",
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="./",
    )
    parser.add_argument(
        "--model_only", dest="model_only", action="store_true", help="Download model only, not any dataset"
    )

    args = parser.parse_args()
    download_model_data(args.version, args.root_dir, args.model_only)
