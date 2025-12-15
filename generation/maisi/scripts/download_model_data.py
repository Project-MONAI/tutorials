import os, subprocess, shutil
import argparse
from tqdm.auto import tqdm
from monai.apps import download_url
from pathlib import Path
from huggingface_hub import snapshot_download
from typing import List, Dict, Optional

def fetch_to_hf_path_cmd(
    items: List[Dict[str, str]],
    root_dir: str = "./",          # staging dir for CLI output
    revision: str = "main",
    overwrite: bool = False,
    token: Optional[str] = None,      # or rely on env HUGGINGFACE_HUB_TOKEN
) -> list[str]:
    """
    items: list of {"repo_id": "...", "filename": "path/in/repo.ext", "path": "local/target.ext"}
    Returns list of saved local paths (in the same order as items).
    """
    saved = []
    root = Path(root_dir)
    root.mkdir(parents=True, exist_ok=True)

    # Env for subprocess; keep Rust fast-path off to avoid notebook progress quirks
    env = os.environ.copy()
    if token:
        env["HUGGINGFACE_HUB_TOKEN"] = token
    env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")      # safer in Jupyter
    env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")   # show CLI progress in terminal

    for it in items:
        repo_id  = it["repo_id"]
        repo_file = it["filename"]
        dst = Path(it["path"])
        dst.parent.mkdir(parents=True, exist_ok=True)

        if dst.exists() and not overwrite:
            saved.append(str(dst))
            continue

        # Build command (no shell=True; no quoting issues)
        cmd = [
            "huggingface-cli", "download",
            repo_id,
            "--include", repo_file,
            "--revision", revision,
            "--local-dir", str(root),
        ]
        # Run
        subprocess.run(cmd, check=True, env=env)

        # Source path where CLI placed the file
        src = root / repo_file
        if not src.exists():
            raise FileNotFoundError(
                f"Expected downloaded file missing: {src}\n"
                f"Tip: authenticate (`huggingface-cli login` or pass token=...),"
                f" and avoid shared-IP 429s."
            )

        # Move to desired target
        if dst.exists() and overwrite:
            dst.unlink()
        if src.resolve() != dst.resolve():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
        saved.append(str(dst))

    return saved



def download_model_data(generate_version,root_dir, model_only=False):
    # TODO: remove the `files` after the files are uploaded to the NGC
    if generate_version == "ddpm-ct" or generate_version == "rflow-ct":
        files = [
            {
                "path": "models/autoencoder_v1.pt",
                "repo_id": "nvidia/NV-Generate-CT",
                "filename":"models/autoencoder_v1.pt",
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
            }]
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
    elif generate_version == "rflow-mr":
        files = [
            {
                "path": "models/autoencoder_v2.pt",
                "repo_id": "nvidia/NV-Generate-MR",
                "filename": "models/autoencoder_v2.pt",
            },
            {
                "path": "models/diff_unet_3d_rflow-mr.pt",
                "repo_id": "nvidia/NV-Generate-MR",
                "filename": "models/diff_unet_3d_rflow-mr.pt",
            }
        ]
    else:
        raise ValueError(f"generate_version has to be chosen from ['ddpm-ct', 'rflow-ct', 'rflow-mr'], yet got {generate_version}.")
    if generate_version == "ddpm-ct":
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
            }]
        if not model_only:
            files += [
                {
                    "path": "datasets/candidate_masks_flexible_size_and_spacing_3000.json",
                    "repo_id": "nvidia/NV-Generate-CT",
                    "filename": "datasets/candidate_masks_flexible_size_and_spacing_3000.json",
                },
            ]
    elif generate_version == "rflow-ct":
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
            }]
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
            path = fetch_to_hf_path_cmd([file],root_dir=root_dir, revision="main")
            print("saved to:", path)
        else:
            download_url(url=file["url"], filepath=file["path"])
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model downloading")
    parser.add_argument(
        "--version",
        type=str,
        default="rflow-ct",
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="./",
    )
    parser.add_argument("--model_only", dest="model_only", action="store_true", help="Download model only, not any dataset")

    args = parser.parse_args()
    download_model_data(args.version, args.root_dir, args.model_only)
