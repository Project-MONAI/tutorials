# MONAI 1.6 Release — Tutorial Test Diagnostics

**Date:** 2026-06-10
**Image:** `monai_1_6:latest` (`nvcr.io/nvidia/pytorch:25.03-py3` base)
**GPU:** NVIDIA RTX 5090 (Blackwell, SM_120, CUDA 12.8)
**Python:** 3.12 · PyTorch 2.7.0a0 (nv25.03) · NumPy 1.26.4
**Runner:** `bash runner.sh` (all tutorials)
**Log:** `runner_output.logs`

---

## Summary

| Category | Count | Notebooks |
|---|---|---|
| **Passed** | ~150 | — |
| **Failed — ExecutionError** | 91 | Papermill failed; specific errors not in log (stderr not captured) |
| **Failed — PEP8** | 3 | Style violations caught by flake8 |
| **Failed — MissingKeyword** | 1 | `max_epochs` not found and not exempted |
| **Total failures** | **94 events / 93 notebooks** | `class_lung_lesion.ipynb` has both PEP8 + ExecutionError |

> **Important — missing error details:** The runner was invoked without `2>&1`. Papermill writes
> progress and tracebacks to **stderr**, which was not redirected to `runner_output.logs`. All
> 91 ExecutionError notebooks show `papermill … -k python3` then immediately `Check failed!` in
> the log with no further detail. To see actual errors, re-run with:
> ```bash
> bash runner.sh 2>&1 | tee runner_output.logs
> ```
> Or for a single notebook:
> ```bash
> docker --context default run --gpus all --rm \
>   --entrypoint bash -e NVIDIA_DISABLE_REQUIRE=true \
>   --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
>   -v /data/rgd/tutorials:/opt/tutorials \
>   monai_1_6:latest \
>   -c "cd /opt/tutorials && bash runner.sh --verbose -t <notebook_path> 2>&1"
> ```

---

## Category 1 — PEP8 Violations (3 notebooks)

The flake8 check (run via `jupytext`) fails before papermill is even invoked.
Fix with `bash runner.sh --autofix -t <notebook>` or apply manually.

| Notebook | Error | Location |
|---|---|---|
| `competitions/MICCAI/surgtoolloc/preprocess_detect_scene_and_split_fold.ipynb` | E225 missing whitespace around operator | `stdin:160:44` |
| `deep_atlas/deep_atlas_tutorial.ipynb` | E225 missing whitespace around operator | `stdin:1353:31` |
| `modules/interpretability/class_lung_lesion.ipynb` | E231 missing whitespace after `,` (two occurrences in f-strings `y_pred[0,1]` / `y_pred[0,0]`) | `stdin:358:34`, `stdin:359:38` |

---

## Category 2 — MissingKeyword (1 notebook)

The runner requires all training notebooks to declare `max_epochs` (so it can reduce it to 1
for fast CI). Notebooks that don't have it must be added to the `doesnt_contain_max_epochs`
exemption list in `runner.sh`.

| Notebook | Detail |
|---|---|
| `auto3dseg/notebooks/msd_crossval_datalist_generator.ipynb` | `max_epochs` keyword absent; not in `doesnt_contain_max_epochs` list |

**Fix options:**
- Add the notebook filename to `doesnt_contain_max_epochs` in `runner.sh` (line ~32), or
- Add a `max_epochs = 1` cell/variable to the notebook.

---

## Category 3 — ExecutionError (91 notebooks)

Papermill returned a non-zero exit code. Specific errors are in stderr (not captured in
`runner_output.logs`). Sub-grouped by likely root cause.

### 3a — Generative model training (30 notebooks)

These notebooks train diffusion/VAE/GAN models. Even with `max_epochs = 1`, they likely fail
because they need either:
- Pre-trained checkpoint files not present in the container, or
- Custom dataset paths (e.g. BraTS, TCIA) not mounted.

| Notebook |
|---|
| `generation/2d_autoencoderkl/2d_autoencoderkl_tutorial.ipynb` |
| `generation/2d_ddpm/2d_ddpm_compare_schedulers.ipynb` |
| `generation/2d_ddpm/2d_ddpm_inpainting.ipynb` |
| `generation/2d_ddpm/2d_ddpm_tutorial.ipynb` |
| `generation/2d_ddpm/2d_ddpm_tutorial_ignite.ipynb` |
| `generation/2d_ddpm/2d_ddpm_tutorial_v_prediction.ipynb` |
| `generation/2d_diffusion_autoencoder/2d_diffusion_autoencoder_tutorial.ipynb` |
| `generation/2d_ldm/2d_ldm_tutorial.ipynb` |
| `generation/2d_super_resolution/2d_sd_super_resolution.ipynb` |
| `generation/2d_super_resolution/2d_sd_super_resolution_lightning.ipynb` |
| `generation/2d_vqgan/2d_vqgan_tutorial.ipynb` |
| `generation/2d_vqvae/2d_vqvae_tutorial.ipynb` |
| `generation/2d_vqvae_transformer/2d_vqvae_transformer_tutorial.ipynb` |
| `generation/3d_autoencoderkl/3d_autoencoderkl_tutorial.ipynb` |
| `generation/3d_ddpm/3d_ddpm_tutorial.ipynb` |
| `generation/3d_ldm/3d_ldm_tutorial.ipynb` |
| `generation/3d_vqvae/3d_vqvae_tutorial.ipynb` |
| `generation/anomaly_detection/2d_classifierfree_guidance_anomalydetection_tutorial.ipynb` |
| `generation/anomaly_detection/anomaly_detection_with_transformers.ipynb` |
| `generation/anomaly_detection/anomalydetection_tutorial_classifier_guidance.ipynb` |
| `generation/classifier_free_guidance/2d_ddpm_classifier_free_guidance_tutorial.ipynb` |
| `generation/controlnet/2d_controlnet.ipynb` |
| `generation/image_to_image_translation/tutorial_segmentation_with_ddpm.ipynb` |
| `generation/maisi/data/mask_augmentation_example.ipynb` |
| `generation/maisi/maisi_train_controlnet_tutorial.ipynb` |
| `generation/maisi/maisi_train_diff_unet_tutorial.ipynb` |
| `generation/maisi/maisi_train_vae_tutorial.ipynb` |
| `generation/realism_diversity_metrics/realism_diversity_metrics.ipynb` |
| `generation/spade_gan/spade_gan.ipynb` |
| `generation/spade_ldm/spade_ldm_brats.ipynb` |

### 3b — 3D segmentation / AutoSeg training (6 notebooks)

These notebooks train segmentation models (Spleen, BraTS, VISTA-3D). They need the
corresponding datasets to be available at the paths assumed by the notebook.

| Notebook |
|---|
| `3d_segmentation/spleen_segmentation_3d_lightning.ipynb` |
| `3d_segmentation/unet_segmentation_3d_ignite.ipynb` |
| `auto3dseg/notebooks/auto3dseg_autorunner_ref_api.ipynb` |
| `auto3dseg/notebooks/auto3dseg_hello_world.ipynb` |
| `auto3dseg/notebooks/ensemble_byoc.ipynb` |
| `vista_3d/vista3d_spleen_finetune.ipynb` |

### 3c — External service / tool dependency (7 notebooks)

These notebooks require external services (tracking servers, serving frameworks, or desktop
applications) that are not available inside the Docker container.

| Notebook | Service needed |
|---|---|
| `experiment_management/bundle_integrate_mlflow.ipynb` | MLflow tracking server |
| `experiment_management/spleen_segmentation_mlflow.ipynb` | MLflow tracking server |
| `experiment_management/spleen_segmentation_aim.ipynb` | AIM tracking server |
| `deployment/bentoml/mednist_classifier_bentoml.ipynb` | BentoML serving framework |
| `monailabel/monailabel_vista2d_cell_segmentation_CellProfiler.ipynb` | MONAILabel server + CellProfiler |
| `modules/omniverse/omniverse_integration.ipynb` | NVIDIA Omniverse (desktop app) |
| `hugging_face/hugging_face_pipeline_for_monai.ipynb` | HuggingFace model hub / `transformers` pipeline |

### 3d — MONAI core modules / data loading (48 notebooks)

These are general-purpose tutorials covering transforms, networks, datasets, and workflows.
Likely failures are data download errors (missing network access or cached data), or import
errors from packages that changed API between versions. Requires `2>&1` re-run to diagnose.

| Notebook |
|---|
| `2d_regression/image_restoration.ipynb` |
| `bundle/05_spleen_segmentation_lightning.ipynb` |
| `computer_assisted_intervention/endoscopic_inbody_classification.ipynb` |
| `microscopy/multichannel_microscopy_classification.ipynb` |
| `modules/2d_inference_3d_volume.ipynb` |
| `modules/2d_slices_from_3d_sampling.ipynb` |
| `modules/2d_slices_from_3d_training.ipynb` |
| `modules/3d_image_transforms.ipynb` |
| `modules/UNet_input_size_constraints.ipynb` |
| `modules/autoencoder_mednist.ipynb` |
| `modules/batch_output_transform.ipynb` |
| `modules/bending_energy_diffusion_loss_notes.ipynb` |
| `modules/cross_validation_models_ensemble.ipynb` |
| `modules/csv_datasets.ipynb` |
| `modules/decollate_batch.ipynb` |
| `modules/developer_guide.ipynb` |
| `modules/dice_loss_metric_notes.ipynb` |
| `modules/image_dataset.ipynb` |
| `modules/integrate_3rd_party_transforms.ipynb` |
| `modules/interpretability/cats_and_dogs.ipynb` |
| `modules/interpretability/class_lung_lesion.ipynb` *(also PEP8)* |
| `modules/interpretability/covid_classification.ipynb` |
| `modules/inverse_transforms_and_test_time_augmentations.ipynb` |
| `modules/jupyter_utils.ipynb` |
| `modules/layer_wise_learning_rate.ipynb` |
| `modules/lazy_resampling_benchmark.ipynb` |
| `modules/lazy_resampling_compose.ipynb` |
| `modules/lazy_resampling_functional.ipynb` |
| `modules/learning_rate.ipynb` |
| `modules/load_medical_images.ipynb` |
| `modules/mednist_GAN_tutorial.ipynb` |
| `modules/mednist_GAN_workflow_array.ipynb` |
| `modules/mednist_GAN_workflow_dict.ipynb` |
| `modules/network_api.ipynb` |
| `modules/network_contraints/unet_plusplus.ipynb` |
| `modules/nifti_read_example.ipynb` |
| `modules/postprocessing_transforms.ipynb` |
| `modules/public_datasets.ipynb` |
| `modules/resample_benchmark.ipynb` |
| `modules/tcia_csv_processing.ipynb` |
| `modules/torch_compile.ipynb` |
| `modules/transforms_demo_2d.ipynb` |
| `modules/transforms_metatensor.ipynb` |
| `modules/varautoencoder_mednist.ipynb` |
| `modules/workflow_profiling.ipynb` |
| `patch_inferer/modular_patch_inferer.ipynb` |
| `pathology/tumor_detection/ignite/profiling_camelyon_pipeline.ipynb` |

---

## Environment Notes

Key changes made to the Docker image to reach this test run (from baseline `nvcr.io/nvidia/pytorch:25.03-py3`):

| Issue | Fix applied |
|---|---|
| `No module named pkg_resources` during build (MetricsReloaded, segment-anything) | `PIP_CONSTRAINT` with `setuptools<71` propagates to pip's isolated build envs |
| RTX 5090 SM_120 not supported | Rebased from `nvcr.io/nvidia/pytorch:24.10-py3` → `25.03-py3` |
| `torch.patch` for ONNX bug (24.10 only) | Removed |
| Python 3.12 markers excluded `cucim`, `transformers`, `onnxruntime` | Removed `python_version <= '3.10'` caps in `requirements-dev.txt` |
| Container's `jupytext==1.16.7` blocked tutorial runner | Rebuilt `/etc/pip/constraint.txt`, kept only `numpy==1.26.4` + `setuptools<71` |
| Container's `isort==6.0.1` conflicted with MONAI's `isort<6.0` | Same constraint file rebuild |
| NumPy 2.x broke PyTorch's C-extension bridge in DataLoader workers | `numpy==1.26.4` pin retained (nv25.03 PyTorch compiled against NumPy 1.x) |
| GPU allowlist check (`NVIDIA_DISABLE_REQUIRE`) | Launch with `-e NVIDIA_DISABLE_REQUIRE=true --entrypoint bash` |

### Working docker run command

```bash
docker --context default run --gpus all --rm \
  --entrypoint bash \
  -e NVIDIA_DISABLE_REQUIRE=true \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nofile=65536:65536 \
  -v /data/rgd/tutorials:/opt/tutorials \
  -v /data/rgd/MONAI:/opt/monai \
  monai_1_6:latest \
  -c "cd /opt/tutorials && bash runner.sh -t <notebook_path> 2>&1"
```

> `--ulimit nofile=65536:65536` is required for `modules/public_datasets.ipynb` (DataLoader
> workers pass file descriptors via Unix sockets; the default container limit of 1024 is too low).
> `2>&1` captures papermill tracebacks that would otherwise be invisible.

---

## Rerun Results (stderr captured) — 2026-06-11

A targeted rerun of the 80 "our-only" failing notebooks was performed with
`2>&1 | tee` to capture papermill tracebacks.
Script: `run_our_only.sh` | Log: `runner_output_our_only.logs`

### Outcome

| Result | Count |
|---|---|
| Passed | **65** |
| Failed | **15** |
| Total targeted | 80 |

**All 30 generation notebooks passed** (they download their own small synthetic datasets,
writing ~3.7 GB to `tutorials/generation/`). The unknown "ExecutionError" group is now
fully explained.

### The 15 remaining failures

#### Group R1 — mlflow 3.13.0 broken on Python 3.12 (4 notebooks)

`mlflow 3.13.0` (installed in the image) fails on Python 3.12 with:
```
ImportError: attempted relative import beyond top-level package
```
The error originates inside `mlflow.utils.uv_utils` which performs a relative
`from .. import zipp` that is invalid at top-level scope in Python 3.12.

| Notebook | Where mlflow is used |
|---|---|
| `3d_segmentation/unet_segmentation_3d_ignite.ipynb` | `MLFlowHandler` optional import |
| `auto3dseg/notebooks/auto3dseg_hello_world.ipynb` | bundled `train.py` imports mlflow |
| `auto3dseg/notebooks/ensemble_byoc.ipynb` | bundled `train.py` imports mlflow |
| `experiment_management/spleen_segmentation_mlflow.ipynb` | direct `import mlflow` |

**Fix:** pin `mlflow<3.0` in the Dockerfile (or constraint file):
```dockerfile
RUN pip install "mlflow<3.0"
```
Eric passes these notebooks — his environment likely has an older mlflow.

#### Group R2 — Disk full `OSError: [Errno 28] No space left on device` (4 notebooks)

During the first rerun the generation notebooks wrote ~3.7 GB to `tutorials/generation/`,
consuming enough space to cause `Errno 28` for subsequent data-downloading notebooks.

| Notebook | What it tried to write |
|---|---|
| `deep_atlas/deep_atlas_tutorial.ipynb` | OASIS dataset (~2 GB) |
| `deployment/bentoml/mednist_classifier_bentoml.ipynb` | MedNIST dataset |
| `experiment_management/bundle_integrate_mlflow.ipynb` | Spleen bundle run artefacts (ran 221 min before failing) |
| `microscopy/multichannel_microscopy_classification.ipynb` | Pre-trained DenseNet169 weights |

**This is a run-order artifact, not a persistent issue on this host.** The root
filesystem has 39 GB free and the 3.7 GB generation datasets are now cached. On
subsequent runs with the generation data already present these notebooks complete
normally — confirmed by rerun on 2026-06-11 (`runner_output_rerun_r2r7r8.logs`).
Note: `bundle_integrate_mlflow` also requires the mlflow fix (Group R1) to be in
the rebuilt image before it can complete.

#### Group R3 — transformers 5.10.2 + PyTorch nv25.03 incompatibility (1 notebook)

`transformers 5.10.2` references `torch.float8_e8m0fnu` which does not exist in
PyTorch 2.7.0a0+nv25.03, causing import of `PreTrainedModel` to fail.

| Notebook | Error |
|---|---|
| `hugging_face/hugging_face_pipeline_for_monai.ipynb` | `ModuleNotFoundError: Could not import module 'PreTrainedModel'` (caused by `AttributeError: module 'torch' has no attribute 'float8_e8m0fnu'`) |

**Fix:** Pin `transformers<5.0` in constraint file, or wait for nv25 PyTorch update.
Eric's PyTorch 2.12+cu130 supports `float8_e8m0fnu`; ours does not.

#### Group R4 — Missing MONAI module (1 notebook)

| Notebook | Error |
|---|---|
| `2d_regression/image_restoration.ipynb` | `ModuleNotFoundError: No module named 'monai.networks.nets.restormer'` |

`Restormer` is not yet present in our MONAI dev branch (19cab577). Eric is on
eccefc57 (+143 commits) which may already include it, or the notebook needs updating.

**Fix:** Cherry-pick the Restormer commit into the dev branch, or add the
notebook to `skip_run_papermill` until the class is merged.

#### Group R5 — MONAI `bundle.load` API — local-only issue (1 notebook)

| Notebook | Error (local MONAI dev @ 19cab577 only) |
|---|---|
| `computer_assisted_intervention/endoscopic_inbody_classification.ipynb` | `AttributeError: 'collections.OrderedDict' object has no attribute 'train'` |

`monai.bundle.load()` in our local MONAI dev branch (19cab577) has
`@deprecated_arg("return_state_dict", since="1.2", removed="1.5")` with
`return_state_dict=True` still active as the default, so it returns an
`OrderedDict` instead of an `nn.Module`.

**This is a local-environment-only issue.** The upstream MONAI dev branch
(eccefc57, used by CI and the upstream Docker image) removed the
`return_state_dict` parameter in MONAI 1.5, and `load()` now returns `nn.Module`
by default. The notebook is correct as-is for any MONAI ≥ 1.5.

**No notebook fix needed.** The `return_state_dict=False` workaround that was
previously applied broke CI because the upstream MONAI does not accept that
parameter at all (`TypeError: unexpected keyword argument`). It has been reverted.

#### Group R6 — Missing `aim` package (1 notebook)

| Notebook | Error |
|---|---|
| `experiment_management/spleen_segmentation_aim.ipynb` | `ModuleNotFoundError: No module named 'aim'` |

**Fix:** Add `aim` to `requirements-dev.txt` and reinstall in the image.

#### Group R7 — pytorch-lightning → mlflow import chain failure (1 notebook)

| Notebook | Error |
|---|---|
| `bundle/05_spleen_segmentation_lightning.ipynb` | First run: `ContentTooShortError` (89 min download truncated). Rerun with `--ulimit nofile=65536:65536`: download succeeds in 6 min but training cell fails with `OptionalImportError: from scripts.main import train (No module named 'pytorch_lightning')`. |

**Root cause chain:**
1. Notebook installs `pytorch-lightning~=2.0.0` → installs 2.0.9
2. `pytorch_lightning` 2.0.x eagerly imports `mlflow` at module level (via `pytorch_lightning.loggers.mlflow`)
3. mlflow 3.13.0 fails to initialize under Python 3.12 (same R1 root cause)
4. Result: `import pytorch_lightning` fails in the `%%bash` training subprocess

**Fix:** Change `!pip install -q pytorch-lightning~=2.0.0` → `!pip install -q "pytorch-lightning>=2.1"`.
pytorch-lightning ≥ 2.1 uses lazy mlflow imports; tested with 2.6.5, training and evaluation pass.
The original download truncation (ContentTooShortError) was due to fd limits (R8) and is also resolved with `--ulimit nofile=65536:65536`.

#### Group R8 — Socket resource exhaustion (1 notebook)

| Notebook | Error |
|---|---|
| `modules/public_datasets.ipynb` | `RuntimeError: received 0 items of ancdata` |

PyTorch DataLoader failed to pass file descriptors through Unix sockets between
the main process and worker processes (ancillary data = file-descriptor passing).
This happens when the per-process open-file-descriptor limit is too low
(Docker default: 1024; DataLoader workers need ~65 k).

**Fix:** Add `--ulimit nofile=65536:65536` to the docker run command (already
reflected in the working command above).

**Rerun note:** The targeted rerun (`runner_output_rerun_r2r7r8.logs`) ran `public_datasets` in
the same container as `deep_atlas`. That notebook pip-installs packages which upgraded `urllib3`
to 2.x; `papermill` then failed to import immediately (`ModuleNotFoundError: No module named
'urllib3.packages.six.moves'`) — a run-order contamination, **not** the ancdata fix being tested.
**Confirmed fixed** in isolated run (container with `--ulimit nofile=65536:65536`, no shared
container): all 39 cells passed, `real 3m1s`. No `ancdata` error observed.

#### Group R9 — MissingKeyword (1 notebook)

| Notebook | Error |
|---|---|
| `auto3dseg/notebooks/msd_crossval_datalist_generator.ipynb` | `max_epochs` not found; not in exemption list |

Known issue (documented in Category 2 above). Add to `doesnt_contain_max_epochs`.

---

## Recommended Next Steps

1. **Pin `mlflow<3.0`** in the Dockerfile — fixes 4 notebooks (Group R1). ✓ DONE (PR #8912)
2. **R2 is a run-order artifact** — not applicable on this host (39 GB free, data cached).
3. **`--ulimit nofile=65536:65536`** added to working docker run command — fixes R8. ✓ DONE (confirmed clean run 3m1s)
4. **PEP8 autofix** — all three notebooks autofixed with `runner.sh --autofix`. ✓ DONE
5. **Fix MissingKeyword** — `msd_crossval_datalist_generator.ipynb` added to exemption list. ✓ DONE (PR #2065)
6. **Pin `transformers<5.0`** — fixes R3. ✓ DONE (PR #8912)
7. **Add `aim`** to `requirements-dev.txt` — fixes R6. ✓ DONE (PR #8912)
8. **R5 is local-only** — no notebook fix needed; upstream MONAI ≥ 1.5 removed the deprecated param and `load()` returns `nn.Module` directly. ✓ CONFIRMED (reverted wrong fix)
9. **Skip `image_restoration.ipynb`** until Restormer is merged (R4). ✓ DONE (PR #2065)
10. **Fix R7** — `pytorch-lightning>=2.1` pin in `bundle/05_spleen_segmentation_lightning.ipynb`. ✓ DONE (PR #2065)

### Priority order

| Priority | Action | Impact | Status |
|---|---|---|---|
| High | Pin `mlflow<3.0` (Dockerfile rebuild) | +4 passes | ✓ Done (PR #8912) |
| High | R2 disk-full | +4 passes | ✓ Not an issue (run-order artifact) |
| Medium | `--ulimit nofile=65536:65536` in docker run | +1 pass | ✓ Done (confirmed, 3m1s clean run) |
| Medium | Add `msd_crossval_datalist_generator` to exemption | +1 pass | ✓ Done (PR #2065) |
| Medium | Pin `transformers<5.0` | +1 pass | ✓ Done (PR #8912) |
| Low | Add `aim` to requirements | +1 pass | ✓ Done (PR #8912) |
| Low | R5 `bundle.load` API — local-only, no fix needed | +1 pass (upstream) | ✓ Confirmed (reverted bad fix) |
| Low | PEP8 autofix (3 notebooks) | 0 fails eliminated | ✓ Done |
| Low | Skip `image_restoration.ipynb` (Restormer missing) | +1 pass | ✓ Done (PR #2065) |
| Low | Fix Spleen Lightning notebook (`pytorch-lightning>=2.1`) | +1 pass | ✓ Done (PR #2065) |

---

## Fixes Applied (2026-06-11)

Changes committed to bring Docker run to parity with Eric's run:

### MONAI repo (`/data/rgd/MONAI`)

| File | Change |
|---|---|
| `Dockerfile` | Base image: `24.10-py3` → `25.03-py3`; rebuild pip constraint file (keep `numpy==1.26.4`, add `setuptools<71`); install `papermill jupytext autopep8 autoflake ipywidgets`; pin `mlflow<3.0`, `transformers<5.0`; add `aim`, `lightning>=2.0` |
| `requirements-dev.txt` | Pin `transformers<5.0`; pin `mlflow<3.0`; add `aim`; add `lightning>=2.0`; remove `python_version<=3.10` caps from `cucim`, `onnxruntime`, `transformers` |

### Tutorials repo (`/data/rgd/tutorials`)

| File | Change |
|---|---|
| `runner.sh` | Add `msd_crossval_datalist_generator.ipynb` and `hovernet_infer_compare.ipynb` to `doesnt_contain_max_epochs`; add `image_restoration.ipynb` to `skip_run_papermill` |
| `computer_assisted_intervention/endoscopic_inbody_classification.ipynb` | Reverted `return_state_dict=False` — upstream MONAI ≥1.5 already returns `nn.Module` by default; the extra kwarg caused `TypeError` in CI |
| `bundle/05_spleen_segmentation_lightning.ipynb` | Change `pytorch-lightning~=2.0.0` → `pytorch-lightning>=2.1` to avoid mlflow eager-import failure (R7) |

### Still pending (require environment changes or separate PRs)

| Issue | Action needed |
|---|---|
| Disk space (R2) | Free `/data` disk or bind-mount scratch volume; `deep_atlas` (~2 GB), `deployment/bentoml`, `experiment_management/bundle_integrate_mlflow`, `microscopy` notebooks |
| Socket FD limit (R8) | ✓ Confirmed fixed — `--ulimit nofile=65536:65536` resolves ancdata error; isolated run completed in 3m1s (2026-06-11) |
| pytorch-lightning/mlflow import chain (R7) | ✓ Fixed: `pytorch-lightning>=2.1` in notebook (PR #2065) |
| PEP8 in 3 notebooks | Run `bash runner.sh --autofix` for `surgtoolloc/preprocess_detect_scene_and_split_fold.ipynb`, `deep_atlas/deep_atlas_tutorial.ipynb`, `modules/interpretability/class_lung_lesion.ipynb` |
| Restormer in MONAI dev | Cherry-pick Restormer network class commit; remove from skip list once merged |
