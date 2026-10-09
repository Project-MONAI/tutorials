# MONAI Tutorials — Notebook Execution Test Log

**Goal:** Verify every notebook in each tutorials folder executes successfully under the
project virtual environment with **MONAI 1.6.1**. Where a notebook fails, make the
*minimal* changes needed to run it (max 5 attempts), and log every change here. If it
still fails after 5 attempts, it is logged as a failure with the final error.

## Environment

| Item | Value |
|------|-------|
| Python venv | `/home/ubuntu/tools/MONAI-test/tutorials/.venv` (Python 3.13) |
| MONAI | 1.6.1 |
| PyTorch | 2.10.0+cu128 |
| GPU | NVIDIA A10G, 23 GB (CUDA available) |
| papermill | 2.7.0 |
| Jupyter kernel | `monai-venv` (registered to the venv interpreter) |
| Data cache | `MONAI_DATA_DIRECTORY=.nbtest/data` |

## Methodology

- Each notebook is executed with **papermill** from its own directory.
- Before execution, long-running loop variables are reduced to 1 (matching the project's
  `runner.sh`): `max_epochs`, `val_interval`, `disc_train_interval`, `disc_train_steps`,
  `num_batches_for_histogram`. This exercises the full code path quickly.
- Harness: `.nbtest/run_nb.py` (does not modify the original notebook; runs a reduced copy).
- A per-notebook wall-clock **timeout** is applied (default 1800 s).
- **Skipped** notebooks are those the project's `runner.sh` lists under `skip_run_papermill`
  — they require external services (3D Slicer, CVAT, QuPath, OHIF), special hardware,
  network/data not available in CI, or are known-blocked upstream. These are not failures.
- Each run uses up to **5 attempts**. Fixes applied between attempts are recorded.

## Status legend

- ✅ **PASS** — executed end-to-end without error.
- 🔧 **PASS (fixed)** — passed after minimal changes (changes listed).
- ⏭️ **SKIP** — on the project skip list / needs unavailable external resource.
- ❌ **FAIL** — still failing after up to 5 attempts (final error listed).

---

## Results by folder

<!-- Results are appended below as each folder is processed. -->

### 2d_classification

| Notebook | Status | Time | Notes |
|----------|--------|------|-------|
| monai_101.ipynb | ✅ PASS | 110s | |
| monai_201.ipynb | ✅ PASS | 262s | |
| mednist_tutorial.ipynb | ✅ PASS | 119s | |

### 2d_registration

| Notebook | Status | Time | Notes |
|----------|--------|------|-------|
| registration_mednist.ipynb | ✅ PASS | 31s | |

### 2d_regression

| Notebook | Status | Time | Notes |
|----------|--------|------|-------|
| image_restoration.ipynb | ⏭️ SKIP | — | On project skip list: `monai.networks.nets.restormer` not yet in released MONAI. |

### 3d_classification

| Notebook | Status | Time | Notes |
|----------|--------|------|-------|
| densenet_training_array.ipynb | ✅ PASS | 250s | |

### 3d_regression

| Notebook | Status | Time | Notes |
|----------|--------|------|-------|
| densenet_training_array.ipynb | ✅ PASS | 36s | |

### patch_inferer

| Notebook | Status | Time | Notes |
|----------|--------|------|-------|
| modular_patch_inferer.ipynb | 🔧 PASS (fixed) | 129s | **Fix:** zarr 3.4.0 was installed, but MONAI 1.6.1's `ZarrAvgMerger` uses the zarr v2 API (`chunks=True` default and `cdata_shape`), which zarr 3.x rejects (`ValueError: True is not a valid chunk input`). Pinned **`zarr<3`** (installed 2.18.7) and updated the notebook's install cell to enforce `zarr<3`. |

### reconstruction

| Notebook | Status | Time | Notes |
|----------|--------|------|-------|
| MRI_reconstruction/unet_demo/inference.ipynb | ⏭️ SKIP | — | On project skip list (`MRI_reconstruction`): requires trained model weights / fastMRI data not available in CI. |
| MRI_reconstruction/varnet_demo/inference.ipynb | ⏭️ SKIP | — | On project skip list (`MRI_reconstruction`). |

### 3d_registration

| Notebook | Status | Time | Notes |
|----------|--------|------|-------|
| learn2reg_nlst_paired_lung_ct.ipynb | ⏭️ SKIP | — | On project skip list: slow test / large Learn2Reg dataset. |
| learn2reg_oasis_unpaired_brain_mr.ipynb | ⏭️ SKIP | — | On project skip list. |
| paired_lung_ct.ipynb | ✅ PASS | 191s | |

### hugging_face

| Notebook | Status | Time | Notes |
|----------|--------|------|-------|
| hugging_face_pipeline_for_monai.ipynb | ✅ PASS | 101s | |
| finetune_vista3d_for_hugging_face_pipeline.ipynb | ⏭️ SKIP | — | On project skip list: requires VISTA3D weights / large data. |

### acceleration

| Notebook | Status | Time | Notes |
|----------|--------|------|-------|
| TensorRT_inference_acceleration.ipynb | ⏭️ SKIP | — | On project skip list: requires TensorRT engine build. |
| automatic_mixed_precision.ipynb | 🔧 PASS (fixed) | 258s | **Fix:** process was OOM-killed (host has 15 GiB RAM, no swap) while building the second `CacheDataset(cache_rate=1.0)` at `num_workers=8`. Confirmed via kernel oom-killer log (anon-rss ~13 GB). Reduced the two CacheDataset `num_workers` to 2 to cap peak RAM during caching. Attempts: 1 fail (OOM), 2 fail (OOM), 3 pass. |
| dataset_type_performance.ipynb | ✅ PASS | 393s | |
| fast_training_tutorial.ipynb | 🔧 PASS (fixed) | 144s | **Fix:** missing dependency `nvtx` (`No module named 'nvtx'`). Installed `nvtx` into the venv and added `!python -c "import nvtx" \|\| pip install -q nvtx` to the notebook's install cell. Attempts: 1 fail, 2 pass. |
| threadbuffer_performance.ipynb | ✅ PASS | 66s | |
| transform_speed.ipynb | ✅ PASS | 301s | |

### bundle

| Notebook | Status | Time | Notes |
|----------|--------|------|-------|
| 01_bundle_intro.ipynb | 🔧 PASS (fixed) | 93s | **Fix:** `python -m monai.bundle init_bundle` failed with `OptionalImportError: import fire`. Installed `fire` into the venv and added `!python -c "import fire" \|\| pip install -q fire` to the install cell. |
| 02_mednist_classification.ipynb | 🔧 PASS (fixed) | 144s | **Fixes:** (1) added `fire` install (same as above). (2) **Environment:** the notebook's `%%bash` cells call bare `python -m monai.bundle run`, which resolved to a *base conda MONAI 1.4.0* on PATH instead of the venv's 1.6.1 — causing `RuntimeError: Failed to instantiate SupervisedTrainer`. Fixed at the harness level by prepending the venv `bin` to `PATH` so shell cells use the kernel interpreter. See "Environment note" below. Attempts: 1–3 fail, 4 pass. |
| 03_mednist_classification_v2.ipynb | 🔧 PASS (fixed) | 190s | Same two fixes as 02 (fire + PATH to venv for `%%bash` cells). |
| 04_integrating_code.ipynb | 🔧 PASS (fixed) | 147s | **Fix:** added `fire` install line. |
| 05_spleen_segmentation_lightning.ipynb | ⏭️ SKIP | — | On project skip list: hardcoded `.to("cuda")` with no CPU fallback / GPU-only. |
| pythonic_usage_guidance/pythonic_bundle_access.ipynb | 🔧 PASS (fixed) | 191s | **Fix:** OOM (15 GiB host) while the downloaded `spleen_ct_segmentation` bundle built `CacheDataset(cache_rate=1.0)` for the full dataset. Added `train#dataset#cache_rate=0.0` and `validate#dataset#cache_rate=0.0` overrides to the four `create_workflow(...)` calls. Attempts: 1 fail (OOM), 2 pass. |

> **Environment note (applies to all `%%bash` / `!python` shell cells):** Many notebooks shell
> out with bare `python`/`pip`. Those subshells inherit `$PATH`, where a base conda env with
> **MONAI 1.4.0** was ahead of the project venv. To run these notebooks correctly the venv must be
> the first Python on `PATH` (i.e. activate the venv, or prepend `.venv/bin`). The test harness does
> this automatically (`run_nb.py` prepends the venv `bin` to `PATH`). This is an environment setup
> requirement, not a defect in the notebooks.

### deepgrow

| Notebook | Status | Time | Notes |
|----------|--------|------|-------|
| ignite/inference.ipynb | ✅ PASS | 29s | |
| ignite/inference_3d.ipynb | ✅ PASS | 63s | |

### deepedit

| Notebook | Status | Time | Notes |
|----------|--------|------|-------|
| ignite/infoANDinference.ipynb | ✅ PASS | 36s | |

### deep_atlas

| Notebook | Status | Time | Notes |
|----------|--------|------|-------|
| deep_atlas_tutorial.ipynb | ✅ PASS | 169s | |

### computer_assisted_intervention

| Notebook | Status | Time | Notes |
|----------|--------|------|-------|
| endoscopic_inbody_classification.ipynb | ✅ PASS | 36s | |
| video_seg.ipynb | ⏭️ SKIP | — | On project skip list. |

### multimodal

| Notebook | Status | Time | Notes |
|----------|--------|------|-------|
| nakaseke_multimodal_early_fusion/multimodal_early_fusion_tutorial.ipynb | ✅ PASS | 28s | |
| openi_multilabel_classification_transchex/transchex_openi_multilabel_classification.ipynb | ⏭️ SKIP | — | On project skip list (`transchex_openi`). |

### active_learning

| Notebook | Status | Time | Notes |
|----------|--------|------|-------|
| liver_tumor_al/results_uncertainty_analysis.ipynb | ⏭️ SKIP | — | On project skip list (`active_learning`). |
| tool_tracking_al/results_uncertainty_analysis.ipynb | ⏭️ SKIP | — | On project skip list. |

### self_supervised_pretraining

| Notebook | Status | Time | Notes |
|----------|--------|------|-------|
| swinunetr_pretrained/swinunetr_finetune.ipynb | ⏭️ SKIP | — | On project skip list (`swinunetr_finetune`). |
| vit_unetr_ssl/ssl_finetune.ipynb | ⏭️ SKIP | — | On project skip list (`ssl_finetune`). |
| vit_unetr_ssl/ssl_train.ipynb | ⏭️ SKIP | — | On project skip list (`ssl_train`). |

### experiment_management

| Notebook | Status | Time | Notes |
|----------|--------|------|-------|
| bundle_integrate_mlflow.ipynb | 🔧 PASS (fixed) | 608s | **Fix:** MONAI 1.6.1's `MLFlowHandler` rejects filesystem (file-store) tracking URIs and requires a `sqlite:///` or remote URI. Changed the two explicit file-store URIs (`./eval/mlruns`) to `sqlite:///./eval/mlruns.db` — in notebook cell with `--tracking_uri`, in the `MLFlowHandler(tracking_uri=...)` cell, and in `experiment_management/mlflow_example.json` (`tracking_uri` now `sqlite:///...mlruns.db`). Attempts: 1 fail, 2 pass. |
| spleen_segmentation_aim.ipynb | ❌ FAIL | — | **Blocked (environment):** requires the `aim` package, which cannot be installed on **Python 3.13**. `aim==3.17.5` (pinned by the notebook) and the latest `aim==3.29.1` both fail to build — their build dep `aimrocks==0.5.*` has no py3.13 wheel/sdist (only 0.2.0 exists) and they pin unavailable Cython alphas. Not fixable via notebook edits; needs an older Python or an `aim` release with py3.13 support. Attempts: 5 (various install strategies: pinned version, latest, `--only-binary`). |
| spleen_segmentation_mlflow.ipynb | ✅ PASS | 93s | |
| unet_segmentation_3d_ignite_clearml.ipynb | ⏭️ SKIP | — | On project skip list (needs ClearML account). |

### 3d_segmentation

| Notebook | Status | Time | Notes |
|----------|--------|------|-------|
| brats_segmentation_3d.ipynb | ⏳ RUNS, exceeds time budget | >20min | Dependency fixed (installed `onnxruntime`; the notebook's install cell already handles it). Already uses `cache_rate=0.0` so no OOM. Executes correctly but trains 1 epoch over the full BraTS set (~388 3D volumes) plus ONNX/TensorRT inference, exceeding the 20-min per-notebook budget. Final generous run in progress (30-min cap); result appended below. BraTS data (7.1 GB) cached under `.nbtest/data`. |
| spleen_segmentation_3d.ipynb | ✅ PASS | 179s | |
| spleen_segmentation_3d_lightning.ipynb | 🔧 PASS (fixed) | 149s | **Fixes (3):** (1) installed `pytorch-lightning` (notebook install cell already requests it). (2) OOM on 15 GiB host → reduced the two `CacheDataset` to `cache_rate=0.1`, `num_workers=2`. (3) `RecursionError` in `rich.style` (Lightning's rich progress bar recurses under the papermill/ZMQ display backend) → added `enable_progress_bar=False` to the `Trainer`. Attempts: 1 fail (dep), 2 fail (OOM), 3 fail (recursion), 4 pass. |
| spleen_segmentation_3d_visualization_basic.ipynb | ⏭️ SKIP | — | On project skip list. |
| swin_unetr_brats21_segmentation_3d.ipynb | ⏭️ SKIP | — | On project skip list (matches `unetr_`). |
| swin_unetr_btcv_segmentation_3d.ipynb | ⏭️ SKIP | — | On project skip list (matches `unetr_`). |
| unet_segmentation_3d_ignite.ipynb | 🔧 PASS (fixed) | 65s | **Fix:** MONAI 1.6.1 `MLFlowHandler` rejects file-store URIs. Changed the two `MLFlowHandler(tracking_uri=Path(mlflow_dir).as_uri())` to `sqlite:///<mlflow_dir>/mlruns.db`. Attempts: 1 fail (mlflow), 2 pass. |
| unetr_btcv_segmentation_3d.ipynb | ⏭️ SKIP | — | On project skip list (`unetr_`). |
| unetr_btcv_segmentation_3d_lightning.ipynb | ⏭️ SKIP | — | On project skip list (`unetr_`). |

### deployment

| Notebook | Status | Time | Notes |
|----------|--------|------|-------|
| bentoml/mednist_classifier_bentoml.ipynb | ❌ FAIL | — | **Blocked (environment):** notebook pins `bentoml==0.13.1` (2021-era). It cannot work on **Python 3.13** — the pinned version force-downgrades `urllib3`/`sqlalchemy` and then fails to import (`No module named 'urllib3.packages.six.moves'`), and the notebook's entire API (`bentoml.frameworks.pytorch`, `bentoml.adapters`, `@bentoml.env`) was removed in bentoml 1.x. Installing it also *corrupted the shared venv* (broke mlflow); I uninstalled it and restored `urllib3>=2`, `sqlalchemy>=1.4`. Not fixable without rewriting the notebook for bentoml 1.x. Attempts: 5. |
| ray/mednist_classifier_ray.ipynb | ⏭️ SKIP | — | On project skip list (tutorials issue #1307). |

### microscopy

| Notebook | Status | Time | Notes |
|----------|--------|------|-------|
| multichannel_microscopy_classification.ipynb | ⏳ (re-run pending) | — | First attempt was aborted by the harness-timeout bug (not a notebook failure). Re-running after the harness fix. |

### Harness fix (mid-run)

> The per-notebook timeout originally used `subprocess.run(timeout=...)`, which raised but left the
> papermill process and its Jupyter kernel running (orphaned). Slow/blocking notebooks therefore ran
> far past their cap, piling up zombie kernels that contended for GPU/RAM. Fixed `run_nb.py` to launch
> papermill in its own process group (`start_new_session=True`) and `killpg(SIGKILL)` the whole tree on
> timeout. Verified: a 120 s cap now returns at ~125 s wall-clock with zero leftover processes.

---

## Summary of progress so far

As of this commit, the following folders have been fully processed:

`2d_classification`, `2d_registration`, `2d_regression`, `3d_classification`,
`3d_regression`, `3d_registration`, `patch_inferer`, `reconstruction`, `hugging_face`,
`acceleration`, `bundle`, `deepgrow`, `deepedit`, `deep_atlas`,
`computer_assisted_intervention`, `multimodal`, `active_learning`,
`self_supervised_pretraining`, `experiment_management`, `3d_segmentation`, `deployment`.

Running totals across processed folders:

| Outcome | Count |
|---------|-------|
| ✅ / 🔧 PASS (executed successfully) | 30 |
| &nbsp;&nbsp;&nbsp;of which required a fix (🔧) | 11 |
| ⏭️ SKIP (project `skip_run_papermill` list) | 22 |
| ❌ FAIL (environment-blocked, not fixable by notebook edits) | 2 |
| ⏳ RUNS but exceeds the 20-min per-notebook time budget | 1 |

### Dependencies added to the venv to make notebooks run

| Package | Reason |
|---------|--------|
| `zarr<3` (2.18.7) | MONAI 1.6.1 `ZarrAvgMerger` uses the zarr v2 API; zarr 3.x rejects it. |
| `nvtx` | Imported by `acceleration/fast_training_tutorial`. |
| `fire` | Required by the `monai.bundle` CLI (`init_bundle`, `run`). |
| `pytorch-lightning` (2.6.6) | Required by the Lightning-based notebooks. |
| `onnxruntime` (1.30.0) | Imported by `3d_segmentation/brats_segmentation_3d`. |

> `mlflow` is kept at its original 3.16.1 — the MLflow failures were due to MONAI 1.6.1's
> `MLFlowHandler` rejecting file-store URIs, fixed in-notebook with `sqlite:///` URIs.

### Environment-blocked failures (not fixable by editing the notebook)

- `experiment_management/spleen_segmentation_aim.ipynb` — `aim` has no Python 3.13 support.
- `deployment/bentoml/mednist_classifier_bentoml.ipynb` — pins `bentoml==0.13.1` (2021), incompatible
  with Python 3.13 and the modern bentoml 1.x API; installing it corrupts the shared env.

> **Recommendation for both:** they are good candidates to add to `skip_run_papermill` in
> `runner.sh` for the current Python 3.13 / MONAI 1.6.1 environment (or to re-run on an older
> Python). `mednist_classifier_ray` is already skipped for a similar reason.

---

## Remaining work (not yet executed)

The following folders still need to be run and logged. Counts are runnable notebooks
(before applying the skip list):

| Folder | Notebooks | Notes |
|--------|-----------|-------|
| `modules` | 51 | Largest folder; mix of CPU demos and GPU training notebooks. |
| `generation` | 31 | GAN / diffusion / MAISI; several are large and GPU-heavy. |
| `auto3dseg` | 9 | AutoML segmentation pipelines; some long-running. |
| `pathology` | 6 | Several already on the skip list (hovernet, nuclei, nuclick). |
| `monailabel` | 10 | Nearly all on the skip list (need 3D Slicer / CVAT / QuPath / OHIF). |
| `competitions` | 3 | All three (`preprocess_*`) are on the skip list. |
| `microscopy` | 1 | `multichannel_microscopy_classification` — in progress. |
| `vista_3d` | 1 | `vista3d_spleen_finetune`. |
| `vista_2d` | 1 | `vista_2d_tutorial_monai` — on the skip list. |
| `model_zoo` | 1 | `TCIA_PROSTATEx...` — on the skip list. |
| `federated_learning` | 1 | On the skip list (needs OpenFL cluster). |
| `full_gpu_inference_pipeline` | 1 | On the skip list (Triton client/server). |

Folders with **no notebooks** (Python-script tutorials, out of scope for notebook testing):
`detection`, `nnunet`, `automl`, `performance_profiling`.

### Notes / caveats carried forward

- **15 GiB RAM ceiling (no swap):** several training notebooks OOM when a `CacheDataset`
  caches a full 3D dataset at `cache_rate=1.0`. The minimal fix applied in each case is to lower
  `cache_rate`/`num_workers`. Expect the same in `generation`, `auto3dseg`, and parts of `modules`.
- **Shell cells use the venv:** notebooks that shell out with bare `python`/`pip` need the venv to
  be first on `PATH` (the harness enforces this). Without it they pick up a base-conda MONAI 1.4.0.
- **Time budget:** per the agreed default, notebooks are capped at ~20 min; anything correct but
  slower is logged as "RUNS, exceeds time budget" rather than a failure.
