# Changes on Branch

**Branch:** `vikash/updated_monai_1_6_release`
**Base:** `origin/main`
**Total Commits:** 45 (41 previously documented + 4 added below)

> Note: earlier commits reference the predecessor branch name
> `vikash/fix/monai_1_6_release_notebooks`; the active branch is now
> `vikash/updated_monai_1_6_release`.

---

## Commits

### (working tree — not yet committed) — MONAI 1.6.0rc1 tutorial fixes + skip
- **Author:** (pending)
- **Date:** 2026-09-21
- **Files changed:**
  - `3d_segmentation/unet_segmentation_3d_ignite.ipynb`
  - `experiment_management/bundle_integrate_mlflow.ipynb`
  - `runner.sh`
  - `VERIFIED_CHANGES.md` (new — source-of-truth for verified fixes)
- **Why:** Results of a full folder-by-folder run against MONAI 1.6.0rc1
  (source build `1.6.0rc1-4-gb89a8af1`, NGC PyTorch base, Python 3.12, A10G GPU).
  See `VERIFIED_CHANGES.md` for the authoritative per-notebook summary.

  **Fixes applied and verified green (single-notebook re-run):**
  - `unet_segmentation_3d_ignite.ipynb` — a prior commit (`1bf0bf9`) imported
    `path_to_sqlite_uri` from `monai.utils`, which does **not exist** in
    1.6.0rc1 (`ImportError`). Removed the import and replaced the call with an
    inline `"sqlite:///" + os.path.join(log_dir, "mlruns.db")`.
  - `bundle_integrate_mlflow.ipynb` — same `path_to_sqlite_uri` regression from
    `1bf0bf9`; replaced with `"sqlite:///" + os.path.abspath("./eval/mlruns.db")`.

  **Skip added to `runner.sh`:**
  - `microscopy/multichannel_microscopy_classification.ipynb` — kernel dies
    mid-run (DeadKernelError ~cell 17-21, likely OOM); not code-fixable here.

  **Corrections to earlier (stale-log) assumptions — NOT skipped after all:**
  - `bundle/04_integrating_code.ipynb` — **passes** in the live run (was wrongly
    proposed for skip based on old logs). No skip applied.
  - `unet_segmentation_3d_ignite.ipynb` — the real failure was the
    `path_to_sqlite_uri` ImportError above, **not** a rich/ClearML RecursionError
    as previously assumed. Fixed rather than skipped.

  **Other failures left unskipped (documented, not code-fixable in this pass):**
  - `deployment/bentoml/mednist_classifier_bentoml.ipynb` — `bentoml==0.13.1`
    uninstallable on Python 3.12; needs a BentoML 1.x rewrite.
  - `bundle/pythonic_usage_guidance/pythonic_bundle_access.ipynb` — DeadKernel/OOM.
  - `vista_3d/vista3d_spleen_finetune.ipynb` — DeadKernel/OOM on 24 GB GPU.
  - `modules/transforms_metatensor.ipynb` — real bug: empty `applied_operations`
    on `DivisiblePadd.inverse` (see VERIFIED_CHANGES.md "Known Issue").

  Note: numerous untracked files in the working tree (datasets, `*_work_dir/`,
  `mlruns/`, `output/`, `predictions.csv`, etc.) are run artifacts and should be
  git-ignored, not committed.

---

### 8ee04e1 — Fix zarr v3 API usage, a dead logo URL and a stale batchgenerators pin
- **Author:** Vikash Gupta
- **Date:** 2026-09-20
- **Files changed:**
  - `modules/integrate_3rd_party_transforms.ipynb`
  - `modules/resample_benchmark.ipynb`
  - `patch_inferer/modular_patch_inferer.ipynb`
- **Why:** Zarr 3.x removed the `compressor=` argument for array creation
  (`TypeError: compressor is not available for Zarr format 3 arrays`), so the
  patch-inferer notebook was updated to the Zarr v3 API. Also refreshed a stale
  `batchgenerators` version pin (fixes `ModuleNotFoundError: No module named
  'batchgenerators'`) and replaced a dead logo URL.

---

### 14724e3 — auto3dseg: use algo_to_json instead of the disabled algo_to_pickle
- **Author:** Vikash Gupta
- **Date:** 2026-09-20
- **Files changed:**
  - `auto3dseg/notebooks/auto3dseg_autorunner_ref_api.ipynb`
  - `auto3dseg/notebooks/hpo_nni.ipynb`
- **Why:** MONAI 1.6 disabled `algo_to_pickle`; switched the Auto3DSeg notebooks
  to `algo_to_json`. (Auto3DSeg is otherwise out of scope for this review pass.)

---

### 1bf0bf9 — Use SQLite MLflow tracking URIs instead of the filesystem backend
- **Author:** Vikash Gupta
- **Date:** 2026-09-20
- **Files changed:**
  - `3d_segmentation/unet_segmentation_3d_ignite.ipynb`
  - `experiment_management/bundle_integrate_mlflow.ipynb`
  - `experiment_management/mlflow_example.json`
- **Why:** The filesystem MLflow backend is unreliable under the newer MLflow in
  the 1.6 container; switched tracking URIs to SQLite so runs persist correctly.

---

### df98606 — Pass hash_type="md5" explicitly to download_and_extract
- **Author:** Vikash Gupta
- **Date:** 2026-09-20
- **Files changed:** 44 files across `2d_classification/`, `3d_classification/`,
  `3d_regression/`, `3d_segmentation/`, `acceleration/`, `bundle/`,
  `computer_assisted_intervention/`, `deep_atlas/`, `deployment/`,
  `experiment_management/`, `full_gpu_inference_pipeline/`, `generation/maisi/`,
  `hugging_face/`, `microscopy/`, `modules/` (incl. `public_datasets.ipynb`),
  `performance_profiling/`, and `vista_3d/` (see `git show df98606 --stat`).
- **Why:** MONAI 1.6 changed `download_and_extract`/`download_url` default hashing
  behaviour; passing `hash_type="md5"` explicitly restores checksum validation and
  fixes download failures (including the `public_datasets.ipynb` HTTP 404/hash path).

---

### 14e1ef8 — fix: replace dead monai.io URL with GitHub raw URL in load_medical_images
- **Author:** Vikash Gupta
- **Date:** 2026-06-24
- **Files changed:**
  - `modules/load_medical_images.ipynb`
- **Why:** The monai.io domain no longer resolves. Replaced the download URL for MONAI-logo_color.png with the equivalent file hosted on raw.githubusercontent.com to restore notebook functionality.

---

### d38d101 — fix: remove non-existent hovernet_infer_compare from doesnt_contain_max_epochs; fix stale pending table
- **Author:** R. Garcia-Dias
- **Date:** 2026-06-11
- **Files changed:**
  - `diagnose_1_6_release.md`
  - `runner.sh`
- **Why:** Cleaned up references to a notebook that no longer exists and updated the diagnostics table to reflect current status.

---

### 3d198d8 — fix: skip deep_atlas_tutorial in CPU CI
- **Author:** R. Garcia-Dias
- **Date:** 2026-06-11
- **Files changed:**
  - `runner.sh`
- **Why:** The deep_atlas_tutorial requires GPU resources and fails in CPU-only CI environments; skip it to prevent false failures.

---

### c62cc06 — fix: skip 05_spleen_segmentation_lightning in CPU CI
- **Author:** R. Garcia-Dias
- **Date:** 2026-06-11
- **Files changed:**
  - `runner.sh`
- **Why:** The lightning segmentation notebook requires GPU and should be skipped in CPU CI to avoid unnecessary failures.

---

### a581217 — style: add trailing newline to endoscopic_inbody_classification.ipynb
- **Author:** R. Garcia-Dias
- **Date:** 2026-06-11
- **Files changed:**
  - `computer_assisted_intervention/endoscopic_inbody_classification.ipynb`
- **Why:** Fix file formatting to comply with pre-commit hooks that require trailing newlines.

---

### f3f49ff — fix: revert return_state_dict change; update R5/R8 diagnostics
- **Author:** R. Garcia-Dias
- **Date:** 2026-06-11
- **Files changed:**
  - `computer_assisted_intervention/endoscopic_inbody_classification.ipynb`
  - `diagnose_1_6_release.md`
- **Why:** Reverted an incompatible API change and updated release diagnostics tracking for requirements R5 and R8.

---

### 252b9d6 — Fix bundle/05_spleen_segmentation_lightning: upgrade pytorch-lightning pin
- **Author:** R. Garcia-Dias
- **Date:** 2026-06-11
- **Files changed:**
  - `bundle/05_spleen_segmentation_lightning.ipynb`
  - `diagnose_1_6_release.md`
- **Why:** The previous pytorch-lightning version pin was incompatible with MONAI 1.6; upgraded to a compatible version.

---

### a862ed1 — fix: apply PEP8 autofix to 3 notebooks (E225/E231 whitespace violations)
- **Author:** R. Garcia-Dias
- **Date:** 2026-06-11
- **Files changed:**
  - `competitions/MICCAI/surgtoolloc/preprocess_detect_scene_and_split_fold.ipynb`
  - `deep_atlas/deep_atlas_tutorial.ipynb`
  - `modules/interpretability/class_lung_lesion.ipynb`
- **Why:** Automated PEP8 whitespace fixes to pass linting checks in CI.

---

### 7877af8 — [pre-commit.ci] auto fixes from pre-commit.com hooks
- **Author:** pre-commit-ci[bot]
- **Date:** 2026-06-11
- **Files changed:**
  - `computer_assisted_intervention/endoscopic_inbody_classification.ipynb`
- **Why:** Automated formatting corrections applied by pre-commit CI hooks.

---

### d6e252e — fix: update runner.sh and endoscopic notebook for MONAI 1.6 compatibility
- **Author:** R. Garcia-Dias
- **Date:** 2026-06-11
- **Files changed:**
  - `computer_assisted_intervention/endoscopic_inbody_classification.ipynb`
  - `runner.sh`
- **Why:** Updated test runner and notebook code to be compatible with MONAI 1.6 API changes.

---

### 7bedfd0 — enh: add notebook demonstrating access to data from Imaging Data Commons (#2063)
- **Author:** Andrey Fedorov
- **Date:** 2026-05-05
- **Files changed:**
  - `README.md`
  - `modules/idc_dataset.ipynb`
  - `runner.sh`
- **Why:** Added a new tutorial notebook showing how to access medical imaging data from the Imaging Data Commons platform.

---

### 5c908aa — update monai nvflare example links (#2062)
- **Author:** Holger Roth
- **Date:** 2026-04-14
- **Files changed:**
  - `federated_learning/nvflare/README.md`
- **Why:** Updated broken or outdated links to MONAI NVFlare examples in the documentation.

---

### 60cf9ac — add opencv-python to requirements.txt (#2061)
- **Author:** Zijian
- **Date:** 2026-04-12
- **Files changed:**
  - `detection/requirements.txt`
- **Why:** Added missing opencv-python dependency required by detection tutorials.

---

### 9292800 — Updating Workflows to Fix Missing `pkg_resources` (#2057)
- **Author:** Eric Kerfoot
- **Date:** 2026-02-14
- **Files changed:**
  - `.github/workflows/copyright.yml`
  - `.github/workflows/guidelines.yml`
  - `.github/workflows/pep8.yml`
  - `.github/workflows/test-modified.yml`
- **Why:** Fixed CI workflows that broke due to the removal of pkg_resources from newer Python/setuptools versions.

---

### 17ef259 — Update MAISI model URL (#2051)
- **Author:** Can Zhao
- **Date:** 2026-01-29
- **Files changed:**
  - `.github/workflows/test-modified.yml`
  - `deployment/fastapi_inference/app/__init__.py`
  - `deployment/fastapi_inference/app/inference.py`
  - `deployment/fastapi_inference/app/main.py`
  - `deployment/fastapi_inference/app/model_loader.py`
  - `deployment/fastapi_inference/app/schemas.py`
  - `deployment/fastapi_inference/examples/client.py`
  - `deployment/fastapi_inference/tests/__init__.py`
  - `deployment/fastapi_inference/tests/test_api.py`
  - `generation/maisi/README.md`
  - `generation/maisi/configs/environment_maisi3d-ddpm.json`
  - `generation/maisi/configs/environment_maisi3d-rflow.json`
  - `generation/maisi/configs/environment_maisi_controlnet_train.json`
  - `generation/maisi/configs/environment_maisi_diff_model.json`
  - `generation/maisi/maisi_inference_tutorial.ipynb`
  - `generation/maisi/maisi_train_controlnet_tutorial.ipynb`
  - `generation/maisi/maisi_train_diff_unet_tutorial.ipynb`
  - `generation/maisi/scripts/download_model_data.py`
  - `generation/maisi/scripts/inference.py`
  - `generation/maisi/scripts/utils.py`
  - `runner.sh`
- **Why:** Updated the MAISI model download URL to a new hosting location and related configuration files.

---

### d6da454 — Add FastAPI deployment tutorial for MONAI models (#2050)
- **Author:** Mohamed Salah
- **Date:** 2025-12-04
- **Files changed:**
  - `deployment/fastapi_inference/README.md`
  - `deployment/fastapi_inference/app/__init__.py`
  - `deployment/fastapi_inference/app/inference.py`
  - `deployment/fastapi_inference/app/main.py`
  - `deployment/fastapi_inference/app/model_loader.py`
  - `deployment/fastapi_inference/app/schemas.py`
  - `deployment/fastapi_inference/docker/Dockerfile`
  - `deployment/fastapi_inference/docker/docker-compose.yml`
  - `deployment/fastapi_inference/examples/client.py`
  - `deployment/fastapi_inference/examples/sample_requests.http`
  - `deployment/fastapi_inference/requirements.txt`
  - `deployment/fastapi_inference/tests/__init__.py`
  - `deployment/fastapi_inference/tests/test_api.py`
- **Why:** Added a complete FastAPI-based deployment tutorial showing how to serve MONAI models as REST APIs with Docker support.

---

### 97f1075 — Fix tcia_utils issues loading metadata df to support latest v3.2.1 release return values (#2046)
- **Author:** Justin Kirby
- **Date:** 2025-11-05
- **Files changed:**
  - `model_zoo/TCIA_PROSTATEx_Prostate_MRI_Anatomy_Model.ipynb`
- **Why:** Updated metadata loading to handle the new SeriesInstanceUID return values in tcia_utils v3.2.1.

---

### 3cacee2 — Updating TCIA MRI Anatomy Model Notebook (#2037)
- **Author:** Eric Kerfoot
- **Date:** 2025-10-30
- **Files changed:**
  - `model_zoo/TCIA_PROSTATEx_Prostate_MRI_Anatomy_Model.ipynb`
- **Why:** General updates to the TCIA MRI Anatomy Model notebook for compatibility and correctness.

---

### 521433e — [pre-commit.ci] pre-commit suggestions (#2040)
- **Author:** pre-commit-ci[bot]
- **Date:** 2025-10-29
- **Files changed:**
  - `.pre-commit-config.yaml`
- **Why:** Automated pre-commit hook version updates suggested by pre-commit CI.

---

### 832f723 — Update README.md for MAISI (#2041)
- **Author:** Can Zhao
- **Date:** 2025-10-29
- **Files changed:**
  - `generation/maisi/README.md`
- **Why:** Updated the MAISI tutorial README with corrected information and instructions.

---

### c4bff94 — Fix no space left on device in build workflow (#2045)
- **Author:** YunLiu
- **Date:** 2025-10-29
- **Files changed:**
  - `.github/workflows/test-modified.yml`
  - `.pre-commit-config.yaml`
- **Why:** Resolved disk space issues in CI by optimizing the build workflow and updating pre-commit config.

---

### d96190e — docs: fix typo "UNet_meatdata" → "UNet_metadata" (#2034)
- **Author:** Minsu Kim
- **Date:** 2025-09-29
- **Files changed:**
  - `README.md`
- **Why:** Corrected a typo in the README (meatdata → metadata).

---

### 1fcee23 — 2015 improve explanation of datalist format (#2019)
- **Author:** Daniël Nobbe
- **Date:** 2025-09-26
- **Files changed:**
  - `auto3dseg/README.md`
  - `auto3dseg/docs/run_with_minimal_input.md`
  - `auto3dseg/notebooks/auto_runner.ipynb`
  - `auto3dseg/notebooks/msd_crossval_datalist_generator.ipynb`
  - `auto3dseg/notebooks/msd_datalist_generator.ipynb`
- **Why:** Improved documentation explaining the expected datalist format for Auto3DSeg workflows.

---

### 9948f26 — Fixed bug in KL_loss calculation for VAE validation step during training (#2000)
- **Author:** Muhammad Nabi Yasinzai
- **Date:** 2025-09-26
- **Files changed:**
  - `generation/maisi/maisi_train_vae_tutorial.ipynb`
- **Why:** Corrected the KL divergence loss calculation in the VAE validation step which was producing incorrect results.

---

### 77ccd31 — Fixed Typo (#2016)
- **Author:** Eric Kerfoot
- **Date:** 2025-09-22
- **Files changed:**
  - `3d_segmentation/spleen_segmentation_3d_visualization_basic.ipynb`
- **Why:** Fixed a typo in the spleen segmentation visualization notebook.

---

### 7b032f1 — docs: add Google Colab setup and troubleshooting section (#2025)
- **Author:** Minsu Kim
- **Date:** 2025-09-22
- **Files changed:**
  - `README.md`
- **Why:** Added documentation for running tutorials in Google Colab with setup instructions and common troubleshooting tips.

---

### 713c6b2 — Skip tcia notebook test for 1.5.1 release (#2033)
- **Author:** YunLiu
- **Date:** 2025-09-22
- **Files changed:**
  - `runner.sh`
- **Why:** Temporarily skipped the TCIA notebook test which was incompatible with the 1.5.1 release.

---

### b448cf2 — Updating workflow to use Github CPU runner (#2032)
- **Author:** Eric Kerfoot
- **Date:** 2025-09-22
- **Files changed:**
  - `.github/workflows/test-modified.yml`
  - `2d_classification/monai_101.ipynb`
  - `requirements.txt`
  - `runner.sh`
- **Why:** Migrated CI workflow to use GitHub-hosted CPU runners and updated dependencies accordingly.

---

### b5cb801 — update maisi readme (#2018)
- **Author:** Can Zhao
- **Date:** 2025-09-22
- **Files changed:**
  - `generation/maisi/README.md`
- **Why:** Updated MAISI readme with current instructions and model information.

---

### 8b90a16 — Fix typo and missing description of content in folder (#2004)
- **Author:** Mingxin Zheng
- **Date:** 2025-06-24
- **Files changed:**
  - `3d_regression/README.md`
  - `README.md`
  - `acceleration/distributed_training/distributed_training.md`
  - `nnunet/README.md`
  - `pathology/tumor_detection/README.MD`
  - `vista_2d/README.md`
  - `vista_3d/README.md`
- **Why:** Fixed typos and added missing folder content descriptions across multiple README files.

---

### ef0ac7d — Fix missspelled words (#2003)
- **Author:** Mingxin Zheng
- **Date:** 2025-06-24
- **Files changed:**
  - `3d_segmentation/swin_unetr_brats21_segmentation_3d.ipynb`
  - `3d_segmentation/swin_unetr_btcv_segmentation_3d.ipynb`
  - `auto3dseg/docs/gpu_opt.md`
  - `deployment/Triton/models/mednist_class/1/model.py`
  - `deployment/Triton/models/monai_covid/1/model.py`
  - `generation/anomaly_detection/anomaly_detection_with_transformers.ipynb`
  - `generation/maisi/scripts/sample.py`
  - `monailabel/monailabel_HelloWorld_radiology_3dslicer.ipynb`
  - `multimodal/openi_multilabel_classification_transchex/transchex_openi_multilabel_classification.ipynb`
  - `self_supervised_pretraining/vit_unetr_ssl/ssl_train.ipynb`
- **Why:** Corrected misspelled words across multiple tutorials and scripts.

---

### 5f12844 — Fix documentation errors in tutorial (#2002)
- **Author:** Mingxin Zheng
- **Date:** 2025-06-24
- **Files changed:**
  - `README.md`
  - `deepedit/ignite/README.md`
  - `deepgrow/ignite/README.md`
  - `pathology/hovernet/README.MD`
  - `pathology/nuclick/README.md`
- **Why:** Fixed documentation errors and broken references in multiple tutorial README files.

---

### a51fdeb — Fix the links of the generative models tutorials (#1999)
- **Author:** Virginia Fernandez
- **Date:** 2025-06-20
- **Files changed:**
  - `README.md`
- **Why:** Updated broken links pointing to generative model tutorials.

---

### 28b462c — Remove deprecated feature for v1.5 (#1992)
- **Author:** YunLiu
- **Date:** 2025-05-25
- **Files changed:**
  - `bundle/pythonic_usage_guidance/pythonic_bundle_access.ipynb`
  - `computer_assisted_intervention/endoscopic_inbody_classification.ipynb`
- **Why:** Removed usage of deprecated MONAI features that were dropped in v1.5.

---

### 9902215 — Add a tutorial demonstrating 2D image restoration using the MONAI Restormer model (#1987)
- **Author:** Cano-Muniz, Santiago
- **Date:** 2025-05-22
- **Files changed:**
  - `2d_regression/image_restoration.ipynb`
- **Why:** Added a new tutorial demonstrating 2D image restoration with the Restormer architecture in MONAI.

---

### 75c44e0 — fix: handle metadata loading and shape calculation in transforms (#1990)
- **Author:** Tristan Kirscher
- **Date:** 2025-05-16
- **Files changed:**
  - `modules/dynunet_pipeline/transforms.py`
- **Why:** Fixed metadata loading and shape calculation logic in the DynUNet pipeline transforms to prevent runtime errors.

---

### 211cfd8 — Remove deprecated feature for v1.5 (#1989)
- **Author:** YunLiu
- **Date:** 2025-05-11
- **Files changed:**
  - `3d_segmentation/spleen_segmentation_3d.ipynb`
  - `3d_segmentation/spleen_segmentation_3d_lightning.ipynb`
  - `3d_segmentation/spleen_segmentation_3d_visualization_basic.ipynb`
  - `3d_segmentation/swin_unetr_brats21_segmentation_3d.ipynb`
  - `3d_segmentation/swin_unetr_btcv_segmentation_3d.ipynb`
  - `3d_segmentation/unetr_btcv_segmentation_3d.ipynb`
  - `3d_segmentation/unetr_btcv_segmentation_3d_lightning.ipynb`
  - `acceleration/automatic_mixed_precision.ipynb`
  - `acceleration/dataset_type_performance.ipynb`
  - `acceleration/fast_training_tutorial.ipynb`
  - `auto3dseg/docs/algorithm_generation.md`
  - `auto3dseg/tasks/hecktor22/hecktor_crop_neck_region.py`
  - `bundle/python_bundle_workflow/scripts/inference.py`
  - `bundle/python_bundle_workflow/scripts/train.py`
  - `bundle/pythonic_usage_guidance/pythonic_bundle_access.ipynb`
  - `deepgrow/ignite/train.py`
  - `deployment/Triton/models/monai_covid/1/model.py`
  - `experiment_management/spleen_segmentation_aim.ipynb`
  - `experiment_management/spleen_segmentation_mlflow.ipynb`
  - `model_zoo/transfer_learning_with_bundle/evaluate.py`
  - `model_zoo/transfer_learning_with_bundle/train.py`
  - `modules/dynunet_pipeline/transforms.py`
  - `modules/integrate_3rd_party_transforms.ipynb`
  - `modules/inverse_transforms_and_test_time_augmentations.ipynb`
  - `modules/postprocessing_transforms.ipynb`
  - `modules/transform_visualization.ipynb`
  - `modules/transforms_metatensor.ipynb`
  - `modules/transforms_update_meta_data.ipynb`
  - `performance_profiling/radiology/train_base_nvtx.py`
  - `performance_profiling/radiology/train_fast_nvtx.py`
  - `self_supervised_pretraining/swinunetr_pretrained/swinunetr_finetune.ipynb`
  - `self_supervised_pretraining/vit_unetr_ssl/multi_gpu/mgpu_ssl_train.py`
  - `self_supervised_pretraining/vit_unetr_ssl/ssl_finetune.ipynb`
  - `self_supervised_pretraining/vit_unetr_ssl/ssl_train.ipynb`
  - `vista_3d/vista3d_spleen_finetune.ipynb`
- **Why:** Bulk removal of deprecated MONAI v1.5 features across many tutorials and scripts to ensure compatibility.

---

### fd86def — update error explanation in parameters fold in tutorials train_controlnet.ipynb (#1974)
- **Author:** SiCheng Li
- **Date:** 2025-05-07
- **Files changed:**
  - `generation/maisi/maisi_train_controlnet_tutorial.ipynb`
- **Why:** Improved the error explanation text in the controlnet training tutorial parameters section.

---

### 84c0a07 — Fix omniverse integration notebook (#1980)
- **Author:** YunLiu
- **Date:** 2025-04-20
- **Files changed:**
  - `modules/omniverse/omniverse_integration.ipynb`
  - `modules/omniverse/utility.py`
- **Why:** Fixed broken functionality in the Omniverse integration notebook.

---

### 1d4b31d — 1978 fix sform issue in omniverse nifti to mesh function (#1979)
- **Author:** Yiheng Wang
- **Date:** 2025-04-18
- **Files changed:**
  - `modules/omniverse/utility.py`
- **Why:** Fixed the sform affine matrix handling in the NIfTI-to-mesh conversion function for Omniverse.

---

### 826c451 — add readme for MONAI + Fed-BioMed integration for federated learning (#1944)
- **Author:** Marc Vesin
- **Date:** 2025-04-18
- **Files changed:**
  - `federated_learning/fedbiomed/README.md`
- **Why:** Added documentation for the MONAI + Fed-BioMed federated learning integration.

---

### b702677 — [pre-commit.ci] pre-commit suggestions (#1972)
- **Author:** pre-commit-ci[bot]
- **Date:** 2025-04-08
- **Files changed:**
  - `.pre-commit-config.yaml`
  - `bundle/hybrid_programming/scripts/train_demo.py`
- **Why:** Automated pre-commit hook updates and formatting fixes applied by CI.

---

## Cumulative Files Modified

| Commits | File |
|---------|------|
| 8 | `runner.sh` |
| 6 | `README.md` |
| 5 | `computer_assisted_intervention/endoscopic_inbody_classification.ipynb` |
| 4 | `.github/workflows/test-modified.yml` |
| 3 | `generation/maisi/README.md` |
| 3 | `diagnose_1_6_release.md` |
| 3 | `.pre-commit-config.yaml` |
| 2 | `self_supervised_pretraining/vit_unetr_ssl/ssl_train.ipynb` |
| 2 | `modules/omniverse/utility.py` |
| 2 | `modules/dynunet_pipeline/transforms.py` |
| 2 | `model_zoo/TCIA_PROSTATEx_Prostate_MRI_Anatomy_Model.ipynb` |
| 2 | `generation/maisi/maisi_train_controlnet_tutorial.ipynb` |
| 2 | `deployment/fastapi_inference/tests/test_api.py` |
| 2 | `deployment/fastapi_inference/tests/__init__.py` |
| 2 | `deployment/fastapi_inference/examples/client.py` |
| 2 | `deployment/fastapi_inference/app/schemas.py` |
| 2 | `deployment/fastapi_inference/app/model_loader.py` |
| 2 | `deployment/fastapi_inference/app/main.py` |
| 2 | `deployment/fastapi_inference/app/inference.py` |
| 2 | `deployment/fastapi_inference/app/__init__.py` |
| 2 | `deployment/Triton/models/monai_covid/1/model.py` |
| 2 | `bundle/pythonic_usage_guidance/pythonic_bundle_access.ipynb` |
| 2 | `3d_segmentation/swin_unetr_btcv_segmentation_3d.ipynb` |
| 2 | `3d_segmentation/swin_unetr_brats21_segmentation_3d.ipynb` |
| 2 | `3d_segmentation/spleen_segmentation_3d_visualization_basic.ipynb` |
| 1 | `vista_3d/vista3d_spleen_finetune.ipynb` |
| 1 | `vista_3d/README.md` |
| 1 | `vista_2d/README.md` |
| 1 | `self_supervised_pretraining/vit_unetr_ssl/ssl_finetune.ipynb` |
| 1 | `self_supervised_pretraining/vit_unetr_ssl/multi_gpu/mgpu_ssl_train.py` |
| 1 | `self_supervised_pretraining/swinunetr_pretrained/swinunetr_finetune.ipynb` |
| 1 | `requirements.txt` |
| 1 | `performance_profiling/radiology/train_fast_nvtx.py` |
| 1 | `performance_profiling/radiology/train_base_nvtx.py` |
| 1 | `pathology/tumor_detection/README.MD` |
| 1 | `pathology/nuclick/README.md` |
| 1 | `pathology/hovernet/README.MD` |
| 1 | `nnunet/README.md` |
| 1 | `multimodal/openi_multilabel_classification_transchex/transchex_openi_multilabel_classification.ipynb` |
| 1 | `monailabel/monailabel_HelloWorld_radiology_3dslicer.ipynb` |
| 1 | `modules/transforms_update_meta_data.ipynb` |
| 1 | `modules/transforms_metatensor.ipynb` |
| 1 | `modules/transform_visualization.ipynb` |
| 1 | `modules/postprocessing_transforms.ipynb` |
| 1 | `modules/omniverse/omniverse_integration.ipynb` |
| 1 | `modules/load_medical_images.ipynb` |
| 1 | `modules/inverse_transforms_and_test_time_augmentations.ipynb` |
| 1 | `modules/interpretability/class_lung_lesion.ipynb` |
| 1 | `modules/integrate_3rd_party_transforms.ipynb` |
| 1 | `modules/idc_dataset.ipynb` |
| 1 | `model_zoo/transfer_learning_with_bundle/train.py` |
| 1 | `model_zoo/transfer_learning_with_bundle/evaluate.py` |
| 1 | `generation/maisi/scripts/utils.py` |
| 1 | `generation/maisi/scripts/sample.py` |
| 1 | `generation/maisi/scripts/inference.py` |
| 1 | `generation/maisi/scripts/download_model_data.py` |
| 1 | `generation/maisi/maisi_train_vae_tutorial.ipynb` |
| 1 | `generation/maisi/maisi_train_diff_unet_tutorial.ipynb` |
| 1 | `generation/maisi/maisi_inference_tutorial.ipynb` |
| 1 | `generation/maisi/configs/environment_maisi_diff_model.json` |
| 1 | `generation/maisi/configs/environment_maisi_controlnet_train.json` |
| 1 | `generation/maisi/configs/environment_maisi3d-rflow.json` |
| 1 | `generation/maisi/configs/environment_maisi3d-ddpm.json` |
| 1 | `generation/anomaly_detection/anomaly_detection_with_transformers.ipynb` |
| 1 | `federated_learning/nvflare/README.md` |
| 1 | `federated_learning/fedbiomed/README.md` |
| 1 | `experiment_management/spleen_segmentation_mlflow.ipynb` |
| 1 | `experiment_management/spleen_segmentation_aim.ipynb` |
| 1 | `detection/requirements.txt` |
| 1 | `deployment/fastapi_inference/requirements.txt` |
| 1 | `deployment/fastapi_inference/examples/sample_requests.http` |
| 1 | `deployment/fastapi_inference/docker/docker-compose.yml` |
| 1 | `deployment/fastapi_inference/docker/Dockerfile` |
| 1 | `deployment/fastapi_inference/README.md` |
| 1 | `deployment/Triton/models/mednist_class/1/model.py` |
| 1 | `deepgrow/ignite/train.py` |
| 1 | `deepgrow/ignite/README.md` |
| 1 | `deepedit/ignite/README.md` |
| 1 | `deep_atlas/deep_atlas_tutorial.ipynb` |
| 1 | `competitions/MICCAI/surgtoolloc/preprocess_detect_scene_and_split_fold.ipynb` |
| 1 | `bundle/python_bundle_workflow/scripts/train.py` |
| 1 | `bundle/python_bundle_workflow/scripts/inference.py` |
| 1 | `bundle/hybrid_programming/scripts/train_demo.py` |
| 1 | `bundle/05_spleen_segmentation_lightning.ipynb` |
| 1 | `auto3dseg/tasks/hecktor22/hecktor_crop_neck_region.py` |
| 1 | `auto3dseg/notebooks/msd_datalist_generator.ipynb` |
| 1 | `auto3dseg/notebooks/msd_crossval_datalist_generator.ipynb` |
| 1 | `auto3dseg/notebooks/auto_runner.ipynb` |
| 1 | `auto3dseg/docs/run_with_minimal_input.md` |
| 1 | `auto3dseg/docs/gpu_opt.md` |
| 1 | `auto3dseg/docs/algorithm_generation.md` |
| 1 | `auto3dseg/README.md` |
| 1 | `acceleration/fast_training_tutorial.ipynb` |
| 1 | `acceleration/distributed_training/distributed_training.md` |
| 1 | `acceleration/dataset_type_performance.ipynb` |
| 1 | `acceleration/automatic_mixed_precision.ipynb` |
| 1 | `3d_segmentation/unetr_btcv_segmentation_3d_lightning.ipynb` |
| 1 | `3d_segmentation/unetr_btcv_segmentation_3d.ipynb` |
| 1 | `3d_segmentation/spleen_segmentation_3d_lightning.ipynb` |
| 1 | `3d_segmentation/spleen_segmentation_3d.ipynb` |
| 1 | `3d_regression/README.md` |
| 1 | `2d_regression/image_restoration.ipynb` |
| 1 | `2d_classification/monai_101.ipynb` |
| 1 | `.github/workflows/pep8.yml` |
| 1 | `.github/workflows/guidelines.yml` |
| 1 | `.github/workflows/copyright.yml` |

**Total unique files modified:** 99
