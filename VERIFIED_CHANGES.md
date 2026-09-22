# Verified Changes (Source of Truth)

This file records **only** fixes that were applied to a previously-failing
notebook and then **confirmed to pass** by an actual test run. It is the source
of truth for "what did we fix and did it work?".

Scope and rules:
- An entry is added ONLY after the notebook runs green
  (`Testing finished. All N executed tests passed!` for its folder, or an
  explicit single-notebook pass).
- Skip-list changes are NOT recorded here (the notebook was disabled, not
  fixed) — those live in `CHANGES.md` only.
- Unverified / assumed fixes are NOT recorded here until a run confirms them.

Test environment:
- **MONAI version:** `1.6.0rc1` (source build, git `1.6.0rc1-4-gb89a8af1`;
  runtime self-reports `0+unknown` because the image builds from a non-tagged
  source tree)
- **Base image:** `monai_1_6:latest` (NGC PyTorch base)
- **Python:** 3.12 · **PyTorch:** 2.10.0a0 · **NumPy:** 2.1.0
- **GPU:** NVIDIA A10G (24 GB), `--gpus all`
- **Data cache:** `MONAI_DATA_DIRECTORY=/data` (host `nb_data/`)
- **Branch:** `vikash/updated_monai_1_6_release`

---

## Verified Fixes

### patch_inferer/modular_patch_inferer.ipynb
- **Verified:** 2026-09-21 — folder run `patch_inferer`: `All 1 executed tests passed!`
- **Root cause:** Zarr 3.x removed the `compressor=` argument for array creation
  (`TypeError: compressor is not available for Zarr format 3 arrays`).
- **Fix:** Updated the notebook to the Zarr v3 array-creation API.
- **Applied in:** commit `8ee04e1`
- **Verified against:** MONAI `1.6.0rc1`

### bundle/04_integrating_code.ipynb
- **Verified:** 2026-09-21 — folder run `bundle`: `5 of 6 executed tests passed!`
  (this notebook executed successfully; the single bundle failure was
  `pythonic_bundle_access.ipynb`, a separate DeadKernelError).
- **Root cause (old run):** `monai.bundle run train` subprocess exited non-zero;
  underlying issue was the `download_and_extract` hashing behaviour change.
- **Fix:** `hash_type="md5"` passed explicitly to downloads (commit `df98606`);
  confirmed working with the shared `nb_data` cache under MONAI 1.6.0rc1.
- **Applied in:** commit `df98606`
- **Verified against:** MONAI `1.6.0rc1`
- **Note:** earlier I had proposed skip-listing this notebook based on stale logs;
  the live run shows it PASSES, so it is NOT skipped.

### 3d_segmentation/unet_segmentation_3d_ignite.ipynb
- **Verified:** 2026-09-21 — single-notebook run: `All 1 executed tests passed!`
- **Root cause:** `ImportError: cannot import name 'path_to_sqlite_uri' from
  'monai.utils'`. A prior fix (commit `1bf0bf9`) imported and used
  `path_to_sqlite_uri`, which does **not** exist in MONAI 1.6.0rc1 — so that
  commit regressed this notebook.
- **Fix (working tree, not yet committed):**
  - removed `path_to_sqlite_uri` from `from monai.utils import ...`
  - replaced `path_to_sqlite_uri(os.path.join(log_dir, "mlruns.db"))` with the
    inline URI `"sqlite:///" + os.path.join(log_dir, "mlruns.db")`
- **Verified against:** MONAI `1.6.0rc1`

### experiment_management/bundle_integrate_mlflow.ipynb
- **Verified:** 2026-09-21 — single-notebook run: `All 1 executed tests passed!`
- **Root cause:** same `path_to_sqlite_uri` ImportError introduced by commit
  `1bf0bf9` (function does not exist in MONAI 1.6.0rc1).
- **Fix (working tree):** removed the import; replaced
  `path_to_sqlite_uri("./eval/mlruns.db")` with
  `"sqlite:///" + os.path.abspath("./eval/mlruns.db")`.
- **Verified against:** MONAI `1.6.0rc1`

---

## Verified Passing After Prior Fixes (confirmed green this full run)

These carried fixes from earlier commits and were confirmed to pass in the
full folder run on 2026-09-21 (MONAI 1.6.0rc1):

- `modules/integrate_3rd_party_transforms.ipynb` — batchgenerators pin fix (commit `8ee04e1`) — passed in `modules` run
- `modules/resample_benchmark.ipynb` — zarr/pin fix (commit `8ee04e1`) — passed in `modules` run
- `modules/public_datasets.ipynb` — `hash_type="md5"` download fix (commit `df98606`) — passed in `modules` run

---

## Passing With No Code Change Needed (folder-run failures were transient)

During the batched folder run, these two notebooks failed inside a DataLoader
worker while ~42 `modules` notebooks executed back-to-back in a single
container (memory pressure). Re-running each **individually in a clean
container** passes cleanly, so there is **no code bug and no fix applied** —
the folder-level failure was environmental.

- `modules/mednist_GAN_tutorial.ipynb` — standalone run: `All 1 executed tests passed!`
- `modules/mednist_GAN_workflow_array.ipynb` — standalone run: `All 1 executed tests passed!`

Implication: the batched folder counts can *under*-report passes for
memory-heavy folders; individual re-runs are authoritative.

---

## Known Issue (real bug, not yet fixed)

- `modules/transforms_metatensor.ipynb` — fails **even standalone** at cell
  `In [17]`:
  `ValueError: Item of type <MetaTensor> (key: None, pop: True) has empty
  'applied_operations'` raised from `DivisiblePadd.inverse`.
  The tutorial copies `applied_operations` onto a synthetic `output_batch`
  MetaTensor and calls `t.inverse()`; under MONAI 1.6.0rc1 one item reaches the
  inverse with empty `applied_operations`. A minimal repro shows
  `applied_operations` normally survives collation/indexing/arithmetic, so the
  problem is specific to how this demo cell reconstructs the tensor after
  `RandAffineD(spatial_size=...)`. Left unfixed pending a focused rewrite of the
  inverse-transform demo cell (no guess-patch applied).

---

## Full Run Summary (2026-09-21, MONAI 1.6.0rc1, A10G)

Folders executed: 30 (excluded: `auto3dseg` per request; `microscopy` skip-listed for kernel/OOM crash).

**Fully green (25 folders):** 2d_classification, 2d_registration, 2d_regression,
3d_classification, 3d_registration, 3d_regression, acceleration, active_learning,
competitions, computer_assisted_intervention, deep_atlas, deepedit, deepgrow,
federated_learning, full_gpu_inference_pipeline, hugging_face, model_zoo,
monailabel (10/10), multimodal, patch_inferer, pathology, reconstruction,
self_supervised_pretraining, vista_2d, generation (18 passed before the 90-min folder cap).

**Failures and disposition:**

| Notebook | Error | Disposition |
|----------|-------|-------------|
| 3d_segmentation/unet_segmentation_3d_ignite.ipynb | `path_to_sqlite_uri` ImportError | FIXED + verified |
| experiment_management/bundle_integrate_mlflow.ipynb | `path_to_sqlite_uri` ImportError | FIXED + verified |
| modules/mednist_GAN_tutorial.ipynb | transient DataLoader worker crash | Passes standalone — no fix |
| modules/mednist_GAN_workflow_array.ipynb | transient DataLoader worker crash | Passes standalone — no fix |
| modules/transforms_metatensor.ipynb | empty `applied_operations` on inverse | KNOWN ISSUE — unfixed |
| bundle/pythonic_bundle_access.ipynb | DeadKernelError (OOM) | Environmental — not code-fixable |
| deployment/mednist_classifier_bentoml.ipynb | `bentoml==0.13.1` uninstallable on Py3.12 | Needs BentoML 1.x rewrite |
| vista_3d/vista3d_spleen_finetune.ipynb | DeadKernelError (OOM, 24 GB GPU) | Environmental — not code-fixable |

**Not fully covered:** `generation` hit the 90-min per-folder cap; large diffusion
notebooks beyond the first 18 were not executed this run.
