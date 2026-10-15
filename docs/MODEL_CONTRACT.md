# Model contract

What the web app must feed the model and what it gets back, as read from the
research repository — not from the build prompt.

- Source: <https://github.com/glioma-ml-research/mu-glioma-segmentation-benchmark>
  at commit `23115233b151f795c949df84fa8dcb82156b4f74` (8 Oct 2026).
- Files read: `README.md`, `manuscript/MU_Glioma_Research_Paper.docx`,
  `protocol/*`, `notebooks/no29_kaggle.ipynb` (nnU-Net, fold 1; `no30`–`no35`
  are the same notebook with a different fold/seed), `results/final_analysis/*`,
  `results/validation_records/no29.json`.

Every statement below is tagged **[repo]** (read directly from the repository)
or **[unverified]** (not recorded in the repository; needs the weights folder or
a sample case to confirm).

## 1. Label map — matches the prompt

**[repo]** `dataset.json` written by notebook cell 8, and the protocol:

| Value | `dataset.json` key | Meaning | Short name used in the app |
|---|---|---|---|
| 0 | `background` | background | — |
| 1 | `non_enhancing_tumor_core` | non-enhancing tumor core / necrosis | NETC |
| 2 | `flair_hyperintensity_edema` | surrounding non-enhancing FLAIR hyperintensity / edema | SNFH |
| 3 | `enhancing_tissue` | enhancing tissue | ET |
| 4 | `resection_cavity` | resection cavity | RC |

Plain multiclass labels, no region-based training. Predictions are validated in
the notebook to contain only `{0,1,2,3,4}`.

Derived targets used by the paper: **tumor-related region = labels 1–3**
(primary outcome) and **all postoperative regions = labels 1–4**.

## 2. Input channels — matches the prompt

**[repo]** notebook cell 6 (`modalities`) and cell 8 (`channel_names`):

| nnU-Net suffix | `channel_names` | Sequence | MU-Glioma-Post file suffix |
|---|---|---|---|
| `_0000` | `T1` | native T1 | `_brain_t1n.nii.gz` |
| `_0001` | `T1CE` | contrast-enhanced T1 | `_brain_t1c.nii.gz` |
| `_0002` | `T2` | T2 | `_brain_t2w.nii.gz` |
| `_0003` | `FLAIR` | T2-FLAIR | `_brain_t2f.nii.gz` |

Reference mask: `<case>_tumorMask.nii.gz`. Case id: `PatientID_XXXX_Timepoint_N`.
File ending `.nii.gz`. No `overwrite_image_reader_writer`, so nnU-Net picks
`SimpleITKIO` for these files.

## 3. nnU-Net identity

**[repo]** unless tagged otherwise.

| Item | Value |
|---|---|
| nnU-Net | `nnunetv2==2.8.1` |
| PyTorch | `2.8.0+cu126` (torchvision `0.23.0`), NumPy `2.0.2`, SciPy `1.14.1`, scikit-image `0.24.0` |
| Dataset | ID `502`, name `Dataset502_MUGliomaPost` |
| Plans | `nnUNetPlans` (default planner, `nnUNetv2_plan_and_preprocess -d 502 -c 3d_fullres`) |
| Configuration | `3d_fullres` |
| Trainer | **`nnUNetTrainer_MUGlioma_noNN`** (`no29` … `no35`), a subclass of `nnUNetTrainer` |
| Folds inside each run | **only `fold_0`** (manual `splits_final.json`: fit-train vs inner-tuning) |
| Checkpoint used for reported results | **`checkpoint_best.pth`** (selected on the inner-tuning patients) |
| Inference command | `nnUNetv2_predict -d Dataset502_MUGliomaPost -c 3d_fullres -f 0 -tr <trainer> -p nnUNetPlans -chk checkpoint_best.pth` → defaults: sliding window step 0.5, Gaussian weighting, mirroring TTA on |
| Patch / batch | 128 × 128 × 128, batch 2 (manuscript). **Run no30's `plans.json` says 128 × 160 × 112** (z, y, x), batch 2, target spacing 1 × 1 × 1 mm. Each run has its own plans. |
| Parameters | 31,200,601 (no29, no32–no35) or 30,791,001 (no30, no31) |

The custom trainer only changes training length, seeding and early stopping
(max 150 epochs, ≥100 before stopping, EMA foreground Dice, Δ 0.002, patience
15). It does **not** override `build_network_architecture`, so the stock
`nnUNetTrainer` builds the identical network for inference.

Runs that make up the reported result:

| Run | Outer fold | Seed | Role | Mean tumor-region Dice on its own outer fold |
|---|---|---|---|---|
| no29 | 1 | 2026 | primary | 0.895 |
| no30 | 2 | 2026 | primary | 0.900 |
| no31 | 3 | 2026 | primary | 0.893 |
| no32 | 4 | 2026 | primary | 0.835 |
| no33 | 5 | 2026 | primary | 0.871 |
| no34 | 1 | 2027 | seed robustness | — |
| no35 | 1 | 2028 | seed robustness | — |

Pooled over the five primary runs (one out-of-fold score per patient, 203
patients): **0.879 (95% CI 0.856–0.900)**.

## 4. Where the repo contradicts the prompt

The app follows the repo on every point below.

### 4.1 There is no `nnUNetTrainer__nnUNetPlans__3d_fullres` folder with `fold_0…fold_4`

The prompt expects one trained folder with several `fold_*` subfolders. The
study instead trained **five independent nnU-Net models**, one per outer fold.
For each, the outer-test patients were excluded from fingerprinting, planning
and training, so each run has **its own `plans.json`**, and each contains only
`fold_0`. The parameter counts differ between runs (31.20 M vs 30.79 M), so the
networks are not even the same shape: they cannot be loaded as "folds" of one
model.

The trainer folder is also named
`nnUNetTrainer_MUGlioma_no29__nnUNetPlans__3d_fullres`, not
`nnUNetTrainer__…`.

### 4.2 Weights are packaged differently from the nnU-Net results layout

Each validated archive (`MU_Glioma_noNN_nnunet_foldF_seedS_reports.zip`, ~220 MiB)
stores the model as:

```
model/checkpoint_best.pth
model/plans.json
model/dataset.json
```

### 4.3 `checkpoint_best.pth`, not `checkpoint_final.pth`

The reported numbers come from the best inner-tuning checkpoint.

### 4.4 The custom trainer class is not importable from a clean nnU-Net install

The notebook writes `nnUNetTrainer_MUGlioma_noNN.py` into the installed
`nnunetv2` package at run time. Stock `nnUNetPredictor.initialize_from_trained_model_folder`
raises "Unable to locate trainer class" on a fresh install.

### 4.5 No reorientation was applied in training

The prompt asks to reorient to RAS before inference "exactly as the notebooks
did". The notebook does **not** reorient: it symlinks the distributed files
straight into `imagesTr`, and `SimpleITKIO` reads arrays as stored. To match
training, already-preprocessed uploads are passed to nnU-Net **untouched**.
RAS reorientation is used only for the viewer volumes (the affine is updated
with it, so world coordinates are unchanged).

### 4.6 Confidence is uncalibrated

The manuscript's limitations state that calibration was not evaluated. The app
shows the maximum softmax probability and labels it "uncalibrated".

### 4.7 Authorship metadata

`README.md` lists three researchers (as in the prompt). `CITATION.cff` and
`LICENSES.md` list a fourth, Alasgar Aliyev. The About page follows the README
and prompt; change `frontend/src/pages/About.tsx` if that is wrong.
`CITATION.cff` also points at `github.com/atilla-m/…` rather than the
`glioma-ml-research` organisation URL.

## 5. How the app resolves this

`NNUNET_MODEL_DIR` may point at any of these; the layout is auto-detected:

| Layout | Example | Behaviour |
|---|---|---|
| A. Study archive `model/` folder | `…/no29/model/` containing `checkpoint_best.pth`, `plans.json`, `dataset.json` | one model |
| B. nnU-Net trainer folder | `…/nnUNetTrainer_MUGlioma_no29__nnUNetPlans__3d_fullres/` with `fold_0/` | one model per `fold_*`; folds that share `plans.json` are ensembled the normal nnU-Net way |
| C. Parent folder of several A/B folders | `…/models/` containing `no29/ … no33/` | **cross-model ensemble**: every model predicts with its own plans, class probabilities are averaged in the uploaded image space, argmax gives the label |

Rules enforced at load time (`backend/app/inference.py`):

- `dataset.json` must have exactly the channel order and label map in §1–§2,
  otherwise the app refuses to run inference (it does not fall back to another
  model, and it does not silently enter demo mode).
- `checkpoint_best.pth` is preferred; `checkpoint_final.pth` is used only if
  best is absent, and the summary says which was used.
- Trainer names matching `nnUNetTrainer_MUGlioma_*` are built with the stock
  `nnUNetTrainer.build_network_architecture`. Any other unknown trainer is an
  error.
- Every model, checkpoint file, SHA-256 and trainer name is reported in
  `/api/jobs/{id}/summary`.

**Important for interpretation.** The paper never evaluated an ensemble of the
five outer-fold models: its 0.879 is a pool of single-model, out-of-fold
predictions. Layout C is a reasonable deployment choice for new patients, but it
is not the configuration the paper validated. For any MU-Glioma-Post patient,
four of the five models saw that patient during training, so an ensemble Dice
on a dataset case is optimistically biased. The end-to-end test
(`backend/scripts/e2e_case.py`) therefore reports which loaded model is
out-of-fold for the test patient, using the frozen splits copied to
`backend/app/resources/outer_test_patients.json`.

## 6. Preprocessing the training data already had

**[repo]** Manuscript, Methods: "The provided images underwent skull stripping,
co-registration, and resampling, and were saved as NIfTI volumes" (citing the
dataset paper). The four sequences of a timepoint share one grid — the Keras
notebooks assert matching affines, and nnU-Net's `--verify_dataset_integrity`
passed.

**[repo]** On top of that, the only preprocessing is nnU-Net's own, driven by
each run's `plans.json`: crop to non-zero, resample to the planned target
spacing, per-channel z-score normalisation (MRI channels), then sliding-window
inference and resampling of the prediction back to the input grid. The app
leaves all of this to `nnunetv2`.

**[verified 15 Oct 2026 on a local copy of the dataset]** All 596 T1c volumes
are 240 × 240 × 155 voxels, 1 mm isotropic, stored LPS, with the affine
`diag(-1, -1, 1)` and translation `(0, 239, 0)`. A sampled case is skull-stripped
(80% zero voxels) and its four sequences and mask share one grid. So the default
`TRAINING_ORIENTATION=LPS` matches the training data. What follows was written
before this check.

**[unverified]** The repository does not record the atlas space, voxel spacing,
matrix size or array orientation of the distributed files. The dataset is
described upstream as following the BraTS-style pipeline (SRI24 atlas, 1 mm
isotropic, 240 × 240 × 155, LPS-stored arrays); the app does not depend on that
being true:

- target spacing is read from `plans.json` at run time and shown in the summary;
- for already-preprocessed uploads, arrays are passed through as stored;
- for raw uploads, images are brought to the axis order named by
  `TRAINING_ORIENTATION` (default `LPS`) before inference. nnU-Net trains and
  tests with mirroring on all three axes, so axis *flips* are immaterial; only
  the axis *order* matters.

These three items should be confirmed against one real case and one
`plans.json`; `backend/scripts/inspect_contract.py` prints them.

## 7. Raw (not preprocessed) uploads

Not part of the study — the model has only ever seen the dataset's own
preprocessing. When "Already preprocessed" is unchecked, the app approximates
it:

1. rigid co-registration of T1, T2, FLAIR onto the T1c grid (SimpleITK, Mattes
   mutual information, multi-resolution);
2. skull stripping with HD-BET on T1c, mask applied to all four sequences;
3. reorientation to `TRAINING_ORIENTATION`; resampling to the planned spacing
   is done by nnU-Net.

This differs from the dataset's pipeline (no atlas registration), is untested
against reference masks, and is flagged as such in the job summary.

## 8. Scope of claims

From the manuscript and README: internal technical validation on
MU-Glioma-Post only; no external testing; not a diagnostic-accuracy study; no
evidence of clinical utility. The model segments tumor-related sub-regions and
the resection cavity. It does not classify tumor type or grade. Reference masks
are called a "reference standard", not ground truth.

Dataset citation: Yaseen D, Garrett F, Gass J, et al. *University of Missouri
Post-operative Glioma Dataset (MU-Glioma-Post), Version 1.* The Cancer Imaging
Archive, 2025. doi:10.7937/7K9K-3C83. Descriptor: Mahmoud E, Gass J, Dhemesh Y,
et al. *Scientific Data* 2025;12:1847. doi:10.1038/s41597-025-06011-7.
