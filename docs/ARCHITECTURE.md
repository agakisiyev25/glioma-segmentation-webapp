# Architecture

A doctor uploads four MRI sequences of one examination. The backend segments
them with the study's nnU-Net, measures the result, and builds a 3D surface of
that patient's own brain. The frontend shows the regions in 3D and on slices,
and reports the label, coordinates and model confidence of any voxel.

What the model expects is in [MODEL_CONTRACT.md](MODEL_CONTRACT.md). This
document covers how the app is put together.

## The one rule: a single world frame

Everything is expressed in **NIfTI world coordinates, RAS+, millimetres**
(x → patient right, y → anterior, z → superior). Voxel indices are never used
as 3D coordinates.

| Thing | How it gets to world mm |
|---|---|
| Mesh vertices (GLB) | marching-cubes vertices → `voxel_to_world` → smoothing in mm |
| Bounding boxes, centroids | voxel values → `voxel_to_world` |
| `GET /voxel?x=&y=&z=` | `world_to_voxel` with the inverse affine |
| 2D slice ↔ mouse | pixel → voxel → `voxelToWorld`; crosshair → `worldToVoxel` |
| 3D scene | scene units *are* world mm; camera up = +z; no transforms in the scene graph |
| 3D picking | ray → voxel space via the inverse affine → hit voxel → `voxelToWorld` |
| Voxel-mode cubes | instance matrix = affine · translate(i, j, k) |

Conversions live in exactly two files and nowhere else multiplies by an affine:

- `backend/app/geometry.py` — `voxel_to_world()`, `world_to_voxel()`
- `frontend/src/lib/geometry.ts` — `voxelToWorld()`, `worldToVoxel()`

The backend sends each volume's affine **and** its inverse in the volume
header, so both sides use the same numbers.

### Three grids, one frame

| Grid | What it is | Used for |
|---|---|---|
| Uploaded T1c grid | the file as the doctor uploaded it | nnU-Net input (untouched when already preprocessed); the downloadable mask |
| nnU-Net input grid | same as above, or reoriented to `TRAINING_ORIENTATION` for raw uploads | inference only |
| Viewer grid | the uploaded T1c grid with axes permuted/flipped to the closest RAS | label, confidence, T1c, FLAIR volumes; slices; meshes |

Moving between these grids is a pure axis permutation/flip with the affine
updated to match (`preprocess.reorient`). No voxel is resampled and no voxel
changes its world coordinate, so the three grids describe the same points.

## Backend (`backend/app`)

```
main.py        FastAPI routes; upload ingestion; optional static frontend
jobs.py        job folders, job.json, single worker thread, 24 h cleanup
pipeline.py    one job, start to finish
io_utils.py    validation, header sanitising, DICOM zip → NIfTI
preprocess.py  reorientation, rigid registration (SimpleITK), HD-BET
inference.py   weight discovery, contract checks, nnU-Net prediction
postprocess.py volumes (mL), 26-connected components, boxes, centroids
meshing.py     mask → marching cubes → Taubin → decimation → GLB
volumes.py     voxel volumes as gzip binary + JSON header
geometry.py    voxel ↔ world
labels.py      label map, colours, disclaimer text
config.py      environment variables
```

### Pipeline (one job)

1. **Upload** (`POST /api/jobs`, synchronous). Each file is streamed to disk
   under its slot name, DICOM zips are converted (`dcm2niix`, SimpleITK as a
   fallback) and deleted, and every volume is validated (3D, 16–1024 voxels per
   axis, 0.2–10 mm spacing, no NaN/Inf) and **re-written with a fresh NIfTI
   header**. The originals are then removed. Validation failures return 400
   before a job is queued.
2. **Preprocess.**
   - *Already preprocessed* (default): the four volumes must share one grid.
     They are passed to nnU-Net exactly as stored. Brain mask = non-zero T1c.
   - *Raw*: T1, T2, FLAIR are rigidly registered onto the T1c grid (Mattes
     mutual information, three resolution levels, linear interpolation); HD-BET
     gives the brain mask from T1c; masked volumes are reoriented to
     `TRAINING_ORIENTATION`. Resampling to the planned spacing is nnU-Net's job.
3. **Inference** (`inference.py`). Arrays are read with nnU-Net's own
   `SimpleITKIO`, predicted with `nnUNetPredictor.predict_single_npy_array`
   (sliding window 0.5, Gaussian weighting, mirroring TTA — nnU-Net's defaults,
   as in the study). Several models → softmax averaged, then argmax. On a CUDA
   out-of-memory error the model is retried on CPU and a warning is recorded.
   In **demo mode** this step loads the uploaded segmentation instead.
4. **Postprocess.** The mask is written in the uploaded T1c grid
   (`segmentation.nii.gz`). Volumes use the determinant of the affine, so they
   are right for anisotropic and oblique grids. Components are 26-connected, the
   lesion definition of the study.
5. **Meshes.** Brain: largest component of the mask with holes filled. Each
   label 1–4 separately. Marching cubes at 0.5 on a padded crop → world mm →
   Taubin (λ 0.5, μ −0.53; 20 passes for the brain, 10 for labels) → quadric
   decimation (80k / 40k faces) → `.glb`. Face winding is corrected for the
   affine's handedness so normals point outward.
6. **Volumes for the viewer.** `labels` (uint8), `confidence` (max softmax ×
   255, uint8), `t1c`, `flair` (int16 with a scale factor). Fortran order,
   gzip. A `.npy` copy is memory-mapped by the `/voxel` endpoint.

Stages reported by `GET /api/jobs/{id}`: `uploading → queued → preprocessing →
inference → meshing → done`, or `error` with a message.

### Jobs

One folder per job under `GLIOMA_DATA_DIR/jobs/<32-hex id>/`:

```
job.json                     status, stage, progress, warnings
input/{t1,t1c,t2,flair}.nii.gz   sanitised uploads
output/segmentation.nii.gz
output/summary.json
output/meshes/{brain,label_1..4}.glb
output/viewer/{labels,confidence,t1c,flair}.{json,bin.gz,npy}
```

A single daemon thread takes job ids from a queue, so only one inference runs
at a time. A second thread deletes jobs older than `JOB_TTL_HOURS` (24) every
ten minutes. Jobs interrupted by a restart are marked as errors. Job ids are
validated against `^[0-9a-f]{32}$` before touching the filesystem.

### Operating modes

| `GET /api/info` → `mode` | When | Behaviour |
|---|---|---|
| `inference` | valid weights in `NNUNET_MODEL_DIR` and torch + nnunetv2 installed | normal |
| `demo` | no `NNUNET_MODEL_DIR`, or no weights in it | a segmentation must be uploaded; UI shows "DEMO — no model inference" |
| `model_error` | weights present but violating the contract, or the inference stack missing | job submission returns 503; **no fallback** to demo or another model |

In demo mode only, `POST /api/demo/sample` starts a job on a synthetic case
generated by `app/sample_case.py` (procedural shapes, no patient data). It goes
through the same validation and pipeline as an upload and its summary carries
`synthetic_sample: true`, which the viewer shows as a chip. The endpoint
returns 409 once a model is loaded, so generated shapes are never passed off as
a test of the model.

### Patient data

- Client file names are used only to detect the format. Files are stored by
  slot name under a random job id.
- NIfTI headers are rebuilt from the voxel data and affine; `descrip`,
  `db_name`, `aux_file`, `intent_name` and header extensions are dropped.
- DICOM files are extracted with flattened names, converted without a BIDS
  sidecar, and deleted in a `finally` block.
- Nothing logs file names, headers or tool output. Access logs show job ids.
- Image voxels themselves are kept for up to 24 h and can be identifying (for
  example a face in a non-skull-stripped scan). The app is a research
  prototype and has no authentication; run it only on a trusted machine or
  network.

## Frontend (`frontend/src`)

```
lib/geometry.ts   voxel ↔ world (the only place)
lib/volumes.ts    binary volumes, world lookup, voxel-mode instance lists
lib/picking.ts    ray marching through the label volume
lib/api.ts        typed API client
lib/files.ts      matching dropped files to sequences by file name
state/viewerStore.ts   crosshair, hover, layers, clip plane (zustand)
components/Scene3D.tsx     react-three-fiber scene
components/SliceView.tsx   one canvas slice view
components/ViewChrome.tsx  cursor read-out, enlarge button
components/Panels.tsx      voxel info, regions table, display controls, result
pages/{Upload,Viewer,About}.tsx
```

- **One crosshair.** `viewerStore.crosshair` is a world-mm point. Each slice
  view shows the slice containing it; the 3D view draws a marker at it. A click
  anywhere writes it.
- **One lookup.** `lookupWorld(world, volumes)` produces the info-box content
  for both 2D and 3D, from volumes held in the browser. Every click is also
  sent to `GET /voxel` and the two answers are compared; the info box shows
  "same voxel and label" or a mismatch.
- **Slices** are drawn with the canvas 2D API (not Niivue), one pixel per voxel,
  scaled by voxel spacing so the aspect ratio is physical. Radiological
  convention: patient right on screen left. Slices follow the voxel axes of the
  RAS viewer grid; for an oblique acquisition they are the acquisition planes,
  not resliced to world axes.
- **3D picking** does not depend on the meshes: the mouse ray is marched
  through the label volume (Amanatides–Woo) to find the first visible tumor
  voxel, so mesh mode and voxel mode pick identically and exactly. If no tumor
  voxel is hit, the brain-surface intersection is used.
- **Voxel mode** draws one `InstancedMesh` per label. "Visible shell only"
  (default) skips voxels whose six face-neighbours share their label; they are
  hidden anyway, and it cuts the cube count by an order of magnitude.

## Tests

| Requirement | Where |
|---|---|
| Alignment: oblique, anisotropic affine, known cube → mesh at the expected world mm; `world_to_voxel(voxel_to_world(ijk)) == ijk` | `backend/tests/test_meshing.py`, `test_geometry.py` |
| Hover/click round trip in the backend | `backend/tests/test_lookup_roundtrip.py` |
| Hover/click round trip in the frontend, same probes | `frontend/src/lib/geometry.test.ts` (reads `backend/tests/fixtures/alignment_case.json`) |
| API, jobs, validation, PHI, expiry | `backend/tests/test_api.py` |
| Registration, reorientation, DICOM | `backend/tests/test_preprocess.py` |
| nnU-Net loading and prediction | `backend/tests/test_inference.py` (tiny random model in the study's layouts; axis handling compared with nnU-Net's own NIfTI export) |
| End to end on a real case | `backend/scripts/e2e_case.py` (needs weights and a MU-Glioma-Post case) |
