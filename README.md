# Postoperative glioma segmentation — web app

Upload a patient's four MRI sequences, segment them with the nnU-Net v2
`3d_fullres` pipeline from our study, and inspect the result on a 3D model of
that patient's own brain and on 2D slices.

> **Research prototype. Trained and internally validated on the MU-Glioma-Post
> dataset only. Not a medical device, not for diagnosis or treatment
> decisions.**
>
> The model segments tumor-related sub-regions and the resection cavity. It does
> not classify tumor type or grade.

Study: *Nested Patient-Grouped Benchmarking of 3D Deep-Learning Pipelines for
Multiclass Postoperative Glioma Segmentation on Longitudinal MRI* —
<https://github.com/glioma-ml-research/mu-glioma-segmentation-benchmark>

- [docs/MODEL_CONTRACT.md](docs/MODEL_CONTRACT.md) — label map, channel order,
  weights layout, and where the research repo differs from the original brief.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — pipeline, coordinate rule, API,
  tests.

## Quick start (Windows, PowerShell)

Needs Python 3.11 and Node 20.

```powershell
# backend
cd backend
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m app
```

```powershell
# frontend (second terminal)
cd frontend
npm install
npm run dev
```

Open <http://localhost:5173>. With no weights configured the app starts in
**demo mode**.

On Linux/macOS use `python3.11 -m venv .venv` and `.venv/bin/python`.

### Single process

`npm run build` in `frontend/` writes `frontend/dist`. If that folder exists the
backend serves it, so the whole app is at <http://localhost:8000> with no Node
process.

## Demo mode — no model needed

Demo mode is what you get when `NNUNET_MODEL_DIR` is unset or holds no weights.
You upload the four sequences **plus an existing segmentation** (labels 0–4).
No inference runs; everything after it (measurements, meshes, slices, voxel
lookups, download) works the same. The UI is labelled "DEMO — no model
inference" and shows no confidence values.

**Fastest way to see it:** on the upload page, click **Open the synthetic
sample**. The server generates a non-patient case (shapes, not anatomy) with all
four regions and a small second lesion, and opens it in the viewer in a few
seconds. The button exists only in demo mode; the viewer marks the result
"Synthetic sample · not a patient".

To exercise the upload path with the same case as files:

```powershell
cd backend
.venv\Scripts\python scripts\make_sample_case.py     # → backend\data\sample_case\
```

Drop all five files on the upload card at once. They are matched to sequences by
file name (`t1`/`t1n`, `t1c`/`t1ce`, `t2`/`t2w`, `flair`/`t2f`, `seg`/`mask`);
anything that cannot be matched is listed so you can drop it on its slot.

## Running the real model

### 1. Install the inference stack

Same versions as the study. Install PyTorch **and torchvision** first, from the
PyTorch index:

```powershell
cd backend
# NVIDIA GPU (CUDA 12.6):
.venv\Scripts\python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
# or CPU only:
# .venv\Scripts\python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu

.venv\Scripts\python -m pip install -c constraints.txt -r requirements-inference.txt
```

Do not skip the torchvision pin: nnU-Net pulls in `timm`, which pulls in an
unpinned torchvision, and pip then replaces torch with a newer CPU-only build
without saying so.

### 2. Put the weights in place

The weights are not in any repository. They are inside the study's validated
result archives, under `model/`. See [models/README.md](models/README.md):

```
models/
  no29/model/checkpoint_best.pth
  no29/model/plans.json
  no29/model/dataset.json
  no30/model/…          (optional: more runs → probability-averaged ensemble)
```

### 3. Start the backend

If the weights are in the project's `models/` folder, nothing else is needed:
the backend loads from there when `NNUNET_MODEL_DIR` is not set. To keep them
elsewhere, set the variable:

```powershell
$env:NNUNET_MODEL_DIR = "C:\path\to\models"
.venv\Scripts\python -m app
```

`GET http://localhost:8000/api/info` reports `mode`:

- `inference` — weights loaded;
- `demo` — no weights found;
- `model_error` — weights found but they are not the study's model (wrong
  channel order or label map, unknown trainer) or PyTorch/nnU-Net is missing.
  The app refuses jobs in this state. It never substitutes another model.

Check a weights folder without starting the server:

```powershell
.venv\Scripts\python scripts\inspect_contract.py --model-dir C:\path\to\models
```

### Which model to load

The study did not train one model with five folds. It trained five independent
nnU-Nets (runs no29–no33), one per outer fold, each with its own plans. The
paper's Dice of 0.879 is a pool of their single-model, out-of-fold predictions.

- **One run** — closest to what the paper evaluated.
- **Several runs** — averaged probabilities. Sensible for a new patient, but an
  ensemble was not evaluated in the paper.

### Runtime

A GPU is detected automatically; without one the app runs on CPU and warns that
it is slow. In the study, inference took about 3 s per case on an A40 and 12 s
on a P100. Expect minutes on a small laptop GPU and considerably longer on CPU.
If the GPU runs out of memory the job retries on CPU instead of failing.

## Uploads

- Four sequences: **T1, T1c, T2, FLAIR**, each `.nii` / `.nii.gz`, or a `.zip`
  with one DICOM series.
- **Already preprocessed (MU-Glioma-Post format)** — ticked by default. Use it
  for skull-stripped sequences that share one voxel grid. They go to the model
  untouched.
- Unticked: T1/T2/FLAIR are rigidly co-registered to T1c and the brain is
  extracted with HD-BET (`pip install -c constraints.txt hd-bet`; it downloads
  its own weights on first use). This approximates the dataset's preprocessing
  and **has not been validated against reference masks**.

Jobs are stored one folder each under `backend/data/jobs/` and deleted after
24 hours. Headers are stripped of free-text fields on arrival; see
ARCHITECTURE.md → Patient data. There is no login: run this on a machine or
network you trust.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `NNUNET_MODEL_DIR` | `models/` if it exists, else demo mode | folder with the model weights |
| `NNUNET_CHECKPOINT` | `checkpoint_best.pth`, then `checkpoint_final.pth` | force a checkpoint file name |
| `NNUNET_DEVICE` | `auto` | `auto`, `cuda` or `cpu` |
| `NNUNET_DISABLE_TTA` | `0` | `1` turns off mirroring (faster; the study used it) |
| `TRAINING_ORIENTATION` | `LPS` | axis order given to the model for raw uploads |
| `GLIOMA_DATA_DIR` | `backend/data` | job storage |
| `GLIOMA_HOST` / `GLIOMA_PORT` | `127.0.0.1` / `8000` | where `python -m app` listens |
| `JOB_TTL_HOURS` | `24` | auto-delete age |
| `MAX_UPLOAD_MB` | `2048` | per-file limit |
| `BRAIN_MESH_FACES` / `LABEL_MESH_FACES` | `80000` / `40000` | decimation targets |
| `CORS_ORIGINS` | `http://localhost:5173,…` | only needed when not using the dev proxy |

## API

| Method and path | Purpose |
|---|---|
| `GET /api/info` | mode, model info, label map, disclaimer |
| `POST /api/jobs` | multipart `t1`, `t1c`, `t2`, `flair`, optional `segmentation`, `already_preprocessed` → `job_id` |
| `GET /api/model/metrics` | the study's published nnU-Net results (per-region Dice with CI, HD95, surface Dice, lesion sensitivity, false alarms, per fold), from `app/resources/study_metrics.json` |
| `POST /api/demo/sample` | demo mode only: start a job on the built-in synthetic case → `job_id` (409 when a model is loaded) |
| `GET /api/jobs/{id}` | status, stage, progress, message |
| `GET /api/jobs/{id}/summary` | volumes (mL), components, boxes, centroids, label map, model info, disclaimer |
| `GET /api/jobs/{id}/meshes/{name}.glb` | `brain`, `label_1` … `label_4`; vertices in world mm |
| `GET /api/jobs/{id}/volumes/{name}` | `labels`, `confidence`, `t1c`, `flair`: gzip-encoded raw voxels, header in `X-Volume-Header` |
| `GET /api/jobs/{id}/volumes/{name}/header` | the same header as JSON (shape, affine, inverse affine, spacing, label map) |
| `GET /api/jobs/{id}/voxel?x=&y=&z=` | world mm → voxel, label, confidence, intensities |
| `GET /api/jobs/{id}/download/segmentation.nii.gz` | mask in the uploaded T1c space |
| `DELETE /api/jobs/{id}` | delete the job and its files |

Interactive docs: <http://localhost:8000/docs>.

## Tests

```powershell
cd backend
.venv\Scripts\python -m pytest            # 59 tests; the 11 inference tests skip without torch + nnunetv2

cd ..\frontend
npm test                                  # 15 tests: geometry, lookup and picking against the shared fixture; file-name matching
npm run build                             # type-check + production build
```

### End-to-end on a MU-Glioma-Post case

Needs the weights and one case folder from the dataset. With the backend
running in inference mode:

```powershell
cd backend
.venv\Scripts\python scripts\e2e_case.py --case-dir "D:\MU-Glioma-Post\PatientID_0005\Timepoint_1"
```

It uploads the four sequences, downloads the mask, checks it is on the
reference grid, and prints Dice for the six targets of the study next to the
paper's numbers. It also tells you which run never saw that patient: every
dataset patient was a training patient for four of the five runs, so a Dice
computed with any other run (or an ensemble) is optimistic. `PatientID_0005` is
an outer-test patient of fold 1 → load run **no29** alone for a fair check.

Before the first real run, confirm the two facts the research repo does not
record (stored orientation of the dataset files and the planned spacing):

```powershell
.venv\Scripts\python scripts\inspect_contract.py --case-dir "D:\MU-Glioma-Post\PatientID_0005\Timepoint_1" --model-dir C:\path\to\models
```

## Permanent public address (Tailscale Funnel)

The app is published at **https://glioma.tail7d9208.ts.net** through Tailscale
Funnel (free; Tailscale machine name `glioma`).

- Double-click **`start-server.bat`** to run the app on that address. Keep the
  window open and the PC awake (Settings → System → Power: sleep "Never").
- The address never changes. Funnel is saved by Tailscale and comes back after
  a reboot, but the app itself only runs while `start-server.bat` is open.
- Turn public access off: `tailscale funnel --https=443 off`. Turn it back on:
  run `start-server.bat`, which runs `tailscale funnel --bg 8000`.
- There is no login: anyone with the address can upload and view results.
  Don't use it with real, identifiable patient scans.

## Temporary public link (Cloudflare quick tunnel)

Double-click **`share.bat`** in the project folder, or run `powershell -ExecutionPolicy Bypass -File share.ps1`.
It builds the website if needed, starts the backend (which serves the website
and runs the model on this PC's GPU), and opens a free Cloudflare quick tunnel.
The link, `https://<random words>.trycloudflare.com`, is printed in the window.

- Keep that window open and the PC awake (Settings → System → Power: sleep
  "Never" while sharing). Closing the window ends the link.
- The link changes every time the script starts. A permanent address needs a
  free Cloudflare account plus your own domain (a "named tunnel").
- There is no login: anyone with the link can upload and view results. Don't
  share it publicly with real, identifiable patient scans.
- Jobs run one at a time. With several users, uploads wait in the queue.
- Cloudflare's free tunnel rejects requests over ~100 MB. The upload page gzips
  `.nii` files in the browser first, so four sequences fit comfortably.
- Install the tunnel client once with `winget install --id Cloudflare.cloudflared`.

## Docker

```bash
docker compose up --build                                              # CPU
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build   # NVIDIA GPU
```

The app is at <http://localhost:8080>. Weights are read from `./models`, or
from the host path in `NNUNET_MODEL_DIR`. Build arguments: `INSTALL_INFERENCE=0`
for a small demo-only image, `INSTALL_HDBET=1` to add skull stripping.

## Viewer controls

| Where | Action |
|---|---|
| 3D | drag to orbit, right-drag to pan, wheel to zoom; **A P R L S I** buttons look from that side; **Reset** |
| 3D and slices | point at a voxel for a read-out at the cursor (label, coordinates, confidence); click to move the crosshair there |
| Slices | wheel or slider to change slice; click/drag to move the crosshair; right-drag for window/level |
| Any view | double-click, or the corner button, to enlarge it; Esc to return to four views |
| Regions | one table: colour, volume (mL), lesion count, show/hide, 3D opacity. Click a region to jump to its largest lesion; click a lesion count above 1 to list and visit each lesion |
| Display | T1c / FLAIR, overlay opacity, window/level, **Voxel mode** (one cube per voxel), **Clip plane** (off by default) |
| Result | download the mask, delete the job, model and run details |

World coordinates are shown as (R, A, S) in millimetres. Slices use the
radiological convention: the patient's right is on the left of the screen.
