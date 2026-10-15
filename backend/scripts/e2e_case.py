"""End-to-end test on one MU-Glioma-Post case.

Uploads the four sequences to a running backend (inference mode), waits for the
job, downloads the segmentation and compares it with the reference mask using
the study's overlap definitions.

    python scripts/e2e_case.py --case-dir "D:/MU-Glioma-Post/PatientID_0005/Timepoint_1"

The case folder must hold ``*_brain_t1n.nii.gz``, ``*_brain_t1c.nii.gz``,
``*_brain_t2w.nii.gz``, ``*_brain_t2f.nii.gz`` and ``*_tumorMask.nii.gz``.
Nothing is written except an optional ``--report`` JSON.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import time
from pathlib import Path

import httpx
import nibabel as nib
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
ROOT = Path(__file__).resolve().parents[1]
SUFFIXES = {"t1": "_brain_t1n", "t1c": "_brain_t1c", "t2": "_brain_t2w", "flair": "_brain_t2f"}
TARGETS = {
    "Non-enhancing tumor core (1)": (1,),
    "FLAIR hyperintensity / edema (2)": (2,),
    "Enhancing tissue (3)": (3,),
    "Resection cavity (4)": (4,),
    "Tumor-related region (1-3)  <- paper's primary target": (1, 2, 3),
    "All postoperative regions (1-4)": (1, 2, 3, 4),
}
PAPER = {"mean": 0.879, "ci": (0.856, 0.900), "fold_means": {1: 0.895, 2: 0.900, 3: 0.893, 4: 0.835, 5: 0.871}}


def find_case_files(case_dir: Path) -> dict[str, Path]:
    files = {}
    for slot, suffix in SUFFIXES.items():
        matches = sorted(case_dir.glob(f"*{suffix}.nii.gz")) + sorted(case_dir.glob(f"*{suffix}.nii"))
        if len(matches) != 1:
            raise SystemExit(f"Expected exactly one *{suffix}.nii.gz in {case_dir}, found {len(matches)}.")
        files[slot] = matches[0]
    masks = sorted(case_dir.glob("*_tumorMask.nii.gz")) + sorted(case_dir.glob("*_tumorMask.nii"))
    if len(masks) != 1:
        raise SystemExit(f"Expected exactly one *_tumorMask.nii.gz in {case_dir}, found {len(masks)}.")
    files["mask"] = masks[0]
    return files


def overlap(truth: np.ndarray, prediction: np.ndarray) -> dict:
    """Study rules: both empty → NaN (excluded); reference empty, prediction present → Dice 0."""
    tp = int(np.logical_and(prediction, truth).sum())
    fp = int(np.logical_and(prediction, ~truth).sum())
    fn = int(np.logical_and(~prediction, truth).sum())
    if not truth.any() and not prediction.any():
        dice = float("nan")
    else:
        dice = 2 * tp / (2 * tp + fp + fn)
    return {"dice": dice, "tp": tp, "fp": fp, "fn": fn, "reference_present": bool(truth.any())}


def fold_of_patient(patient_id: str | None) -> tuple[int, str] | None:
    if patient_id is None:
        return None
    table = json.loads((ROOT / "app" / "resources" / "outer_test_patients.json").read_text())
    for fold, entry in table["folds"].items():
        if patient_id in entry["outer_test_patients"]:
            return int(fold), entry["primary_run"]
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout-minutes", type=float, default=180.0)
    parser.add_argument("--report", type=Path, help="write the results as JSON here")
    parser.add_argument("--keep-job", action="store_true", help="do not delete the job afterwards")
    args = parser.parse_args()

    files = find_case_files(args.case_dir)
    match = re.search(r"PatientID_\d+", str(args.case_dir.resolve()))
    patient_id = match.group(0) if match else None

    with httpx.Client(base_url=args.api, timeout=600.0) as client:
        info = client.get("/api/info").json()
        if info["mode"] != "inference":
            raise SystemExit(
                f"Backend is in '{info['mode']}' mode ({info['message']}). The end-to-end test needs "
                "model inference: set NNUNET_MODEL_DIR and restart the backend."
            )
        members = [m["name"] for m in info["model"]["members"]]
        print(f"Backend model: {info['message']}")
        print(f"Members: {', '.join(members)}")

        held_out = fold_of_patient(patient_id)
        leakage_note = None
        if held_out is None:
            print("Patient id not found in the study's splits; cannot tell whether this case was used in training.")
        else:
            fold, run = held_out
            print(f"{patient_id} is an outer-test patient of fold {fold}: only run {run} never saw it.")
            if len(members) > 1 or not any(run in name for name in members):
                leakage_note = (
                    f"At least one loaded model was trained on {patient_id}. This Dice is optimistically "
                    f"biased and is not comparable to the paper. For a fair check, point NNUNET_MODEL_DIR "
                    f"at run {run} only."
                )
                print("WARNING:", leakage_note)

        handles = {slot: open(files[slot], "rb") for slot in SUFFIXES}
        try:
            upload = {slot: (f"{slot}.nii.gz", handle, "application/gzip") for slot, handle in handles.items()}
            started = time.time()
            response = client.post("/api/jobs", files=upload, data={"already_preprocessed": "true"})
        finally:
            for handle in handles.values():
                handle.close()
        response.raise_for_status()
        job_id = response.json()["job_id"]

        last = None
        deadline = time.time() + args.timeout_minutes * 60
        while True:
            try:
                state = client.get(f"/api/jobs/{job_id}").json()
            except httpx.TransportError:
                time.sleep(3)  # the server is busy with the job; ask again
                continue
            line =f"{state['status']:<14} {state['progress']:>5.0%}  {state['message']}"
            if line != last:
                print(line)
                last = line
            if state["status"] in ("done", "error"):
                break
            if time.time() > deadline:
                raise SystemExit("Timed out waiting for the job.")
            time.sleep(3)
        if state["status"] == "error":
            raise SystemExit(f"Job failed: {state['error']}")
        elapsed = time.time() - started

        summary = client.get(f"/api/jobs/{job_id}/summary").json()
        with tempfile.TemporaryDirectory() as scratch:
            target = Path(scratch) / "prediction.nii.gz"
            target.write_bytes(client.get(f"/api/jobs/{job_id}/download/segmentation.nii.gz").content)
            prediction_image = nib.load(str(target))
            prediction = np.asanyarray(prediction_image.dataobj).astype(np.uint8)
            prediction_affine = prediction_image.affine
        if not args.keep_job:
            client.delete(f"/api/jobs/{job_id}")

    truth_image = nib.load(str(files["mask"]))
    truth = np.asanyarray(truth_image.dataobj)
    if prediction.shape != truth.shape or not np.allclose(prediction_affine, truth_image.affine, atol=1e-3):
        raise SystemExit(
            f"FAIL: prediction grid {prediction.shape} does not match the reference mask grid {truth.shape} "
            "(the downloaded mask must be in the uploaded T1c space)."
        )
    truth = np.rint(truth).astype(np.uint8)

    print(f"\nGrid check: prediction and reference share shape {truth.shape} and affine. OK")
    print(f"Wall time: {elapsed / 60:.1f} min\n")
    print(f"{'Target':<58}{'Dice':>8}{'ref voxels':>12}{'pred voxels':>13}")
    results = {}
    for name, labels in TARGETS.items():
        metrics = overlap(np.isin(truth, labels), np.isin(prediction, labels))
        results[name] = metrics
        dice = "   n/a" if np.isnan(metrics["dice"]) else f"{metrics['dice']:.4f}"
        print(f"{name:<58}{dice:>8}{metrics['tp'] + metrics['fn']:>12}{metrics['tp'] + metrics['fp']:>13}")

    primary = results["Tumor-related region (1-3)  <- paper's primary target"]["dice"]
    print(
        f"\nPaper (203 patients, out-of-fold): mean Dice {PAPER['mean']:.3f} "
        f"(95% CI {PAPER['ci'][0]:.3f}-{PAPER['ci'][1]:.3f}) for the tumor-related region."
    )
    if held_out is not None:
        print(f"Fold {held_out[0]} mean in the paper: {PAPER['fold_means'][held_out[0]]:.3f}.")
    print(
        "A single case is one draw from a wide per-patient distribution (per-fold SD 0.13-0.17), "
        "so 'in line with the paper' means a plausible value, not equality with the mean."
    )
    if leakage_note:
        print("WARNING:", leakage_note)

    if args.report:
        args.report.write_text(
            json.dumps(
                {
                    "case_dir": args.case_dir.name,
                    "patient_id": patient_id,
                    "outer_test_fold": held_out[0] if held_out else None,
                    "out_of_fold_run": held_out[1] if held_out else None,
                    "leakage_warning": leakage_note,
                    "model": summary["model"],
                    "wall_seconds": elapsed,
                    "primary_dice_tumor_related_1_3": primary,
                    "targets": results,
                    "paper": PAPER,
                },
                indent=2,
            )
        )
        print(f"Report written to {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
