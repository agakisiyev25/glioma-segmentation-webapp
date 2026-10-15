"""One job, start to finish: preprocess → segment → measure → mesh → export."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Callable

import nibabel as nib
import numpy as np
from nibabel import processing as nib_processing

from . import __version__
from .config import Settings
from .geometry import voxel_spacing
from .io_utils import InputError, load_segmentation, load_volume, same_grid, save_nifti
from .jobs import JobState
from .labels import (
    CONFIDENCE_NOTE,
    DISCLAIMER,
    FOREGROUND_LABELS,
    LABEL_BY_VALUE,
    SCOPE_NOTE,
    SEQUENCE_DISPLAY,
    SEQUENCES,
    label_map_json,
)
from .meshing import clean_brain_mask, export_glb, mask_to_mesh
from .postprocess import label_statistics
from .preprocess import (
    axcodes_of,
    brain_mask_from_nonzero,
    reorient,
    reorient_like,
    rigid_register_to,
    skull_strip_hd_bet,
)
from .volumes import write_volume

log = logging.getLogger(__name__)

Report = Callable[[str, float, str], None]

VIEWER_ORIENTATION = "RAS"


def _prepare_preprocessed(input_dir: Path, warnings: list[str]):
    """Uploads that are already co-registered and skull-stripped: pass through untouched."""
    volumes, affines = {}, {}
    for slot in SEQUENCES:
        volumes[slot], affines[slot] = load_volume(input_dir / f"{slot}.nii.gz", SEQUENCE_DISPLAY[slot])
    for slot in SEQUENCES:
        if not same_grid(volumes[slot].shape, affines[slot], volumes["t1c"].shape, affines["t1c"]):
            raise InputError(
                f"{SEQUENCE_DISPLAY[slot]} is not on the same voxel grid as T1c "
                f"(shape {tuple(volumes[slot].shape)} vs {tuple(volumes['t1c'].shape)}, or a different affine). "
                "Untick 'Already preprocessed' to co-register the sequences."
            )
    zero_fraction = float((volumes["t1c"] == 0).mean())
    if zero_fraction < 0.2:
        warnings.append(
            f"T1c has only {zero_fraction:.0%} zero-valued background, so it does not look "
            "skull-stripped. The model was trained on skull-stripped images and the 3D surface "
            "will be the whole image block rather than the brain. Untick 'Already preprocessed' "
            "for raw scans."
        )
    brain_mask = brain_mask_from_nonzero(volumes["t1c"])
    nnunet_inputs = [input_dir / f"{slot}.nii.gz" for slot in SEQUENCES]
    return volumes, affines["t1c"], brain_mask, nnunet_inputs, affines["t1c"]


def _prepare_raw(input_dir: Path, work_dir: Path, settings: Settings, device: str, report: Report, warnings: list[str]):
    """Raw scans: rigid co-registration to T1c, HD-BET skull strip, training orientation."""
    t1c, affine = load_volume(input_dir / "t1c.nii.gz", "T1c")
    volumes = {"t1c": t1c}
    moving = [slot for slot in SEQUENCES if slot != "t1c"]
    for index, slot in enumerate(moving):
        report("preprocessing", 0.08 + 0.12 * index / len(moving), f"Co-registering {SEQUENCE_DISPLAY[slot]} to T1c")
        registered = work_dir / "coreg" / f"{slot}.nii.gz"
        rigid_register_to(input_dir / "t1c.nii.gz", input_dir / f"{slot}.nii.gz", registered)
        data, registered_affine = load_volume(registered, SEQUENCE_DISPLAY[slot])
        if not same_grid(data.shape, registered_affine, t1c.shape, affine):
            raise RuntimeError("Co-registration output is not on the T1c grid.")
        volumes[slot] = data

    report("preprocessing", 0.22, "Skull stripping (HD-BET)")
    brain_mask = skull_strip_hd_bet(input_dir / "t1c.nii.gz", work_dir / "bet", device)

    nnunet_inputs, nnunet_affine = [], affine
    for slot in SEQUENCES:
        stripped = np.where(brain_mask, volumes[slot], 0.0).astype(np.float32)
        oriented, nnunet_affine = reorient(stripped, affine, settings.training_orientation)
        path = work_dir / "nnunet_in" / f"{slot}.nii.gz"
        save_nifti(oriented, nnunet_affine, path)
        nnunet_inputs.append(path)
    warnings.append(
        "Raw-scan preprocessing (rigid co-registration to T1c, HD-BET skull stripping) only "
        "approximates the MU-Glioma-Post preprocessing the model was trained on and has not been "
        "validated against reference masks."
    )
    return volumes, affine, brain_mask, nnunet_inputs, nnunet_affine


def _demo_labels(input_dir: Path, shape, affine, warnings: list[str]) -> np.ndarray:
    path = input_dir / "segmentation.nii.gz"
    if not path.exists():
        raise InputError("Demo mode needs an existing segmentation NIfTI (labels 0–4) alongside the four sequences.")
    labels, seg_affine = load_segmentation(path, set(LABEL_BY_VALUE))
    if same_grid(labels.shape, seg_affine, shape, affine):
        return labels
    # Different grid: nearest-neighbour resampling in world space onto T1c.
    resampled = nib_processing.resample_from_to(
        nib.Nifti1Image(labels, seg_affine), (tuple(shape), affine), order=0, mode="constant", cval=0
    )
    warnings.append(
        "The uploaded segmentation was on a different voxel grid from T1c and was resampled "
        "onto the T1c grid with nearest-neighbour interpolation."
    )
    return np.asarray(resampled.dataobj).astype(np.uint8)


def run_pipeline(state: JobState, job_dir: Path, settings: Settings, model, report: Report) -> None:
    started = time.time()
    input_dir, work_dir, output_dir = job_dir / "input", job_dir / "work", job_dir / "output"
    warnings = state.warnings
    demo = state.mode == "demo"
    device = model.device if model is not None else "cpu"

    # ------------------------------------------------------------ preprocess
    report("preprocessing", 0.03, "Checking uploaded volumes")
    if state.already_preprocessed:
        volumes, native_affine, brain_mask, nnunet_inputs, nnunet_affine = _prepare_preprocessed(input_dir, warnings)
    else:
        volumes, native_affine, brain_mask, nnunet_inputs, nnunet_affine = _prepare_raw(
            input_dir, work_dir, settings, device, report, warnings
        )
    native_shape = volumes["t1c"].shape

    # ------------------------------------------------------------- inference
    confidence_native = None
    model_info: dict
    if demo:
        report("inference", 0.30, "DEMO — no model inference; using the uploaded segmentation")
        labels_native = _demo_labels(input_dir, native_shape, native_affine, warnings)
        model_info = {
            "mode": "demo",
            "description": "DEMO — no model inference. The segmentation shown is the file that was uploaded.",
        }
    else:
        report("inference", 0.30, f"Running nnU-Net on {device.upper()}")
        if device == "cpu":
            warnings.append("No GPU was available: inference ran on CPU, which is much slower.")

        def inference_progress(fraction: float, message: str) -> None:
            report("inference", 0.30 + 0.45 * fraction, message)

        result = model.predict(nnunet_inputs, inference_progress)
        warnings.extend(result.warnings)
        labels_native = reorient_like(result.labels, nnunet_affine, native_affine, native_shape)
        confidence_native = reorient_like(result.confidence, nnunet_affine, native_affine, native_shape)
        model_info = {"mode": "inference", **result.info}

    # Mask in the T1c space the user uploaded (same shape and affine).
    output_dir.mkdir(parents=True, exist_ok=True)
    save_nifti(labels_native.astype(np.uint8), native_affine, output_dir / "segmentation.nii.gz")

    # --------------------------------------------------- viewer grid (RAS)
    report("meshing", 0.78, "Measuring regions")
    labels, affine = reorient(labels_native, native_affine, VIEWER_ORIENTATION)
    brain = clean_brain_mask(reorient(brain_mask, native_affine, VIEWER_ORIENTATION)[0])
    t1c = reorient(volumes["t1c"], native_affine, VIEWER_ORIENTATION)[0]
    flair = reorient(volumes["flair"], native_affine, VIEWER_ORIENTATION)[0]

    statistics = label_statistics(labels, affine)

    # ----------------------------------------------------------------- meshes
    meshes = {}
    mesh_dir = output_dir / "meshes"
    report("meshing", 0.82, "Building the brain surface")
    brain_mesh = mask_to_mesh(brain, affine, smooth_iterations=20, target_faces=settings.brain_mesh_faces)
    if brain_mesh is not None:
        meshes["brain"] = export_glb(brain_mesh, mesh_dir / "brain.glb", name="brain")
    for index, info in enumerate(FOREGROUND_LABELS):
        report("meshing", 0.86 + 0.02 * index, f"Building the {info.short} surface")
        mesh = mask_to_mesh(labels == info.value, affine, smooth_iterations=10, target_faces=settings.label_mesh_faces)
        if mesh is not None:
            meshes[f"label_{info.value}"] = export_glb(
                mesh, mesh_dir / f"label_{info.value}.glb", name=f"label_{info.value}"
            )

    # ---------------------------------------------------------------- volumes
    report("meshing", 0.95, "Writing voxel volumes")
    viewer_dir = output_dir / "viewer"
    header = write_volume(viewer_dir, "labels", labels, affine, kind="labels")
    write_volume(viewer_dir, "t1c", t1c, affine, kind="intensity")
    write_volume(viewer_dir, "flair", flair, affine, kind="intensity")
    if confidence_native is not None:
        confidence = reorient(confidence_native, native_affine, VIEWER_ORIENTATION)[0]
        write_volume(viewer_dir, "confidence", confidence, affine, kind="confidence")

    summary = {
        "job_id": state.id,
        "mode": state.mode,
        "demo": demo,
        "synthetic_sample": state.source == "sample",
        "banner": "DEMO — no model inference" if demo else None,
        "disclaimer": DISCLAIMER,
        "scope_note": SCOPE_NOTE,
        "confidence_note": CONFIDENCE_NOTE if not demo else "No model confidence in demo mode.",
        "confidence_available": confidence_native is not None,
        "already_preprocessed": state.already_preprocessed,
        "label_map": label_map_json(),
        "model": model_info,
        "statistics": statistics,
        "meshes": meshes,
        "volumes_available": [
            name for name in ("labels", "confidence", "t1c", "flair") if (viewer_dir / f"{name}.json").exists()
        ],
        "grid": {
            "viewer": {
                "shape": header["shape"],
                "affine": header["affine"],
                "inverse_affine": header["inverse_affine"],
                "spacing_mm": header["spacing_mm"],
                "axcodes": header["axcodes"],
            },
            "uploaded_t1c": {
                "shape": [int(n) for n in native_shape],
                "affine": np.asarray(native_affine).tolist(),
                "spacing_mm": list(voxel_spacing(native_affine)),
                "axcodes": axcodes_of(native_affine),
            },
            "world_frame": "RAS+ mm (NIfTI scanner/aligned space)",
            "note": (
                "The viewer grid is the uploaded T1c grid with axes reordered to RAS. "
                "No resampling; every voxel keeps its world coordinate."
            ),
        },
        "warnings": warnings,
        "runtime_seconds": round(time.time() - started, 1),
        "app_version": __version__,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
