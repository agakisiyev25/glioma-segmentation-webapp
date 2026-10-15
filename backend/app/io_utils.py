"""Reading, validating and de-identifying uploads.

Nothing here logs file names, headers or pixel data: uploads are referred to
only by their sequence slot (t1, t1c, t2, flair, segmentation).
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np

from .geometry import voxel_spacing

log = logging.getLogger(__name__)

MIN_DIM, MAX_DIM = 16, 1024
MIN_SPACING_MM, MAX_SPACING_MM = 0.2, 10.0
MAX_DICOM_FILES = 5000
MAX_DICOM_UNCOMPRESSED_BYTES = 4 * 1024**3


class InputError(ValueError):
    """A problem with what the user uploaded. The message is shown in the UI."""


def kind_of_upload(filename: str) -> str:
    lower = (filename or "").lower()
    if lower.endswith(".nii.gz") or lower.endswith(".nii"):
        return "nifti"
    if lower.endswith(".zip"):
        return "dicom_zip"
    raise InputError("Unsupported file type. Upload .nii, .nii.gz, or a .zip of one DICOM series.")


def nifti_suffix(filename: str) -> str:
    return ".nii.gz" if filename.lower().endswith(".nii.gz") else ".nii"


def load_volume(path: Path, slot: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a NIfTI as (float32 array, affine) and validate it."""
    try:
        image = nib.load(str(path))
        data = np.asanyarray(image.dataobj)
    except Exception as error:
        raise InputError(f"{slot}: not a readable NIfTI file.") from error

    # Accept trailing singleton dimensions (e.g. 240x240x155x1).
    while data.ndim > 3 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise InputError(f"{slot}: expected a 3D volume, got {data.ndim}D with shape {tuple(data.shape)}.")
    if any(size < MIN_DIM or size > MAX_DIM for size in data.shape):
        raise InputError(
            f"{slot}: implausible matrix size {tuple(data.shape)} (each axis must be {MIN_DIM}–{MAX_DIM} voxels)."
        )

    affine = np.asarray(image.affine, dtype=np.float64)
    if not np.all(np.isfinite(affine)) or abs(np.linalg.det(affine[:3, :3])) < 1e-9:
        raise InputError(f"{slot}: the NIfTI affine is singular or not finite.")
    spacing = voxel_spacing(affine)
    if any(value < MIN_SPACING_MM or value > MAX_SPACING_MM for value in spacing):
        raise InputError(
            f"{slot}: implausible voxel spacing {tuple(round(v, 3) for v in spacing)} mm "
            f"(expected {MIN_SPACING_MM}–{MAX_SPACING_MM} mm)."
        )

    data = np.asarray(data, dtype=np.float32)
    if not np.all(np.isfinite(data)):
        raise InputError(f"{slot}: the volume contains NaN or infinite values.")
    return data, affine


def save_nifti(data: np.ndarray, affine: np.ndarray, path: Path) -> None:
    """Write a NIfTI with a freshly built header.

    Only the voxel data, the affine and the data type survive. Free-text header
    fields (descrip, db_name, aux_file, intent_name) and header extensions —
    where identifiers can hide — are never copied. qform and sform are both set
    so nibabel and SimpleITK read the same geometry.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(np.ascontiguousarray(data), affine)
    image.set_sform(affine, code=1)
    image.set_qform(affine, code=1)
    image.header.set_xyzt_units("mm")
    nib.save(image, str(path))


def sanitize_nifti(source: Path, destination: Path, slot: str) -> None:
    data, affine = load_volume(source, slot)
    save_nifti(data, affine, destination)


def load_segmentation(path: Path, allowed: set[int]) -> tuple[np.ndarray, np.ndarray]:
    try:
        image = nib.load(str(path))
        raw = np.asanyarray(image.dataobj)
    except Exception as error:
        raise InputError("segmentation: not a readable NIfTI file.") from error
    while raw.ndim > 3 and raw.shape[-1] == 1:
        raw = raw[..., 0]
    if raw.ndim != 3:
        raise InputError(f"segmentation: expected a 3D volume, got shape {tuple(raw.shape)}.")
    if not np.all(np.isfinite(raw)) or not np.all(raw == np.rint(raw)):
        raise InputError("segmentation: values must be integer labels.")
    observed = {int(value) for value in np.unique(raw)}
    if not observed.issubset(allowed):
        raise InputError(
            f"segmentation: unexpected label values {sorted(observed - allowed)}; "
            f"expected only {sorted(allowed)}."
        )
    return raw.astype(np.uint8), np.asarray(image.affine, dtype=np.float64)


def sanitize_segmentation(source: Path, destination: Path, allowed: set[int]) -> None:
    labels, affine = load_segmentation(source, allowed)
    save_nifti(labels, affine, destination)


def same_grid(shape_a, affine_a, shape_b, affine_b, *, atol: float = 1e-3) -> bool:
    return tuple(shape_a) == tuple(shape_b) and np.allclose(affine_a, affine_b, atol=atol)


# --------------------------------------------------------------------------- DICOM


def _safe_extract(archive_path: Path, target: Path) -> int:
    """Extract a zip without following paths outside ``target``."""
    target.mkdir(parents=True, exist_ok=True)
    root = target.resolve()
    extracted = 0
    try:
        with zipfile.ZipFile(archive_path) as archive:
            members = [m for m in archive.infolist() if not m.is_dir()]
            if len(members) > MAX_DICOM_FILES:
                raise InputError(f"DICOM zip has more than {MAX_DICOM_FILES} files; upload one series per zip.")
            if sum(m.file_size for m in members) > MAX_DICOM_UNCOMPRESSED_BYTES:
                raise InputError("DICOM zip is too large when uncompressed.")
            for index, member in enumerate(members):
                # Flatten: member names can carry patient names and are never used.
                destination = (root / f"{index:05d}.dcm").resolve()
                if root not in destination.parents:
                    raise InputError("DICOM zip contains an unsafe path.")
                with archive.open(member) as source, open(destination, "wb") as sink:
                    shutil.copyfileobj(source, sink)
                extracted += 1
    except zipfile.BadZipFile as error:
        raise InputError("The uploaded .zip file is not a valid zip archive.") from error
    if extracted == 0:
        raise InputError("The uploaded .zip file is empty.")
    return extracted


def _dcm2niix_binary() -> str | None:
    found = shutil.which("dcm2niix")
    if found:
        return found
    try:
        import dcm2niix  # pip package that bundles the binary

        return str(dcm2niix.bin)
    except Exception:
        return None


def _convert_with_dcm2niix(dicom_dir: Path, output_dir: Path) -> Path | None:
    binary = _dcm2niix_binary()
    if binary is None:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    # -b n: no BIDS sidecar (it would carry acquisition metadata);
    # -f: fixed output name so nothing from the DICOM header reaches a filename.
    command = [binary, "-z", "y", "-b", "n", "-f", "series_%s", "-o", str(output_dir), str(dicom_dir)]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=600)
    outputs = sorted(output_dir.glob("*.nii.gz"), key=lambda p: p.stat().st_size, reverse=True)
    if completed.returncode != 0 or not outputs:
        # stdout/stderr may contain header content — do not log it.
        log.warning("dcm2niix failed (exit code %s); trying the SimpleITK reader", completed.returncode)
        return None
    if len(outputs) > 1:
        raise InputError(
            f"The DICOM zip produced {len(outputs)} volumes. Upload exactly one series per sequence."
        )
    return outputs[0]


def _convert_with_simpleitk(dicom_dir: Path, output_path: Path) -> Path:
    import SimpleITK as sitk

    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(dicom_dir))
    if not series_ids:
        raise InputError("No DICOM series found in the uploaded zip.")
    if len(series_ids) > 1:
        raise InputError(
            f"The DICOM zip contains {len(series_ids)} series. Upload exactly one series per sequence."
        )
    reader.SetFileNames(reader.GetGDCMSeriesFileNames(str(dicom_dir), series_ids[0]))
    image = reader.Execute()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(output_path))
    return output_path


def dicom_zip_to_nifti(archive_path: Path, work_dir: Path, slot: str) -> Path:
    """Convert a zip holding one DICOM series to NIfTI.

    The extracted DICOM files (which contain patient identifiers) are deleted
    before returning, whether or not conversion succeeds.
    """
    dicom_dir = work_dir / "dicom"
    converted_dir = work_dir / "converted"
    try:
        count = _safe_extract(archive_path, dicom_dir)
        log.info("%s: extracted %d DICOM files", slot, count)
        result = _convert_with_dcm2niix(dicom_dir, converted_dir)
        if result is None:
            result = _convert_with_simpleitk(dicom_dir, converted_dir / "series.nii.gz")
        return result
    except InputError as error:
        raise InputError(f"{slot}: {error}") from error
    except subprocess.TimeoutExpired as error:
        raise InputError(f"{slot}: DICOM conversion timed out.") from error
    finally:
        shutil.rmtree(dicom_dir, ignore_errors=True)
