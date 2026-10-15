"""Orientation handling, rigid co-registration and skull stripping.

See docs/MODEL_CONTRACT.md §4.5 and §7 for why already-preprocessed uploads are
never reoriented before inference.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel import orientations as nio
from scipy import ndimage

from .io_utils import InputError

log = logging.getLogger(__name__)


# ------------------------------------------------------------------ orientation


def reorient(data: np.ndarray, affine: np.ndarray, axcodes: str) -> tuple[np.ndarray, np.ndarray]:
    """Permute/flip array axes so they run along ``axcodes`` (e.g. "RAS").

    The affine is updated with the array, so every voxel keeps its world
    coordinate. No interpolation takes place.
    """
    current = nio.io_orientation(affine)
    target = nio.axcodes2ornt(tuple(axcodes))
    transform = nio.ornt_transform(current, target)
    moved = nio.apply_orientation(data, transform)
    new_affine = affine @ nio.inv_ornt_aff(transform, data.shape)
    return np.ascontiguousarray(moved), new_affine


def axcodes_of(affine: np.ndarray) -> str:
    return "".join(nio.aff2axcodes(affine))


def reorient_like(
    data: np.ndarray, affine: np.ndarray, reference_affine: np.ndarray, reference_shape
) -> np.ndarray:
    """Bring an array back onto a reference grid that differs only by axis flips/permutations."""
    moved, new_affine = reorient(data, affine, axcodes_of(reference_affine))
    if tuple(moved.shape) != tuple(reference_shape) or not np.allclose(new_affine, reference_affine, atol=1e-3):
        raise RuntimeError("Reorientation did not land on the reference grid.")
    return moved


# ------------------------------------------------------------------- brain mask


def brain_mask_from_nonzero(volume: np.ndarray) -> np.ndarray:
    """Brain mask of a skull-stripped image: its non-zero region."""
    return np.asarray(volume) != 0


# --------------------------------------------------------------- registration


def rigid_register_to(fixed_path: Path, moving_path: Path, output_path: Path) -> None:
    """Rigidly align ``moving`` to ``fixed`` and resample it onto the fixed grid.

    SimpleITK, Mattes mutual information, three-level multi-resolution, linear
    interpolation. The fixed image (T1c) is never resampled.
    """
    import SimpleITK as sitk

    fixed = sitk.ReadImage(str(fixed_path), sitk.sitkFloat32)
    moving = sitk.ReadImage(str(moving_path), sitk.sitkFloat32)

    initial = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    method = sitk.ImageRegistrationMethod()
    method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    method.SetMetricSamplingStrategy(method.RANDOM)
    method.SetMetricSamplingPercentage(0.2, seed=2026)
    method.SetInterpolator(sitk.sitkLinear)
    method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0,
        minStep=1e-4,
        numberOfIterations=200,
        gradientMagnitudeTolerance=1e-8,
    )
    method.SetOptimizerScalesFromPhysicalShift()
    method.SetShrinkFactorsPerLevel([4, 2, 1])
    method.SetSmoothingSigmasPerLevel([2.0, 1.0, 0.0])
    method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    method.SetInitialTransform(initial, inPlace=False)
    transform = method.Execute(fixed, moving)

    resampled = sitk.Resample(moving, fixed, transform, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(resampled, str(output_path))
    log.info("registration finished: metric %.4f", method.GetMetricValue())


# -------------------------------------------------------------- skull stripping


def hd_bet_available() -> bool:
    return _hd_bet_command() is not None


def _hd_bet_command() -> list[str] | None:
    scripts_dir = Path(sys.executable).parent
    for name in ("hd-bet", "hd-bet.exe"):
        candidate = scripts_dir / name
        if candidate.exists():
            return [str(candidate)]
    found = shutil.which("hd-bet")
    return [found] if found else None


def skull_strip_hd_bet(image_path: Path, work_dir: Path, device: str) -> np.ndarray:
    """Run HD-BET on one image and return the brain mask as a boolean array.

    The mask is returned on the grid of ``image_path``.
    """
    command = _hd_bet_command()
    if command is None:
        raise InputError(
            "Skull stripping needs HD-BET, which is not installed on this server "
            "(pip install hd-bet). Upload skull-stripped, co-registered images and tick "
            "'Already preprocessed', or install HD-BET."
        )
    work_dir.mkdir(parents=True, exist_ok=True)
    output_path = work_dir / "bet.nii.gz"
    arguments = command + ["-i", str(image_path), "-o", str(output_path), "--save_bet_mask"]
    arguments += ["-device", "cuda" if device == "cuda" else "cpu"]
    if device != "cuda":
        arguments.append("--disable_tta")  # CPU runtime
    completed = subprocess.run(arguments, capture_output=True, text=True, timeout=3600)
    if completed.returncode != 0:
        raise RuntimeError(f"HD-BET failed with exit code {completed.returncode}.")

    candidates = [p for p in work_dir.glob("*.nii.gz") if p.name.endswith(("_bet.nii.gz", "_mask.nii.gz"))]
    if not candidates:
        raise RuntimeError("HD-BET did not write a brain mask.")
    mask_image = nib.load(str(candidates[0]))
    reference = nib.load(str(image_path))
    mask = np.asanyarray(mask_image.dataobj) > 0
    if mask.shape != reference.shape[:3] or not np.allclose(mask_image.affine, reference.affine, atol=1e-3):
        raise RuntimeError("HD-BET mask is not on the input grid.")
    if not mask.any():
        raise RuntimeError("HD-BET returned an empty brain mask.")
    return ndimage.binary_fill_holes(mask)
