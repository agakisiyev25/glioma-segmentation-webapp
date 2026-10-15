"""A synthetic, non-patient sample case for demo mode.

The images are procedurally generated shapes: not anatomy, and not derived from
any person. The grid is deliberately awkward — slightly oblique, anisotropic,
stored LPS — so the viewer's coordinate handling is exercised.
"""

from __future__ import annotations

import threading
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

from .geometry import voxel_to_world
from .labels import SEQUENCES

SHAPE = (150, 180, 110)
SPACING = (1.0, 1.0, 1.3)
HEAD_CENTRE_MM = (2.0, -18.0, 12.0)
SLOTS = SEQUENCES + ("segmentation",)

_lock = threading.Lock()


def sample_affine() -> np.ndarray:
    """LPS-stored (as BraTS-style files are), tilted 8° about x and 5° about z."""
    ax, az = np.deg2rad(8.0), np.deg2rad(5.0)
    rot_x = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    rot_z = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    direction = rot_z @ rot_x @ np.diag([-1.0, -1.0, 1.0])  # i → Left, j → Posterior, k → Superior
    affine = np.eye(4)
    affine[:3, :3] = direction @ np.diag(SPACING)
    centre_voxel = (np.array(SHAPE) - 1) / 2.0
    affine[:3, 3] = np.array(HEAD_CENTRE_MM) - affine[:3, :3] @ centre_voxel
    return affine


def build(seed: int = 7) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """Return (affine, {t1, t1c, t2, flair}, labels)."""
    rng = np.random.default_rng(seed)
    affine = sample_affine()
    grid = np.stack(np.meshgrid(*[np.arange(n) for n in SHAPE], indexing="ij"), axis=-1).astype(np.float32)
    world = voxel_to_world(grid, affine).astype(np.float32)  # all shapes are defined in world mm
    centre = np.array(HEAD_CENTRE_MM, dtype=np.float32)

    def ellipsoid(c, radii):
        return (((world - np.asarray(c, np.float32)) / np.asarray(radii, np.float32)) ** 2).sum(-1)

    # "Brain": an ellipsoid with a longitudinal fissure and a wavy surface.
    theta = np.arctan2(world[..., 1] - centre[1], world[..., 0] - centre[0])
    wobble = 1.0 + 0.035 * np.sin(9 * theta) * np.cos(0.11 * (world[..., 2] - centre[2]))
    brain = ellipsoid(centre, (66, 82, 62)) <= wobble
    fissure = (np.abs(world[..., 0] - centre[0]) < 1.6) & (world[..., 2] > centre[2] + 18)
    brain &= ~fissure
    brain = ndimage.binary_opening(brain, iterations=1)

    # Regions, in world mm (right frontal lobe).
    tumor_centre = centre + np.array([28.0, 22.0, 14.0], dtype=np.float32)
    bumpy = ndimage.gaussian_filter(rng.standard_normal(SHAPE).astype(np.float32), 5) * 9.0
    edema = (ellipsoid(tumor_centre, (25, 29, 21)) + bumpy * 0.06) <= 1.0
    enhancing_outer = (ellipsoid(tumor_centre + [2, 1, 0], (13, 14, 12)) + bumpy * 0.05) <= 1.0
    core = ellipsoid(tumor_centre + [2, 1, 0], (8.5, 9.5, 8)) <= 1.0
    cavity = ellipsoid(tumor_centre + [-13, -19, -3], (9, 8, 9)) <= 1.0
    satellite = ellipsoid(centre + np.array([-31.0, -8.0, 26.0]), (4.2, 4.2, 4.2)) <= 1.0

    labels = np.zeros(SHAPE, np.uint8)
    labels[edema] = 2
    labels[enhancing_outer] = 3
    labels[core] = 1
    labels[cavity] = 4
    labels[satellite] = 3
    labels[~brain] = 0

    texture = ndimage.gaussian_filter(rng.standard_normal(SHAPE).astype(np.float32), 2.5)
    depth = np.clip(1.15 - ellipsoid(centre, (66, 82, 62)), 0, 1).astype(np.float32)

    def image(base, by_label, noise=28.0):
        data = base + 140.0 * depth * 0.35 + texture * 60.0 + rng.standard_normal(SHAPE).astype(np.float32) * noise
        for value, intensity in by_label.items():
            data[labels == value] = intensity + texture[labels == value] * 30.0
        data = ndimage.gaussian_filter(data, 0.7)
        return np.where(brain, np.clip(data, 20.0, None), 0.0).astype(np.float32)

    images = {
        "t1": image(520.0, {1: 300.0, 2: 440.0, 3: 500.0, 4: 180.0}),
        "t1c": image(540.0, {1: 310.0, 2: 450.0, 3: 980.0, 4: 190.0}),
        "t2": image(430.0, {1: 760.0, 2: 820.0, 3: 640.0, 4: 1050.0}),
        "flair": image(420.0, {1: 560.0, 2: 900.0, 3: 700.0, 4: 230.0}),
    }
    return affine, images, labels


def sample_paths(directory: Path) -> dict[str, Path]:
    return {slot: directory / f"synthetic_{slot}.nii.gz" for slot in SLOTS}


def ensure_sample_case(directory: Path) -> dict[str, Path]:
    """Write the sample case to ``directory`` if it is not already there."""
    paths = sample_paths(directory)
    with _lock:
        if all(path.is_file() and path.stat().st_size > 0 for path in paths.values()):
            return paths
        directory.mkdir(parents=True, exist_ok=True)
        affine, images, labels = build()
        for slot, data in images.items():
            nib.save(nib.Nifti1Image(data, affine), str(paths[slot]))
        nib.save(nib.Nifti1Image(labels, affine), str(paths["segmentation"]))
    return paths
