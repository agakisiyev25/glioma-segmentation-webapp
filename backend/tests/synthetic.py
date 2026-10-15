"""Synthetic cases with an oblique, anisotropic affine and known label boxes."""

from __future__ import annotations

import numpy as np

SHAPE = (40, 48, 36)
SPACING = (0.9, 1.2, 2.5)  # anisotropic
ROTATION_DEG = (17.0, -11.0, 23.0)  # oblique about x, y, z
TRANSLATION = (-71.3, 42.8, -15.6)

# label value -> inclusive voxel box (lo, hi)
BOXES = {
    3: ((10, 20, 8), (17, 29, 13)),
    2: ((22, 10, 20), (30, 18, 27)),
    4: ((5, 34, 25), (9, 40, 30)),
}


def rotation_matrix(degrees=ROTATION_DEG) -> np.ndarray:
    rx, ry, rz = np.deg2rad(degrees)
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return z @ y @ x


def oblique_affine(*, left_handed: bool = False) -> np.ndarray:
    """Rotation x anisotropic scaling + translation (orthogonal axes, as real scanners produce)."""
    scale = np.diag(SPACING)
    if left_handed:
        scale = scale @ np.diag([-1.0, 1.0, 1.0])
    affine = np.eye(4)
    affine[:3, :3] = rotation_matrix() @ scale
    affine[:3, 3] = TRANSLATION
    return affine


def sheared_affine() -> np.ndarray:
    """Adds shear; only for pure-geometry tests (SimpleITK rejects sheared NIfTI)."""
    shear = np.eye(4)
    shear[0, 1], shear[1, 2] = 0.15, -0.08
    return oblique_affine() @ shear


def label_volume() -> np.ndarray:
    labels = np.zeros(SHAPE, dtype=np.uint8)
    for value, (lo, hi) in BOXES.items():
        labels[lo[0] : hi[0] + 1, lo[1] : hi[1] + 1, lo[2] : hi[2] + 1] = value
    return labels


def brain_volume(seed: int = 0) -> np.ndarray:
    """A skull-stripped-like image: positive inside an ellipsoid, zero outside."""
    grid = np.stack(np.meshgrid(*[np.arange(n) for n in SHAPE], indexing="ij"), axis=-1).astype(np.float64)
    centre = (np.array(SHAPE) - 1) / 2.0
    radii = np.array(SHAPE) * 0.46
    inside = (((grid - centre) / radii) ** 2).sum(axis=-1) <= 1.0
    rng = np.random.default_rng(seed)
    image = np.clip(400.0 + 60.0 * rng.standard_normal(SHAPE), 50.0, None)
    return np.where(inside, image, 0.0).astype(np.float32)
