"""The single place where voxel indices and world coordinates are converted.

World frame: NIfTI scanner/aligned space in millimetres, RAS+ (x → patient
Right, y → Anterior, z → Superior). Voxel indices ``(i, j, k)`` address voxel
*centres*, as in NIfTI.

Everything in the backend that needs a coordinate conversion — mesh vertices,
bounding boxes, centroids, the voxel lookup endpoint — goes through
:func:`voxel_to_world` and :func:`world_to_voxel`. Nothing else multiplies by
an affine.
"""

from __future__ import annotations

import numpy as np


def _as_affine(affine) -> np.ndarray:
    matrix = np.asarray(affine, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"affine must be 4x4, got {matrix.shape}")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0]):
        raise ValueError("affine bottom row must be [0, 0, 0, 1]")
    return matrix


def voxel_to_world(ijk, affine) -> np.ndarray:
    """Map voxel indices to world mm (RAS).

    ``ijk`` may be a single point ``(3,)`` or an array ``(..., 3)``; fractional
    indices are allowed. Returns float64 with the same leading shape.
    """
    matrix = _as_affine(affine)
    points = np.asarray(ijk, dtype=np.float64)
    if points.shape[-1] != 3:
        raise ValueError(f"ijk must have a trailing dimension of 3, got {points.shape}")
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def world_to_voxel(xyz, affine, *, round_to_index: bool = False) -> np.ndarray:
    """Map world mm (RAS) to voxel indices using the inverse affine.

    Returns continuous float64 indices by default. With ``round_to_index`` the
    result is the index of the voxel whose cell contains the point (round half
    away from the lower voxel, i.e. ``floor(x + 0.5)``), as int64.
    """
    matrix = _as_affine(affine)
    points = np.asarray(xyz, dtype=np.float64)
    if points.shape[-1] != 3:
        raise ValueError(f"xyz must have a trailing dimension of 3, got {points.shape}")
    inverse = np.linalg.inv(matrix)
    voxels = points @ inverse[:3, :3].T + inverse[:3, 3]
    if round_to_index:
        return np.floor(voxels + 0.5).astype(np.int64)
    return voxels


def inverse_affine(affine) -> np.ndarray:
    """Inverse of a voxel→world affine (world→voxel), for shipping to the frontend."""
    return np.linalg.inv(_as_affine(affine))


def voxel_spacing(affine) -> tuple[float, float, float]:
    """Voxel edge lengths in mm along i, j, k (column norms of the affine)."""
    matrix = _as_affine(affine)
    return tuple(float(v) for v in np.linalg.norm(matrix[:3, :3], axis=0))


def voxel_volume_mm3(affine) -> float:
    """Volume of one voxel in mm³. Exact for sheared/oblique affines too."""
    return float(abs(np.linalg.det(_as_affine(affine)[:3, :3])))


def is_in_bounds(ijk, shape) -> bool:
    index = np.asarray(ijk)
    return bool(np.all(index >= 0) and np.all(index < np.asarray(shape[:3])))


def is_left_handed(affine) -> bool:
    """True when the voxel→world mapping flips handedness (negative determinant)."""
    return bool(np.linalg.det(_as_affine(affine)[:3, :3]) < 0)
