"""Binary mask → surface mesh in world mm (RAS) → GLB.

Pipeline: marching cubes on the voxel grid → vertices mapped to world mm with
the NIfTI affine → Taubin smoothing (in mm, so it is isotropic regardless of
voxel anisotropy) → quadric decimation → GLB.

Vertices in the exported file are world millimetres. Voxel indices never leave
this module as coordinates.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import trimesh
from scipy import ndimage, sparse
from skimage import measure

from .geometry import is_left_handed, voxel_to_world

log = logging.getLogger(__name__)

MAX_FACES_BEFORE_SMOOTHING = 1_500_000


def marching_cubes_world(mask: np.ndarray, affine: np.ndarray) -> trimesh.Trimesh | None:
    """Iso-surface of a binary mask, vertices in world mm. No smoothing.

    The surface sits half a voxel outside the outermost voxel centres, i.e. on
    the voxel faces of the mask. Returns ``None`` for an empty mask.
    """
    binary = np.asarray(mask).astype(bool)
    if not binary.any():
        return None

    # Crop to the bounding box (speed) and pad by one voxel so the surface is
    # closed where the mask touches the volume border.
    box = ndimage.find_objects(binary.astype(np.uint8))[0]
    offset = np.array([axis.start for axis in box], dtype=np.float64)
    padded = np.pad(binary[box], 1, mode="constant", constant_values=False)

    vertices, faces, _, _ = measure.marching_cubes(
        padded.astype(np.float32), level=0.5, allow_degenerate=False
    )
    vertices_voxel = vertices - 1.0 + offset  # undo padding and crop
    vertices_world = voxel_to_world(vertices_voxel, affine)

    # For a mask (inside = high) skimage winds faces clockwise seen from
    # outside, i.e. inward normals in index space. Reverse them for a
    # right-handed affine; a left-handed affine mirrors the mesh, which already
    # reverses the winding. tests/test_meshing.py checks the signed volume.
    if not is_left_handed(affine):
        faces = faces[:, ::-1]

    return trimesh.Trimesh(vertices=vertices_world, faces=faces, process=False)


def _umbrella_laplacian(mesh: trimesh.Trimesh) -> sparse.csr_matrix:
    """Row-normalised vertex adjacency: each vertex → mean of its neighbours.

    The same equal-weight operator trimesh builds, but assembled with sparse
    matrices instead of per-vertex Python loops. That matters here: meshing
    runs in the worker thread, and long pure-Python loops starve the API
    thread of the interpreter while a job is running.
    """
    edges = mesh.edges_unique
    count = len(mesh.vertices)
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    columns = np.concatenate([edges[:, 1], edges[:, 0]])
    adjacency = sparse.coo_matrix((np.ones(len(rows)), (rows, columns)), shape=(count, count)).tocsr()
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    degree[degree == 0] = 1.0
    return (sparse.diags(1.0 / degree) @ adjacency).tocsr()


def _decimate(mesh: trimesh.Trimesh, face_count: int) -> trimesh.Trimesh:
    try:
        return mesh.simplify_quadric_decimation(face_count=face_count)
    except Exception as error:  # decimation is an optimisation, never fatal
        log.warning("Mesh decimation skipped: %r", error)
        return mesh


def smooth_and_decimate(
    mesh: trimesh.Trimesh, *, smooth_iterations: int, target_faces: int
) -> trimesh.Trimesh:
    result = mesh
    # Safety valve for pathological inputs (e.g. a noisy prediction scattered
    # through the whole volume): bring the mesh to a workable size first.
    if len(result.faces) > MAX_FACES_BEFORE_SMOOTHING:
        log.warning("Very large surface (%d faces); pre-decimating", len(result.faces))
        result = _decimate(result, MAX_FACES_BEFORE_SMOOTHING)
    if smooth_iterations > 0 and len(result.faces) >= 4:
        # Taubin alternates a shrinking and an inflating Laplacian step, so
        # unlike plain Laplacian smoothing it does not systematically shrink
        # the enclosed volume.
        result = trimesh.smoothing.filter_taubin(
            result.copy(),
            lamb=0.5,
            nu=0.53,
            iterations=smooth_iterations,
            laplacian_operator=_umbrella_laplacian(result),
        )
    if target_faces > 0 and len(result.faces) > target_faces:
        result = _decimate(result, target_faces)
    result.remove_unreferenced_vertices()
    return result


def mask_to_mesh(
    mask: np.ndarray,
    affine: np.ndarray,
    *,
    smooth_iterations: int = 10,
    target_faces: int = 40000,
) -> trimesh.Trimesh | None:
    mesh = marching_cubes_world(mask, affine)
    if mesh is None:
        return None
    return smooth_and_decimate(mesh, smooth_iterations=smooth_iterations, target_faces=target_faces)


def clean_brain_mask(mask: np.ndarray) -> np.ndarray:
    """Largest connected component with enclosed holes filled.

    Keeps the outer brain surface and drops internal surfaces (e.g. zero-valued
    voxels inside the brain) that would clutter a semi-transparent rendering.
    """
    binary = np.asarray(mask).astype(bool)
    if not binary.any():
        return binary
    labelled, count = ndimage.label(binary)
    if count > 1:
        sizes = ndimage.sum_labels(binary, labelled, index=np.arange(1, count + 1))
        binary = labelled == (int(np.argmax(sizes)) + 1)
    return ndimage.binary_fill_holes(binary)


def export_glb(mesh: trimesh.Trimesh, path: Path, *, name: str) -> dict:
    """Write a GLB whose vertex positions are world mm (RAS). Returns mesh stats."""
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.metadata["name"] = name
    scene = trimesh.Scene()
    scene.add_geometry(mesh, geom_name=name, node_name=name)
    path.write_bytes(scene.export(file_type="glb", include_normals=True))
    bounds = mesh.bounds
    return {
        "name": name,
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "bounds_world_mm": {"min": bounds[0].tolist(), "max": bounds[1].tolist()},
        "bytes": path.stat().st_size,
    }
