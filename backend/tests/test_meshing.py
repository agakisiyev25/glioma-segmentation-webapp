"""Alignment test: mesh vertices must land at the expected world mm."""

import numpy as np
import pytest
import trimesh

from app.geometry import voxel_to_world, voxel_volume_mm3, world_to_voxel
from app.meshing import clean_brain_mask, export_glb, marching_cubes_world, mask_to_mesh

from . import synthetic

AFFINES = {
    "oblique": synthetic.oblique_affine(),
    "oblique-left-handed": synthetic.oblique_affine(left_handed=True),
}


def _box(label):
    lo, hi = synthetic.BOXES[label]
    return np.array(lo, dtype=float), np.array(hi, dtype=float)


@pytest.mark.parametrize("name", AFFINES)
@pytest.mark.parametrize("label", synthetic.BOXES)
def test_raw_surface_lies_on_the_box_faces(name, label):
    """Unsmoothed marching cubes: vertices map back to exactly the voxel-face box."""
    affine = AFFINES[name]
    lo, hi = _box(label)
    mesh = marching_cubes_world(synthetic.label_volume() == label, affine)

    back = world_to_voxel(mesh.vertices, affine)
    assert np.allclose(back.min(axis=0), lo - 0.5, atol=1e-6)
    assert np.allclose(back.max(axis=0), hi + 0.5, atol=1e-6)

    # Each of the eight expected world corners is within half a voxel diagonal
    # of some mesh vertex (marching cubes chamfers the corners).
    corners = np.array(
        [
            [x, y, z]
            for x in (lo[0] - 0.5, hi[0] + 0.5)
            for y in (lo[1] - 0.5, hi[1] + 0.5)
            for z in (lo[2] - 0.5, hi[2] + 0.5)
        ]
    )
    half_diagonal = 0.5 * np.linalg.norm(synthetic.SPACING)
    for corner in voxel_to_world(corners, affine):
        assert np.linalg.norm(mesh.vertices - corner, axis=1).min() <= half_diagonal + 1e-6


@pytest.mark.parametrize("name", AFFINES)
def test_raw_surface_is_closed_outward_and_volume_correct(name):
    affine = AFFINES[name]
    lo, hi = _box(3)
    mesh = marching_cubes_world(synthetic.label_volume() == 3, affine)
    assert mesh.is_watertight
    # Positive signed volume == outward-facing normals (also for a mirrored affine).
    assert mesh.volume > 0
    expected = np.prod(hi - lo + 1) * voxel_volume_mm3(affine)
    assert abs(mesh.volume - expected) / expected < 0.05


@pytest.mark.parametrize("name", AFFINES)
def test_smoothed_mesh_stays_aligned(name):
    affine = AFFINES[name]
    lo, hi = _box(3)
    mesh = mask_to_mesh(synthetic.label_volume() == 3, affine, smooth_iterations=10, target_faces=400)
    assert len(mesh.faces) <= 400

    expected_centre = voxel_to_world((lo + hi) / 2.0, affine)
    assert np.linalg.norm(mesh.center_mass - expected_centre) < 0.3  # mm

    back = world_to_voxel(mesh.vertices, affine)
    assert np.all(back.min(axis=0) >= lo - 0.75)
    assert np.all(back.max(axis=0) <= hi + 0.75)
    assert np.all(back.min(axis=0) <= lo + 0.75)  # still reaches the faces
    assert np.all(back.max(axis=0) >= hi - 0.75)

    expected_volume = np.prod(hi - lo + 1) * voxel_volume_mm3(affine)
    assert mesh.volume > 0
    assert abs(mesh.volume - expected_volume) / expected_volume < 0.12


def test_mesh_is_not_in_voxel_index_space():
    """Guards against exporting voxel indices as coordinates."""
    mesh = marching_cubes_world(synthetic.label_volume() == 3, synthetic.oblique_affine())
    lo, hi = _box(3)
    assert np.linalg.norm(mesh.centroid - (lo + hi) / 2.0) > 20.0


def test_empty_mask_gives_no_mesh():
    assert mask_to_mesh(np.zeros(synthetic.SHAPE, bool), synthetic.oblique_affine()) is None


def test_mask_touching_the_border_is_closed():
    mask = np.zeros(synthetic.SHAPE, bool)
    mask[0:5, 0:5, 0:5] = True
    mesh = marching_cubes_world(mask, synthetic.oblique_affine())
    assert mesh.is_watertight
    back = world_to_voxel(mesh.vertices, synthetic.oblique_affine())
    assert np.allclose(back.min(axis=0), -0.5, atol=1e-6)


def test_glb_round_trip_keeps_world_coordinates(tmp_path):
    affine = synthetic.oblique_affine()
    mesh = mask_to_mesh(synthetic.label_volume() == 2, affine, smooth_iterations=10, target_faces=600)
    stats = export_glb(mesh, tmp_path / "label_2.glb", name="label_2")
    loaded = trimesh.load(tmp_path / "label_2.glb", force="mesh", process=False)
    assert stats["faces"] == len(loaded.faces)
    assert np.allclose(loaded.bounds, mesh.bounds, atol=1e-3)  # float32 in the file
    assert np.allclose(np.sort(loaded.vertices, axis=0), np.sort(mesh.vertices, axis=0), atol=1e-3)


def test_vectorised_laplacian_equals_trimesh_operator():
    from trimesh.smoothing import laplacian_calculation

    from app.meshing import _umbrella_laplacian

    mesh = marching_cubes_world(synthetic.brain_volume() != 0, synthetic.oblique_affine())
    assert abs(_umbrella_laplacian(mesh) - laplacian_calculation(mesh).tocsr()).max() < 1e-12


def test_oversized_surface_is_reduced_before_smoothing(monkeypatch):
    import app.meshing as meshing

    monkeypatch.setattr(meshing, "MAX_FACES_BEFORE_SMOOTHING", 2000)
    mesh = mask_to_mesh(synthetic.brain_volume() != 0, synthetic.oblique_affine(), smooth_iterations=10, target_faces=1000)
    assert len(mesh.faces) <= 1000 and mesh.volume > 0
    expected_centre = voxel_to_world((np.array(synthetic.SHAPE) - 1) / 2.0, synthetic.oblique_affine())
    assert np.linalg.norm(mesh.center_mass - expected_centre) < 1.0


def test_brain_mask_cleanup_fills_holes_and_drops_specks():
    brain = synthetic.brain_volume() != 0
    brain[18:22, 22:26, 16:20] = False  # internal hole
    brain[0, 0, 0] = True  # isolated speck
    cleaned = clean_brain_mask(brain)
    assert cleaned[19, 23, 17]
    assert not cleaned[0, 0, 0]
    mesh = mask_to_mesh(cleaned, synthetic.oblique_affine(), smooth_iterations=20, target_faces=3000)
    assert mesh.is_watertight and mesh.volume > 0
    assert len(mesh.split(only_watertight=False)) == 1
