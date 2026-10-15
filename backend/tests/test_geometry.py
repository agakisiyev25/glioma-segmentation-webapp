import numpy as np
import pytest

from app.geometry import (
    inverse_affine,
    is_in_bounds,
    is_left_handed,
    voxel_spacing,
    voxel_to_world,
    voxel_volume_mm3,
    world_to_voxel,
)

from . import synthetic


@pytest.mark.parametrize(
    "affine",
    [synthetic.oblique_affine(), synthetic.oblique_affine(left_handed=True), synthetic.sheared_affine()],
    ids=["oblique", "oblique-left-handed", "sheared"],
)
def test_round_trip_every_voxel(affine):
    """world_to_voxel(voxel_to_world(ijk)) == ijk for all voxels of the grid."""
    ijk = np.stack(np.meshgrid(*[np.arange(n) for n in synthetic.SHAPE], indexing="ij"), axis=-1)
    world = voxel_to_world(ijk, affine)
    back = world_to_voxel(world, affine)
    assert np.abs(back - ijk).max() < 1e-9
    assert np.array_equal(world_to_voxel(world, affine, round_to_index=True), ijk)


def test_matches_explicit_matrix_product():
    affine = synthetic.oblique_affine()
    expected = (affine @ np.array([7, 31, 12, 1.0]))[:3]
    assert np.allclose(voxel_to_world([7, 31, 12], affine), expected)
    assert np.allclose(inverse_affine(affine) @ np.append(expected, 1.0), [7, 31, 12, 1.0])


def test_origin_and_axis_directions():
    affine = synthetic.oblique_affine()
    assert np.allclose(voxel_to_world([0, 0, 0], affine), synthetic.TRANSLATION)
    step = voxel_to_world([1, 0, 0], affine) - voxel_to_world([0, 0, 0], affine)
    assert np.isclose(np.linalg.norm(step), synthetic.SPACING[0])


def test_rounding_picks_the_containing_voxel():
    affine = synthetic.oblique_affine()
    centre = np.array([12.0, 22.0, 9.0])
    for offset in ([0.49, -0.49, 0.49], [-0.49, 0.49, -0.49], [0.0, 0.0, 0.0]):
        world = voxel_to_world(centre + offset, affine)
        assert world_to_voxel(world, affine, round_to_index=True).tolist() == [12, 22, 9]
    world = voxel_to_world(centre + [0.51, 0, 0], affine)
    assert world_to_voxel(world, affine, round_to_index=True).tolist() == [13, 22, 9]


def test_spacing_volume_handedness_bounds():
    affine = synthetic.oblique_affine()
    assert np.allclose(voxel_spacing(affine), synthetic.SPACING)
    assert np.isclose(voxel_volume_mm3(affine), np.prod(synthetic.SPACING))
    assert not is_left_handed(affine)
    assert is_left_handed(synthetic.oblique_affine(left_handed=True))
    assert is_in_bounds([0, 0, 0], synthetic.SHAPE)
    assert is_in_bounds([39, 47, 35], synthetic.SHAPE)
    assert not is_in_bounds([40, 0, 0], synthetic.SHAPE)
    assert not is_in_bounds([0, -1, 0], synthetic.SHAPE)


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        voxel_to_world([1, 2, 3], np.eye(3))
    with pytest.raises(ValueError):
        world_to_voxel([1, 2], np.eye(4))
