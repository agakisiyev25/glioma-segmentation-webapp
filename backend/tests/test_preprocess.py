"""Reorientation, rigid co-registration and DICOM conversion."""

import gzip
import zipfile

import nibabel as nib
import numpy as np
import pytest
import SimpleITK as sitk
from scipy import ndimage

from app.geometry import voxel_to_world, world_to_voxel
from app.io_utils import InputError, dicom_zip_to_nifti, load_volume, save_nifti
from app.preprocess import axcodes_of, reorient, reorient_like, rigid_register_to

from . import synthetic


@pytest.mark.parametrize("target", ["RAS", "LPS", "PIR", "SAL"])
def test_reorient_keeps_every_voxel_at_its_world_coordinate(target):
    affine = synthetic.oblique_affine()
    data = np.arange(np.prod(synthetic.SHAPE), dtype=np.float32).reshape(synthetic.SHAPE)
    moved, new_affine = reorient(data, affine, target)
    assert axcodes_of(new_affine) == target

    rng = np.random.default_rng(0)
    for _ in range(50):
        ijk = rng.integers(0, np.array(moved.shape))
        world = voxel_to_world(ijk, new_affine)
        original = np.rint(np.linalg.inv(affine) @ np.append(world, 1.0))[:3].astype(int)
        assert moved[tuple(ijk)] == data[tuple(original)]

    back = reorient_like(moved, new_affine, affine, data.shape)
    assert np.array_equal(back, data)


def _structured_image(shape=(48, 56, 44)) -> np.ndarray:
    image = np.zeros(shape, np.float32)
    grid = np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), axis=-1).astype(float)
    centre = (np.array(shape) - 1) / 2.0
    image[(((grid - centre) / (np.array(shape) * 0.42)) ** 2).sum(-1) <= 1.0] = 300.0
    image[14:22, 18:30, 12:20] = 700.0
    image[28:36, 30:40, 24:34] = 120.0
    image[20:26, 12:18, 28:36] = 520.0
    return ndimage.gaussian_filter(image, 1.0)


def test_rigid_registration_recovers_a_known_misalignment(tmp_path):
    fixed = _structured_image()
    fixed_affine = np.diag([1.0, 1.0, 1.2, 1.0])
    fixed_affine[:3, 3] = [-24.0, -28.0, -26.0]
    save_nifti(fixed, fixed_affine, tmp_path / "fixed.nii.gz")

    # Moving image: same anatomy, different contrast, stored with a header
    # that is rotated 6 degrees and shifted 4-5 mm from the truth.
    moving = np.where(fixed > 0, 1000.0 - fixed, 0.0).astype(np.float32)
    angle = np.deg2rad(6.0)
    wrong = np.eye(4)
    wrong[:3, :3] = [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    wrong[:3, 3] = [4.0, -5.0, 3.0]
    save_nifti(moving, wrong @ fixed_affine, tmp_path / "moving.nii.gz")

    def resampled_without_registration():
        f = sitk.ReadImage(str(tmp_path / "fixed.nii.gz"), sitk.sitkFloat32)
        m = sitk.ReadImage(str(tmp_path / "moving.nii.gz"), sitk.sitkFloat32)
        return sitk.GetArrayFromImage(sitk.Resample(m, f, sitk.Transform(), sitk.sitkLinear, 0.0)).transpose(2, 1, 0)

    def similarity(candidate):  # contrast is inverted, so use |correlation| inside the head
        mask = fixed > 50
        return abs(np.corrcoef(candidate[mask], moving[mask])[0, 1])

    before = similarity(resampled_without_registration())
    rigid_register_to(tmp_path / "fixed.nii.gz", tmp_path / "moving.nii.gz", tmp_path / "out.nii.gz")
    registered, registered_affine = load_volume(tmp_path / "out.nii.gz", "moving")

    assert registered.shape == fixed.shape and np.allclose(registered_affine, fixed_affine, atol=1e-4)
    after = similarity(registered)
    assert before < 0.8, before
    assert after > 0.97, (before, after)


def _write_dicom_series(directory, volume_zyx, spacing_xyz, origin):
    directory.mkdir(parents=True)
    image = sitk.GetImageFromArray(volume_zyx.astype(np.int16))
    image.SetSpacing(spacing_xyz)
    image.SetOrigin(origin)
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()
    series_uid = "1.2.826.0.1.3680043.2.1125.1.99999"
    direction = image.GetDirection()
    for index in range(image.GetDepth()):
        slice_image = image[:, :, index]
        tags = {
            "0008|0060": "MR",
            # Fixed study date/time: SimpleITK otherwise stamps each slice with
            # its own write time and dcm2niix treats them as separate studies.
            "0008|0020": "20260101",
            "0008|0030": "120000",
            "0018|5100": "HFS",
            "0010|0010": "DOE^JOHN",
            "0010|0020": "MRN123456",
            "0010|0030": "19610203",
            "0020|000e": series_uid,
            "0020|000d": series_uid + ".1",
            "0008|103e": "AX T1 POST DOE",
            "0020|0037": "\\".join(str(v) for v in (direction[0], direction[3], direction[6], direction[1], direction[4], direction[7])),
            "0020|0032": "\\".join(str(v) for v in image.TransformIndexToPhysicalPoint((0, 0, index))),
            "0020|0013": str(index + 1),
            "0018|0050": str(spacing_xyz[2]),
        }
        for tag, value in tags.items():
            slice_image.SetMetaData(tag, value)
        writer.SetFileName(str(directory / f"DOE_JOHN_{index:03d}.dcm"))
        writer.Execute(slice_image)


def test_dicom_zip_is_converted_and_identifiers_are_dropped(tmp_path):
    rng = np.random.default_rng(1)
    volume = (rng.random((20, 32, 28)) * 800).astype(np.int16)  # z, y, x
    spacing, origin = (0.9, 1.1, 3.0), (-12.0, 20.0, -31.0)
    _write_dicom_series(tmp_path / "series", volume, spacing, origin)
    archive = tmp_path / "upload.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        for path in sorted((tmp_path / "series").iterdir()):
            zipped.write(path, arcname=f"DOE_JOHN/{path.name}")

    work = tmp_path / "work"
    converted = dicom_zip_to_nifti(archive, work, "T1c")
    clean = tmp_path / "t1c.nii.gz"
    data, affine = load_volume(converted, "T1c")
    save_nifti(data, affine, clean)

    assert not (work / "dicom").exists(), "extracted DICOM files must be deleted"
    image = nib.load(str(clean))
    assert sorted(image.shape) == sorted(volume.shape)
    assert np.allclose(sorted(image.header.get_zooms()), sorted(spacing), atol=1e-3)
    assert np.isclose(float(np.asanyarray(image.dataobj).sum()), float(volume.sum()), rtol=1e-6)
    # Geometry: every DICOM voxel must be found at its own world position.
    # DICOM patient coordinates are LPS; NIfTI world is RAS → x and y flip sign.
    converted_data = np.asanyarray(image.dataobj)
    for _ in range(40):
        z, y, x = (int(rng.integers(0, n)) for n in volume.shape)
        lps = np.array(origin) + np.array(spacing) * np.array([x, y, z])
        ras = np.array([-lps[0], -lps[1], lps[2]])
        index = world_to_voxel(ras, image.affine, round_to_index=True)
        assert np.allclose(world_to_voxel(ras, image.affine), index, atol=1e-3)  # lands on a voxel centre
        assert converted_data[tuple(index)] == volume[z, y, x]

    assert b"DOE" not in gzip.decompress(clean.read_bytes())[:400] and "DOE" not in clean.name


def test_bad_zips_are_rejected(tmp_path):
    not_zip = tmp_path / "x.zip"
    not_zip.write_bytes(b"nope")
    with pytest.raises(InputError):
        dicom_zip_to_nifti(not_zip, tmp_path / "w1", "T1")

    empty = tmp_path / "empty.zip"
    with zipfile.ZipFile(empty, "w"):
        pass
    with pytest.raises(InputError):
        dicom_zip_to_nifti(empty, tmp_path / "w2", "T1")

    junk = tmp_path / "junk.zip"
    with zipfile.ZipFile(junk, "w") as zipped:
        zipped.writestr("../../evil.txt", "not dicom")
    with pytest.raises(InputError):
        dicom_zip_to_nifti(junk, tmp_path / "w3", "T1")
    assert not (tmp_path / "evil.txt").exists() and not (tmp_path.parent / "evil.txt").exists()
