"""nnU-Net loading and prediction, exercised with a tiny randomly initialised model."""

import dataclasses
import json

import nibabel as nib
import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("nnunetv2")

from fastapi.testclient import TestClient  # noqa: E402

from app.config import load_settings  # noqa: E402
from app.inference import discover_members, resolve_model  # noqa: E402
from app.main import create_app  # noqa: E402

from . import synthetic, tiny_model  # noqa: E402
from .conftest import run_job, submit, write_case  # noqa: E402

pytestmark = pytest.mark.inference


@pytest.fixture()
def model_settings(tmp_path, monkeypatch):
    def make(model_dir):
        monkeypatch.setenv("NNUNET_MODEL_DIR", str(model_dir))
        monkeypatch.setenv("NNUNET_DEVICE", "cpu")
        monkeypatch.setenv("GLIOMA_DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setenv("FRONTEND_DIST", str(tmp_path / "no-frontend"))
        return dataclasses.replace(load_settings(), brain_mesh_faces=4000, label_mesh_faces=2000)

    return make


def test_discovery_of_all_three_layouts(tmp_path):
    archive = tiny_model.write_member(tmp_path / "a" / "model", seed=1, layout="archive")
    assert [m.name for m in discover_members(archive)] == ["."]
    assert discover_members(archive)[0].checkpoints[0].name == "checkpoint_best.pth"

    nnunet = tiny_model.write_member(
        tmp_path / "b" / "nnUNetTrainer_MUGlioma_no29__nnUNetPlans__3d_fullres", seed=1, layout="nnunet"
    )
    members = discover_members(nnunet)
    assert len(members) == 1 and members[0].checkpoints[0].parent.name == "fold_0"

    parent = tmp_path / "c"
    tiny_model.write_member(parent / "no29" / "model", seed=1)
    tiny_model.write_member(parent / "no30" / "model", seed=2)
    assert [m.name for m in discover_members(parent)] == ["no29/model", "no30/model"]

    assert discover_members(tmp_path / "a" / "model", "checkpoint_final.pth") == []


def test_best_checkpoint_is_preferred_over_final(tmp_path):
    directory = tiny_model.write_member(tmp_path / "m", seed=1, checkpoint_name="checkpoint_final.pth")
    assert discover_members(directory)[0].checkpoints[0].name == "checkpoint_final.pth"
    tiny_model.write_member(directory, seed=1, checkpoint_name="checkpoint_best.pth")
    assert discover_members(directory)[0].checkpoints[0].name == "checkpoint_best.pth"


def test_no_weights_means_demo_mode(tmp_path, model_settings):
    (tmp_path / "empty").mkdir()
    assert resolve_model(model_settings(tmp_path / "empty")).mode == "demo"
    assert resolve_model(model_settings(tmp_path / "missing")).mode == "demo"


def test_wrong_channel_order_is_refused_not_demoted_to_demo(tmp_path, model_settings):
    wrong = dict(tiny_model.DATASET_JSON, channel_names={"0": "T1CE", "1": "T1", "2": "T2", "3": "FLAIR"})
    directory = tiny_model.write_member(tmp_path / "m", seed=1, dataset_json=wrong)
    status = resolve_model(model_settings(directory))
    assert status.mode == "model_error" and "channel_names" in status.message

    with TestClient(create_app(model_settings(directory))) as client:
        assert client.get("/api/info").json()["mode"] == "model_error"
        paths = write_case(tmp_path / "case", synthetic.oblique_affine())
        assert submit(client, paths).status_code == 503  # even with a segmentation: no silent fallback


def test_wrong_label_map_is_refused(tmp_path, model_settings):
    wrong = dict(tiny_model.DATASET_JSON, labels={"background": 0, "tumor": 1})
    directory = tiny_model.write_member(tmp_path / "m", seed=1, dataset_json=wrong)
    assert resolve_model(model_settings(directory)).mode == "model_error"


def test_unknown_trainer_is_refused(tmp_path, model_settings):
    directory = tiny_model.write_member(tmp_path / "m", seed=1, trainer_name="SomeOtherTrainer")
    status = resolve_model(model_settings(directory))
    assert status.mode == "inference"  # trainer name is only known once the checkpoint is opened
    paths = write_case(tmp_path / "case", synthetic.oblique_affine())
    with pytest.raises(Exception, match="unknown trainer class"):
        status.model.predict([paths[s] for s in ("t1", "t1c", "t2", "flair")], lambda *_: None)


@pytest.mark.parametrize("left_handed", [False, True], ids=["RAS", "LAS"])
def test_axis_handling_matches_nnunets_own_nifti_export(tmp_path, model_settings, left_handed):
    """Our [z,y,x]→[i,j,k] handling must equal what nnU-Net itself writes to NIfTI."""
    directory = tiny_model.write_member(tmp_path / "m", seed=3)
    model = resolve_model(model_settings(directory)).model
    affine = synthetic.oblique_affine(left_handed=left_handed)
    paths = write_case(tmp_path / "case", affine)
    inputs = [paths[s] for s in ("t1", "t1c", "t2", "flair")]

    result = model.predict(inputs, lambda *_: None)
    assert result.labels.shape == synthetic.SHAPE and result.confidence.shape == synthetic.SHAPE
    assert len(np.unique(result.labels)) >= 2, "random model should predict more than one class"
    assert 0.2 <= float(result.confidence.min()) and float(result.confidence.max()) <= 1.0 + 1e-6

    # Reference: let nnU-Net write the segmentation file itself, then read it with nibabel.
    from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

    images, properties = SimpleITKIO().read_images([str(p) for p in inputs])
    predictor = model._predictor(model.members[0])
    predictor.predict_single_npy_array(images, properties, None, str(tmp_path / "reference"), False)
    reference = nib.load(str(tmp_path / "reference.nii.gz"))
    assert np.allclose(reference.affine, affine, atol=1e-3)
    assert np.array_equal(np.asanyarray(reference.dataobj), result.labels)

    assert result.info["members"][0]["trainer"] == "nnUNetTrainer_MUGlioma_no29"
    assert result.info["members"][0]["checkpoints"][0]["file"] == "checkpoint_best.pth"
    assert len(result.info["members"][0]["checkpoints"][0]["sha256"]) == 64


def test_cross_model_ensemble_averages_probabilities(tmp_path, model_settings):
    parent = tmp_path / "models"
    tiny_model.write_member(parent / "no29" / "model", seed=1)
    tiny_model.write_member(parent / "no30" / "model", seed=2, features=(4, 8))  # different architecture
    paths = write_case(tmp_path / "case", synthetic.oblique_affine())
    inputs = [paths[s] for s in ("t1", "t1c", "t2", "flair")]

    ensemble = resolve_model(model_settings(parent)).model
    assert ensemble.is_cross_model_ensemble
    combined = ensemble.predict(inputs, lambda *_: None)
    assert any("not evaluated in the paper" in note for note in combined.info["notes"])

    singles = [
        resolve_model(model_settings(parent / name / "model")).model.predict(inputs, lambda *_: None)
        for name in ("no29", "no30")
    ]
    # Where both members agree on the label, the ensemble must agree too and its
    # confidence is the mean of theirs.
    agree = singles[0].labels == singles[1].labels
    assert agree.mean() > 0.05
    assert np.array_equal(combined.labels[agree], singles[0].labels[agree])
    mean_confidence = (singles[0].confidence + singles[1].confidence) / 2.0
    assert np.allclose(combined.confidence[agree], mean_confidence[agree], atol=1e-3)


def test_inference_job_through_the_api(tmp_path, model_settings):
    directory = tiny_model.write_member(tmp_path / "m", seed=3)
    affine = synthetic.oblique_affine(left_handed=True)  # uploaded as LAS; viewer shows RAS
    with TestClient(create_app(model_settings(directory))) as client:
        info = client.get("/api/info").json()
        assert info["mode"] == "inference" and info["banner"] is None
        paths = write_case(tmp_path / "case", affine)
        job_id = run_job(client, paths, with_segmentation=False)

        summary = client.get(f"/api/jobs/{job_id}/summary").json()
        assert summary["demo"] is False and summary["confidence_available"] is True
        assert summary["model"]["mode"] == "inference"
        assert summary["model"]["members"][0]["trainer"] == "nnUNetTrainer_MUGlioma_no29"
        assert "uncalibrated" in summary["confidence_note"]
        assert any("CPU" in warning for warning in summary["warnings"])

        # Downloaded mask is on the uploaded grid and equals a direct prediction.
        (tmp_path / "out.nii.gz").write_bytes(client.get(f"/api/jobs/{job_id}/download/segmentation.nii.gz").content)
        image = nib.load(str(tmp_path / "out.nii.gz"))
        assert np.allclose(image.affine, affine, atol=1e-4)
        direct = resolve_model(model_settings(directory)).model.predict(
            [paths[s] for s in ("t1", "t1c", "t2", "flair")], lambda *_: None
        )
        downloaded = np.asanyarray(image.dataobj)
        assert np.array_equal(downloaded, direct.labels)

        # Confidence volume + voxel endpoint agree with the direct prediction at world points.
        header = client.get(f"/api/jobs/{job_id}/volumes/confidence/header").json()
        assert header["dtype"] == "uint8" and header["axcodes"] == "RAS"
        rng = np.random.default_rng(0)
        for _ in range(25):
            ijk = rng.integers(0, np.array(synthetic.SHAPE))
            world = (affine @ np.append(ijk, 1.0))[:3]
            hit = client.get(f"/api/jobs/{job_id}/voxel", params=dict(zip("xyz", world))).json()
            assert hit["label"] == int(direct.labels[tuple(ijk)])
            assert abs(hit["confidence"] - float(direct.confidence[tuple(ijk)])) <= 0.5 / 255 + 1e-6
        assert json.dumps(summary).count("DOE") == 0


def test_sample_case_is_refused_when_a_model_is_loaded(tmp_path, model_settings):
    directory = tiny_model.write_member(tmp_path / "m", seed=3)
    with TestClient(create_app(model_settings(directory))) as client:
        assert client.get("/api/info").json()["sample_case_available"] is False
        assert client.post("/api/demo/sample").status_code == 409
