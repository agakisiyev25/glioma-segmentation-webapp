"""API + job system in demo mode (uploaded segmentation, no model)."""

import gzip
import io
import json
import time

import nibabel as nib
import numpy as np
import trimesh

from app.geometry import voxel_to_world, voxel_volume_mm3, world_to_voxel
from app.labels import DISCLAIMER

from . import synthetic
from .conftest import run_job, submit, wait_for, write_case


def _fetch_volume(client, job_id, name):
    header = client.get(f"/api/jobs/{job_id}/volumes/{name}/header").json()
    response = client.get(f"/api/jobs/{job_id}/volumes/{name}")
    assert response.status_code == 200
    assert json.loads(response.headers["x-volume-header"])["shape"] == header["shape"]
    raw = response.content  # httpx already decoded Content-Encoding: gzip
    dtype = np.dtype("<u1") if header["dtype"] == "uint8" else np.dtype("<i2")
    assert len(raw) == np.prod(header["shape"]) * dtype.itemsize
    return header, np.frombuffer(raw, dtype=dtype).reshape(header["shape"], order="F")


def test_info_reports_demo_mode(client):
    info = client.get("/api/info").json()
    assert info["mode"] == "demo"
    assert info["banner"] == "DEMO — no model inference"
    assert info["disclaimer"] == DISCLAIMER
    assert [label["value"] for label in info["label_map"]] == [0, 1, 2, 3, 4]
    assert [s["slot"] for s in info["sequences"]] == ["t1", "t1c", "t2", "flair"]


def test_demo_job_end_to_end(client, tmp_path):
    affine = synthetic.oblique_affine()
    job_id = run_job(client, write_case(tmp_path / "case", affine))

    summary = client.get(f"/api/jobs/{job_id}/summary").json()
    assert summary["demo"] is True and summary["banner"] == "DEMO — no model inference"
    assert summary["disclaimer"] == DISCLAIMER
    assert summary["confidence_available"] is False
    assert summary["grid"]["viewer"]["axcodes"] == "RAS"

    # Volumes (mL), component counts, centroids.
    voxel_ml = voxel_volume_mm3(affine) / 1000.0
    by_value = {entry["value"]: entry for entry in summary["statistics"]["labels"]}
    for value, (lo, hi) in synthetic.BOXES.items():
        voxels = int(np.prod(np.array(hi) - np.array(lo) + 1))
        entry = by_value[value]
        assert entry["voxels"] == voxels
        assert np.isclose(entry["volume_ml"], voxels * voxel_ml)
        assert entry["component_count"] == 1
        expected_centre = voxel_to_world((np.array(lo) + np.array(hi)) / 2.0, affine)
        assert np.allclose(entry["centroid_world_mm"], expected_centre, atol=1e-6)
        assert entry["bounding_box"]["voxel_min"] == list(lo)
        assert entry["bounding_box"]["voxel_max"] == list(hi)
    assert by_value[1]["voxels"] == 0 and by_value[1]["component_count"] == 0
    tumor = summary["statistics"]["composites"]["tumor_related_1_3"]
    assert tumor["voxels"] == by_value[2]["voxels"] + by_value[3]["voxels"]

    # Label volume served to the frontend equals the uploaded segmentation.
    header, labels = _fetch_volume(client, job_id, "labels")
    assert np.allclose(header["affine"], affine)
    assert np.allclose(np.array(header["inverse_affine"]) @ np.array(header["affine"]), np.eye(4), atol=1e-9)
    assert np.array_equal(labels, synthetic.label_volume())

    t1c_header, t1c = _fetch_volume(client, job_id, "t1c")
    assert t1c_header["dtype"] == "int16" and t1c.shape == tuple(synthetic.SHAPE)
    assert client.get(f"/api/jobs/{job_id}/volumes/flair").status_code == 200
    assert client.get(f"/api/jobs/{job_id}/volumes/confidence").status_code == 404
    assert client.get(f"/api/jobs/{job_id}/volumes/nope").status_code == 404

    # Meshes: present for non-empty labels, in world mm, aligned with the boxes.
    assert client.get(f"/api/jobs/{job_id}/meshes/label_1.glb").status_code == 404  # empty label
    assert client.get(f"/api/jobs/{job_id}/meshes/skull.glb").status_code == 404
    for value, (lo, hi) in synthetic.BOXES.items():
        response = client.get(f"/api/jobs/{job_id}/meshes/label_{value}.glb")
        assert response.status_code == 200
        mesh = trimesh.load(io.BytesIO(response.content), file_type="glb", force="mesh", process=False)
        expected_centre = voxel_to_world((np.array(lo) + np.array(hi)) / 2.0, affine)
        assert np.linalg.norm(mesh.center_mass - expected_centre) < 0.4
        back = world_to_voxel(mesh.vertices, affine)
        assert np.all(back.min(axis=0) >= np.array(lo) - 0.8) and np.all(back.max(axis=0) <= np.array(hi) + 0.8)
    brain = trimesh.load(
        io.BytesIO(client.get(f"/api/jobs/{job_id}/meshes/brain.glb").content), file_type="glb", force="mesh", process=False
    )
    brain_centre = voxel_to_world((np.array(synthetic.SHAPE) - 1) / 2.0, affine)
    assert np.linalg.norm(brain.center_mass - brain_centre) < 1.0
    assert brain.volume > 0

    # Download: identical grid and labels to what was uploaded.
    download = client.get(f"/api/jobs/{job_id}/download/segmentation.nii.gz")
    assert download.status_code == 200
    (tmp_path / "out.nii.gz").write_bytes(download.content)
    image = nib.load(str(tmp_path / "out.nii.gz"))
    assert np.allclose(image.affine, affine, atol=1e-4)
    assert np.array_equal(np.asanyarray(image.dataobj), synthetic.label_volume())


def test_uploaded_orientation_is_preserved_in_download_and_world_lookups(client, tmp_path):
    """A non-RAS (mirrored) upload: viewer reorients, world coordinates do not move."""
    affine = synthetic.oblique_affine(left_handed=True)
    job_id = run_job(client, write_case(tmp_path / "case", affine))
    summary = client.get(f"/api/jobs/{job_id}/summary").json()
    assert summary["grid"]["uploaded_t1c"]["axcodes"] == "LAS"
    assert summary["grid"]["viewer"]["axcodes"] == "RAS"

    for value, (lo, hi) in synthetic.BOXES.items():
        world = voxel_to_world((np.array(lo) + np.array(hi)) / 2.0, affine)
        hit = client.get(f"/api/jobs/{job_id}/voxel", params=dict(zip("xyz", world))).json()
        assert hit["label"] == value
        # The voxel index reported is on the viewer grid, whose affine maps it back to the same point.
        viewer_affine = np.array(summary["grid"]["viewer"]["affine"])
        assert np.linalg.norm(voxel_to_world(hit["voxel"], viewer_affine) - world) < max(synthetic.SPACING)

    (tmp_path / "out.nii.gz").write_bytes(client.get(f"/api/jobs/{job_id}/download/segmentation.nii.gz").content)
    image = nib.load(str(tmp_path / "out.nii.gz"))
    assert np.allclose(image.affine, affine, atol=1e-4)  # still the uploaded (LAS) grid
    assert np.array_equal(np.asanyarray(image.dataobj), synthetic.label_volume())


def test_segmentation_on_another_grid_is_resampled_with_a_warning(client, tmp_path):
    affine = synthetic.oblique_affine()
    paths = write_case(tmp_path / "case", affine)
    # Same labels stored with the first axis flipped (different grid, same world content).
    flipped_affine = affine @ np.array(
        [[-1, 0, 0, synthetic.SHAPE[0] - 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]]
    )
    nib.save(nib.Nifti1Image(synthetic.label_volume()[::-1], flipped_affine), str(paths["segmentation"]))
    job_id = run_job(client, paths)
    summary = client.get(f"/api/jobs/{job_id}/summary").json()
    assert any("resampled" in warning for warning in summary["warnings"])
    _, labels = _fetch_volume(client, job_id, "labels")
    assert np.array_equal(labels, synthetic.label_volume())


def test_no_phi_is_stored(client, settings, tmp_path):
    job_id = run_job(client, write_case(tmp_path / "case", synthetic.oblique_affine()))
    job_dir = settings.jobs_dir / job_id
    stored = [p for p in job_dir.rglob("*") if p.is_file()]
    assert stored and not (job_dir / "incoming").exists() and not (job_dir / "work").exists()
    for path in stored:
        assert "DOE" not in path.name.upper()
        blob = path.read_bytes()
        if path.suffix == ".gz":
            blob = gzip.decompress(blob)
        assert b"DOE" not in blob and b"1961" not in blob[:400]
    header = nib.load(str(job_dir / "input" / "t1c.nii.gz")).header
    assert bytes(header["descrip"]).strip(b"\x00") == b""


def test_validation_errors(client, tmp_path):
    affine = synthetic.oblique_affine()
    paths = write_case(tmp_path / "case", affine)

    # Demo mode without a segmentation.
    assert submit(client, paths, with_segmentation=False).status_code == 400

    # NaNs.
    bad = synthetic.brain_volume()
    bad[5, 5, 5] = np.nan
    nib.save(nib.Nifti1Image(bad, affine), str(paths["t2"]))
    response = submit(client, paths)
    assert response.status_code == 400 and "NaN" in response.json()["detail"]
    assert "T2" in response.json()["detail"]

    # 4D volume.
    paths = write_case(tmp_path / "case4d", affine)
    nib.save(nib.Nifti1Image(np.zeros(synthetic.SHAPE + (3,), np.float32), affine), str(paths["flair"]))
    response = submit(client, paths)
    assert response.status_code == 400 and "3D" in response.json()["detail"]

    # Implausible spacing.
    paths = write_case(tmp_path / "casespacing", affine)
    nib.save(nib.Nifti1Image(synthetic.brain_volume(), np.diag([50.0, 1, 1, 1])), str(paths["t1"]))
    response = submit(client, paths)
    assert response.status_code == 400 and "spacing" in response.json()["detail"]

    # Label values outside 0–4.
    paths = write_case(tmp_path / "caselabels", affine, labels=synthetic.label_volume() * 2)
    response = submit(client, paths)
    assert response.status_code == 400 and "label" in response.json()["detail"]

    # Not a NIfTI.
    paths = write_case(tmp_path / "casejunk", affine)
    paths["t1"].write_bytes(b"not a nifti")
    assert submit(client, paths).status_code == 400

    # Unsupported extension.
    junk = tmp_path / "scan.png"
    junk.write_bytes(b"x")
    paths["t1"] = junk
    assert submit(client, paths).status_code == 400


def test_sequences_on_different_grids_fail_with_a_clear_message(client, tmp_path):
    affine = synthetic.oblique_affine()
    paths = write_case(tmp_path / "case", affine)
    shifted = affine.copy()
    shifted[:3, 3] += 4.0
    nib.save(nib.Nifti1Image(synthetic.brain_volume(), shifted), str(paths["flair"]))
    response = submit(client, paths)
    assert response.status_code == 201
    state = wait_for(client, response.json()["job_id"])
    assert state["status"] == "error"
    assert "FLAIR" in state["error"] and "Already preprocessed" in state["error"]


def test_status_delete_and_unknown_jobs(client, tmp_path):
    job_id = run_job(client, write_case(tmp_path / "case", synthetic.oblique_affine()))
    state = client.get(f"/api/jobs/{job_id}").json()
    assert state["status"] == "done" and state["progress"] == 1.0
    assert state["expires_at"] - state["created_at"] == 24 * 3600

    assert client.delete(f"/api/jobs/{job_id}").status_code == 204
    assert client.get(f"/api/jobs/{job_id}").status_code == 404
    assert client.get(f"/api/jobs/{job_id}/summary").status_code == 404
    assert client.get("/api/jobs/not-a-job").status_code == 404
    assert client.get("/api/jobs/..%2F..%2Fetc/summary").status_code == 404


def test_expired_jobs_are_deleted(client, settings, tmp_path):
    job_id = run_job(client, write_case(tmp_path / "case", synthetic.oblique_affine()))
    store = client.app.state.store
    state = store.load(job_id)
    state.created_at = time.time() - 25 * 3600
    store.save(state)
    assert store.cleanup_expired(settings.job_ttl_hours) == 1
    assert client.get(f"/api/jobs/{job_id}").status_code == 404


def test_one_click_sample_case_in_demo_mode(client, settings):
    assert client.get("/api/info").json()["sample_case_available"] is True
    response = client.post("/api/demo/sample")
    assert response.status_code == 201, response.text
    job_id = response.json()["job_id"]
    state = wait_for(client, job_id)
    assert state["status"] == "done", state
    assert state["source"] == "sample"

    summary = client.get(f"/api/jobs/{job_id}/summary").json()
    assert summary["demo"] is True and summary["synthetic_sample"] is True
    by_short = {entry["short"]: entry for entry in summary["statistics"]["labels"]}
    assert all(by_short[name]["voxels"] > 0 for name in ("NETC", "SNFH", "ET", "RC"))
    assert by_short["ET"]["component_count"] == 2  # enhancing rim + a small satellite
    assert summary["grid"]["uploaded_t1c"]["axcodes"] == "LPS" and summary["grid"]["viewer"]["axcodes"] == "RAS"
    assert set(summary["meshes"]) == {"brain", "label_1", "label_2", "label_3", "label_4"}
    # Generated once, then reused.
    assert (settings.data_dir / "sample_case" / "synthetic_t1c.nii.gz").is_file()
    assert client.post("/api/demo/sample").status_code == 201


def test_uploaded_jobs_are_not_marked_as_the_sample(client, tmp_path):
    job_id = run_job(client, write_case(tmp_path / "case", synthetic.oblique_affine()))
    assert client.get(f"/api/jobs/{job_id}").json()["source"] == "upload"
    assert client.get(f"/api/jobs/{job_id}/summary").json()["synthetic_sample"] is False


def test_study_metrics_are_served(client):
    metrics = client.get("/api/model/metrics").json()
    assert metrics["model"] == "nnU-Net v2 3d_fullres"
    assert metrics["primary"]["patients"] == 203
    assert metrics["primary"]["mean_dice"] == 0.8791 and metrics["primary"]["ci95"] == [0.8563, 0.8997]
    by_key = {target["key"]: target for target in metrics["targets"]}
    assert set(by_key) == {"label_1", "label_2", "label_3", "label_4", "tumor_related_1_3", "all_postoperative_1_4"}
    assert by_key["label_3"]["labels"] == [3] and 0 < by_key["label_3"]["dice"] < 1
    assert [fold["fold"] for fold in metrics["folds"]] == [1, 2, 3, 4, 5]
