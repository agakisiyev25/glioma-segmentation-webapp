from __future__ import annotations

import dataclasses
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import load_settings
from app.main import create_app

from . import synthetic


def write_case(directory: Path, affine: np.ndarray, *, labels: np.ndarray | None = None) -> dict[str, Path]:
    """Four co-registered, skull-stripped-like sequences plus a segmentation."""
    directory.mkdir(parents=True, exist_ok=True)
    labels = synthetic.label_volume() if labels is None else labels
    paths = {}
    for index, slot in enumerate(("t1", "t1c", "t2", "flair")):
        image = synthetic.brain_volume(seed=index)
        image[labels > 0] += 150.0 * (index + 1)
        paths[slot] = directory / f"patient_DOE_JOHN_{slot}.nii.gz"  # PHI-looking name on purpose
        header = nib.Nifti1Header()
        header["descrip"] = b"DOE^JOHN 1961-02-03"
        nib.save(nib.Nifti1Image(image, affine, header), str(paths[slot]))
    paths["segmentation"] = directory / "seg.nii.gz"
    nib.save(nib.Nifti1Image(labels, affine), str(paths["segmentation"]))
    return paths


@pytest.fixture()
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("NNUNET_MODEL_DIR", str(tmp_path / "no-weights"))  # demo mode, whatever is in models/
    monkeypatch.setenv("GLIOMA_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("FRONTEND_DIST", str(tmp_path / "no-frontend"))
    return dataclasses.replace(load_settings(), brain_mesh_faces=4000, label_mesh_faces=2000)


@pytest.fixture()
def client(settings):
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def submit(client: TestClient, paths: dict[str, Path], *, already_preprocessed: bool = True, with_segmentation: bool = True):
    handles = {}
    try:
        for slot in ("t1", "t1c", "t2", "flair") + (("segmentation",) if with_segmentation else ()):
            handles[slot] = open(paths[slot], "rb")
        files = {slot: (paths[slot].name, handle, "application/gzip") for slot, handle in handles.items()}
        return client.post("/api/jobs", files=files, data={"already_preprocessed": str(already_preprocessed).lower()})
    finally:
        for handle in handles.values():
            handle.close()


def wait_for(client: TestClient, job_id: str, timeout: float = 120.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        state = client.get(f"/api/jobs/{job_id}").json()
        if state["status"] in ("done", "error"):
            return state
        time.sleep(0.2)
    raise AssertionError(f"job did not finish in {timeout}s: {state}")


def run_job(client: TestClient, paths: dict[str, Path], **kwargs) -> str:
    response = submit(client, paths, **kwargs)
    assert response.status_code == 201, response.text
    job_id = response.json()["job_id"]
    state = wait_for(client, job_id)
    assert state["status"] == "done", state
    return job_id
