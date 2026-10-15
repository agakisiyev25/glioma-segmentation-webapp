"""Hover/click round trip: a world point returns the right label from the backend.

The probes come from tests/fixtures/alignment_case.json, which the frontend
tests (frontend/src/lib/geometry.test.ts) read as well, so both lookups are
held to the same expected answers.
"""

import json
from pathlib import Path

import numpy as np

from app.geometry import voxel_to_world, world_to_voxel
from scripts.make_alignment_fixture import build

from . import synthetic
from .conftest import run_job, write_case

FIXTURE = Path(__file__).parent / "fixtures" / "alignment_case.json"


def test_committed_fixture_is_current():
    committed = json.loads(FIXTURE.read_text(encoding="utf-8"))
    fresh = build()
    assert committed["shape"] == fresh["shape"] and committed["boxes"] == fresh["boxes"]
    assert np.allclose(committed["affine"], fresh["affine"])
    assert len(committed["probes"]) == len(fresh["probes"])
    for a, b in zip(committed["probes"], fresh["probes"]):
        assert a["voxel"] == b["voxel"] and a["label"] == b["label"] and a["in_bounds"] == b["in_bounds"]
        assert np.allclose(a["world_mm"], b["world_mm"])


def test_geometry_functions_agree_with_fixture():
    case = json.loads(FIXTURE.read_text(encoding="utf-8"))
    labels = synthetic.label_volume()
    for probe in case["probes"]:
        index = world_to_voxel(probe["world_mm"], case["affine"], round_to_index=True)
        assert index.tolist() == probe["voxel"], probe["note"]
        if probe["in_bounds"]:
            assert int(labels[tuple(index)]) == probe["label"], probe["note"]


def test_backend_voxel_endpoint_matches_fixture(client, tmp_path):
    case = json.loads(FIXTURE.read_text(encoding="utf-8"))
    affine = np.array(case["affine"])
    job_id = run_job(client, write_case(tmp_path / "case", affine))

    seen_labels = set()
    for probe in case["probes"]:
        x, y, z = probe["world_mm"]
        hit = client.get(f"/api/jobs/{job_id}/voxel", params={"x": x, "y": y, "z": z}).json()
        assert hit["voxel"] == probe["voxel"], probe["note"]
        assert hit["in_bounds"] == probe["in_bounds"], probe["note"]
        assert hit["label"] == probe["label"], probe["note"]
        if probe["in_bounds"]:
            assert hit["label_name"] and hit["color"].startswith("#")
            assert hit["confidence"] is None  # demo mode: no model confidence
            assert hit["intensities"]["t1c"] is not None and hit["intensities"]["flair"] is not None
            assert np.allclose(voxel_to_world(hit["voxel_continuous"], affine), probe["world_mm"], atol=1e-6)
            seen_labels.add(hit["label"])
    assert seen_labels == {0, 2, 3, 4}


def test_voxel_endpoint_requires_coordinates(client, tmp_path):
    job_id = run_job(client, write_case(tmp_path / "case", synthetic.oblique_affine()))
    assert client.get(f"/api/jobs/{job_id}/voxel", params={"x": 1, "y": 2}).status_code == 422
