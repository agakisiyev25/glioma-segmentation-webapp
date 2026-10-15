"""FastAPI application."""

from __future__ import annotations

import json
import logging
import os
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from . import __version__
from .config import Settings, load_settings
from .geometry import is_in_bounds, world_to_voxel
from .inference import ModelStatus, resolve_model
from .io_utils import (
    InputError,
    dicom_zip_to_nifti,
    kind_of_upload,
    nifti_suffix,
    sanitize_nifti,
    sanitize_segmentation,
)
from .jobs import ACTIVE_STAGES, CleanupThread, JobStore, Worker
from .labels import (
    CONFIDENCE_NOTE,
    DISCLAIMER,
    LABEL_BY_VALUE,
    SCOPE_NOTE,
    SEQUENCE_DISPLAY,
    SEQUENCES,
    label_map_json,
)
from .pipeline import run_pipeline
from .sample_case import ensure_sample_case
from .volumes import VOLUME_NAMES, open_array, read_header

log = logging.getLogger(__name__)

STUDY_METRICS = Path(__file__).resolve().parent / "resources" / "study_metrics.json"
MESH_NAMES = ("brain", "label_1", "label_2", "label_3", "label_4")
CHUNK = 1024 * 1024


async def _save_upload(upload: UploadFile, destination: Path, limit_bytes: int, slot: str) -> None:
    written = 0
    with open(destination, "wb") as sink:
        while chunk := await upload.read(CHUNK):
            written += len(chunk)
            if written > limit_bytes:
                raise InputError(f"{slot}: file is larger than the {limit_bytes // 1024**2} MB upload limit.")
            sink.write(chunk)
    if written == 0:
        raise InputError(f"{slot}: the uploaded file is empty.")


def _ingest(incoming: Path, kind: str, destination: Path, scratch: Path, slot: str) -> None:
    """Convert (if DICOM), validate, and re-write with a clean header."""
    source = dicom_zip_to_nifti(incoming, scratch, slot) if kind == "dicom_zip" else incoming
    sanitize_nifti(source, destination, slot)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    store = JobStore(settings.jobs_dir)
    model_status: ModelStatus = resolve_model(settings)

    def run(state, report) -> None:
        run_pipeline(state, store.path(state.id), settings, model_status.model, report)

    worker = Worker(store, run)
    cleanup = CleanupThread(store, settings.job_ttl_hours)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        store.fail_interrupted()
        worker.start()
        cleanup.start()
        log.info("mode=%s  %s", model_status.mode, model_status.message)
        yield
        worker.stop()
        cleanup.stop()

    app = FastAPI(title="Postoperative glioma segmentation", version=__version__, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Volume-Header"],
    )
    app.state.settings, app.state.store, app.state.model_status = settings, store, model_status

    # ------------------------------------------------------------- helpers

    def job_or_404(job_id: str):
        try:
            return store.load(job_id)
        except KeyError:
            raise HTTPException(404, "Job not found (jobs are deleted automatically after 24 h).")

    def done_job_dir(job_id: str) -> Path:
        state = job_or_404(job_id)
        if state.status != "done":
            raise HTTPException(409, f"Job is not finished (status: {state.status}).")
        return store.path(job_id)

    # ---------------------------------------------------------------- info

    @app.get("/api/info")
    def info():
        return {
            "app_version": __version__,
            "mode": model_status.mode,  # "inference" | "demo" | "model_error"
            "message": model_status.message,
            "banner": "DEMO — no model inference" if model_status.mode == "demo" else None,
            "model": model_status.describe(),
            "disclaimer": DISCLAIMER,
            "scope_note": SCOPE_NOTE,
            "confidence_note": CONFIDENCE_NOTE,
            "label_map": label_map_json(),
            "sequences": [{"slot": slot, "name": SEQUENCE_DISPLAY[slot]} for slot in SEQUENCES],
            "hd_bet_available": model_status.hd_bet_available,
            "job_ttl_hours": settings.job_ttl_hours,
            "max_upload_mb": settings.max_upload_mb,
            "queue_length": worker.queue_length(),
            "sample_case_available": model_status.mode == "demo",
        }

    @app.get("/api/model/metrics")
    def study_metrics():
        """The study's published nnU-Net results (results/final_analysis of the research repo)."""
        return JSONResponse(json.loads(STUDY_METRICS.read_text(encoding="utf-8")))

    @app.post("/api/demo/sample", status_code=201)
    async def create_sample_job():
        """Demo mode only: run the built-in synthetic (non-patient) case. No upload needed."""
        if model_status.mode != "demo":
            raise HTTPException(
                409,
                "The synthetic sample is only offered in demo mode. With a model loaded it would "
                "mean running the model on generated shapes, which says nothing about the model.",
            )
        state = store.create(mode="demo", already_preprocessed=True, source="sample")
        try:
            await run_in_threadpool(_ingest_sample, settings.data_dir / "sample_case", store.path(state.id) / "input")
        except Exception:
            store.delete(state.id)
            log.exception("could not prepare the sample case for job %s", state.id)
            raise HTTPException(500, "The sample case could not be prepared.")
        state.status, state.message = "queued", "Waiting for the worker"
        store.save(state)
        worker.submit(state.id)
        return {"job_id": state.id, "status": state.status, "mode": state.mode}

    # ---------------------------------------------------------------- jobs

    @app.post("/api/jobs", status_code=201)
    async def create_job(
        t1: UploadFile = File(...),
        t1c: UploadFile = File(...),
        t2: UploadFile = File(...),
        flair: UploadFile = File(...),
        segmentation: UploadFile | None = File(None),
        already_preprocessed: bool = Form(True),
    ):
        if model_status.mode == "model_error":
            raise HTTPException(503, f"The model could not be loaded: {model_status.message}")
        demo = model_status.mode == "demo"
        if demo and segmentation is None:
            raise HTTPException(
                400,
                "DEMO — no model inference: no model weights are configured, so an existing "
                "segmentation NIfTI must be uploaded with the four sequences.",
            )

        state = store.create(mode="demo" if demo else "inference", already_preprocessed=already_preprocessed)
        job_dir = store.path(state.id)
        incoming = job_dir / "incoming"
        incoming.mkdir()
        limit = settings.max_upload_mb * 1024**2
        uploads = {"t1": t1, "t1c": t1c, "t2": t2, "flair": flair}
        if demo:
            uploads["segmentation"] = segmentation
        try:
            for slot, upload in uploads.items():
                display = SEQUENCE_DISPLAY.get(slot, "segmentation")
                # The client file name is used only to detect the format; it is
                # never stored or logged.
                try:
                    kind = kind_of_upload(upload.filename or "")
                except InputError as error:
                    raise InputError(f"{display}: {error}") from error
                if slot == "segmentation" and kind != "nifti":
                    raise InputError("segmentation: must be a NIfTI file (.nii or .nii.gz).")
                suffix = ".zip" if kind == "dicom_zip" else nifti_suffix(upload.filename or "")
                raw_path = incoming / f"{slot}{suffix}"
                await _save_upload(upload, raw_path, limit, display)
                if slot == "segmentation":
                    await run_in_threadpool(
                        sanitize_segmentation, raw_path, job_dir / "input" / "segmentation.nii.gz", set(LABEL_BY_VALUE)
                    )
                else:
                    await run_in_threadpool(
                        _ingest, raw_path, kind, job_dir / "input" / f"{slot}.nii.gz", incoming / f"{slot}_scratch", display
                    )
        except InputError as error:
            store.delete(state.id)
            raise HTTPException(400, str(error))
        except Exception:
            store.delete(state.id)
            log.exception("upload failed for job %s", state.id)
            raise HTTPException(500, "The upload could not be processed.")
        finally:
            # Original files (with their original headers) are not kept.
            shutil.rmtree(incoming, ignore_errors=True)

        state.status, state.message = "queued", "Waiting for the worker"
        store.save(state)
        worker.submit(state.id)
        return {"job_id": state.id, "status": state.status, "mode": state.mode}

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str):
        state = job_or_404(job_id)
        data = state.public(settings.job_ttl_hours)
        data["queue_length"] = worker.queue_length() if state.status in ACTIVE_STAGES else 0
        return data

    @app.delete("/api/jobs/{job_id}", status_code=204)
    def delete_job(job_id: str):
        job_or_404(job_id)
        store.delete(job_id)
        return Response(status_code=204)

    @app.get("/api/jobs/{job_id}/summary")
    def job_summary(job_id: str):
        path = done_job_dir(job_id) / "output" / "summary.json"
        return JSONResponse(json.loads(path.read_text(encoding="utf-8")))

    @app.get("/api/jobs/{job_id}/meshes/{name}.glb")
    def job_mesh(job_id: str, name: str):
        if name not in MESH_NAMES:
            raise HTTPException(404, f"Unknown mesh; expected one of {', '.join(MESH_NAMES)}.")
        path = done_job_dir(job_id) / "output" / "meshes" / f"{name}.glb"
        if not path.exists():
            raise HTTPException(404, f"No {name} mesh: that region is empty in this segmentation.")
        return FileResponse(path, media_type="model/gltf-binary")

    @app.get("/api/jobs/{job_id}/volumes/{name}/header")
    def job_volume_header(job_id: str, name: str):
        return _volume_header(done_job_dir(job_id), name)

    @app.get("/api/jobs/{job_id}/volumes/{name}")
    def job_volume(job_id: str, name: str):
        """Raw voxels, gzip-encoded; the JSON header is also sent as X-Volume-Header."""
        job_dir = done_job_dir(job_id)
        header = _volume_header(job_dir, name)
        path = job_dir / "output" / "viewer" / f"{name}.bin.gz"
        return FileResponse(
            path,
            media_type="application/octet-stream",
            headers={
                "Content-Encoding": "gzip",
                "X-Volume-Header": json.dumps({k: v for k, v in header.items() if k != "label_map"}),
                "Cache-Control": "private, max-age=3600",
            },
        )

    @app.get("/api/jobs/{job_id}/voxel")
    def job_voxel(
        job_id: str,
        x: float = Query(..., description="world mm, RAS: +x = patient right"),
        y: float = Query(..., description="world mm, RAS: +y = anterior"),
        z: float = Query(..., description="world mm, RAS: +z = superior"),
    ):
        viewer = done_job_dir(job_id) / "output" / "viewer"
        header = read_header(viewer, "labels")
        affine = np.array(header["affine"])
        continuous = world_to_voxel([x, y, z], affine)
        index = world_to_voxel([x, y, z], affine, round_to_index=True)
        result = {
            "world_mm": [x, y, z],
            "voxel": index.tolist(),
            "voxel_continuous": continuous.tolist(),
            "in_bounds": is_in_bounds(index, header["shape"]),
            "label": None,
            "label_short": None,
            "label_name": None,
            "color": None,
            "confidence": None,
            "intensities": {"t1c": None, "flair": None},
        }
        if not result["in_bounds"]:
            return result
        i, j, k = (int(v) for v in index)
        label = int(open_array(viewer, "labels")[i, j, k])
        info = LABEL_BY_VALUE[label]
        result.update(label=label, label_short=info.short, label_name=info.name, color=info.color)
        confidence = open_array(viewer, "confidence")
        if confidence is not None:
            result["confidence"] = float(confidence[i, j, k]) / 255.0
        for name in ("t1c", "flair"):
            array, volume_header = open_array(viewer, name), read_header(viewer, name)
            if array is not None:
                result["intensities"][name] = (
                    float(array[i, j, k]) * volume_header["scl_slope"] + volume_header["scl_inter"]
                )
        return result

    @app.get("/api/jobs/{job_id}/download/segmentation.nii.gz")
    def job_download(job_id: str):
        path = done_job_dir(job_id) / "output" / "segmentation.nii.gz"
        return FileResponse(path, media_type="application/gzip", filename=f"segmentation_{job_id[:8]}.nii.gz")

    # --------------------------------------------- optional built frontend
    dist = Path(os.environ.get("FRONTEND_DIST", Path(__file__).resolve().parents[2] / "frontend" / "dist"))
    if (dist / "index.html").exists():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str):
            if path.startswith("api/"):
                raise HTTPException(404, "Not found")
            candidate = (dist / path).resolve()
            if path and dist.resolve() in candidate.parents and candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(dist / "index.html")

    return app


def _ingest_sample(cache_dir: Path, input_dir: Path) -> None:
    """Copy the synthetic case into a job through the same validation as an upload."""
    paths = ensure_sample_case(cache_dir)
    for slot in SEQUENCES:
        sanitize_nifti(paths[slot], input_dir / f"{slot}.nii.gz", SEQUENCE_DISPLAY[slot])
    sanitize_segmentation(paths["segmentation"], input_dir / "segmentation.nii.gz", set(LABEL_BY_VALUE))


def _volume_header(job_dir: Path, name: str) -> dict:
    if name not in VOLUME_NAMES:
        raise HTTPException(404, f"Unknown volume; expected one of {', '.join(VOLUME_NAMES)}.")
    header = read_header(job_dir / "output" / "viewer", name)
    if header is None:
        detail = (
            "No confidence volume: this job ran in demo mode, without model inference."
            if name == "confidence"
            else f"Volume {name} is not available."
        )
        raise HTTPException(404, detail)
    return header


def get_app() -> FastAPI:
    """Factory for ``uvicorn app.main:get_app --factory``."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return create_app()
