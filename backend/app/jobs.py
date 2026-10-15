"""Job storage (one folder per job) and the single background worker.

Jobs run strictly one at a time so two inferences never compete for GPU memory.
"""

from __future__ import annotations

import json
import logging
import queue
import re
import shutil
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)

JOB_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
ACTIVE_STAGES = ("queued", "preprocessing", "inference", "meshing")
STAGES = ("uploading",) + ACTIVE_STAGES + ("done", "error")


@dataclass
class JobState:
    id: str
    status: str = "uploading"
    progress: float = 0.0
    message: str = ""
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    mode: str = "inference"  # "inference" | "demo"
    already_preprocessed: bool = True
    source: str = "upload"  # "upload" | "sample" (the built-in synthetic case)
    warnings: list[str] = field(default_factory=list)

    def public(self, ttl_hours: float) -> dict:
        data = asdict(self)
        data["stages"] = list(STAGES)
        data["expires_at"] = self.created_at + ttl_hours * 3600.0
        return data


class JobStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def path(self, job_id: str) -> Path:
        if not JOB_ID_PATTERN.match(job_id):
            raise KeyError(job_id)
        return self.root / job_id

    def create(self, *, mode: str, already_preprocessed: bool, source: str = "upload") -> JobState:
        state = JobState(id=uuid.uuid4().hex, mode=mode, already_preprocessed=already_preprocessed, source=source)
        directory = self.path(state.id)
        for name in ("input", "work", "output"):
            (directory / name).mkdir(parents=True, exist_ok=True)
        self.save(state)
        return state

    def save(self, state: JobState) -> None:
        state.updated_at = time.time()
        directory = self.path(state.id)
        with self._lock:
            if not directory.exists():  # deleted while running
                return
            temporary = directory / "job.json.tmp"
            temporary.write_text(json.dumps(asdict(state)), encoding="utf-8")
            temporary.replace(directory / "job.json")

    def load(self, job_id: str) -> JobState:
        file = self.path(job_id) / "job.json"
        with self._lock:
            if not file.exists():
                raise KeyError(job_id)
            return JobState(**json.loads(file.read_text(encoding="utf-8")))

    def exists(self, job_id: str) -> bool:
        try:
            return (self.path(job_id) / "job.json").exists()
        except KeyError:
            return False

    def delete(self, job_id: str) -> None:
        directory = self.path(job_id)
        with self._lock:
            shutil.rmtree(directory, ignore_errors=True)

    def all_ids(self) -> list[str]:
        return [p.name for p in self.root.iterdir() if p.is_dir() and JOB_ID_PATTERN.match(p.name)]

    def cleanup_expired(self, ttl_hours: float) -> int:
        removed = 0
        cutoff = time.time() - ttl_hours * 3600.0
        for job_id in self.all_ids():
            try:
                created = self.load(job_id).created_at
            except Exception:
                created = (self.root / job_id).stat().st_mtime
            if created < cutoff:
                self.delete(job_id)
                removed += 1
        if removed:
            log.info("auto-deleted %d expired job(s)", removed)
        return removed

    def fail_interrupted(self) -> None:
        """Jobs left mid-run by a server restart cannot be resumed."""
        for job_id in self.all_ids():
            try:
                state = self.load(job_id)
            except Exception:
                continue
            if state.status in ACTIVE_STAGES or state.status == "uploading":
                state.status = "error"
                state.error = "The server restarted while this job was running. Please submit it again."
                self.save(state)


class Worker:
    """One daemon thread pulling job ids off a queue."""

    def __init__(self, store: JobStore, run: Callable[[JobState, Callable], None]):
        self.store = store
        self._run = run
        self._queue: queue.Queue[str] = queue.Queue()
        self._thread = threading.Thread(target=self._loop, name="glioma-worker", daemon=True)
        self._stop = threading.Event()

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._queue.put("")

    def submit(self, job_id: str) -> None:
        self._queue.put(job_id)

    def queue_length(self) -> int:
        return self._queue.qsize()

    def _loop(self) -> None:
        while not self._stop.is_set():
            job_id = self._queue.get()
            if not job_id:
                continue
            try:
                state = self.store.load(job_id)
            except KeyError:
                continue  # deleted before it started

            def report(stage: str, progress: float, message: str = "", _state=state) -> None:
                _state.status, _state.progress, _state.message = stage, float(progress), message
                self.store.save(_state)

            try:
                self._run(state, report)
                state.status, state.progress, state.message = "done", 1.0, "Finished"
            except Exception as error:
                from .io_utils import InputError

                state.status = "error"
                if isinstance(error, InputError):
                    state.error = str(error)
                else:
                    # Exception text is safe to show: the pipeline never puts
                    # file names or header content into messages.
                    state.error = f"{type(error).__name__}: {error}"
                    log.exception("job %s failed", job_id)
            finally:
                shutil.rmtree(self.store.path(job_id) / "work", ignore_errors=True)
                self.store.save(state)


class CleanupThread:
    def __init__(self, store: JobStore, ttl_hours: float, interval_seconds: float = 600.0):
        self._store, self._ttl, self._interval = store, ttl_hours, interval_seconds
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="glioma-cleanup", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._store.cleanup_expired(self._ttl)
            except Exception:
                log.exception("cleanup failed")
            self._stop.wait(self._interval)
