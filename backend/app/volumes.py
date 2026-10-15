"""Voxel volumes for the frontend: compact binary + JSON header.

Every volume of a job lives on the same *viewer grid*: the uploaded T1c grid
with its axes permuted/flipped to the closest RAS orientation (no
interpolation). The header carries the affine and its inverse, so the frontend
converts between voxel indices and world mm exactly as the backend does.

Binary layout: little-endian, Fortran order (i fastest):
``offset = i + nx * (j + ny * k)``. Stored gzip-compressed and served with
``Content-Encoding: gzip``.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

from .geometry import inverse_affine, voxel_spacing
from .labels import label_map_json
from .preprocess import axcodes_of

VOLUME_NAMES = ("labels", "confidence", "t1c", "flair")


def _intensity_to_int16(data: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Quantise an intensity image to int16; real value = raw * slope + inter."""
    finite_min, finite_max = float(data.min()), float(data.max())
    integral = bool(np.all(data == np.rint(data)))
    if integral and finite_min >= -32768 and finite_max <= 32767:
        return data.astype("<i2"), 1.0, 0.0
    peak = max(abs(finite_min), abs(finite_max), 1e-12)
    slope = peak / 32767.0
    return np.rint(data / slope).astype("<i2"), slope, 0.0


def _display_range(data: np.ndarray) -> list[float]:
    foreground = data[data != 0]
    if foreground.size == 0:
        return [0.0, 1.0]
    low, high = np.percentile(foreground, [0.5, 99.5])
    if high <= low:
        high = low + 1.0
    return [float(low), float(high)]


def write_volume(directory: Path, name: str, data: np.ndarray, affine: np.ndarray, *, kind: str) -> dict:
    """Write ``<name>.bin.gz``, ``<name>.npy`` and ``<name>.json``. Returns the header.

    kind: "labels" (uint8), "confidence" (float 0–1 → uint8 0–255) or
    "intensity" (→ int16 with a scale factor).
    """
    directory.mkdir(parents=True, exist_ok=True)
    slope, inter, display = 1.0, 0.0, None
    if kind == "labels":
        raw = np.asarray(data).astype(np.uint8)
    elif kind == "confidence":
        raw = np.rint(np.clip(np.asarray(data, dtype=np.float32), 0.0, 1.0) * 255.0).astype(np.uint8)
        slope = 1.0 / 255.0
    elif kind == "intensity":
        raw, slope, inter = _intensity_to_int16(np.asarray(data, dtype=np.float32))
        display = _display_range(np.asarray(data, dtype=np.float32))
    else:
        raise ValueError(kind)

    header = {
        "name": name,
        "kind": kind,
        "dtype": "uint8" if raw.dtype == np.uint8 else "int16",
        "byte_order": "little",
        "shape": [int(n) for n in raw.shape],
        "order": "F",
        "index_formula": "offset = i + nx * (j + ny * k)",
        "encoding": "gzip",
        "affine": np.asarray(affine, dtype=np.float64).tolist(),
        "inverse_affine": inverse_affine(affine).tolist(),
        "spacing_mm": list(voxel_spacing(affine)),
        "axcodes": axcodes_of(affine),
        "world_frame": "RAS+ mm (NIfTI scanner/aligned space)",
        "scl_slope": slope,
        "scl_inter": inter,
    }
    if display is not None:
        header["display_range"] = display
    if kind == "labels":
        header["label_map"] = label_map_json()

    np.save(directory / f"{name}.npy", raw)
    payload = np.asfortranarray(raw).tobytes(order="F")
    (directory / f"{name}.bin.gz").write_bytes(gzip.compress(payload, compresslevel=6))
    header["bytes_uncompressed"] = len(payload)
    (directory / f"{name}.json").write_text(json.dumps(header, indent=2), encoding="utf-8")
    return header


def read_header(directory: Path, name: str) -> dict | None:
    path = directory / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def open_array(directory: Path, name: str) -> np.ndarray | None:
    """Memory-mapped raw array indexed ``[i, j, k]``, or None if absent."""
    path = directory / f"{name}.npy"
    return np.load(path, mmap_mode="r") if path.exists() else None
