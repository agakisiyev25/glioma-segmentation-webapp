"""Per-label measurements on the label volume."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .geometry import voxel_to_world, voxel_volume_mm3
from .labels import COMPOSITE_TARGETS, FOREGROUND_LABELS

# 26-connectivity, the lesion definition used by the study (protocol/ZIP_CONTRACT.md).
CONNECTIVITY_26 = np.ones((3, 3, 3), dtype=bool)
MAX_COMPONENTS_LISTED = 25


def _box_json(slices, affine) -> dict:
    low = np.array([axis.start for axis in slices], dtype=np.int64)
    high = np.array([axis.stop - 1 for axis in slices], dtype=np.int64)
    # World-space box: the eight voxel-cell corners of the index box, mapped
    # through the affine (an index box is not axis-aligned in world space when
    # the affine is oblique).
    corners_voxel = np.array(
        [
            [x, y, z]
            for x in (low[0] - 0.5, high[0] + 0.5)
            for y in (low[1] - 0.5, high[1] + 0.5)
            for z in (low[2] - 0.5, high[2] + 0.5)
        ]
    )
    corners_world = voxel_to_world(corners_voxel, affine)
    return {
        "voxel_min": low.tolist(),
        "voxel_max": high.tolist(),
        "world_min_mm": corners_world.min(axis=0).tolist(),
        "world_max_mm": corners_world.max(axis=0).tolist(),
    }


def _region_stats(mask: np.ndarray, affine: np.ndarray, voxel_ml: float) -> dict:
    voxels = int(mask.sum())
    if voxels == 0:
        return {
            "voxels": 0,
            "volume_ml": 0.0,
            "component_count": 0,
            "components": [],
            "bounding_box": None,
            "centroid_world_mm": None,
        }

    labelled, count = ndimage.label(mask, structure=CONNECTIVITY_26)
    indices = np.arange(1, count + 1)
    sizes = ndimage.sum_labels(mask, labelled, index=indices).astype(np.int64)
    centroids = np.atleast_2d(ndimage.center_of_mass(mask, labelled, indices))
    boxes = ndimage.find_objects(labelled)

    order = np.argsort(-sizes)
    components = []
    for rank in order[:MAX_COMPONENTS_LISTED]:
        components.append(
            {
                "voxels": int(sizes[rank]),
                "volume_ml": float(sizes[rank] * voxel_ml),
                "centroid_world_mm": voxel_to_world(centroids[rank], affine).tolist(),
                "bounding_box": _box_json(boxes[rank], affine),
            }
        )

    whole_box = ndimage.find_objects(mask.astype(np.uint8))[0]
    centroid_voxel = np.array(ndimage.center_of_mass(mask))
    return {
        "voxels": voxels,
        "volume_ml": float(voxels * voxel_ml),
        "component_count": int(count),
        "components": components,
        "components_truncated": bool(count > MAX_COMPONENTS_LISTED),
        "bounding_box": _box_json(whole_box, affine),
        "centroid_world_mm": voxel_to_world(centroid_voxel, affine).tolist(),
    }


def label_statistics(labels: np.ndarray, affine: np.ndarray) -> dict:
    """Volumes (mL), 26-connected components, bounding boxes and centroids."""
    voxel_ml = voxel_volume_mm3(affine) / 1000.0
    per_label = []
    for info in FOREGROUND_LABELS:
        stats = _region_stats(labels == info.value, affine, voxel_ml)
        per_label.append(
            {"value": info.value, "short": info.short, "name": info.name, "color": info.color, **stats}
        )
    composites = {}
    for key, target in COMPOSITE_TARGETS.items():
        stats = _region_stats(np.isin(labels, target["labels"]), affine, voxel_ml)
        composites[key] = {"name": target["name"], "labels": list(target["labels"]), **stats}
    return {
        "voxel_volume_mm3": voxel_volume_mm3(affine),
        "connectivity": "26-connected",
        "labels": per_label,
        "composites": composites,
    }
