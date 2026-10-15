"""Label map and fixed colours. Source of truth: docs/MODEL_CONTRACT.md §1–§2."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LabelInfo:
    value: int
    key: str  # key in the study's nnU-Net dataset.json
    short: str
    name: str
    color: str  # hex, fixed across 2D, 3D and legend


LABELS: tuple[LabelInfo, ...] = (
    LabelInfo(0, "background", "BG", "Background", "#000000"),
    LabelInfo(1, "non_enhancing_tumor_core", "NETC", "Non-enhancing tumor core / necrosis", "#F2C200"),
    LabelInfo(2, "flair_hyperintensity_edema", "SNFH", "Surrounding non-enhancing FLAIR hyperintensity (edema)", "#2FA84F"),
    LabelInfo(3, "enhancing_tissue", "ET", "Enhancing tissue", "#E0332B"),
    LabelInfo(4, "resection_cavity", "RC", "Resection cavity", "#2F7DE1"),
)

LABEL_BY_VALUE = {label.value: label for label in LABELS}
FOREGROUND_LABELS = tuple(label for label in LABELS if label.value != 0)

# Composite targets used by the paper (protocol/ZIP_CONTRACT.md).
COMPOSITE_TARGETS = {
    "tumor_related_1_3": {"name": "Tumor-related region (labels 1–3)", "labels": (1, 2, 3)},
    "all_postoperative_1_4": {"name": "All postoperative regions (labels 1–4)", "labels": (1, 2, 3, 4)},
}

# Contract the weights' dataset.json must satisfy before inference is allowed.
EXPECTED_CHANNEL_NAMES = {"0": "T1", "1": "T1CE", "2": "T2", "3": "FLAIR"}
EXPECTED_DATASET_LABELS = {label.key: label.value for label in LABELS}

# Upload field name -> nnU-Net channel index.
SEQUENCES = ("t1", "t1c", "t2", "flair")
SEQUENCE_DISPLAY = {"t1": "T1", "t1c": "T1c", "t2": "T2", "flair": "FLAIR"}

DISCLAIMER = (
    "Research prototype. Trained and internally validated on the MU-Glioma-Post "
    "dataset only. Not a medical device, not for diagnosis or treatment decisions."
)
SCOPE_NOTE = (
    "The model segments tumor-related sub-regions and the resection cavity. "
    "It does not classify tumor type or grade and does not provide a diagnosis."
)
CONFIDENCE_NOTE = (
    "Confidence is the maximum softmax probability of the model. It is "
    "uncalibrated: calibration was not evaluated in the study."
)


def label_map_json() -> list[dict]:
    return [
        {"value": l.value, "key": l.key, "short": l.short, "name": l.name, "color": l.color}
        for l in LABELS
    ]
