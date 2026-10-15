"""Extract the nnU-Net results of the study into app/resources/study_metrics.json.

    python scripts/build_study_metrics.py <path to mu-glioma-segmentation-benchmark checkout>

Reads only the published summary tables under results/final_analysis/. The app
shows these numbers as the model's validated performance; nothing here is
recomputed.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURE = "nnU-Net v2 3D fullres"
TARGETS = {
    "Non-enhancing tumor core": {"key": "label_1", "labels": [1]},
    "FLAIR hyperintensity / edema": {"key": "label_2", "labels": [2]},
    "Enhancing tissue": {"key": "label_3", "labels": [3]},
    "Resection cavity": {"key": "label_4", "labels": [4]},
    "Tumor-related region (1-3)": {"key": "tumor_related_1_3", "labels": [1, 2, 3]},
    "All postoperative regions (1-4)": {"key": "all_postoperative_1_4", "labels": [1, 2, 3, 4]},
}


def rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row.get("architecture", row.get("comparator")) == ARCHITECTURE]


def number(value: str):
    return None if value in ("", "nan", "NaN") else round(float(value), 4)


def build(repo: Path) -> dict:
    results = repo / "results" / "final_analysis"
    commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()

    primary = rows(results / "primary_model_summary.csv")[0]
    targets = []
    for row in rows(results / "target_metrics_summary.csv"):
        target = TARGETS[row["target"]]
        targets.append(
            {
                "key": target["key"],
                "name": row["target"],
                "labels": target["labels"],
                "patients": int(row["reference_present_patients"]),
                "scans": int(row["reference_present_scans"]),
                "dice": number(row["mean_patient_dice"]),
                "dice_ci95": [number(row["dice_bootstrap_ci95_low"]), number(row["dice_bootstrap_ci95_high"])],
                "precision": number(row["mean_patient_pooled_precision"]),
                "recall": number(row["mean_patient_pooled_recall"]),
                "hd95_mm_median": number(row["median_patient_hd95_mm"]),
                "surface_dice_1mm": number(row["mean_patient_surface_dice_1mm"]),
                "lesion_sensitivity": number(row["mean_patient_lesion_sensitivity"]),
            }
        )
    absent = {}
    for row in rows(results / "absent_reference_summary.csv"):
        absent[TARGETS[row["target"]]["key"]] = {
            "absent_scans": int(row["absent_reference_scans"]),
            "false_positive_scans": int(row["false_positive_scans"]),
            "false_positive_scan_rate": number(row["false_positive_scan_rate"]),
            "rate_ci95": [number(row["scan_rate_wilson_ci95_low"]), number(row["scan_rate_wilson_ci95_high"])],
            "median_predicted_ml": number(row["median_predicted_volume_ml_all_absent_scans"]),
        }
    for target in targets:
        target["when_absent"] = absent.get(target["key"])

    folds = [
        {"fold": int(row["cv_fold"]), "patients": int(row["n_patients"]), "mean_dice": number(row["mean_dice"]), "median_dice": number(row["median_dice"])}
        for row in rows(results / "fold_summary.csv")
    ]
    seeds = rows(results / "seed_sensitivity_summary.csv")[0]
    complementary = rows(results / "complementary_162_patient_summary.csv")[0]
    with open(results / "primary_paired_comparisons.csv", newline="", encoding="utf-8") as handle:
        versus = next(r for r in csv.DictReader(handle) if r["comparator"] == ARCHITECTURE)

    return {
        "source": {
            "repository": "https://github.com/glioma-ml-research/mu-glioma-segmentation-benchmark",
            "commit": commit,
            "files": "results/final_analysis/*.csv",
        },
        "model": "nnU-Net v2 3d_fullres",
        "evaluation": (
            "Out-of-fold predictions of five patient-grouped outer folds, one score per patient, "
            "internal validation on MU-Glioma-Post only. Overlap metrics use reference-present scans."
        ),
        "primary": {
            "target": "Tumor-related region (labels 1-3)",
            "patients": int(primary["n_patients"]),
            "mean_dice": number(primary["mean_dice"]),
            "ci95": [number(primary["bootstrap_ci95_low"]), number(primary["bootstrap_ci95_high"])],
            "median_dice": number(primary["median_dice"]),
            "iqr": [number(primary["q1_dice"]), number(primary["q3_dice"])],
            "sd": number(primary["sd_dice"]),
        },
        "targets": targets,
        "folds": folds,
        "seed_range_fold1": number(seeds["range_across_seed_means"]),
        "unseen_162_patients": {
            "patients": int(complementary["n_patients"]),
            "mean_dice": number(complementary["mean_dice"]),
            "ci95": [number(complementary["bootstrap_ci95_low"]), number(complementary["bootstrap_ci95_high"])],
        },
        "versus_residual_unet": {
            "median_paired_difference": number(versus["median_paired_difference"]),
            "ci95": [number(versus["bootstrap_ci95_low"]), number(versus["bootstrap_ci95_high"])],
        },
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    metrics = build(Path(sys.argv[1]))
    target = ROOT / "app" / "resources" / "study_metrics.json"
    target.write_text(json.dumps(metrics, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {target}: primary Dice {metrics['primary']['mean_dice']} {metrics['primary']['ci95']}")
