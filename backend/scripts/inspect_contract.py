"""Print the facts docs/MODEL_CONTRACT.md marks as [unverified].

    python scripts/inspect_contract.py --case-dir "D:/MU-Glioma-Post/PatientID_0005/Timepoint_1"
    python scripts/inspect_contract.py --model-dir "D:/weights"

Reads headers and JSON only; prints no file names from inside the case folder.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.inference import ModelContractError, discover_members, validate_dataset_json  # noqa: E402

SUFFIXES = {"T1": "_brain_t1n", "T1c": "_brain_t1c", "T2": "_brain_t2w", "FLAIR": "_brain_t2f", "mask": "_tumorMask"}


def inspect_case(case_dir: Path) -> None:
    print(f"Case folder: {case_dir.name}")
    reference = None
    for name, suffix in SUFFIXES.items():
        matches = sorted(case_dir.glob(f"*{suffix}.nii*"))
        if not matches:
            print(f"  {name:<6} missing")
            continue
        image = nib.load(str(matches[0]))
        data = np.asanyarray(image.dataobj)
        spacing = tuple(round(float(v), 4) for v in image.header.get_zooms()[:3])
        codes = "".join(nib.aff2axcodes(image.affine))
        extra = (
            f"labels {sorted(int(v) for v in np.unique(data))}"
            if name == "mask"
            else f"zero voxels {float((data == 0).mean()):.0%}"
        )
        same = "" if reference is None else ("  same grid as T1" if np.allclose(image.affine, reference, atol=1e-3) else "  DIFFERENT GRID")
        reference = image.affine if reference is None else reference
        print(f"  {name:<6} shape {image.shape}  spacing {spacing} mm  stored orientation {codes}  dtype {data.dtype}  {extra}{same}")
    if reference is not None:
        print("  affine of the first sequence:")
        print("   ", np.array2string(np.asarray(reference), precision=4, suppress_small=True).replace("\n", "\n    "))
    print("  → set TRAINING_ORIENTATION to the stored orientation above if it is not LPS.")


def inspect_models(model_dir: Path) -> None:
    members = discover_members(model_dir)
    print(f"Model folder: {len(members)} model(s) found")
    for member in members:
        plans = json.loads((member.folder / "plans.json").read_text())
        dataset = json.loads((member.folder / "dataset.json").read_text())
        configuration = plans["configurations"].get("3d_fullres", {})
        try:
            validate_dataset_json(dataset, member.name)
            contract = "matches the contract"
        except ModelContractError as error:
            contract = f"CONTRACT VIOLATION: {error}"
        print(f"  {member.name}")
        print(f"    checkpoints      {[p.name if p.parent == member.folder else f'{p.parent.name}/{p.name}' for p in member.checkpoints]}")
        print(f"    dataset          {plans.get('dataset_name')}  plans {plans.get('plans_name')}")
        print(f"    channel_names    {dataset.get('channel_names')}")
        print(f"    labels           {dataset.get('labels')}  → {contract}")
        print(f"    reader/writer    {plans.get('image_reader_writer')}")
        print(f"    target spacing   {configuration.get('spacing')} (z, y, x)")
        print(f"    patch size       {configuration.get('patch_size')}  batch {configuration.get('batch_size')}")
        print(f"    median shape     {configuration.get('median_image_size_in_voxels')}")
        print(f"    normalization    {configuration.get('normalization_schemes')}")
        print(f"    transpose        {plans.get('transpose_forward')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case-dir", type=Path)
    parser.add_argument("--model-dir", type=Path)
    arguments = parser.parse_args()
    if not arguments.case_dir and not arguments.model_dir:
        parser.error("give --case-dir and/or --model-dir")
    if arguments.case_dir:
        inspect_case(arguments.case_dir)
    if arguments.model_dir:
        inspect_models(arguments.model_dir)
