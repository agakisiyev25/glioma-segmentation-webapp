"""Write the synthetic, non-patient sample case to disk.

    python scripts/make_sample_case.py            # → backend/data/sample_case/

Produces t1 / t1c / t2 / flair and a segmentation with labels 0–4, for trying
the upload page by hand. (In demo mode the upload page also has a one-click
"synthetic sample" button that needs no files.) See app/sample_case.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.sample_case import SHAPE, SPACING, ensure_sample_case  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "sample_case")
    arguments = parser.parse_args()

    paths = ensure_sample_case(arguments.out)
    image = nib.load(str(paths["segmentation"]))
    labels = np.asanyarray(image.dataobj)
    voxel_ml = abs(np.linalg.det(image.affine[:3, :3])) / 1000.0
    print(f"sample case in {arguments.out}")
    print(f"grid {SHAPE}, spacing {SPACING} mm, stored orientation {''.join(nib.aff2axcodes(image.affine))}")
    print("label volumes (mL):", {v: round(float((labels == v).sum()) * voxel_ml, 2) for v in range(1, 5)})
