"""Write tests/fixtures/alignment_case.json.

The same file is read by the backend tests (pytest) and the frontend tests
(vitest), so both lookups are checked against one set of world-mm probes.

    python scripts/make_alignment_fixture.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.geometry import inverse_affine, voxel_to_world  # noqa: E402
from tests import synthetic  # noqa: E402


def build() -> dict:
    affine = synthetic.oblique_affine()
    labels = synthetic.label_volume()
    probes = []

    def add(ijk_continuous, note):
        index = np.floor(np.asarray(ijk_continuous, dtype=float) + 0.5).astype(int)
        inside = bool(np.all(index >= 0) and np.all(index < np.array(synthetic.SHAPE)))
        probes.append(
            {
                "note": note,
                "world_mm": voxel_to_world(ijk_continuous, affine).tolist(),
                "voxel": index.tolist(),
                "in_bounds": inside,
                "label": int(labels[tuple(index)]) if inside else None,
            }
        )

    for value, (lo, hi) in synthetic.BOXES.items():
        lo, hi = np.array(lo, float), np.array(hi, float)
        # Voxel centre nearest the box centre (the exact centre can sit on a
        # voxel boundary, where the containing voxel is ambiguous).
        add(np.floor((lo + hi) / 2.0), f"label {value}: central voxel")
        add(lo, f"label {value}: lowest corner voxel centre")
        add(hi, f"label {value}: highest corner voxel centre")
        add(lo + [0.4, -0.4, 0.4], f"label {value}: off-centre inside the lowest corner voxel")
        add(hi + [0.45, 0.45, 0.45], f"label {value}: just inside the outer face")
        add(hi + [0.55, 0.0, 0.0], f"label {value}: just outside the outer face (background)")
        add(lo - [1.0, 0.0, 0.0], f"label {value}: one voxel outside (background)")
    add([0, 0, 0], "volume corner (background)")
    add(np.array(synthetic.SHAPE) - 1.0, "opposite volume corner (background)")
    add([-3.0, 5.0, 5.0], "outside the volume")
    add(np.array(synthetic.SHAPE) + 2.0, "outside the volume")

    return {
        "description": "Oblique, anisotropic affine with known label boxes. Voxel boxes are inclusive.",
        "shape": list(synthetic.SHAPE),
        "spacing_mm": list(synthetic.SPACING),
        "affine": affine.tolist(),
        "inverse_affine": inverse_affine(affine).tolist(),
        "boxes": [
            {"label": value, "voxel_min": list(lo), "voxel_max": list(hi)}
            for value, (lo, hi) in synthetic.BOXES.items()
        ],
        "probes": probes,
    }


if __name__ == "__main__":
    target = ROOT / "tests" / "fixtures" / "alignment_case.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(build(), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {target} ({len(build()['probes'])} probes)")
