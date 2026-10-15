/**
 * Alignment and hover/click round-trip tests for the frontend lookups.
 *
 * The probes come from backend/tests/fixtures/alignment_case.json — the same
 * file the backend tests check their own lookups against — so a world point
 * must give the same voxel and label on both sides.
 */
import { describe, expect, it } from "vitest";

import fixture from "../../../backend/tests/fixtures/alignment_case.json";
import {
  invertAffine,
  roundToIndex,
  voxelOffset,
  voxelToWorld,
  worldToVoxel,
  worldToVoxelIndex,
  type Mat4,
  type Vec3,
} from "./geometry";
import { pickVoxelAlongRay } from "./picking";
import type { Volume, VolumeHeader } from "./types";
import { collectLabelVoxels, lookupWorld, makeVolume, type VolumeSet } from "./volumes";

const shape = fixture.shape;
const affine = fixture.affine as Mat4;
const inverse = fixture.inverse_affine as Mat4;

function header(name: string, dtype: "uint8" | "int16", slope = 1): VolumeHeader {
  return {
    name,
    kind: dtype === "uint8" ? "labels" : "intensity",
    dtype,
    shape,
    order: "F",
    affine,
    inverse_affine: inverse,
    spacing_mm: fixture.spacing_mm,
    axcodes: "RAS",
    scl_slope: slope,
    scl_inter: 0,
  };
}

/** Label volume built from the fixture's boxes, in the byte layout the backend serves. */
function labelVolume(): Volume {
  const data = new Uint8Array(shape[0] * shape[1] * shape[2]);
  for (const box of fixture.boxes) {
    for (let k = box.voxel_min[2]; k <= box.voxel_max[2]; k++)
      for (let j = box.voxel_min[1]; j <= box.voxel_max[1]; j++)
        for (let i = box.voxel_min[0]; i <= box.voxel_max[0]; i++) data[voxelOffset([i, j, k], shape)] = box.label;
  }
  return makeVolume(header("labels", "uint8"), data.buffer);
}

function volumeSet(): VolumeSet {
  const voxels = shape[0] * shape[1] * shape[2];
  const intensity = new Int16Array(voxels).map((_, index) => index % 1000);
  const confidence = new Uint8Array(voxels).fill(204);
  return {
    labels: labelVolume(),
    t1c: makeVolume(header("t1c", "int16", 0.5), intensity.buffer),
    flair: makeVolume(header("flair", "int16"), intensity.slice().buffer),
    confidence: makeVolume({ ...header("confidence", "uint8", 1 / 255), kind: "confidence" }, confidence.buffer),
  };
}

describe("voxel ↔ world", () => {
  it("round-trips every voxel of an oblique, anisotropic grid", () => {
    let worst = 0;
    for (let k = 0; k < shape[2]; k++)
      for (let j = 0; j < shape[1]; j++)
        for (let i = 0; i < shape[0]; i++) {
          const world = voxelToWorld([i, j, k], affine);
          const back = worldToVoxel(world, inverse);
          worst = Math.max(worst, Math.abs(back[0] - i), Math.abs(back[1] - j), Math.abs(back[2] - k));
          expect(worldToVoxelIndex(world, inverse)).toEqual([i, j, k]);
        }
    expect(worst).toBeLessThan(1e-9);
  });

  it("computes the same inverse affine as the backend", () => {
    const computed = invertAffine(affine);
    for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) expect(computed[r][c]).toBeCloseTo(inverse[r][c], 10);
  });

  it("rounds to the containing voxel like the backend (floor(x + 0.5))", () => {
    expect(roundToIndex([12.49, 21.51, -0.49])).toEqual([12, 22, 0]);
    expect(roundToIndex([-0.51, 0, 0])).toEqual([-1, 0, 0]);
  });

  it("is not the identity: world mm differ from voxel indices", () => {
    const world = voxelToWorld([13, 24, 10], affine);
    expect(Math.hypot(world[0] - 13, world[1] - 24, world[2] - 10)).toBeGreaterThan(20);
  });
});

describe("hover/click lookup", () => {
  const volumes = volumeSet();

  it("returns the fixture's voxel and label for every world probe", () => {
    const seen = new Set<number>();
    for (const probe of fixture.probes) {
      const info = lookupWorld(probe.world_mm as Vec3, volumes, "2d");
      expect(info.voxel, probe.note).toEqual(probe.voxel);
      expect(info.inBounds, probe.note).toBe(probe.in_bounds);
      expect(info.label, probe.note).toBe(probe.label);
      if (info.label !== null) seen.add(info.label);
    }
    expect([...seen].sort()).toEqual([0, 2, 3, 4]);
  });

  it("scales stored values to real units and reports nothing outside the volume", () => {
    const inside = lookupWorld(voxelToWorld([7, 0, 0], affine), volumes, "3d");
    expect(inside.t1c).toBeCloseTo(3.5);
    expect(inside.flair).toBe(7);
    expect(inside.confidence).toBeCloseTo(0.8);
    const outside = lookupWorld(voxelToWorld([-5, 0, 0], affine), volumes, "3d");
    expect(outside.inBounds).toBe(false);
    expect([outside.label, outside.confidence, outside.t1c, outside.flair]).toEqual([null, null, null, null]);
  });
});

describe("3D picking by ray marching", () => {
  const labels = labelVolume();
  const acceptAll = () => true;

  it("a ray aimed at each box hits that label at the box surface", () => {
    for (const box of fixture.boxes) {
      const centre: Vec3 = [
        Math.floor((box.voxel_min[0] + box.voxel_max[0]) / 2),
        Math.floor((box.voxel_min[1] + box.voxel_max[1]) / 2),
        Math.floor((box.voxel_min[2] + box.voxel_max[2]) / 2),
      ];
      // Start far outside along +i in voxel space, converted to a world ray.
      const target = voxelToWorld(centre, affine);
      const origin = voxelToWorld([shape[0] + 30, centre[1], centre[2]], affine);
      const direction: Vec3 = [target[0] - origin[0], target[1] - origin[1], target[2] - origin[2]];
      const hit = pickVoxelAlongRay(origin, direction, labels, acceptAll);
      expect(hit).not.toBeNull();
      expect(hit!.label).toBe(box.label);
      expect(hit!.voxel).toEqual([box.voxel_max[0], centre[1], centre[2]]); // first voxel met from +i
      // The reported world point looks up to the same label (click → crosshair → lookup).
      expect(lookupWorld(hit!.world, { ...volumeSet(), labels }, "3d").label).toBe(box.label);
      const expectedDistance = Math.hypot(...direction) * ((shape[0] + 30 - (box.voxel_max[0] + 0.5)) / (shape[0] + 30 - centre[0]));
      expect(hit!.distance).toBeCloseTo(expectedDistance, 4);
    }
  });

  it("skips labels the caller rejects (hidden layers) and returns null on a miss", () => {
    const box = fixture.boxes[0];
    const centre: Vec3 = [15, Math.floor((box.voxel_min[1] + box.voxel_max[1]) / 2), Math.floor((box.voxel_min[2] + box.voxel_max[2]) / 2)];
    const origin = voxelToWorld([shape[0] + 30, centre[1], centre[2]], affine);
    const target = voxelToWorld(centre, affine);
    const direction: Vec3 = [target[0] - origin[0], target[1] - origin[1], target[2] - origin[2]];
    expect(pickVoxelAlongRay(origin, direction, labels, (label) => label !== box.label)).toBeNull();

    const away: Vec3 = [-direction[0], -direction[1], -direction[2]];
    expect(pickVoxelAlongRay(origin, away, labels, acceptAll)).toBeNull();
    const corner = voxelToWorld([0, 0, 0], affine);
    const miss: Vec3 = [corner[0] - origin[0], corner[1] - origin[1], corner[2] - origin[2]];
    expect(pickVoxelAlongRay(origin, miss, labels, acceptAll)).toBeNull();
  });

  it("works from an oblique viewpoint, starting inside or outside the volume", () => {
    const box = fixture.boxes[1];
    const centre: Vec3 = [26, 14, 23];
    const target = voxelToWorld(centre, affine);
    for (const origin of [
      [target[0] + 180, target[1] - 140, target[2] + 95] as Vec3,
      voxelToWorld([20, 24, 18], affine),
    ]) {
      const direction: Vec3 = [target[0] - origin[0], target[1] - origin[1], target[2] - origin[2]];
      const hit = pickVoxelAlongRay(origin, direction, labels, (label) => label === box.label);
      expect(hit?.label).toBe(box.label);
      for (let axis = 0; axis < 3; axis++) {
        expect(hit!.voxel[axis]).toBeGreaterThanOrEqual(box.voxel_min[axis]);
        expect(hit!.voxel[axis]).toBeLessThanOrEqual(box.voxel_max[axis]);
      }
    }
  });
});

describe("voxel-mode instances", () => {
  it("lists every tumor voxel, or only the visible shell", () => {
    const labels = labelVolume();
    const all = collectLabelVoxels(labels, false);
    const shell = collectLabelVoxels(labels, true);
    for (const box of fixture.boxes) {
      const size = [0, 1, 2].map((a) => box.voxel_max[a] - box.voxel_min[a] + 1);
      const total = size[0] * size[1] * size[2];
      const interior = Math.max(0, size[0] - 2) * Math.max(0, size[1] - 2) * Math.max(0, size[2] - 2);
      expect(all.get(box.label)!.length / 3).toBe(total);
      expect(shell.get(box.label)!.length / 3).toBe(total - interior);
    }
    expect(all.has(0)).toBe(false);
  });
});
