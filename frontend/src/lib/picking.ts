import { transformDirection, voxelOffset, voxelToWorld, worldToVoxel, type Vec3 } from "./geometry";
import type { Volume } from "./types";

export interface RayHit {
  voxel: Vec3;
  /** World mm of the hit voxel's centre. */
  world: Vec3;
  label: number;
  /** Distance from the ray origin to where the ray enters the voxel, in mm. */
  distance: number;
}

/**
 * First voxel along a world-space ray whose label passes `accept`.
 *
 * The ray is mapped into voxel space with the inverse affine and walked cell by
 * cell (Amanatides–Woo), so the result is the exact voxel under the cursor —
 * no dependence on mesh smoothing or decimation. `accept` receives the world
 * position of the voxel centre so callers can honour a clipping plane.
 */
export function pickVoxelAlongRay(
  origin: Vec3,
  direction: Vec3,
  labels: Volume,
  accept: (label: number, world: Vec3) => boolean,
): RayHit | null {
  const { shape, affine, inverse_affine: inverse } = labels.header;
  const length = Math.hypot(direction[0], direction[1], direction[2]);
  if (length === 0) return null;
  const unit: Vec3 = [direction[0] / length, direction[1] / length, direction[2] / length];

  // Ray in voxel space, parameterised by t in mm along the world ray.
  const start = worldToVoxel(origin, inverse);
  const step = transformDirection(inverse, unit);

  // Clip to the grid's cell bounds [-0.5, n - 0.5] on each axis.
  let tEnter = 0;
  let tExit = Infinity;
  for (let axis = 0; axis < 3; axis++) {
    const low = -0.5;
    const high = shape[axis] - 0.5;
    if (Math.abs(step[axis]) < 1e-12) {
      if (start[axis] < low || start[axis] > high) return null;
      continue;
    }
    let t0 = (low - start[axis]) / step[axis];
    let t1 = (high - start[axis]) / step[axis];
    if (t0 > t1) [t0, t1] = [t1, t0];
    tEnter = Math.max(tEnter, t0);
    tExit = Math.min(tExit, t1);
    if (tEnter > tExit) return null;
  }

  const epsilon = 1e-6;
  let t = tEnter + epsilon;
  const cell: Vec3 = [0, 0, 0];
  const delta: Vec3 = [0, 0, 0];
  const next: Vec3 = [0, 0, 0];
  const direction3: Vec3 = [0, 0, 0];
  for (let axis = 0; axis < 3; axis++) {
    const position = start[axis] + step[axis] * t;
    cell[axis] = Math.min(shape[axis] - 1, Math.max(0, Math.floor(position + 0.5)));
    if (Math.abs(step[axis]) < 1e-12) {
      direction3[axis] = 0;
      delta[axis] = Infinity;
      next[axis] = Infinity;
    } else {
      direction3[axis] = step[axis] > 0 ? 1 : -1;
      delta[axis] = Math.abs(1 / step[axis]);
      const boundary = cell[axis] + 0.5 * direction3[axis];
      next[axis] = (boundary - start[axis]) / step[axis];
    }
  }

  const maxSteps = shape[0] + shape[1] + shape[2] + 3;
  for (let iteration = 0; iteration < maxSteps; iteration++) {
    const label = labels.data[voxelOffset(cell, shape)];
    if (label !== 0) {
      const world = voxelToWorld(cell, affine);
      if (accept(label, world)) return { voxel: [cell[0], cell[1], cell[2]], world, label, distance: t };
    }
    const axis = next[0] <= next[1] ? (next[0] <= next[2] ? 0 : 2) : next[1] <= next[2] ? 1 : 2;
    t = next[axis];
    if (t > tExit) return null;
    cell[axis] += direction3[axis];
    if (cell[axis] < 0 || cell[axis] >= shape[axis]) return null;
    next[axis] += delta[axis];
  }
  return null;
}
