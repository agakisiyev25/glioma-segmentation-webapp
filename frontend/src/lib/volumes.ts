import { inBounds, voxelOffset, worldToVoxelIndex, type Vec3 } from "./geometry";
import type { Volume, VolumeHeader, VoxelInfo } from "./types";

/** Wrap the raw bytes served by the backend (little-endian, Fortran order). */
export function makeVolume(header: VolumeHeader, buffer: ArrayBuffer): Volume {
  const voxels = header.shape[0] * header.shape[1] * header.shape[2];
  const data = header.dtype === "uint8" ? new Uint8Array(buffer) : new Int16Array(buffer);
  if (data.length !== voxels) {
    throw new Error(`Volume ${header.name}: expected ${voxels} voxels, received ${data.length}.`);
  }
  return { header, data };
}

/** Stored value at a voxel index, scaled to real units (raw * slope + inter). */
export function sampleVolume(volume: Volume, ijk: Vec3): number | null {
  if (!inBounds(ijk, volume.header.shape)) return null;
  const raw = volume.data[voxelOffset(ijk, volume.header.shape)];
  return raw * volume.header.scl_slope + volume.header.scl_inter;
}

export interface VolumeSet {
  labels: Volume;
  t1c: Volume;
  flair: Volume;
  confidence: Volume | null;
}

/**
 * World mm → everything the info box needs. This is the one lookup used by
 * both the 2D slices and the 3D scene, so they cannot disagree.
 */
export function lookupWorld(world: Vec3, volumes: VolumeSet, source: "2d" | "3d"): VoxelInfo {
  const header = volumes.labels.header;
  const voxel = worldToVoxelIndex(world, header.inverse_affine);
  const inside = inBounds(voxel, header.shape);
  return {
    world,
    voxel,
    inBounds: inside,
    label: inside ? sampleVolume(volumes.labels, voxel) : null,
    confidence: inside && volumes.confidence ? sampleVolume(volumes.confidence, voxel) : null,
    t1c: inside ? sampleVolume(volumes.t1c, voxel) : null,
    flair: inside ? sampleVolume(volumes.flair, voxel) : null,
    source,
  };
}

/**
 * Tumor voxels grouped by label, for voxel-mode rendering.
 *
 * With `surfaceOnly`, voxels whose six face neighbours all carry the same label
 * are skipped: they are completely hidden behind their neighbours, so the
 * picture is identical while far fewer cubes are drawn.
 */
export function collectLabelVoxels(labels: Volume, surfaceOnly: boolean): Map<number, Int16Array> {
  const [nx, ny, nz] = labels.header.shape;
  const data = labels.data;
  const lists = new Map<number, number[]>();
  const strideJ = nx;
  const strideK = nx * ny;
  for (let k = 0; k < nz; k++) {
    for (let j = 0; j < ny; j++) {
      let offset = nx * (j + ny * k);
      for (let i = 0; i < nx; i++, offset++) {
        const value = data[offset];
        if (value === 0) continue;
        if (
          surfaceOnly &&
          i > 0 && i < nx - 1 && j > 0 && j < ny - 1 && k > 0 && k < nz - 1 &&
          data[offset - 1] === value && data[offset + 1] === value &&
          data[offset - strideJ] === value && data[offset + strideJ] === value &&
          data[offset - strideK] === value && data[offset + strideK] === value
        ) {
          continue;
        }
        let list = lists.get(value);
        if (!list) lists.set(value, (list = []));
        list.push(i, j, k);
      }
    }
  }
  const result = new Map<number, Int16Array>();
  for (const [value, list] of lists) result.set(value, Int16Array.from(list));
  return result;
}
