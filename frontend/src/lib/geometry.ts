/**
 * The single place in the frontend where voxel indices and world coordinates
 * are converted. Mirrors backend/app/geometry.py.
 *
 * World frame: NIfTI scanner/aligned space in millimetres, RAS+
 * (x → patient Right, y → Anterior, z → Superior). Voxel indices address voxel
 * centres. Matrices are 4x4, row-major, exactly as sent by the backend.
 *
 * Slice views, the 3D scene, hover lookups and the crosshair all go through
 * voxelToWorld / worldToVoxel. Nothing else multiplies by an affine.
 */

export type Vec3 = [number, number, number];
export type Mat4 = number[][];

function apply(matrix: Mat4, p: Vec3): Vec3 {
  return [
    matrix[0][0] * p[0] + matrix[0][1] * p[1] + matrix[0][2] * p[2] + matrix[0][3],
    matrix[1][0] * p[0] + matrix[1][1] * p[1] + matrix[1][2] * p[2] + matrix[1][3],
    matrix[2][0] * p[0] + matrix[2][1] * p[1] + matrix[2][2] * p[2] + matrix[2][3],
  ];
}

/** Voxel index (may be fractional) → world mm. */
export function voxelToWorld(ijk: Vec3, affine: Mat4): Vec3 {
  return apply(affine, ijk);
}

/** World mm → continuous voxel index, using the inverse affine from the volume header. */
export function worldToVoxel(xyz: Vec3, inverseAffine: Mat4): Vec3 {
  return apply(inverseAffine, xyz);
}

/** Index of the voxel whose cell contains the point: floor(x + 0.5), as in the backend. */
export function roundToIndex(ijk: Vec3): Vec3 {
  return [Math.floor(ijk[0] + 0.5), Math.floor(ijk[1] + 0.5), Math.floor(ijk[2] + 0.5)];
}

export function worldToVoxelIndex(xyz: Vec3, inverseAffine: Mat4): Vec3 {
  return roundToIndex(worldToVoxel(xyz, inverseAffine));
}

export function inBounds(ijk: Vec3, shape: number[]): boolean {
  return ijk[0] >= 0 && ijk[1] >= 0 && ijk[2] >= 0 && ijk[0] < shape[0] && ijk[1] < shape[1] && ijk[2] < shape[2];
}

/** Offset into a Fortran-ordered buffer (i fastest), the layout the backend serves. */
export function voxelOffset(ijk: Vec3, shape: number[]): number {
  return ijk[0] + shape[0] * (ijk[1] + shape[1] * ijk[2]);
}

/** Direction (not point) transform: applies only the 3x3 part. */
export function transformDirection(matrix: Mat4, d: Vec3): Vec3 {
  return [
    matrix[0][0] * d[0] + matrix[0][1] * d[1] + matrix[0][2] * d[2],
    matrix[1][0] * d[0] + matrix[1][1] * d[1] + matrix[1][2] * d[2],
    matrix[2][0] * d[0] + matrix[2][1] * d[1] + matrix[2][2] * d[2],
  ];
}

/** General 4x4 affine inverse; used by tests to check the header's inverse_affine. */
export function invertAffine(m: Mat4): Mat4 {
  const [a, b, c] = [m[0][0], m[0][1], m[0][2]];
  const [d, e, f] = [m[1][0], m[1][1], m[1][2]];
  const [g, h, i] = [m[2][0], m[2][1], m[2][2]];
  const det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
  if (Math.abs(det) < 1e-12) throw new Error("affine is singular");
  const r = [
    [(e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det],
    [(f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det],
    [(d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det],
  ];
  const t: Vec3 = [m[0][3], m[1][3], m[2][3]];
  return [
    [...r[0], -(r[0][0] * t[0] + r[0][1] * t[1] + r[0][2] * t[2])],
    [...r[1], -(r[1][0] * t[0] + r[1][1] * t[1] + r[1][2] * t[2])],
    [...r[2], -(r[2][0] * t[0] + r[2][1] * t[1] + r[2][2] * t[2])],
    [0, 0, 0, 1],
  ];
}
