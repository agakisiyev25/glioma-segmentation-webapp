import type { Mat4, Vec3 } from "./geometry";

export interface LabelInfo {
  value: number;
  key: string;
  short: string;
  name: string;
  color: string;
}

export type AppMode = "inference" | "demo" | "model_error";

export interface ModelMember {
  name: string;
  trainer: string | null;
  configuration: string | null;
  target_spacing_mm: number[] | null;
  patch_size: number[] | null;
  checkpoints: { file: string; sha256: string | null }[];
}

export interface ModelInfo {
  mode?: "inference" | "demo";
  description?: string;
  family?: string;
  configuration?: string;
  nnunetv2_version?: string;
  torch_version?: string;
  device?: string;
  gpu?: string | null;
  ensemble?: string;
  test_time_mirroring?: boolean;
  members?: ModelMember[];
  notes?: string[];
  paper?: { title: string; reported_value: number; reported_ci95: [number, number]; primary_metric: string; evaluation: string };
}

export interface ApiInfo {
  app_version: string;
  mode: AppMode;
  message: string;
  banner: string | null;
  model: ModelInfo | null;
  disclaimer: string;
  scope_note: string;
  confidence_note: string;
  label_map: LabelInfo[];
  sequences: { slot: string; name: string }[];
  hd_bet_available: boolean;
  job_ttl_hours: number;
  max_upload_mb: number;
  queue_length: number;
  sample_case_available: boolean;
}

export type JobStage = "uploading" | "queued" | "preprocessing" | "inference" | "meshing" | "done" | "error";

export interface JobStatus {
  id: string;
  status: JobStage;
  progress: number;
  message: string;
  error: string | null;
  mode: "inference" | "demo";
  already_preprocessed: boolean;
  source: "upload" | "sample";
  warnings: string[];
  created_at: number;
  expires_at: number;
  queue_length: number;
}

export interface BoundingBox {
  voxel_min: number[];
  voxel_max: number[];
  world_min_mm: number[];
  world_max_mm: number[];
}

export interface RegionStats {
  voxels: number;
  volume_ml: number;
  component_count: number;
  components: { voxels: number; volume_ml: number; centroid_world_mm: number[]; bounding_box: BoundingBox }[];
  components_truncated?: boolean;
  bounding_box: BoundingBox | null;
  centroid_world_mm: number[] | null;
}

export interface LabelStats extends RegionStats {
  value: number;
  short: string;
  name: string;
  color: string;
}

export interface MeshStats {
  name: string;
  vertices: number;
  faces: number;
  bounds_world_mm: { min: number[]; max: number[] };
}

export interface Summary {
  job_id: string;
  mode: "inference" | "demo";
  demo: boolean;
  synthetic_sample: boolean;
  banner: string | null;
  disclaimer: string;
  scope_note: string;
  confidence_note: string;
  confidence_available: boolean;
  already_preprocessed: boolean;
  label_map: LabelInfo[];
  model: ModelInfo;
  statistics: {
    voxel_volume_mm3: number;
    connectivity: string;
    labels: LabelStats[];
    composites: Record<string, RegionStats & { name: string; labels: number[] }>;
  };
  meshes: Record<string, MeshStats>;
  volumes_available: string[];
  grid: {
    viewer: { shape: number[]; affine: Mat4; inverse_affine: Mat4; spacing_mm: number[]; axcodes: string };
    uploaded_t1c: { shape: number[]; affine: Mat4; spacing_mm: number[]; axcodes: string };
    world_frame: string;
    note: string;
  };
  warnings: string[];
  runtime_seconds: number;
}

export interface VolumeHeader {
  name: string;
  kind: "labels" | "confidence" | "intensity";
  dtype: "uint8" | "int16";
  shape: number[];
  order: "F";
  affine: Mat4;
  inverse_affine: Mat4;
  spacing_mm: number[];
  axcodes: string;
  scl_slope: number;
  scl_inter: number;
  display_range?: [number, number];
}

export interface Volume {
  header: VolumeHeader;
  data: Uint8Array | Int16Array;
}

/** What the info box shows for one point. Produced identically for 2D and 3D. */
export interface VoxelInfo {
  world: Vec3;
  voxel: Vec3;
  inBounds: boolean;
  label: number | null;
  confidence: number | null;
  t1c: number | null;
  flair: number | null;
  source: "2d" | "3d";
}

/** Response of GET /api/jobs/{id}/voxel. */
export interface BackendVoxel {
  world_mm: number[];
  voxel: number[];
  voxel_continuous: number[];
  in_bounds: boolean;
  label: number | null;
  label_short: string | null;
  label_name: string | null;
  color: string | null;
  confidence: number | null;
  intensities: { t1c: number | null; flair: number | null };
}

export interface StudyTarget {
  key: string;
  name: string;
  labels: number[];
  patients: number;
  scans: number;
  dice: number;
  dice_ci95: [number, number];
  precision: number;
  recall: number;
  hd95_mm_median: number;
  surface_dice_1mm: number;
  lesion_sensitivity: number;
  when_absent: { absent_scans: number; false_positive_scans: number; false_positive_scan_rate: number; rate_ci95: [number, number]; median_predicted_ml: number } | null;
}

/** The study's published nnU-Net results (GET /api/model/metrics). */
export interface StudyMetrics {
  source: { repository: string; commit: string; files: string };
  model: string;
  evaluation: string;
  primary: { target: string; patients: number; mean_dice: number; ci95: [number, number]; median_dice: number; iqr: [number, number]; sd: number };
  targets: StudyTarget[];
  folds: { fold: number; patients: number; mean_dice: number; median_dice: number }[];
  seed_range_fold1: number;
  unseen_162_patients: { patients: number; mean_dice: number; ci95: [number, number] };
  versus_residual_unet: { median_paired_difference: number; ci95: [number, number] };
}
