import { create } from "zustand";

import type { Vec3 } from "../lib/geometry";
import type { VoxelInfo } from "../lib/types";

export type BaseVolume = "t1c" | "flair";
export type Verification = "idle" | "pending" | "match" | "mismatch" | "unavailable";
export type ViewId = "3d" | "axial" | "coronal" | "sagittal";

export interface LayerState {
  visible: boolean;
  opacity: number;
}

export interface ClipState {
  enabled: boolean;
  /** World axis the plane is perpendicular to: 0 = R/L, 1 = A/P, 2 = S/I. */
  axis: 0 | 1 | 2;
  /** 0–1 along the scene bounds on that axis. */
  fraction: number;
  flip: boolean;
}

/** Where the 3D camera sits relative to the head, as a world (RAS) direction. */
export const CAMERA_VIEWS: { key: string; title: string; direction: Vec3 }[] = [
  { key: "A", title: "From the front (anterior)", direction: [0, 1, 0] },
  { key: "P", title: "From behind (posterior)", direction: [0, -1, 0] },
  { key: "R", title: "From the patient's right", direction: [1, 0, 0] },
  { key: "L", title: "From the patient's left", direction: [-1, 0, 0] },
  // Slightly off the vertical so that anterior stays "up" on screen.
  { key: "S", title: "From above (superior)", direction: [0, -0.02, 1] },
  { key: "I", title: "From below (inferior)", direction: [0, 0.02, -1] },
];
const DEFAULT_CAMERA_DIRECTION: Vec3 = [0.9, 1.35, 0.75]; // anterior-right-superior

interface ViewerState {
  /** The one shared crosshair, in world mm (RAS). 2D and 3D both read and write it. */
  crosshair: Vec3 | null;
  hover: VoxelInfo | null;
  picked: VoxelInfo | null;
  verification: Verification;

  base: BaseVolume;
  windows: Record<BaseVolume, { low: number; high: number }>;
  overlayOpacity: number;

  brain: LayerState;
  labels: Record<number, LayerState>;
  voxelMode: boolean;
  surfaceVoxelsOnly: boolean;
  clip: ClipState;
  camera: { direction: Vec3; nonce: number };
  maximized: ViewId | null;

  setCrosshair: (world: Vec3) => void;
  setHover: (info: VoxelInfo | null) => void;
  setPicked: (info: VoxelInfo | null) => void;
  setVerification: (state: Verification) => void;
  setBase: (base: BaseVolume) => void;
  setWindow: (base: BaseVolume, low: number, high: number) => void;
  setOverlayOpacity: (opacity: number) => void;
  setBrain: (patch: Partial<LayerState>) => void;
  setLabel: (value: number, patch: Partial<LayerState>) => void;
  setVoxelMode: (enabled: boolean) => void;
  setSurfaceVoxelsOnly: (enabled: boolean) => void;
  setClip: (patch: Partial<ClipState>) => void;
  lookFrom: (direction: Vec3) => void;
  resetView: () => void;
  toggleMaximized: (view: ViewId) => void;
  resetAll: () => void;
}

const initial = {
  crosshair: null as Vec3 | null,
  hover: null as VoxelInfo | null,
  picked: null as VoxelInfo | null,
  verification: "idle" as Verification,
  base: "t1c" as BaseVolume,
  windows: { t1c: { low: 0, high: 1 }, flair: { low: 0, high: 1 } },
  overlayOpacity: 0.5,
  brain: { visible: true, opacity: 0.22 },
  labels: {
    1: { visible: true, opacity: 1 },
    2: { visible: true, opacity: 0.55 }, // edema usually wraps the others
    3: { visible: true, opacity: 1 },
    4: { visible: true, opacity: 1 },
  } as Record<number, LayerState>,
  voxelMode: false,
  surfaceVoxelsOnly: true,
  clip: { enabled: false, axis: 2, fraction: 0.5, flip: false } as ClipState,
  maximized: null as ViewId | null,
};

export const useViewerStore = create<ViewerState>((set) => ({
  ...initial,
  camera: { direction: DEFAULT_CAMERA_DIRECTION, nonce: 0 },
  setCrosshair: (crosshair) => set({ crosshair }),
  setHover: (hover) => set({ hover }),
  setPicked: (picked) => set({ picked }),
  setVerification: (verification) => set({ verification }),
  setBase: (base) => set({ base }),
  setWindow: (base, low, high) => set((state) => ({ windows: { ...state.windows, [base]: { low, high } } })),
  setOverlayOpacity: (overlayOpacity) => set({ overlayOpacity }),
  setBrain: (patch) => set((state) => ({ brain: { ...state.brain, ...patch } })),
  setLabel: (value, patch) =>
    set((state) => ({ labels: { ...state.labels, [value]: { ...state.labels[value], ...patch } } })),
  setVoxelMode: (voxelMode) => set({ voxelMode }),
  setSurfaceVoxelsOnly: (surfaceVoxelsOnly) => set({ surfaceVoxelsOnly }),
  setClip: (patch) => set((state) => ({ clip: { ...state.clip, ...patch } })),
  lookFrom: (direction) => set((state) => ({ camera: { direction, nonce: state.camera.nonce + 1 } })),
  resetView: () => set((state) => ({ camera: { direction: DEFAULT_CAMERA_DIRECTION, nonce: state.camera.nonce + 1 } })),
  toggleMaximized: (view) => set((state) => ({ maximized: state.maximized === view ? null : view })),
  resetAll: () =>
    set((state) => ({ ...initial, camera: { direction: DEFAULT_CAMERA_DIRECTION, nonce: state.camera.nonce + 1 } })),
}));
