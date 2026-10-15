import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { inBounds, roundToIndex, voxelOffset, voxelToWorld, worldToVoxel, type Vec3 } from "../lib/geometry";
import type { LabelInfo } from "../lib/types";
import { lookupWorld, type VolumeSet } from "../lib/volumes";
import { useViewerStore } from "../state/viewerStore";
import { HoverTip, MaximizeButton, type Tip } from "./ViewChrome";

export type Plane = "axial" | "coronal" | "sagittal";

/**
 * Slices are taken along the voxel axes of the viewer grid, which the backend
 * has reoriented to RAS (i → Right, j → Anterior, k → Superior). Displayed in
 * radiological convention: patient right on screen left.
 */
const PLANES: Record<Plane, { fixed: 0 | 1 | 2; h: 0 | 1 | 2; v: 0 | 1 | 2; edges: [string, string, string, string]; title: string }> = {
  //                                         left, right, top, bottom
  axial: { fixed: 2, h: 0, v: 1, edges: ["R", "L", "A", "P"], title: "Axial" },
  coronal: { fixed: 1, h: 0, v: 2, edges: ["R", "L", "S", "I"], title: "Coronal" },
  sagittal: { fixed: 0, h: 1, v: 2, edges: ["A", "P", "S", "I"], title: "Sagittal" },
};

function hexToRgb(hex: string): [number, number, number] {
  const value = parseInt(hex.slice(1), 16);
  return [(value >> 16) & 255, (value >> 8) & 255, value & 255];
}

interface Props {
  plane: Plane;
  volumes: VolumeSet;
  labelMap: LabelInfo[];
}

export function SliceView({ plane, volumes, labelMap }: Props) {
  const config = PLANES[plane];
  const header = volumes.labels.header;
  const shape = header.shape;
  const nh = shape[config.h];
  const nv = shape[config.v];

  const crosshair = useViewerStore((state) => state.crosshair);
  const base = useViewerStore((state) => state.base);
  const range = useViewerStore((state) => state.windows[base]);
  const overlayOpacity = useViewerStore((state) => state.overlayOpacity);
  const layers = useViewerStore((state) => state.labels);

  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [box, setBox] = useState({ width: 300, height: 300 });
  const [tip, setTip] = useState<Tip | null>(null);

  // Crosshair in continuous voxel coordinates; the slice shown is the one containing it.
  const crossVoxel: Vec3 = useMemo(
    () => (crosshair ? worldToVoxel(crosshair, header.inverse_affine) : [shape[0] / 2, shape[1] / 2, shape[2] / 2]),
    [crosshair, header, shape],
  );
  const slice = Math.min(shape[config.fixed] - 1, Math.max(0, roundToIndex(crossVoxel)[config.fixed]));

  const palette = useMemo(() => {
    const colours: Record<number, [number, number, number]> = {};
    for (const label of labelMap) colours[label.value] = hexToRgb(label.color);
    return colours;
  }, [labelMap]);

  // Offscreen bitmap at native voxel resolution: one pixel per voxel.
  const bitmap = useMemo(() => {
    const canvas = document.createElement("canvas");
    canvas.width = nh;
    canvas.height = nv;
    const context = canvas.getContext("2d")!;
    const image = context.createImageData(nh, nv);
    const volume = volumes[base];
    const { scl_slope: slope, scl_inter: inter } = volume.header;
    const span = Math.max(1e-9, range.high - range.low);
    const ijk: Vec3 = [0, 0, 0];
    ijk[config.fixed] = slice;
    for (let py = 0; py < nv; py++) {
      ijk[config.v] = nv - 1 - py; // S or A at the top
      for (let px = 0; px < nh; px++) {
        ijk[config.h] = nh - 1 - px; // R (axial/coronal) or A (sagittal) on the left
        const offset = voxelOffset(ijk, shape);
        const value = volume.data[offset] * slope + inter;
        let grey = Math.round(((value - range.low) / span) * 255);
        grey = grey < 0 ? 0 : grey > 255 ? 255 : grey;
        let r = grey, g = grey, b = grey;
        const label = volumes.labels.data[offset];
        if (label !== 0 && layers[label]?.visible && overlayOpacity > 0) {
          const colour = palette[label];
          if (colour) {
            r = Math.round(r + (colour[0] - r) * overlayOpacity);
            g = Math.round(g + (colour[1] - g) * overlayOpacity);
            b = Math.round(b + (colour[2] - b) * overlayOpacity);
          }
        }
        const target = 4 * (py * nh + px);
        image.data[target] = r;
        image.data[target + 1] = g;
        image.data[target + 2] = b;
        image.data[target + 3] = 255;
      }
    }
    context.putImageData(image, 0, 0);
    return canvas;
  }, [volumes, base, range, overlayOpacity, layers, palette, slice, config, nh, nv, shape]);

  useEffect(() => {
    const element = containerRef.current;
    if (!element) return;
    const observer = new ResizeObserver(([entry]) => {
      setBox({ width: entry.contentRect.width, height: entry.contentRect.height });
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  // Fit the slice in the panel with the true physical aspect ratio (mm, not voxels).
  const layout = useMemo(() => {
    const physicalWidth = nh * header.spacing_mm[config.h];
    const physicalHeight = nv * header.spacing_mm[config.v];
    const scale = Math.min(box.width / physicalWidth, box.height / physicalHeight);
    const width = Math.max(1, physicalWidth * scale);
    const height = Math.max(1, physicalHeight * scale);
    return { width, height, left: (box.width - width) / 2, top: (box.height - height) / 2 };
  }, [box, nh, nv, header, config]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.round(box.width * ratio);
    canvas.height = Math.round(box.height * ratio);
    const context = canvas.getContext("2d")!;
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.fillStyle = "#000";
    context.fillRect(0, 0, box.width, box.height);
    context.imageSmoothingEnabled = false;
    context.drawImage(bitmap, layout.left, layout.top, layout.width, layout.height);

    // Crosshair: same world point as every other view.
    const px = (nh - 1 - crossVoxel[config.h] + 0.5) / nh;
    const py = (nv - 1 - crossVoxel[config.v] + 0.5) / nv;
    if (crosshair && px >= 0 && px <= 1 && py >= 0 && py <= 1) {
      const x = layout.left + px * layout.width;
      const y = layout.top + py * layout.height;
      context.strokeStyle = "rgba(0, 229, 255, 0.85)";
      context.lineWidth = 1;
      const gap = 6;
      context.beginPath();
      context.moveTo(layout.left, y); context.lineTo(x - gap, y);
      context.moveTo(x + gap, y); context.lineTo(layout.left + layout.width, y);
      context.moveTo(x, layout.top); context.lineTo(x, y - gap);
      context.moveTo(x, y + gap); context.lineTo(x, layout.top + layout.height);
      context.stroke();
    }

    context.fillStyle = "rgba(255,255,255,0.8)";
    context.font = "600 12px system-ui, sans-serif";
    context.textAlign = "center";
    context.textBaseline = "middle";
    const [left, right, top, bottom] = config.edges;
    context.fillText(left, layout.left + 10, layout.top + layout.height / 2);
    context.fillText(right, layout.left + layout.width - 10, layout.top + layout.height / 2);
    context.fillText(top, layout.left + layout.width / 2, layout.top + 10);
    context.fillText(bottom, layout.left + layout.width / 2, layout.top + layout.height - 10);
  }, [bitmap, layout, box, crossVoxel, crosshair, config, nh, nv]);

  /** Mouse position → world mm, through the affine. */
  const eventToWorld = useCallback(
    (event: { clientX: number; clientY: number }): Vec3 | null => {
      const canvas = canvasRef.current;
      if (!canvas) return null;
      const rect = canvas.getBoundingClientRect();
      const fx = (event.clientX - rect.left - layout.left) / layout.width;
      const fy = (event.clientY - rect.top - layout.top) / layout.height;
      if (fx < 0 || fx >= 1 || fy < 0 || fy >= 1) return null;
      const ijk: Vec3 = [0, 0, 0];
      ijk[config.fixed] = slice;
      ijk[config.h] = nh - 1 - (fx * nh - 0.5);
      ijk[config.v] = nv - 1 - (fy * nv - 0.5);
      return voxelToWorld(ijk, header.affine);
    },
    [layout, config, slice, nh, nv, header],
  );

  const dragRef = useRef<{ mode: "crosshair" | "window"; x: number; y: number; low: number; high: number } | null>(null);

  const moveCrosshair = (event: React.PointerEvent) => {
    const world = eventToWorld(event);
    if (!world) return;
    const store = useViewerStore.getState();
    store.setCrosshair(world);
    store.setPicked(lookupWorld(world, volumes, "2d"));
  };

  const onPointerDown = (event: React.PointerEvent) => {
    (event.target as HTMLElement).setPointerCapture(event.pointerId);
    setTip(null);
    if (event.button === 2) {
      dragRef.current = { mode: "window", x: event.clientX, y: event.clientY, low: range.low, high: range.high };
    } else if (event.button === 0) {
      dragRef.current = { mode: "crosshair", x: event.clientX, y: event.clientY, low: range.low, high: range.high };
      moveCrosshair(event);
    }
  };

  const onPointerMove = (event: React.PointerEvent) => {
    const drag = dragRef.current;
    if (drag?.mode === "window") {
      // Right-drag: horizontal = window width, vertical = level.
      const full = (volumes[base].header.display_range?.[1] ?? 1) - (volumes[base].header.display_range?.[0] ?? 0);
      const width = Math.max(full * 0.01, drag.high - drag.low + ((event.clientX - drag.x) / 300) * full);
      const centre = (drag.high + drag.low) / 2 - ((event.clientY - drag.y) / 300) * full;
      useViewerStore.getState().setWindow(base, centre - width / 2, centre + width / 2);
      return;
    }
    if (drag?.mode === "crosshair") {
      moveCrosshair(event);
      return;
    }
    const world = eventToWorld(event);
    const info = world ? lookupWorld(world, volumes, "2d") : null;
    useViewerStore.getState().setHover(info);
    const rect = event.currentTarget.getBoundingClientRect();
    setTip(info ? { x: event.clientX - rect.left, y: event.clientY - rect.top, info } : null);
  };

  const onPointerUp = () => {
    dragRef.current = null;
  };

  // Scroll: move the shared crosshair one voxel along this view's normal.
  const step = useCallback(
    (delta: number) => {
      const next: Vec3 = [crossVoxel[0], crossVoxel[1], crossVoxel[2]];
      next[config.fixed] = Math.min(shape[config.fixed] - 1, Math.max(0, slice + delta));
      if (!inBounds(roundToIndex(next), shape)) return;
      useViewerStore.getState().setCrosshair(voxelToWorld(next, header.affine));
    },
    [crossVoxel, config, shape, slice, header],
  );

  useEffect(() => {
    const element = containerRef.current;
    if (!element) return;
    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      step(event.deltaY > 0 ? -1 : 1);
    };
    element.addEventListener("wheel", onWheel, { passive: false });
    return () => element.removeEventListener("wheel", onWheel);
  }, [step]);

  return (
    <div className="slice-view">
      <div className="slice-title">
        <span>{config.title}</span>
        <span className="slice-index">
          {slice + 1} / {shape[config.fixed]}
          <MaximizeButton view={plane} />
        </span>
      </div>
      <div className="slice-canvas" ref={containerRef}>
        <canvas
          ref={canvasRef}
          style={{ width: box.width, height: box.height }}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerLeave={() => {
            useViewerStore.getState().setHover(null);
            setTip(null);
          }}
          onDoubleClick={() => useViewerStore.getState().toggleMaximized(plane)}
          onContextMenu={(event) => event.preventDefault()}
        />
        <HoverTip tip={tip} width={box.width} height={box.height} labelMap={labelMap} />
      </div>
      <input
        className="slice-slider"
        type="range"
        min={0}
        max={shape[config.fixed] - 1}
        value={slice}
        onChange={(event) => step(Number(event.target.value) - slice)}
        aria-label={`${config.title} slice`}
      />
    </div>
  );
}
