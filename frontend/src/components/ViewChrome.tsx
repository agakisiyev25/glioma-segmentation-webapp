import type { LabelInfo, VoxelInfo } from "../lib/types";
import { useViewerStore, type ViewId } from "../state/viewerStore";

export interface Tip {
  x: number;
  y: number;
  info: VoxelInfo;
}

/** Small read-out that follows the cursor inside a view: what is under the pointer, right where you look. */
export function HoverTip({ tip, width, height, labelMap }: { tip: Tip | null; width: number; height: number; labelMap: LabelInfo[] }) {
  if (!tip || !tip.info.inBounds) return null;
  const { x, y, info } = tip;
  const label = labelMap.find((entry) => entry.value === info.label);
  const tumor = label !== undefined && label.value !== 0;
  const flipX = x > width - 220;
  const flipY = y > height - 76;
  return (
    <div
      className="hover-tip"
      style={{
        left: flipX ? undefined : x + 14,
        right: flipX ? width - x + 14 : undefined,
        top: flipY ? undefined : y + 14,
        bottom: flipY ? height - y + 14 : undefined,
      }}
    >
      <div className="hover-tip-label">
        <span className="swatch" style={{ background: tumor ? label.color : "transparent" }} />
        {tumor ? (
          <>
            <b>{label.short}</b> {label.name}
          </>
        ) : (
          "Background"
        )}
      </div>
      <div className="hover-tip-detail">
        {info.confidence !== null && <span>confidence {(info.confidence * 100).toFixed(0)}%</span>}
        <span>
          R {info.world[0].toFixed(1)} · A {info.world[1].toFixed(1)} · S {info.world[2].toFixed(1)} mm
        </span>
      </div>
    </div>
  );
}

export function MaximizeButton({ view }: { view: ViewId }) {
  const maximized = useViewerStore((state) => state.maximized === view);
  const toggle = useViewerStore((state) => state.toggleMaximized);
  return (
    <button
      className="icon-button"
      onClick={() => toggle(view)}
      title={maximized ? "Back to four views (or double-click the view)" : "Enlarge this view (or double-click it)"}
      aria-label={maximized ? "Restore four views" : "Enlarge this view"}
    >
      <svg viewBox="0 0 16 16" width="13" height="13" aria-hidden="true">
        {maximized ? (
          <path d="M6 2v4H2M10 2v4h4M6 14v-4H2M10 14v-4h4" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
        ) : (
          <path d="M2 6V2h4M14 6V2h-4M2 10v4h4M14 10v4h-4" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
        )}
      </svg>
    </button>
  );
}
