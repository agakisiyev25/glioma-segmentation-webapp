import { useState } from "react";

import { downloadUrl } from "../lib/api";
import { dice, targetForLabel, useStudyMetrics } from "../lib/metrics";
import type { LabelStats, Summary, VoxelInfo } from "../lib/types";
import { useViewerStore, type BaseVolume } from "../state/viewerStore";

export function Disclaimer({ text, scope }: { text: string; scope?: string }) {
  return (
    <div className="disclaimer" role="note">
      <strong>{text}</strong>
      {scope && <span> {scope}</span>}
    </div>
  );
}

/** One slim line above the views: mode chips and the required disclaimer, always visible. */
export function ViewerStrip({ summary }: { summary: Summary }) {
  return (
    <div className="viewer-strip" role="note">
      {summary.demo && <span className="chip chip-demo">{summary.banner ?? "DEMO — no model inference"}</span>}
      {summary.synthetic_sample && <span className="chip chip-sample">Synthetic sample · not a patient</span>}
      <span className="strip-text">
        <strong>{summary.disclaimer}</strong> <span title={summary.scope_note}>Segments regions only — no tumor type, grade or diagnosis.</span>
      </span>
    </div>
  );
}

const format = (value: number, digits = 1) => value.toFixed(digits);

function Row({ name, children }: { name: string; children: React.ReactNode }) {
  return (
    <div className="info-row">
      <span className="info-name">{name}</span>
      <span className="info-value">{children}</span>
    </div>
  );
}

/** Voxel details for the hovered point, or the last clicked point when nothing is hovered. */
export function InfoBox({ summary }: { summary: Summary }) {
  const hover = useViewerStore((state) => state.hover);
  const picked = useViewerStore((state) => state.picked);
  const verification = useViewerStore((state) => state.verification);
  const info: VoxelInfo | null = hover ?? picked;
  const showingPicked = !hover && picked !== null;
  const label = info?.label != null ? summary.label_map.find((entry) => entry.value === info.label) : undefined;
  const tumor = label !== undefined && label.value !== 0;
  const metrics = useStudyMetrics();
  const study = tumor ? targetForLabel(metrics, label.value) : undefined;

  return (
    <section className="panel info-box">
      <h3>
        Voxel
        <span className="tag">{info ? (showingPicked ? "clicked point" : `under cursor · ${info.source.toUpperCase()}`) : "—"}</span>
      </h3>
      {!info && <p className="muted empty">Point at a slice or the 3D view. Click to move the crosshair there.</p>}
      {info && !info.inBounds && <p className="muted empty">Outside the image volume.</p>}
      {info && info.inBounds && (
        <>
          <div className={`info-label${tumor ? " is-tumor" : ""}`} style={tumor ? { borderColor: label.color } : undefined}>
            <span className="swatch" style={{ background: tumor ? label.color : "transparent" }} />
            {label ? (tumor ? <><b>{label.short}</b>&nbsp;{label.name}</> : "Background") : "?"}
          </div>
          <Row name="Model confidence">
            {info.confidence === null ? (
              <span className="muted">n/a — no model inference</span>
            ) : (
              <>
                {format(info.confidence * 100)}% <span className="muted">uncalibrated</span>
              </>
            )}
          </Row>
          {study && (
            <Row name={`Study Dice for ${label!.short}`}>
              <span title={`Mean patient-level Dice in the study, 95% CI ${dice(study.dice_ci95[0])}–${dice(study.dice_ci95[1])}`}>
                {dice(study.dice)}
                {study.dice < 0.7 && <span className="muted"> · less reliable region</span>}
              </span>
            </Row>
          )}
          <Row name="World mm (R, A, S)">{info.world.map((value) => format(value)).join(", ")}</Row>
          <Row name="Voxel (i, j, k)">{info.voxel.join(", ")}</Row>
          <Row name="T1c · FLAIR intensity">
            {info.t1c === null ? "—" : format(info.t1c, 0)} · {info.flair === null ? "—" : format(info.flair, 0)}
          </Row>
          {showingPicked && (
            <Row name="Backend check">
              {verification === "match" && <span className="ok">same voxel and label</span>}
              {verification === "mismatch" && <span className="bad">MISMATCH with backend lookup</span>}
              {verification === "pending" && <span className="muted">checking…</span>}
              {verification === "unavailable" && <span className="muted">backend not reachable</span>}
              {verification === "idle" && <span className="muted">—</span>}
            </Row>
          )}
        </>
      )}
    </section>
  );
}

/** Names short enough for the regions table; the full name is the tooltip. */
const CONCISE_NAMES: Record<number, string> = {
  1: "Non-enhancing core",
  2: "Edema (FLAIR)",
  3: "Enhancing tissue",
  4: "Resection cavity",
};

function Slider({ value, onChange, min = 0, max = 1, step = 0.01, label }: { value: number; onChange: (v: number) => void; min?: number; max?: number; step?: number; label: string }) {
  return (
    <input type="range" min={min} max={max} step={step} value={value} aria-label={label} title={label} onChange={(event) => onChange(Number(event.target.value))} />
  );
}

/**
 * Legend, measurements and layer controls in one table: fixed colour, volume
 * and lesion count per region, with show/hide and 3D opacity.
 */
export function RegionsPanel({ summary }: { summary: Summary }) {
  const brain = useViewerStore((state) => state.brain);
  const layers = useViewerStore((state) => state.labels);
  const setBrain = useViewerStore((state) => state.setBrain);
  const setLabel = useViewerStore((state) => state.setLabel);
  const setCrosshair = useViewerStore((state) => state.setCrosshair);
  const [expanded, setExpanded] = useState<number | null>(null);

  const metrics = useStudyMetrics();
  const stats = summary.statistics.labels;
  const largest = Math.max(...stats.map((entry) => entry.volume_ml), 1e-9);
  const tumor = summary.statistics.composites.tumor_related_1_3;
  const jump = (target: number[] | null | undefined) => {
    if (target) setCrosshair([target[0], target[1], target[2]]);
  };

  return (
    <section className="panel">
      <h3>Regions</h3>
      <div className="regions">
        <div className="region-head">
          <span />
          <span>Region</span>
          <span className="num">mL</span>
          <span className="num" title={`${summary.statistics.connectivity} components`}>
            lesions
          </span>
          <span className="num">opacity</span>
        </div>
        {stats.map((entry: LabelStats) => {
          const empty = entry.voxels === 0;
          const layer = layers[entry.value];
          return (
            <div key={entry.value} className={`region${empty ? " is-empty" : ""}`}>
              <div className="region-row">
                <input
                  type="checkbox"
                  checked={layer.visible}
                  onChange={(event) => setLabel(entry.value, { visible: event.target.checked })}
                  aria-label={`Show ${entry.short}`}
                  title="Show in 3D and on the slices"
                />
                <button className="region-name" disabled={empty} onClick={() => jump(entry.components[0]?.centroid_world_mm ?? entry.centroid_world_mm)} title={`${entry.name}${targetForLabel(metrics, entry.value) ? ` · study Dice ${dice(targetForLabel(metrics, entry.value)!.dice)}` : ""} — click to go to the largest lesion`}>
                  <span className="swatch" style={{ background: entry.color }} />
                  <span className="region-text">
                    <b>{entry.short}</b>
                    <span className="region-full">{CONCISE_NAMES[entry.value] ?? entry.name}</span>
                  </span>
                </button>
                <span className="num">{empty ? "—" : entry.volume_ml.toFixed(2)}</span>
                <span className="num">
                  {entry.component_count > 1 ? (
                    <button className="link" onClick={() => setExpanded(expanded === entry.value ? null : entry.value)} title="List the lesions">
                      {entry.component_count} {expanded === entry.value ? "▴" : "▾"}
                    </button>
                  ) : (
                    entry.component_count
                  )}
                </span>
                <Slider label={`${entry.short} opacity in 3D`} value={layer.opacity} onChange={(opacity) => setLabel(entry.value, { opacity })} />
              </div>
              <div className="region-bar" aria-hidden="true">
                <i style={{ width: `${(entry.volume_ml / largest) * 100}%`, background: entry.color }} />
              </div>
              {expanded === entry.value && (
                <ol className="lesions">
                  {entry.components.map((component, index) => (
                    <li key={index}>
                      <button className="link" onClick={() => jump(component.centroid_world_mm)}>
                        Lesion {index + 1}
                      </button>
                      <span className="num">{component.volume_ml < 0.01 ? "<0.01" : component.volume_ml.toFixed(2)} mL</span>
                    </li>
                  ))}
                  {entry.components_truncated && <li className="muted">…only the largest {entry.components.length} are listed</li>}
                </ol>
              )}
            </div>
          );
        })}
        <div className="region-row region-total">
          <span />
          <span title={tumor.name}>Tumor-related (labels 1–3)</span>
          <span className="num">{tumor.volume_ml.toFixed(2)}</span>
          <span className="num">{tumor.component_count}</span>
          <span />
        </div>
        <div className="region-row region-brain">
          <input type="checkbox" checked={brain.visible} onChange={(event) => setBrain({ visible: event.target.checked })} aria-label="Show brain surface" />
          <span className="region-name static">
            <span className="swatch" style={{ background: "#cfc6bd" }} />
            Brain surface
          </span>
          <span />
          <span />
          <Slider label="Brain surface opacity" value={brain.opacity} onChange={(opacity) => setBrain({ opacity })} />
        </div>
      </div>
      <p className="muted small">
        Volume = voxel count × {summary.statistics.voxel_volume_mm3.toFixed(3)} mm³. Lesions are {summary.statistics.connectivity} components.
        Segmented regions only — not a tumor type or grade.
      </p>
    </section>
  );
}

export function DisplayPanel({ ranges }: { ranges: Record<BaseVolume, [number, number]> }) {
  const state = useViewerStore();
  const [fullLow, fullHigh] = ranges[state.base];
  const span = Math.max(1e-9, fullHigh - fullLow);
  const window = state.windows[state.base];
  const width = window.high - window.low;
  const level = (window.high + window.low) / 2;
  const setWidthLevel = (nextWidth: number, nextLevel: number) =>
    state.setWindow(state.base, nextLevel - nextWidth / 2, nextLevel + nextWidth / 2);

  return (
    <section className="panel">
      <h3>Display</h3>
      <div className="control-row">
        <span>Slice image</span>
        <div className="segmented">
          {(["t1c", "flair"] as BaseVolume[]).map((base) => (
            <button key={base} className={state.base === base ? "active" : ""} onClick={() => state.setBase(base)}>
              {base === "t1c" ? "T1c" : "FLAIR"}
            </button>
          ))}
        </div>
      </div>
      <div className="control-row">
        <span>Overlay</span>
        <Slider label="Overlay opacity on the slices" value={state.overlayOpacity} onChange={state.setOverlayOpacity} />
      </div>
      <div className="control-row">
        <span>Window</span>
        <Slider label="Window width" min={span * 0.01} max={span * 2} step={span / 500} value={width} onChange={(v) => setWidthLevel(v, level)} />
      </div>
      <div className="control-row">
        <span>Level</span>
        <Slider label="Window level" min={fullLow - span * 0.5} max={fullHigh + span * 0.5} step={span / 500} value={level} onChange={(v) => setWidthLevel(width, v)} />
      </div>
      <div className="control-row checks">
        <label title="Draw every tumor voxel as its own cube instead of a smoothed surface">
          <input type="checkbox" checked={state.voxelMode} onChange={(event) => state.setVoxelMode(event.target.checked)} /> Voxel mode
        </label>
        <label className={state.voxelMode ? "" : "disabled"} title="Skip voxels hidden inside a region; the picture is the same with far fewer cubes">
          <input type="checkbox" disabled={!state.voxelMode} checked={state.surfaceVoxelsOnly} onChange={(event) => state.setSurfaceVoxelsOnly(event.target.checked)} /> shell only
        </label>
        <label>
          <input type="checkbox" checked={state.clip.enabled} onChange={(event) => state.setClip({ enabled: event.target.checked })} /> Clip plane
        </label>
      </div>
      {state.clip.enabled && (
        <>
          <div className="control-row">
            <span>Cut along</span>
            <div className="segmented">
              {(["R–L", "A–P", "S–I"] as const).map((name, axis) => (
                <button key={name} className={state.clip.axis === axis ? "active" : ""} onClick={() => state.setClip({ axis: axis as 0 | 1 | 2 })}>
                  {name}
                </button>
              ))}
              <button className={state.clip.flip ? "active" : ""} onClick={() => state.setClip({ flip: !state.clip.flip })} title="Keep the other side">
                flip
              </button>
            </div>
          </div>
          <div className="control-row">
            <span>Position</span>
            <Slider label="Clipping plane position" value={state.clip.fraction} onChange={(fraction) => state.setClip({ fraction })} />
          </div>
          {state.voxelMode && state.surfaceVoxelsOnly && (
            <p className="muted small">With "shell only", cut regions look hollow. Untick it to see the cut filled.</p>
          )}
        </>
      )}
      <p className="muted small hint">
        Slices: scroll to move through, click or drag to set the crosshair, right-drag for window/level. Double-click any view to enlarge it.
      </p>
    </section>
  );
}

export function ResultPanel({ summary, jobId, onDelete }: { summary: Summary; jobId: string; onDelete: () => void }) {
  const model = summary.model;
  return (
    <section className="panel">
      <h3>Result</h3>
      <div className="button-row tight">
        <a className="button" href={downloadUrl(jobId)} download>
          Download segmentation
        </a>
        <button className="button secondary" onClick={onDelete}>
          Delete job
        </button>
      </div>
      <p className="muted small">
        NIfTI mask in the uploaded T1c space ({summary.grid.uploaded_t1c.shape.join(" × ")} voxels, stored {summary.grid.uploaded_t1c.axcodes}).
      </p>

      {summary.warnings.length > 0 && (
        <div className="warnings">
          {summary.warnings.map((warning, index) => (
            <p key={index}>⚠ {warning}</p>
          ))}
        </div>
      )}

      <details>
        <summary>Model &amp; run details</summary>
        {summary.demo ? (
          <p className="small">{model.description}</p>
        ) : (
          <ul className="small plain">
            <li>
              {model.family} {model.configuration} · {model.ensemble}
            </li>
            <li>
              nnunetv2 {model.nnunetv2_version} · torch {model.torch_version} · {model.device?.toUpperCase()}
              {model.gpu ? ` (${model.gpu})` : ""}
            </li>
            <li>Test-time mirroring: {model.test_time_mirroring ? "on" : "off"}</li>
            {model.members?.map((member) => (
              <li key={member.name}>
                <code>{member.name}</code> — {member.trainer}; {member.checkpoints.map((c) => `${c.file} (sha256 ${c.sha256?.slice(0, 12)}…)`).join(", ")}
              </li>
            ))}
            {model.notes?.map((note, index) => <li key={index}>{note}</li>)}
          </ul>
        )}
        <p className="small muted">
          Runtime {summary.runtime_seconds}s · viewer grid {summary.grid.viewer.axcodes}, {summary.grid.viewer.spacing_mm.map((value) => value.toFixed(2)).join(" × ")} mm
        </p>
        <p className="small muted">{summary.confidence_note}</p>
      </details>
    </section>
  );
}
