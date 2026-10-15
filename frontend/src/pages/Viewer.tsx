import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";

import { DisplayPanel, InfoBox, RegionsPanel, ResultPanel, ViewerStrip } from "../components/Panels";
import { Scene3D } from "../components/Scene3D";
import { SliceView } from "../components/SliceView";
import { MaximizeButton } from "../components/ViewChrome";
import { deleteJob, getJob, getSummary, getVolume, getVoxel } from "../lib/api";
import { voxelToWorld, type Vec3 } from "../lib/geometry";
import type { Summary } from "../lib/types";
import type { VolumeSet } from "../lib/volumes";
import { useViewerStore, type BaseVolume, type ViewId } from "../state/viewerStore";

interface Loaded {
  summary: Summary;
  volumes: VolumeSet;
}

const LOAD_STEPS = ["Reading the result", "Loading voxel volumes", "Preparing the views"];

export default function Viewer() {
  const { jobId = "" } = useParams();
  const navigate = useNavigate();
  const [loaded, setLoaded] = useState<Loaded | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [step, setStep] = useState(0);

  useEffect(() => {
    let cancelled = false;
    setLoaded(null);
    setError(null);
    setStep(0);
    useViewerStore.getState().resetAll();
    (async () => {
      const job = await getJob(jobId);
      if (job.status !== "done") {
        navigate(`/?job=${jobId}`, { replace: true });
        return;
      }
      const summary = await getSummary(jobId);
      if (cancelled) return;
      setStep(1);
      const [labels, t1c, flair, confidence] = await Promise.all([
        getVolume(jobId, "labels"),
        getVolume(jobId, "t1c"),
        getVolume(jobId, "flair"),
        summary.confidence_available ? getVolume(jobId, "confidence") : Promise.resolve(null),
      ]);
      if (cancelled) return;
      setStep(2);

      const store = useViewerStore.getState();
      for (const [base, volume] of [["t1c", t1c], ["flair", flair]] as const) {
        const [low, high] = volume.header.display_range ?? [0, 1];
        store.setWindow(base, low, high);
      }
      // Start on the largest tumor-related lesion, or the middle of the volume.
      const tumor = summary.statistics.composites.tumor_related_1_3;
      const start = tumor.components[0]?.centroid_world_mm ?? summary.statistics.composites.all_postoperative_1_4.centroid_world_mm;
      if (start) {
        store.setCrosshair([start[0], start[1], start[2]]);
      } else {
        const { shape, affine } = summary.grid.viewer;
        store.setCrosshair(voxelToWorld([(shape[0] - 1) / 2, (shape[1] - 1) / 2, (shape[2] - 1) / 2], affine));
      }
      setLoaded({ summary, volumes: { labels, t1c, flair, confidence } });
    })().catch((reason: Error) => {
      if (!cancelled) setError(reason.message);
    });
    return () => {
      cancelled = true;
    };
  }, [jobId, navigate]);

  // Every click (2D or 3D) is cross-checked against the backend's own lookup.
  const picked = useViewerStore((state) => state.picked);
  useEffect(() => {
    if (!picked) return;
    const controller = new AbortController();
    useViewerStore.getState().setVerification("pending");
    getVoxel(jobId, picked.world as Vec3, controller.signal)
      .then((remote) => {
        const sameVoxel = remote.voxel.every((value, axis) => value === picked.voxel[axis]);
        const sameLabel = remote.label === picked.label;
        useViewerStore.getState().setVerification(sameVoxel && sameLabel && remote.in_bounds === picked.inBounds ? "match" : "mismatch");
      })
      .catch((reason: Error) => {
        if (reason.name !== "AbortError") useViewerStore.getState().setVerification("unavailable");
      });
    return () => controller.abort();
  }, [picked, jobId]);

  // Esc leaves the enlarged view.
  const maximized = useViewerStore((state) => state.maximized);
  useEffect(() => {
    if (!maximized) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") useViewerStore.getState().toggleMaximized(maximized);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [maximized]);

  const ranges = useMemo(() => {
    if (!loaded) return null;
    return {
      t1c: loaded.volumes.t1c.header.display_range ?? [0, 1],
      flair: loaded.volumes.flair.header.display_range ?? [0, 1],
    } as Record<BaseVolume, [number, number]>;
  }, [loaded]);

  if (error) {
    return (
      <main className="viewer viewer-message">
        <div className="panel error-panel">
          <h2>Could not open this result</h2>
          <p>{error}</p>
          <Link className="button" to="/">
            Back to upload
          </Link>
        </div>
      </main>
    );
  }
  if (!loaded || !ranges) {
    return (
      <main className="viewer viewer-message">
        <div className="loading-card">
          <div className="spinner" aria-hidden="true" />
          <ol>
            {LOAD_STEPS.map((title, index) => (
              <li key={title} className={index < step ? "past" : index === step ? "current" : ""}>
                {title}
              </li>
            ))}
          </ol>
        </div>
      </main>
    );
  }

  const { summary, volumes } = loaded;
  const onDelete = async () => {
    if (!window.confirm("Delete this job and all its files from the server?")) return;
    await deleteJob(jobId);
    try {
      window.localStorage.removeItem("glioma.lastJob");
    } catch {
      /* storage unavailable */
    }
    navigate("/");
  };
  const viewClass = (view: ViewId) => `viewport${view === "3d" ? " viewport-3d" : ""}${maximized === view ? " is-max" : ""}`;

  return (
    <main className="viewer">
      <ViewerStrip summary={summary} />
      <div className="viewer-body">
        <div className={`viewports${maximized ? " has-max" : ""}`}>
          <div className={viewClass("3d")}>
            <div className="slice-title">
              <span>3D · this patient's brain surface and segmented regions</span>
              <span className="slice-index">
                world mm (RAS)
                <MaximizeButton view="3d" />
              </span>
            </div>
            <Scene3D jobId={jobId} summary={summary} volumes={volumes} />
          </div>
          {(["axial", "coronal", "sagittal"] as const).map((plane) => (
            <div className={viewClass(plane)} key={plane}>
              <SliceView plane={plane} volumes={volumes} labelMap={summary.label_map} />
            </div>
          ))}
        </div>
        <aside className="sidebar">
          <InfoBox summary={summary} />
          <RegionsPanel summary={summary} />
          <DisplayPanel ranges={ranges} />
          <ResultPanel summary={summary} jobId={jobId} onDelete={onDelete} />
        </aside>
      </div>
    </main>
  );
}
