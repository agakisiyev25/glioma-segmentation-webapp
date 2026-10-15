import { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";

import { Disclaimer } from "../components/Panels";
import { StudySummaryCard } from "../components/StudyMetrics";
import { ApiError, createJob, createSampleJob, getInfo, getJob } from "../lib/api";
import { assignFiles, isAcceptedUpload, isNifti, SLOT_TITLES, type Slot } from "../lib/files";
import type { ApiInfo, JobStage, JobStatus } from "../lib/types";

const SEQUENCES: { slot: Slot; hint: string }[] = [
  { slot: "t1", hint: "native, pre-contrast" },
  { slot: "t1c", hint: "contrast-enhanced · reference space" },
  { slot: "t2", hint: "T2-weighted" },
  { slot: "flair", hint: "T2-FLAIR" },
];

const STAGES: { stage: JobStage; title: string }[] = [
  { stage: "uploading", title: "Upload" },
  { stage: "queued", title: "Queue" },
  { stage: "preprocessing", title: "Preprocess" },
  { stage: "inference", title: "Segment" },
  { stage: "meshing", title: "Build 3D" },
  { stage: "done", title: "Done" },
];

type Files = Record<Slot, File | null>;
const EMPTY: Files = { t1: null, t1c: null, t2: null, flair: null, segmentation: null };

function SlotCard({ slot, hint, file, disabled, onFiles, onClear }: { slot: Slot; hint: string; file: File | null; disabled: boolean; onFiles: (files: File[], slot: Slot) => void; onClear: () => void }) {
  const [over, setOver] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const open = () => !disabled && input.current?.click();
  return (
    <div
      className={`slot${over ? " over" : ""}${file ? " filled" : ""}${disabled ? " disabled" : ""}`}
      onDragOver={(event) => {
        event.preventDefault();
        event.stopPropagation();
        if (!disabled) setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(event) => {
        event.preventDefault();
        event.stopPropagation();
        setOver(false);
        if (!disabled) onFiles([...event.dataTransfer.files], slot);
      }}
      onClick={open}
      role="button"
      tabIndex={0}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") open();
      }}
    >
      <input
        ref={input}
        type="file"
        accept={slot === "segmentation" ? ".nii,.gz" : ".nii,.gz,.zip"}
        hidden
        onChange={(event) => {
          onFiles([...(event.target.files ?? [])], slot);
          event.target.value = "";
        }}
      />
      <div className="slot-head">
        <span className="slot-title">{SLOT_TITLES[slot]}</span>
        <span className="slot-state">{file ? "✓" : ""}</span>
      </div>
      <div className="slot-hint">{hint}</div>
      {file ? (
        <div className="slot-file">
          <span className="slot-file-name" title={file.name}>
            {file.name}
          </span>
          <span className="muted">{(file.size / 1024 / 1024).toFixed(1)} MB</span>
          {!disabled && (
            <button
              className="link"
              onClick={(event) => {
                event.stopPropagation();
                onClear();
              }}
            >
              remove
            </button>
          )}
        </div>
      ) : (
        <div className="slot-empty">Drop or click to choose</div>
      )}
    </div>
  );
}

function Progress({ stage, fraction, message, queue }: { stage: JobStage; fraction: number; message: string; queue: number }) {
  const active = STAGES.findIndex((entry) => entry.stage === stage);
  return (
    <div className="progress" role="status" aria-live="polite">
      <ol className="stages">
        {STAGES.map((entry, index) => (
          <li key={entry.stage} className={index < active ? "past" : index === active ? "current" : ""}>
            <span className="dot" />
            {entry.title}
          </li>
        ))}
      </ol>
      <div className="bar">
        <div className="bar-fill" style={{ width: `${Math.round(fraction * 100)}%` }} />
      </div>
      <p className="muted">
        {message}
        {stage === "queued" && queue > 0 ? ` (${queue} job${queue === 1 ? "" : "s"} waiting; one runs at a time)` : ""}
      </p>
    </div>
  );
}

export default function Upload() {
  const navigate = useNavigate();
  const [params, setParams] = useSearchParams();
  const [info, setInfo] = useState<ApiInfo | null>(null);
  const [infoError, setInfoError] = useState<string | null>(null);
  const [files, setFiles] = useState<Files>(EMPTY);
  const [notice, setNotice] = useState<string | null>(null);
  const [preprocessed, setPreprocessed] = useState(true);
  const [running, setRunning] = useState<{ stage: JobStage; fraction: number; message: string; queue: number } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const multiInput = useRef<HTMLInputElement>(null);

  useEffect(() => {
    getInfo()
      .then(setInfo)
      .catch((reason: Error) => setInfoError(`Cannot reach the backend: ${reason.message}`));
  }, []);

  const follow = useCallback(
    (jobId: string) => {
      let stopped = false;
      const poll = async () => {
        if (stopped) return;
        let job: JobStatus;
        try {
          job = await getJob(jobId);
        } catch (reason) {
          if (reason instanceof ApiError && reason.status === 404) {
            setRunning(null);
            setError("That job no longer exists (jobs are deleted after 24 h).");
            setParams({}, { replace: true });
            return;
          }
          setTimeout(poll, 3000);
          return;
        }
        if (job.status === "done") {
          navigate(`/viewer/${jobId}`);
          return;
        }
        if (job.status === "error") {
          setRunning(null);
          setError(job.error ?? "The job failed.");
          setParams({}, { replace: true });
          return;
        }
        setRunning({ stage: job.status, fraction: job.progress, message: job.message, queue: job.queue_length });
        setTimeout(poll, 1000);
      };
      void poll();
      return () => {
        stopped = true;
      };
    },
    [navigate, setParams],
  );

  // Resume following a job after a reload (?job=<id>).
  const resumeId = params.get("job");
  useEffect(() => {
    if (!resumeId) return;
    setRunning((current) => current ?? { stage: "queued", fraction: 0, message: "Reconnecting…", queue: 0 });
    return follow(resumeId);
  }, [resumeId, follow]);

  const demo = info?.mode === "demo";
  const blocked = info?.mode === "model_error";
  const busy = running !== null;
  const slots: Slot[] = demo ? ["t1", "t1c", "t2", "flair", "segmentation"] : ["t1", "t1c", "t2", "flair"];

  /** One file on a slot goes to that slot; several files are matched to slots by name. */
  const takeFiles = (incoming: File[], target: Slot | null) => {
    if (incoming.length === 0) return;
    setNotice(null);
    if (incoming.length === 1 && target) {
      const [file] = incoming;
      const valid = target === "segmentation" ? isNifti(file.name) : isAcceptedUpload(file.name);
      if (!valid) {
        setNotice(`${SLOT_TITLES[target]}: needs ${target === "segmentation" ? ".nii or .nii.gz" : ".nii, .nii.gz or a DICOM .zip"}.`);
        return;
      }
      setFiles((current) => ({ ...current, [target]: file }));
      return;
    }
    const accepted = incoming.filter((file) => isAcceptedUpload(file.name));
    const { assigned, unmatched } = assignFiles(accepted, slots);
    const skipped = [...unmatched, ...incoming.filter((file) => !isAcceptedUpload(file.name)).map((file) => file.name)];
    setFiles((current) => ({ ...current, ...assigned }));
    const placed = Object.keys(assigned).length;
    if (skipped.length > 0) {
      setNotice(
        `Matched ${placed} file${placed === 1 ? "" : "s"} by name. Could not place: ${skipped.join(", ")} — drop ${skipped.length === 1 ? "it" : "each"} on its sequence.`,
      );
    } else if (placed > 0) {
      setNotice(`Matched ${placed} file${placed === 1 ? "" : "s"} to sequences by file name. Check they are right.`);
    }
  };

  const needsSkullStrip = !preprocessed && info !== null && !info.hd_bet_available;
  const missing = slots.filter((slot) => !files[slot]);
  const ready = info !== null && !blocked && !busy && missing.length === 0 && !needsSkullStrip;

  const begin = (job_id: string) => {
    try {
      window.localStorage.setItem("glioma.lastJob", job_id);
    } catch {
      /* storage unavailable */
    }
    setParams({ job: job_id }, { replace: true });
  };

  const run = async () => {
    if (!files.t1 || !files.t1c || !files.t2 || !files.flair) return;
    setError(null);
    setRunning({ stage: "uploading", fraction: 0, message: "Preparing files (compressing .nii in the browser)…", queue: 0 });
    try {
      const { job_id } = await createJob(
        { t1: files.t1, t1c: files.t1c, t2: files.t2, flair: files.flair, segmentation: demo ? files.segmentation : null },
        preprocessed,
        (fraction) =>
          setRunning({
            stage: "uploading",
            fraction: fraction * 0.9,
            message: fraction < 1 ? `Uploading files… ${Math.round(fraction * 100)}%` : "Checking and de-identifying on the server…",
            queue: 0,
          }),
      );
      begin(job_id);
    } catch (reason) {
      setRunning(null);
      setError(reason instanceof Error ? reason.message : "Upload failed.");
    }
  };

  const runSample = async () => {
    setError(null);
    setRunning({ stage: "uploading", fraction: 0.5, message: "Preparing the synthetic sample on the server…", queue: 0 });
    try {
      begin((await createSampleJob()).job_id);
    } catch (reason) {
      setRunning(null);
      setError(reason instanceof Error ? reason.message : "The sample could not be started.");
    }
  };

  return (
    <main className="page upload">
      {info && <Disclaimer text={info.disclaimer} scope={info.scope_note} />}
      {infoError && <div className="panel error-panel">{infoError}</div>}
      {blocked && (
        <div className="panel error-panel">
          <h3>The model could not be loaded</h3>
          <p>{info!.message}</p>
          <p className="muted">The app does not fall back to a different model. Fix NNUNET_MODEL_DIR and restart the backend.</p>
        </div>
      )}

      <div className="upload-grid">
        <section
          className={`panel upload-card${dragging ? " dragging" : ""}`}
          onDragOver={(event) => {
            event.preventDefault();
            if (!busy) setDragging(true);
          }}
          onDragLeave={(event) => {
            if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setDragging(false);
          }}
          onDrop={(event) => {
            event.preventDefault();
            setDragging(false);
            if (!busy) takeFiles([...event.dataTransfer.files], null);
          }}
        >
          <div className="card-head">
            <div>
              <h2>Segment one MRI examination</h2>
              <p className="muted">
                Four sequences of the same examination, as <code>.nii</code> / <code>.nii.gz</code> or a <code>.zip</code> of one DICOM series each.
              </p>
            </div>
            {info && (
              <span className={`chip ${demo ? "chip-demo" : blocked ? "chip-error" : "chip-ok"}`}>
                {demo ? "DEMO — no model inference" : blocked ? "Model error" : info.message}
              </span>
            )}
          </div>

          <div className="drop-all">
            <span>
              <strong>Drop all files here at once</strong> — they are matched to sequences by file name
              <span className="muted"> (t1 / t1n, t1c / t1ce, t2 / t2w, flair / t2f{demo ? ", seg / mask" : ""})</span>
            </span>
            <button className="button ghost" disabled={busy} onClick={() => multiInput.current?.click()}>
              Choose files…
            </button>
            <input
              ref={multiInput}
              type="file"
              multiple
              hidden
              accept=".nii,.gz,.zip"
              onChange={(event) => {
                takeFiles([...(event.target.files ?? [])], null);
                event.target.value = "";
              }}
            />
          </div>

          <div className="slots">
            {SEQUENCES.map(({ slot, hint }) => (
              <SlotCard key={slot} slot={slot} hint={hint} file={files[slot]} disabled={busy} onFiles={takeFiles} onClear={() => setFiles((current) => ({ ...current, [slot]: null }))} />
            ))}
          </div>

          {demo && (
            <div className="demo-seg">
              <div>
                <strong>Demo mode needs an existing segmentation.</strong>
                <p className="muted">
                  No model weights are configured, so nothing is predicted. Add a label map for this examination (0 background, 1 NETC, 2 SNFH, 3
                  ET, 4 RC). Measurements, 3D surfaces and voxel lookups then work exactly as they will with the model.
                </p>
              </div>
              <SlotCard
                slot="segmentation"
                hint="NIfTI label map, values 0–4"
                file={files.segmentation}
                disabled={busy}
                onFiles={takeFiles}
                onClear={() => setFiles((current) => ({ ...current, segmentation: null }))}
              />
            </div>
          )}

          {notice && <p className="notice">{notice}</p>}

          <label className="checkbox">
            <input type="checkbox" checked={preprocessed} disabled={busy} onChange={(event) => setPreprocessed(event.target.checked)} />
            <span>
              <strong>Already preprocessed (MU-Glioma-Post format)</strong>
              <span className="muted"> — skull-stripped, all four sequences co-registered on one voxel grid.</span>
            </span>
          </label>
          {!preprocessed && (
            <p className="note">
              The server will rigidly co-register T1, T2 and FLAIR to T1c and skull-strip with HD-BET (a few minutes on CPU). This only approximates
              the preprocessing the model was trained on and has not been validated.
              {needsSkullStrip && <strong> HD-BET is not installed on this server, so raw scans cannot be processed.</strong>}
            </p>
          )}

          <div className="button-row">
            <button className="button large" disabled={!ready} onClick={run}>
              {demo ? "Open in viewer" : "Run segmentation"}
            </button>
            {!busy && missing.length > 0 && <span className="muted">Still needed: {missing.map((slot) => SLOT_TITLES[slot]).join(", ")}</span>}
          </div>

          {running && <Progress stage={running.stage} fraction={running.fraction} message={running.message} queue={running.queue} />}
          {error && (
            <div className="error-panel inline" role="alert">
              {error}
            </div>
          )}
        </section>

        <aside className="upload-side">
          {info?.sample_case_available && (
            <section className="panel sample-card">
              <h3>No files at hand?</h3>
              <p>
                Open a <strong>synthetic sample</strong> generated on the server: a brain-shaped volume with all four regions and a small second
                lesion. It is made of shapes, not anatomy, and comes from no person.
              </p>
              <button className="button" disabled={busy} onClick={runSample}>
                Open the synthetic sample
              </button>
              <p className="muted small">Takes 10–20 seconds. Good for trying the 3D view, the slices and the voxel read-out.</p>
            </section>
          )}
          <StudySummaryCard labelMap={info?.label_map} />
          <section className="panel">
            <h3>What you get</h3>
            <ul className="checks">
              <li>A 3D surface of this patient's own brain, built from the uploaded scan, with the segmented regions inside it.</li>
              <li>Axial, coronal and sagittal slices with the overlay, linked to the 3D view by one crosshair.</li>
              <li>For any voxel you point at: label, coordinates, intensities{demo ? "" : " and the model's (uncalibrated) confidence"}.</li>
              <li>Volume in mL and number of lesions per region, and the mask as a NIfTI file.</li>
            </ul>
          </section>
          <section className="panel">
            <h3>Your files</h3>
            <p className="muted small">
              De-identified on arrival (headers rebuilt, DICOM converted and deleted), stored under a random id, and deleted automatically after{" "}
              {info?.job_ttl_hours ?? 24} hours — or immediately when you delete the job.
            </p>
          </section>
        </aside>
      </div>
    </main>
  );
}
