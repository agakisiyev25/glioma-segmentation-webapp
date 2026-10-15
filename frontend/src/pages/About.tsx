import { useEffect, useState } from "react";

import { getInfo } from "../lib/api";
import type { ApiInfo } from "../lib/types";
import { StudyPerformance } from "../components/StudyMetrics";

const REPO = "https://github.com/glioma-ml-research/mu-glioma-segmentation-benchmark";

export default function About() {
  const [info, setInfo] = useState<ApiInfo | null>(null);
  useEffect(() => {
    getInfo().then(setInfo).catch(() => setInfo(null));
  }, []);

  return (
    <main className="page narrow about">
      <section className="panel">
        <h2>About this prototype</h2>
        <p>
          This app runs the best-performing pipeline from our benchmark study on an uploaded postoperative brain MRI examination and shows the
          result on the patient's own anatomy. It is a research demonstration of a segmentation model. It is not a diagnostic tool.
        </p>
        {info && <p className="callout">{info.disclaimer}</p>}
      </section>

      <section className="panel">
        <h3>The paper</h3>
        <p>
          <em>
            Nested Patient-Grouped Benchmarking of 3D Deep-Learning Pipelines for Multiclass Postoperative Glioma Segmentation on Longitudinal MRI.
          </em>
        </p>
        <p>
          The study compares 3D U-Net, Residual 3D U-Net, 3D V-Net and 3D U-Net++ in one controlled pipeline, and evaluates nnU-Net v2 (3D full
          resolution) as a whole-pipeline comparator, on 594 longitudinal examinations from 203 patients of the public MU-Glioma-Post collection.
          All splits are grouped by patient: five outer folds for evaluation, with a separate inner split for checkpoint selection.
        </p>
        <p>
          Manuscript, notebooks, frozen splits and results: <a href={REPO} target="_blank" rel="noreferrer">{REPO.replace("https://", "")}</a>. The
          manuscript has not yet completed journal peer review.
        </p>
      </section>

      <section className="panel">
        <h3>The model</h3>
        <p>
          nnU-Net v2, <code>3d_fullres</code> configuration, trained on four co-registered, skull-stripped sequences (T1, contrast-enhanced T1, T2,
          FLAIR).
        </p>
      </section>

      <section className="panel">
        <h3>Performance in the study</h3>
        <StudyPerformance labelMap={info?.label_map} />
      </section>

      <section className="panel">
        <h3>What it does and does not do</h3>
        <p>
          The model outlines tumor-related sub-regions and the resection cavity. It does not classify tumor type or grade, and it does not
          distinguish recurrence from treatment effect.
        </p>
        {info?.model && (
          <p className="muted small">
            Loaded on this server: {info.message}. nnunetv2 {info.model.nnunetv2_version}, torch {info.model.torch_version}.
            {info.model.notes?.map((note) => ` ${note}`)}
          </p>
        )}
        {info?.mode === "demo" && <p className="muted small">This server is running in demo mode: no model weights are loaded.</p>}
      </section>

      <section className="panel">
        <h3>Limitations</h3>
        <ul>
          <li>
            <strong>Internal validation only.</strong> Trained and tested on one public dataset from one institution. Performance on other scanners,
            protocols, field strengths or populations is unknown.
          </li>
          <li>
            <strong>Not evaluated clinically.</strong> The study is a technical comparison, not a diagnostic-accuracy study. It provides no evidence
            of clinical utility.
          </li>
          <li>
            <strong>Reference masks were machine-initialised.</strong> The dataset's masks started from an automated ensemble that included nnU-Net
            and were then corrected and reviewed by radiologists. Agreement with them may overstate agreement with independent annotations.
          </li>
          <li>
            <strong>Confidence is not calibrated.</strong> The percentage shown per voxel is the model's maximum softmax probability. Calibration was
            not evaluated, so it is not a probability of being correct.
          </li>
          <li>
            <strong>Preprocessing matters.</strong> The model has only seen skull-stripped, co-registered images prepared as in MU-Glioma-Post. The
            app's own preprocessing of raw scans only approximates that and is untested.
          </li>
          <li>
            <strong>Small and absent regions.</strong> Lesion-level sensitivity was lower than for the simpler models (0.746), and the model sometimes
            predicts a region that is absent in the reference.
          </li>
          <li>Longitudinal volume change, fairness across demographic or molecular groups, and effect on decisions were not studied.</li>
        </ul>
      </section>

      <section className="panel">
        <h3>Team</h3>
        <ul className="plain">
          <li>Atilla Mustafa</li>
          <li>Elkhan Aghakishiyev</li>
          <li>Jamila Jabrayilli</li>
        </ul>
      </section>

      <section className="panel">
        <h3>Dataset</h3>
        <p>
          Yaseen D, Garrett F, Gass J, et al. <em>University of Missouri Post-operative Glioma Dataset (MU-Glioma-Post), Version 1</em> [data set].
          The Cancer Imaging Archive, 2025. <a href="https://doi.org/10.7937/7K9K-3C83" target="_blank" rel="noreferrer">doi:10.7937/7K9K-3C83</a>
        </p>
        <p>
          Mahmoud E, Gass J, Dhemesh Y, et al. MU-Glioma Post: a comprehensive dataset of automated MR multi-sequence segmentation and clinical
          features. <em>Scientific Data</em> 2025;12:1847. <a href="https://doi.org/10.1038/s41597-025-06011-7" target="_blank" rel="noreferrer">doi:10.1038/s41597-025-06011-7</a>
        </p>
        <p>
          Isensee F, Jaeger PF, Kohl SAA, Petersen J, Maier-Hein KH. nnU-Net: a self-configuring method for deep learning-based biomedical image
          segmentation. <em>Nature Methods</em> 2021;18:203–211.
        </p>
      </section>

      <section className="panel">
        <h3>Privacy</h3>
        <p>
          Uploaded files are re-written with a clean header on arrival (free-text header fields are dropped, DICOM files are converted and
          deleted), stored under a random job id, never logged by name, and deleted after {info?.job_ttl_hours ?? 24} hours or when you delete the
          job. Image data itself can still be identifying; do not use this prototype with data you are not permitted to process on this server.
        </p>
      </section>
    </main>
  );
}
