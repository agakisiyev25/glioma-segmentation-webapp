import { dice, pct, useStudyMetrics } from "../lib/metrics";
import type { LabelInfo, StudyMetrics } from "../lib/types";

const SHORT: Record<string, string> = {
  label_1: "NETC",
  label_2: "SNFH",
  label_3: "ET",
  label_4: "RC",
  tumor_related_1_3: "Labels 1–3",
  all_postoperative_1_4: "Labels 1–4",
};

function colourOf(labels: number[], labelMap?: LabelInfo[]) {
  return labels.length === 1 ? labelMap?.find((entry) => entry.value === labels[0])?.color : undefined;
}

/** Full table of the study's nnU-Net results, for the About page. */
export function StudyPerformance({ labelMap }: { labelMap?: LabelInfo[] }) {
  const metrics = useStudyMetrics();
  if (!metrics) return <p className="muted">Loading the study's results…</p>;
  return (
    <>
      <p>
        Mean patient-level Dice for the tumor-related region (labels 1–3): <strong>{dice(metrics.primary.mean_dice)}</strong> (95% CI{" "}
        {dice(metrics.primary.ci95[0])}–{dice(metrics.primary.ci95[1])}), median {dice(metrics.primary.median_dice)}, over {metrics.primary.patients}{" "}
        patients, each scored by a model that never saw them.
      </p>
      <div className="table-scroll">
        <table className="metrics-table">
          <thead>
            <tr>
              <th>Region</th>
              <th title="Mean patient-level Dice with bootstrap 95% CI">Dice (95% CI)</th>
              <th title="Median patient-level 95th-percentile Hausdorff distance">HD95</th>
              <th title="Mean patient-level surface Dice at 1 mm tolerance">Surface Dice 1 mm</th>
              <th title="Share of reference lesions (26-connected) that the model touched">Lesions found</th>
              <th title="Scans with no reference region where the model still predicted some">False alarm when absent</th>
            </tr>
          </thead>
          <tbody>
            {metrics.targets.map((target) => {
              const colour = colourOf(target.labels, labelMap);
              const composite = target.labels.length > 1;
              return (
                <tr key={target.key} className={composite ? "composite" : ""}>
                  <td>
                    {colour && <span className="swatch" style={{ background: colour }} />}
                    <b>{SHORT[target.key]}</b> <span className="muted">{target.name.replace(/\s*\(.*\)$/, "")}</span>
                  </td>
                  <td>
                    <DiceBar value={target.dice} colour={colour} />
                    {dice(target.dice)} <span className="muted small">({dice(target.dice_ci95[0])}–{dice(target.dice_ci95[1])})</span>
                  </td>
                  <td>{target.hd95_mm_median.toFixed(1)} mm</td>
                  <td>{dice(target.surface_dice_1mm)}</td>
                  <td>{pct(target.lesion_sensitivity)}</td>
                  <td>
                    {target.when_absent ? (
                      <>
                        {pct(target.when_absent.false_positive_scan_rate)}{" "}
                        <span className="muted small">
                          ({target.when_absent.false_positive_scans}/{target.when_absent.absent_scans})
                        </span>
                      </>
                    ) : (
                      "—"
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="muted small">
        Per outer fold (labels 1–3): {metrics.folds.map((fold) => `${fold.fold}: ${dice(fold.mean_dice)}`).join(" · ")}. On the 162 patients never
        inspected during protocol development: {dice(metrics.unseen_162_patients.mean_dice)} ({dice(metrics.unseen_162_patients.ci95[0])}–
        {dice(metrics.unseen_162_patients.ci95[1])}). Range across three training seeds on fold 1: {metrics.seed_range_fold1.toFixed(3)}.
      </p>
      <p className="muted small">
        Source: <a href={`${metrics.source.repository}/tree/${metrics.source.commit}/results/final_analysis`} target="_blank" rel="noreferrer">
          results/final_analysis
        </a>{" "}
        at commit <code>{metrics.source.commit.slice(0, 7)}</code>. {metrics.evaluation}
      </p>
    </>
  );
}

function DiceBar({ value, colour }: { value: number; colour?: string }) {
  return (
    <span className="dice-bar" aria-hidden="true">
      <i style={{ width: `${value * 100}%`, background: colour ?? "#5b6878" }} />
    </span>
  );
}

/** Compact version for the upload page. */
export function StudySummaryCard({ labelMap }: { labelMap?: LabelInfo[] }) {
  const metrics = useStudyMetrics();
  if (!metrics) return null;
  const single = metrics.targets.filter((target) => target.labels.length === 1);
  return (
    <section className="panel">
      <h3>How well the model did in the study</h3>
      <p className="study-headline">
        <span className="big">{dice(metrics.primary.mean_dice)}</span>
        <span>
          mean Dice, tumor-related region
          <br />
          <span className="muted small">
            95% CI {dice(metrics.primary.ci95[0])}–{dice(metrics.primary.ci95[1])} · {metrics.primary.patients} patients
          </span>
        </span>
      </p>
      <div className="study-bars">
        {single.map((target) => (
          <div key={target.key} className="study-bar-row" title={target.name}>
            <span>{SHORT[target.key]}</span>
            <DiceBar value={target.dice} colour={colourOf(target.labels, labelMap)} />
            <span className="num">{dice(target.dice)}</span>
          </div>
        ))}
      </div>
      <p className="muted small">Internal validation on MU-Glioma-Post only. Small regions (NETC, RC) are much less reliable than edema.</p>
    </section>
  );
}

export type { StudyMetrics };
