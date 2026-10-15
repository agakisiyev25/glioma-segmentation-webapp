import { useEffect, useState } from "react";

import { getJsonPublic } from "./api";
import type { StudyMetrics, StudyTarget } from "./types";

let cached: Promise<StudyMetrics> | null = null;

/** The study's nnU-Net results, fetched once per page load. Null until loaded or if unavailable. */
export function useStudyMetrics(): StudyMetrics | null {
  const [metrics, setMetrics] = useState<StudyMetrics | null>(null);
  useEffect(() => {
    cached ??= getJsonPublic<StudyMetrics>("/api/model/metrics");
    let alive = true;
    cached.then((value) => alive && setMetrics(value)).catch(() => {
      cached = null;
    });
    return () => {
      alive = false;
    };
  }, []);
  return metrics;
}

export function targetForLabel(metrics: StudyMetrics | null, label: number | null): StudyTarget | undefined {
  if (!metrics || label === null) return undefined;
  return metrics.targets.find((target) => target.labels.length === 1 && target.labels[0] === label);
}

export const pct = (value: number, digits = 0) => `${(value * 100).toFixed(digits)}%`;
export const dice = (value: number) => value.toFixed(2);
