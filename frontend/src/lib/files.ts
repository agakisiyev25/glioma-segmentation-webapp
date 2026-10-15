export type Slot = "t1" | "t1c" | "t2" | "flair" | "segmentation";

export const SLOT_TITLES: Record<Slot, string> = {
  t1: "T1",
  t1c: "T1c",
  t2: "T2",
  flair: "FLAIR",
  segmentation: "Segmentation",
};

// A token must stand alone ("_t1_", "-T2.", "t1n") so that names such as
// "patient1" or "set2" are not mistaken for a sequence.
const token = (body: string) => new RegExp(`(^|[^a-z0-9])(${body})([^a-z0-9]|$)`);

const RULES: [Slot, RegExp[]][] = [
  ["segmentation", [/seg|mask|label/]],
  ["t1c", [token("t1w?[-_ ]?(c|ce|gd|post|contrast)"), /contrast|gadolinium/]],
  ["flair", [/flair/, token("t2f")]],
  ["t2", [token("t2w?")]],
  ["t1", [token("t1[nw]?")]],
];

/** Which sequence a file is, judged from its name. Null when it is not clear. */
export function guessSlot(fileName: string): Slot | null {
  const name = fileName.toLowerCase().replace(/\.(nii\.gz|nii|zip)$/, "");
  for (const [slot, patterns] of RULES) {
    if (patterns.some((pattern) => pattern.test(name))) return slot;
  }
  return null;
}

export interface Assignment<T extends { name: string }> {
  assigned: Partial<Record<Slot, T>>;
  /** Names that matched no sequence, or a sequence that was already filled. */
  unmatched: string[];
}

/** Match several dropped files to sequence slots by name. The first match per slot wins. */
export function assignFiles<T extends { name: string }>(files: T[], allowed: Slot[]): Assignment<T> {
  const assigned: Partial<Record<Slot, T>> = {};
  const unmatched: string[] = [];
  for (const file of files) {
    const slot = guessSlot(file.name);
    if (slot === null || !allowed.includes(slot) || assigned[slot]) unmatched.push(file.name);
    else assigned[slot] = file;
  }
  return { assigned, unmatched };
}

export const isNifti = (name: string) => /\.(nii|nii\.gz)$/i.test(name);
export const isAcceptedUpload = (name: string) => /\.(nii|nii\.gz|zip)$/i.test(name);
