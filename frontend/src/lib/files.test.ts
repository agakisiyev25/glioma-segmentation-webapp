import { describe, expect, it } from "vitest";

import { assignFiles, guessSlot, type Slot } from "./files";

const ALL: Slot[] = ["t1", "t1c", "t2", "flair", "segmentation"];

describe("matching dropped files to sequences by name", () => {
  it("recognises MU-Glioma-Post / BraTS names", () => {
    expect(guessSlot("PatientID_0005_Timepoint_1_brain_t1n.nii.gz")).toBe("t1");
    expect(guessSlot("PatientID_0005_Timepoint_1_brain_t1c.nii.gz")).toBe("t1c");
    expect(guessSlot("PatientID_0005_Timepoint_1_brain_t2w.nii.gz")).toBe("t2");
    expect(guessSlot("PatientID_0005_Timepoint_1_brain_t2f.nii.gz")).toBe("flair");
    expect(guessSlot("PatientID_0005_Timepoint_1_tumorMask.nii.gz")).toBe("segmentation");
  });

  it("recognises common clinical spellings", () => {
    expect(guessSlot("AX_T1_POST.zip")).toBe("t1c");
    expect(guessSlot("sub-01_T1w_ce.nii")).toBe("t1c");
    expect(guessSlot("t1gd.nii.gz")).toBe("t1c");
    expect(guessSlot("T1ce.nii.gz")).toBe("t1c");
    expect(guessSlot("AX T2 FLAIR.zip")).toBe("flair");
    expect(guessSlot("sub-01_T2w.nii.gz")).toBe("t2");
    expect(guessSlot("sub-01_T1w.nii.gz")).toBe("t1");
    expect(guessSlot("synthetic_segmentation.nii.gz")).toBe("segmentation");
    expect(guessSlot("labels.nii")).toBe("segmentation");
  });

  it("does not guess from digits that are part of another word", () => {
    expect(guessSlot("patient1_scan.nii.gz")).toBeNull();
    expect(guessSlot("set2.nii")).toBeNull();
    expect(guessSlot("Timepoint_1.nii.gz")).toBeNull();
    expect(guessSlot("brain.nii.gz")).toBeNull();
  });

  it("assigns a whole drop, reporting what it could not place", () => {
    const files = ["a_flair.nii.gz", "a_t1.nii.gz", "a_t2.nii.gz", "a_t1c.nii.gz", "a_seg.nii.gz", "notes.nii", "b_t1.nii.gz"].map(
      (name) => ({ name }),
    );
    const { assigned, unmatched } = assignFiles(files, ALL);
    expect(Object.fromEntries(Object.entries(assigned).map(([slot, file]) => [slot, file!.name]))).toEqual({
      t1: "a_t1.nii.gz",
      t1c: "a_t1c.nii.gz",
      t2: "a_t2.nii.gz",
      flair: "a_flair.nii.gz",
      segmentation: "a_seg.nii.gz",
    });
    expect(unmatched).toEqual(["notes.nii", "b_t1.nii.gz"]);
  });

  it("leaves out slots that are not on offer (no segmentation outside demo mode)", () => {
    const { assigned, unmatched } = assignFiles([{ name: "x_seg.nii.gz" }, { name: "x_t2.nii.gz" }], ["t1", "t1c", "t2", "flair"]);
    expect(Object.keys(assigned)).toEqual(["t2"]);
    expect(unmatched).toEqual(["x_seg.nii.gz"]);
  });
});
