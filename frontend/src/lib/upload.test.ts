import { gunzipSync } from "node:zlib";
import { describe, expect, it } from "vitest";

import { gzipIfUncompressed } from "./api";

describe("browser-side gzip of uploads", () => {
  it("compresses a raw .nii losslessly and renames it .nii.gz", async () => {
    const bytes = new Uint8Array(200_000).map((_, index) => (index % 97 === 0 ? index % 251 : 0));
    const file = new File([bytes], "scan_t1c.nii");
    const out = await gzipIfUncompressed(file);
    expect(out.name).toBe("scan_t1c.nii.gz");
    expect(out.size).toBeLessThan(file.size / 3);
    expect(new Uint8Array(gunzipSync(Buffer.from(await out.arrayBuffer())))).toEqual(bytes);
  });

  it("leaves already-compressed files and DICOM zips alone", async () => {
    for (const name of ["a.nii.gz", "series.zip"]) {
      const file = new File([new Uint8Array(10)], name);
      expect(await gzipIfUncompressed(file)).toBe(file);
    }
  });
});
