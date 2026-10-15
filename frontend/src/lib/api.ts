import type { Vec3 } from "./geometry";
import type { ApiInfo, BackendVoxel, JobStatus, Summary, Volume, VolumeHeader } from "./types";
import { makeVolume } from "./volumes";

const BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? "";

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
  ) {
    super(message);
  }
}

async function detailOf(response: Response): Promise<string> {
  try {
    const body = await response.json();
    if (typeof body?.detail === "string") return body.detail;
    if (Array.isArray(body?.detail)) return body.detail.map((d: { msg?: string }) => d.msg).join("; ");
  } catch {
    /* not JSON */
  }
  return `Request failed (${response.status})`;
}

export async function getJsonPublic<T>(path: string): Promise<T> {
  return getJson<T>(path);
}

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(BASE + path);
  if (!response.ok) throw new ApiError(await detailOf(response), response.status);
  return (await response.json()) as T;
}

export const getInfo = () => getJson<ApiInfo>("/api/info");
export const getJob = (id: string) => getJson<JobStatus>(`/api/jobs/${id}`);
export const getSummary = (id: string) => getJson<Summary>(`/api/jobs/${id}/summary`);
export const meshUrl = (id: string, name: string) => `${BASE}/api/jobs/${id}/meshes/${name}.glb`;
export const downloadUrl = (id: string) => `${BASE}/api/jobs/${id}/download/segmentation.nii.gz`;

export async function deleteJob(id: string): Promise<void> {
  const response = await fetch(`${BASE}/api/jobs/${id}`, { method: "DELETE" });
  if (!response.ok && response.status !== 404) throw new ApiError(await detailOf(response), response.status);
}

export async function getVolume(id: string, name: string): Promise<Volume> {
  const header = await getJson<VolumeHeader>(`/api/jobs/${id}/volumes/${name}/header`);
  // The body is gzip-encoded on the wire; the browser decodes it transparently.
  const response = await fetch(`${BASE}/api/jobs/${id}/volumes/${name}`);
  if (!response.ok) throw new ApiError(await detailOf(response), response.status);
  return makeVolume(header, await response.arrayBuffer());
}

export function getVoxel(id: string, world: Vec3, signal?: AbortSignal): Promise<BackendVoxel> {
  const query = `x=${world[0]}&y=${world[1]}&z=${world[2]}`;
  return fetch(`${BASE}/api/jobs/${id}/voxel?${query}`, { signal }).then(async (response) => {
    if (!response.ok) throw new ApiError(await detailOf(response), response.status);
    return (await response.json()) as BackendVoxel;
  });
}

/** Demo mode: start a job on the server's built-in synthetic case. */
export async function createSampleJob(): Promise<{ job_id: string }> {
  const response = await fetch(`${BASE}/api/demo/sample`, { method: "POST" });
  if (!response.ok) throw new ApiError(await detailOf(response), response.status);
  return (await response.json()) as { job_id: string };
}

export interface UploadFiles {
  t1: File;
  t1c: File;
  t2: File;
  flair: File;
  segmentation?: File | null;
}

/**
 * Gzip an uncompressed .nii in the browser before upload. Four raw MRI volumes
 * are ~140 MB; compressed they are 3–4× smaller, which keeps uploads fast and
 * under the ~100 MB request limit of tunnels such as Cloudflare's.
 */
export async function gzipIfUncompressed(file: File): Promise<File> {
  if (!/\.nii$/i.test(file.name) || typeof CompressionStream === "undefined") return file;
  const compressed = await new Response(file.stream().pipeThrough(new CompressionStream("gzip"))).blob();
  return new File([compressed], `${file.name}.gz`, { type: "application/gzip" });
}

/** POST /api/jobs with upload progress (fetch cannot report it, XHR can). */
export async function createJob(
  files: UploadFiles,
  alreadyPreprocessed: boolean,
  onProgress: (fraction: number) => void,
): Promise<{ job_id: string }> {
  const [t1, t1c, t2, flair, segmentation] = await Promise.all(
    [files.t1, files.t1c, files.t2, files.flair, files.segmentation ?? null].map((file) => (file ? gzipIfUncompressed(file) : null)),
  );
  return sendJob({ t1: t1!, t1c: t1c!, t2: t2!, flair: flair!, segmentation }, alreadyPreprocessed, onProgress);
}

function sendJob(
  files: UploadFiles,
  alreadyPreprocessed: boolean,
  onProgress: (fraction: number) => void,
): Promise<{ job_id: string }> {
  return new Promise((resolve, reject) => {
    const form = new FormData();
    form.append("t1", files.t1);
    form.append("t1c", files.t1c);
    form.append("t2", files.t2);
    form.append("flair", files.flair);
    if (files.segmentation) form.append("segmentation", files.segmentation);
    form.append("already_preprocessed", String(alreadyPreprocessed));

    const request = new XMLHttpRequest();
    request.open("POST", `${BASE}/api/jobs`);
    request.upload.onprogress = (event) => {
      if (event.lengthComputable) onProgress(event.loaded / event.total);
    };
    request.onerror = () => reject(new ApiError("Could not reach the server.", 0));
    request.onload = () => {
      let body: { job_id?: string; detail?: unknown } = {};
      try {
        body = JSON.parse(request.responseText);
      } catch {
        /* not JSON */
      }
      if (request.status >= 200 && request.status < 300 && body.job_id) {
        resolve({ job_id: body.job_id });
      } else {
        const detail = typeof body.detail === "string" ? body.detail : `Upload failed (${request.status})`;
        reject(new ApiError(detail, request.status));
      }
    };
    request.send(form);
  });
}
