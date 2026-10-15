import { GizmoHelper, GizmoViewport, OrbitControls } from "@react-three/drei";
import { Canvas, useFrame, useLoader, useThree } from "@react-three/fiber";
import { Suspense, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";

import { meshUrl } from "../lib/api";
import { voxelToWorld, type Mat4, type Vec3 } from "../lib/geometry";
import { pickVoxelAlongRay } from "../lib/picking";
import type { LabelInfo, Summary, VoxelInfo } from "../lib/types";
import { collectLabelVoxels, lookupWorld, type VolumeSet } from "../lib/volumes";
import { CAMERA_VIEWS, useViewerStore } from "../state/viewerStore";
import { HoverTip, type Tip } from "./ViewChrome";

/**
 * Scene coordinates ARE world millimetres (RAS): the GLB vertices are already
 * in world mm, the camera's up vector is +z (Superior), and nothing in the
 * scene graph carries a transform. A raycast hit point is therefore a world-mm
 * point with no conversion.
 */

const BRAIN_COLOR = "#cfc6bd";
export const MAX_VOXEL_INSTANCES = 700_000;

interface Bounds {
  centre: THREE.Vector3;
  radius: number;
  min: number[];
  max: number[];
}

function useGlbGeometry(url: string): THREE.BufferGeometry {
  const gltf = useLoader(GLTFLoader, url);
  return useMemo(() => {
    gltf.scene.updateMatrixWorld(true);
    let geometry: THREE.BufferGeometry | null = null;
    gltf.scene.traverse((object) => {
      const mesh = object as THREE.Mesh;
      if (mesh.isMesh && geometry === null) {
        geometry = mesh.geometry.clone();
        geometry.applyMatrix4(mesh.matrixWorld); // identity for our files; kept for safety
      }
    });
    if (geometry === null) throw new Error(`No mesh in ${url}`);
    const result = geometry as THREE.BufferGeometry;
    if (!result.getAttribute("normal")) result.computeVertexNormals();
    result.computeBoundingSphere();
    return result;
  }, [gltf, url]);
}

function useClipPlanes(bounds: Bounds): THREE.Plane[] {
  const clip = useViewerStore((state) => state.clip);
  return useMemo(() => {
    if (!clip.enabled) return [];
    const position = bounds.min[clip.axis] + clip.fraction * (bounds.max[clip.axis] - bounds.min[clip.axis]);
    const normal = new THREE.Vector3();
    // three.js keeps the half-space where normal·p + constant >= 0.
    normal.setComponent(clip.axis, clip.flip ? 1 : -1);
    return [new THREE.Plane(normal, clip.flip ? -position : position)];
  }, [clip, bounds]);
}

function BrainMesh({ url, planes, meshRef }: { url: string; planes: THREE.Plane[]; meshRef: React.MutableRefObject<THREE.Mesh | null> }) {
  const geometry = useGlbGeometry(url);
  const brain = useViewerStore((state) => state.brain);
  return (
    <mesh ref={meshRef} geometry={geometry} visible={brain.visible} renderOrder={10}>
      {/* Drawn after the opaque tumor meshes and without writing depth, so what is inside stays visible. */}
      <meshStandardMaterial
        color={BRAIN_COLOR}
        transparent
        opacity={brain.opacity}
        depthWrite={false}
        roughness={0.75}
        metalness={0}
        // Front faces only keeps the shell clean; when it is cut open, the far
        // inner wall has to be drawn too.
        side={planes.length > 0 ? THREE.DoubleSide : THREE.FrontSide}
        clippingPlanes={planes}
      />
    </mesh>
  );
}

function LabelMesh({ url, label, planes }: { url: string; label: LabelInfo; planes: THREE.Plane[] }) {
  const geometry = useGlbGeometry(url);
  const layer = useViewerStore((state) => state.labels[label.value]);
  const translucent = layer.opacity < 0.999;
  return (
    <mesh geometry={geometry} visible={layer.visible} renderOrder={translucent ? 5 : 0}>
      <meshStandardMaterial
        color={label.color}
        transparent={translucent}
        opacity={layer.opacity}
        depthWrite={!translucent}
        roughness={0.55}
        metalness={0}
        clippingPlanes={planes}
      />
    </mesh>
  );
}

/** One cube per voxel. Each instance matrix is affine · translate(i, j, k), i.e. the voxel's own cell in world mm. */
function VoxelInstances({ voxels, affine, label, planes }: { voxels: Int16Array; affine: Mat4; label: LabelInfo; planes: THREE.Plane[] }) {
  const ref = useRef<THREE.InstancedMesh>(null);
  const layer = useViewerStore((state) => state.labels[label.value]);
  const count = voxels.length / 3;
  const translucent = layer.opacity < 0.999;

  useLayoutEffect(() => {
    const mesh = ref.current;
    if (!mesh) return;
    const voxelToWorldMatrix = new THREE.Matrix4().set(
      affine[0][0], affine[0][1], affine[0][2], affine[0][3],
      affine[1][0], affine[1][1], affine[1][2], affine[1][3],
      affine[2][0], affine[2][1], affine[2][2], affine[2][3],
      0, 0, 0, 1,
    );
    const translate = new THREE.Matrix4();
    const matrix = new THREE.Matrix4();
    for (let index = 0; index < count; index++) {
      translate.makeTranslation(voxels[3 * index], voxels[3 * index + 1], voxels[3 * index + 2]);
      matrix.multiplyMatrices(voxelToWorldMatrix, translate);
      mesh.setMatrixAt(index, matrix);
    }
    mesh.instanceMatrix.needsUpdate = true;
  }, [voxels, affine, count]);

  return (
    <instancedMesh
      key={count}
      ref={ref}
      args={[undefined, undefined, count]}
      visible={layer.visible}
      frustumCulled={false}
      renderOrder={translucent ? 5 : 0}
    >
      <boxGeometry args={[1, 1, 1]} />
      <meshLambertMaterial
        color={label.color}
        transparent={translucent}
        opacity={layer.opacity}
        depthWrite={!translucent}
        clippingPlanes={planes}
      />
    </instancedMesh>
  );
}

function CrosshairMarker({ size }: { size: number }) {
  const crosshair = useViewerStore((state) => state.crosshair);
  const lines = useMemo(() => {
    const geometry = new THREE.BufferGeometry();
    const s = size;
    geometry.setAttribute(
      "position",
      new THREE.Float32BufferAttribute([-s, 0, 0, s, 0, 0, 0, -s, 0, 0, s, 0, 0, 0, -s, 0, 0, s], 3),
    );
    return geometry;
  }, [size]);
  if (!crosshair) return null;
  return (
    <group position={crosshair}>
      <lineSegments geometry={lines} renderOrder={999}>
        <lineBasicMaterial color="#ffffff" depthTest={false} transparent opacity={0.9} />
      </lineSegments>
      <mesh renderOrder={1000}>
        <sphereGeometry args={[Math.max(0.8, size * 0.06), 16, 16]} />
        <meshBasicMaterial color="#00e5ff" depthTest={false} transparent />
      </mesh>
    </group>
  );
}

function Headlight() {
  const light = useRef<THREE.DirectionalLight>(null);
  const { camera } = useThree();
  useFrame(() => {
    if (light.current) light.current.position.copy(camera.position);
  });
  return (
    <>
      <hemisphereLight args={["#ffffff", "#3a3f4a", 0.9]} />
      <directionalLight ref={light} intensity={1.6} />
    </>
  );
}

function CameraRig({ bounds }: { bounds: Bounds }) {
  const view = useViewerStore((state) => state.camera);
  const { camera, controls } = useThree();
  useEffect(() => {
    const direction = new THREE.Vector3(...view.direction).normalize();
    camera.position.copy(bounds.centre).addScaledVector(direction, bounds.radius * 2.75);
    camera.up.set(0, 0, 1);
    camera.near = Math.max(0.5, bounds.radius / 100);
    camera.far = bounds.radius * 40;
    camera.lookAt(bounds.centre);
    camera.updateProjectionMatrix();
    const orbit = controls as unknown as { target: THREE.Vector3; update: () => void } | null;
    if (orbit) {
      orbit.target.copy(bounds.centre);
      orbit.update();
    }
  }, [view, bounds, camera, controls]);
  return null;
}

/**
 * Hover and click in 3D → world mm.
 *
 * Tumor voxels are found by marching the ray through the label volume, which
 * returns the exact voxel under the cursor in both mesh and voxel mode. If the
 * ray meets no visible tumor voxel, the brain surface is used.
 */
function Picker({ volumes, planes, brainRef, onTip }: { volumes: VolumeSet; planes: THREE.Plane[]; brainRef: React.MutableRefObject<THREE.Mesh | null>; onTip: (info: VoxelInfo | null, x: number, y: number) => void }) {
  const { camera, gl } = useThree();
  const stateRef = useRef({ volumes, planes });
  stateRef.current = { volumes, planes };

  useEffect(() => {
    const element = gl.domElement;
    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();
    const scratch = new THREE.Vector3();
    let frame = 0;
    let lastEvent: PointerEvent | null = null;
    let down: { x: number; y: number } | null = null;

    const pick = (event: PointerEvent): Vec3 | null => {
      const rect = element.getBoundingClientRect();
      pointer.set(((event.clientX - rect.left) / rect.width) * 2 - 1, -((event.clientY - rect.top) / rect.height) * 2 + 1);
      raycaster.setFromCamera(pointer, camera);
      const { origin, direction } = raycaster.ray;
      const { volumes: current, planes: clipPlanes } = stateRef.current;
      const store = useViewerStore.getState();
      const kept = (world: Vec3) => clipPlanes.every((plane) => plane.distanceToPoint(scratch.set(world[0], world[1], world[2])) >= 0);

      const tumor = pickVoxelAlongRay(
        [origin.x, origin.y, origin.z],
        [direction.x, direction.y, direction.z],
        current.labels,
        (label, world) => Boolean(store.labels[label]?.visible) && kept(world),
      );
      if (tumor) return tumor.world;

      const brain = brainRef.current;
      if (brain && store.brain.visible) {
        const hit = raycaster.intersectObject(brain, false).find((h) => kept([h.point.x, h.point.y, h.point.z]));
        if (hit) {
          // Step just inside the surface so the lookup is not on the boundary.
          const inside = hit.point.clone().addScaledVector(direction, 0.5);
          return [inside.x, inside.y, inside.z];
        }
      }
      return null;
    };

    const onMove = (event: PointerEvent) => {
      lastEvent = event;
      if (frame) return;
      frame = requestAnimationFrame(() => {
        frame = 0;
        if (!lastEvent || lastEvent.buttons !== 0) return; // not while orbiting
        const world = pick(lastEvent);
        const info = world ? lookupWorld(world, stateRef.current.volumes, "3d") : null;
        useViewerStore.getState().setHover(info);
        const rect = element.getBoundingClientRect();
        onTip(info, lastEvent.clientX - rect.left, lastEvent.clientY - rect.top);
      });
    };
    const onLeave = () => {
      useViewerStore.getState().setHover(null);
      onTip(null, 0, 0);
    };
    const onDown = (event: PointerEvent) => {
      down = { x: event.clientX, y: event.clientY };
      onTip(null, 0, 0); // no read-out while orbiting
    };
    const onUp = (event: PointerEvent) => {
      if (!down || event.button !== 0) return;
      const moved = Math.hypot(event.clientX - down.x, event.clientY - down.y);
      down = null;
      if (moved > 4) return; // that was a drag (orbit), not a click
      const world = pick(event);
      if (!world) return;
      const store = useViewerStore.getState();
      store.setCrosshair(world);
      store.setPicked(lookupWorld(world, stateRef.current.volumes, "3d"));
    };

    element.addEventListener("pointermove", onMove);
    element.addEventListener("pointerleave", onLeave);
    element.addEventListener("pointerdown", onDown);
    element.addEventListener("pointerup", onUp);
    return () => {
      cancelAnimationFrame(frame);
      element.removeEventListener("pointermove", onMove);
      element.removeEventListener("pointerleave", onLeave);
      element.removeEventListener("pointerdown", onDown);
      element.removeEventListener("pointerup", onUp);
    };
  }, [camera, gl, brainRef, onTip]);
  return null;
}

function SceneContents({ jobId, summary, volumes, bounds, voxelsByLabel, onTip }: Scene3DProps & { bounds: Bounds; voxelsByLabel: Map<number, Int16Array> | null; onTip: (info: VoxelInfo | null, x: number, y: number) => void }) {
  const planes = useClipPlanes(bounds);
  const brainRef = useRef<THREE.Mesh | null>(null);
  const voxelMode = useViewerStore((state) => state.voxelMode);
  const foreground = summary.label_map.filter((label) => label.value !== 0);
  const affine = volumes.labels.header.affine;

  return (
    <>
      <Headlight />
      <CameraRig bounds={bounds} />
      <Suspense fallback={null}>
        {summary.meshes.brain && <BrainMesh url={meshUrl(jobId, "brain")} planes={planes} meshRef={brainRef} />}
      </Suspense>
      {foreground.map((label) =>
        voxelMode && voxelsByLabel ? (
          voxelsByLabel.get(label.value) ? (
            <VoxelInstances key={`voxels-${label.value}`} voxels={voxelsByLabel.get(label.value)!} affine={affine} label={label} planes={planes} />
          ) : null
        ) : summary.meshes[`label_${label.value}`] ? (
          <Suspense key={`mesh-${label.value}`} fallback={null}>
            <LabelMesh url={meshUrl(jobId, `label_${label.value}`)} label={label} planes={planes} />
          </Suspense>
        ) : null,
      )}
      <CrosshairMarker size={bounds.radius * 0.16} />
      <Picker volumes={volumes} planes={planes} brainRef={brainRef} onTip={onTip} />
      <OrbitControls makeDefault enableDamping dampingFactor={0.12} rotateSpeed={0.9} />
      <GizmoHelper alignment="bottom-left" margin={[58, 58]}>
        <GizmoViewport axisColors={["#e0564c", "#4fb06a", "#4a8fe0"]} labels={["R", "A", "S"]} labelColor="#ffffff" />
      </GizmoHelper>
    </>
  );
}

export interface Scene3DProps {
  jobId: string;
  summary: Summary;
  volumes: VolumeSet;
}

export function sceneBounds(summary: Summary): Bounds {
  // Prefer the brain mesh; fall back to the eight corners of the volume, mapped through the affine.
  let min: number[];
  let max: number[];
  const brain = summary.meshes.brain;
  if (brain) {
    min = brain.bounds_world_mm.min;
    max = brain.bounds_world_mm.max;
  } else {
    const { shape, affine } = summary.grid.viewer;
    min = [Infinity, Infinity, Infinity];
    max = [-Infinity, -Infinity, -Infinity];
    for (const i of [-0.5, shape[0] - 0.5])
      for (const j of [-0.5, shape[1] - 0.5])
        for (const k of [-0.5, shape[2] - 0.5]) {
          const corner = voxelToWorld([i, j, k], affine);
          for (let axis = 0; axis < 3; axis++) {
            min[axis] = Math.min(min[axis], corner[axis]);
            max[axis] = Math.max(max[axis], corner[axis]);
          }
        }
  }
  const centre = new THREE.Vector3((min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2);
  const radius = Math.hypot(max[0] - min[0], max[1] - min[1], max[2] - min[2]) / 2;
  return { centre, radius, min, max };
}

export function Scene3D({ jobId, summary, volumes }: Scene3DProps) {
  const bounds = useMemo(() => sceneBounds(summary), [summary]);
  const voxelMode = useViewerStore((state) => state.voxelMode);
  const surfaceOnly = useViewerStore((state) => state.surfaceVoxelsOnly);
  const voxelsByLabel = useMemo(
    () => (voxelMode ? collectLabelVoxels(volumes.labels, surfaceOnly) : null),
    [voxelMode, surfaceOnly, volumes.labels],
  );
  const instanceCount = voxelsByLabel ? [...voxelsByLabel.values()].reduce((sum, list) => sum + list.length / 3, 0) : 0;
  const tooMany = instanceCount > MAX_VOXEL_INSTANCES;

  const wrapper = useRef<HTMLDivElement>(null);
  const [tip, setTip] = useState<Tip | null>(null);
  const onTip = useCallback((info: VoxelInfo | null, x: number, y: number) => setTip(info ? { x, y, info } : null), []);
  const lookFrom = useViewerStore((state) => state.lookFrom);
  const resetView = useViewerStore((state) => state.resetView);
  const toggleMaximized = useViewerStore((state) => state.toggleMaximized);

  return (
    <div className="scene3d" ref={wrapper} onDoubleClick={() => toggleMaximized("3d")}>
      <Canvas
        camera={{ fov: 32, up: [0, 0, 1], near: 1, far: 5000, position: [300, 400, 250] }}
        gl={{ antialias: true, localClippingEnabled: true }}
        dpr={[1, 2]}
      >
        <color attach="background" args={["#0b0e13"]} />
        <SceneContents jobId={jobId} summary={summary} volumes={volumes} bounds={bounds} voxelsByLabel={tooMany ? null : voxelsByLabel} onTip={onTip} />
      </Canvas>
      <HoverTip tip={tip} width={wrapper.current?.clientWidth ?? 0} height={wrapper.current?.clientHeight ?? 0} labelMap={summary.label_map} />
      <div className="scene-tools" onDoubleClick={(event) => event.stopPropagation()}>
        {CAMERA_VIEWS.map((view) => (
          <button key={view.key} title={view.title} onClick={() => lookFrom(view.direction)}>
            {view.key}
          </button>
        ))}
        <button title="Reset the view" onClick={resetView} className="wide">
          Reset
        </button>
      </div>
      {voxelMode && (
        <div className="scene-note">
          {tooMany
            ? `Voxel mode is off: ${instanceCount.toLocaleString()} cubes exceed the limit of ${MAX_VOXEL_INSTANCES.toLocaleString()}. Turn on "visible shell only".`
            : `${instanceCount.toLocaleString()} voxel cubes${surfaceOnly ? " (visible shell)" : ""}`}
        </div>
      )}
    </div>
  );
}
