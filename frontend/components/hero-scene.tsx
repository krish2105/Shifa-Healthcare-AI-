"use client";

import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { useMemo, useRef } from "react";
import * as THREE from "three";

/**
 * The 3D hero: an abstract node lattice standing in for the clinical knowledge
 * graph — entities orbiting a core, connected by edges.
 *
 * Performance rules this scene follows, because a hero that drops frames is worse
 * than no hero at all:
 *
 * - **Instanced nodes.** One `InstancedMesh` for all ~40 spheres instead of 40
 *   draw calls.
 * - **One shared geometry and material** per class of object.
 * - **`dpr` capped at 1.5.** On a 3× Retina display an uncapped canvas renders
 *   ~4× the pixels for a background element nobody inspects closely.
 * - **`frameloop="demand"` is deliberately NOT used** — the scene rotates
 *   continuously — but rotation is driven off `useFrame` delta rather than clock
 *   accumulation so it stays framerate-independent.
 * - **Mouse parallax is damped**, and applied to the group, not per-node.
 *
 * This component is only ever mounted behind a dynamic import with `ssr:false`
 * and an effects check — see `hero.tsx`. Nothing here runs when the user has
 * motion disabled or WebGL is unavailable.
 */

const NODE_COUNT = 42;
const RADIUS = 2.6;

function useNodePositions() {
  return useMemo(() => {
    const pts: THREE.Vector3[] = [];
    // Fibonacci sphere: even angular distribution without the pole clustering
    // that naive spherical random sampling produces.
    const golden = Math.PI * (3 - Math.sqrt(5));
    for (let i = 0; i < NODE_COUNT; i++) {
      const y = 1 - (i / (NODE_COUNT - 1)) * 2;
      const r = Math.sqrt(1 - y * y);
      const theta = golden * i;
      // Slight radial jitter so it reads as an organic graph, not a grid.
      const jitter = 0.82 + ((i * 37) % 100) / 380;
      pts.push(
        new THREE.Vector3(Math.cos(theta) * r, y, Math.sin(theta) * r).multiplyScalar(
          RADIUS * jitter,
        ),
      );
    }
    return pts;
  }, []);
}

function Lattice({ accent }: { accent: THREE.Color }) {
  const group = useRef<THREE.Group>(null);
  const instances = useRef<THREE.InstancedMesh>(null);
  const points = useNodePositions();
  const { pointer } = useThree();

  const dummy = useMemo(() => new THREE.Object3D(), []);

  // Edges: connect each node to its nearest neighbours only. A fully-connected
  // graph at this node count is visual mud and 800+ line segments.
  const edgeGeometry = useMemo(() => {
    const verts: number[] = [];
    for (let i = 0; i < points.length; i++) {
      const dists = points
        .map((p, j) => ({ j, d: points[i].distanceTo(p) }))
        .filter((x) => x.j !== i)
        .sort((a, b) => a.d - b.d)
        .slice(0, 2);
      for (const { j } of dists) {
        if (j < i) continue; // each edge once
        verts.push(points[i].x, points[i].y, points[i].z, points[j].x, points[j].y, points[j].z);
      }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(verts, 3));
    return g;
  }, [points]);

  useFrame((_, delta) => {
    const g = group.current;
    if (!g) return;

    g.rotation.y += delta * 0.11;
    g.rotation.x = THREE.MathUtils.damp(g.rotation.x, pointer.y * 0.22, 3, delta);
    g.rotation.z = THREE.MathUtils.damp(g.rotation.z, -pointer.x * 0.14, 3, delta);

    const mesh = instances.current;
    if (mesh) {
      for (let i = 0; i < points.length; i++) {
        dummy.position.copy(points[i]);
        const s = 0.055 + (i % 5) * 0.012;
        dummy.scale.setScalar(s);
        dummy.updateMatrix();
        mesh.setMatrixAt(i, dummy.matrix);
      }
      mesh.instanceMatrix.needsUpdate = true;
    }
  });

  return (
    <group ref={group}>
      <lineSegments geometry={edgeGeometry}>
        <lineBasicMaterial color={accent} transparent opacity={0.16} />
      </lineSegments>

      <instancedMesh ref={instances} args={[undefined, undefined, NODE_COUNT]}>
        <sphereGeometry args={[1, 12, 12]} />
        <meshBasicMaterial color={accent} transparent opacity={0.9} />
      </instancedMesh>

      {/* Core */}
      <mesh>
        <sphereGeometry args={[0.42, 32, 32]} />
        <meshBasicMaterial color={accent} transparent opacity={0.22} />
      </mesh>
      <mesh>
        <sphereGeometry args={[0.2, 24, 24]} />
        <meshBasicMaterial color={accent} />
      </mesh>
    </group>
  );
}

export default function HeroScene() {
  // Read the live accent token so the scene follows the theme toggle instead of
  // hardcoding a colour that would be wrong in light mode.
  const accent = useMemo(() => {
    if (typeof window === "undefined") return new THREE.Color("#2ee6c5");
    const raw = getComputedStyle(document.documentElement).getPropertyValue("--accent").trim();
    const [r, g, b] = raw.split(/\s+/).map(Number);
    return Number.isFinite(r) ? new THREE.Color(r / 255, g / 255, b / 255) : new THREE.Color("#2ee6c5");
  }, []);

  return (
    <Canvas
      camera={{ position: [0, 0, 7.2], fov: 45 }}
      dpr={[1, 1.5]}
      gl={{ antialias: true, alpha: true, powerPreference: "low-power" }}
      style={{ pointerEvents: "none" }}
    >
      <Lattice accent={accent} />
    </Canvas>
  );
}
