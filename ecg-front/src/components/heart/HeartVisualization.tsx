import { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

interface Props {
    bpm: number | null;
    severity: 'normal' | 'warning' | 'critical';
}

const SEVERITY_COLOR: Record<string, number> = {
    normal:   0x22c55e,   // green
    warning:  0xeab308,   // yellow
    critical: 0xef4444,   // red
};

export default function HeartVisualization({ bpm, severity }: Props) {
    const mountRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        const mount = mountRef.current;
        if (!mount) return;

        // ── Scene setup ───────────────────────────────────────────────
        const scene    = new THREE.Scene();
        scene.background = new THREE.Color(0x0a0a0f);

        const camera = new THREE.PerspectiveCamera(45, mount.clientWidth / mount.clientHeight, 0.1, 100);
        camera.position.set(0, 0, 3.5);

        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(mount.clientWidth, mount.clientHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        mount.appendChild(renderer.domElement);

        // ── Lighting ──────────────────────────────────────────────────
        scene.add(new THREE.AmbientLight(0xffffff, 0.6));
        const dir = new THREE.DirectionalLight(0xffffff, 1.2);
        dir.position.set(3, 5, 3);
        scene.add(dir);

        // Coloured point light that matches severity
        const accent = new THREE.PointLight(SEVERITY_COLOR[severity] ?? 0x22c55e, 2, 8);
        accent.position.set(-2, 1, 2);
        scene.add(accent);

        // ── Controls ──────────────────────────────────────────────────
        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.minDistance   = 1.5;
        controls.maxDistance   = 8;

        // ── Load GLB heart model ──────────────────────────────────────
        let heart: THREE.Object3D | null = null;
        const loader = new GLTFLoader();
        loader.load(
            '/heart.glb',
            (gltf) => {
                heart = gltf.scene;
                // Apply accent colour to all meshes
                heart.traverse((child) => {
                    if ((child as THREE.Mesh).isMesh) {
                        const mesh = child as THREE.Mesh;
                        const mat  = new THREE.MeshStandardMaterial({
                            color:     SEVERITY_COLOR[severity] ?? 0x22c55e,
                            roughness: 0.4,
                            metalness: 0.1,
                        });
                        mesh.material = mat;
                    }
                });
                // Centre the model
                const box = new THREE.Box3().setFromObject(heart);
                const ctr = box.getCenter(new THREE.Vector3());
                heart.position.sub(ctr);
                scene.add(heart);
            },
            undefined,
            (err) => console.warn('heart.glb load error', err),
        );

        // ── Pulse animation ───────────────────────────────────────────
        // Beat interval in ms derived from BPM (default 72 if null)
        const beatMs  = 60_000 / (bpm ?? 72);
        let   elapsed = 0;
        let   last    = performance.now();

        const animate = () => {
            const now   = performance.now();
            elapsed += now - last;
            last     = now;

            if (heart) {
                // Slow continuous rotation
                heart.rotation.y += 0.003;

                // Pulse: quick scale up then back down over one beat cycle
                const phase = (elapsed % beatMs) / beatMs;         // 0 → 1
                const pulse = phase < 0.15
                    ? 1 + 0.08 * (phase / 0.15)                    // expand
                    : phase < 0.35
                        ? 1 + 0.08 * (1 - (phase - 0.15) / 0.20)  // contract
                        : 1.0;                                      // rest
                heart.scale.setScalar(pulse);
            }

            controls.update();
            renderer.render(scene, camera);
            animId = requestAnimationFrame(animate);
        };

        let animId = requestAnimationFrame(animate);

        // ── Resize handler ────────────────────────────────────────────
        const onResize = () => {
            if (!mount) return;
            camera.aspect = mount.clientWidth / mount.clientHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(mount.clientWidth, mount.clientHeight);
        };
        window.addEventListener('resize', onResize);

        // ── Cleanup ───────────────────────────────────────────────────
        return () => {
            cancelAnimationFrame(animId);
            window.removeEventListener('resize', onResize);
            controls.dispose();
            renderer.dispose();
            mount.removeChild(renderer.domElement);
        };
    }, [bpm, severity]);

    return (
        <div className="w-full h-full flex flex-col">
            <div className="px-3 pt-3 pb-1 flex items-center justify-between">
                <span className="text-xs font-semibold tracking-widest uppercase text-zinc-400">
                    Cardiac Digital Twin
                </span>
                {bpm !== null && (
                    <span className="text-xs font-mono text-zinc-300">
                        {Math.round(bpm)} <span className="text-zinc-500">bpm</span>
                    </span>
                )}
            </div>
            <div ref={mountRef} className="flex-1 w-full" />
        </div>
    );
}
