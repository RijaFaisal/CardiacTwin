import { useState, useRef, useEffect } from 'react';
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { ecgClient, explainFetchError, isProductionApiMissing } from '../../api/ecgClient';

interface Props {
    onUploadSuccess: (id: string) => void;
}

// ── Inline 3D heart for landing (no bpm prop needed, fixed normal state) ──
function LandingHeart() {
    const mountRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        const mount = mountRef.current;
        if (!mount) return;

        const scene    = new THREE.Scene();
        scene.background = new THREE.Color(0x080810);

        const camera = new THREE.PerspectiveCamera(42, mount.clientWidth / mount.clientHeight, 0.1, 100);
        camera.position.set(0, 0.2, 3.8);

        const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        renderer.setSize(mount.clientWidth, mount.clientHeight);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        renderer.toneMapping       = THREE.ACESFilmicToneMapping;
        renderer.toneMappingExposure = 1.1;
        mount.appendChild(renderer.domElement);

        // Lighting — clinical blue-white with emerald heart accent
        scene.add(new THREE.AmbientLight(0xffffff, 0.5));
        const key = new THREE.DirectionalLight(0xffffff, 1.4);
        key.position.set(4, 6, 4);
        scene.add(key);

        const rimLight = new THREE.DirectionalLight(0x6ee7f7, 0.6);
        rimLight.position.set(-4, 2, -3);
        scene.add(rimLight);

        // Green pulse glow — signals healthy / ready state
        const pulse = new THREE.PointLight(0x22c55e, 3.5, 8);
        pulse.position.set(-1.5, 1, 2);
        scene.add(pulse);

        const fill = new THREE.PointLight(0xff3344, 0.7, 5);
        fill.position.set(0, -2, 1);
        scene.add(fill);

        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping    = true;
        controls.dampingFactor    = 0.06;
        controls.autoRotate       = true;
        controls.autoRotateSpeed  = 0.6;
        controls.minDistance      = 2;
        controls.maxDistance      = 7;
        controls.enablePan        = false;

        let heart: THREE.Object3D | null = null;
        new GLTFLoader().load(
            '/heart.glb',
            (gltf) => {
                heart = gltf.scene;
                const box = new THREE.Box3().setFromObject(heart);
                heart.position.sub(box.getCenter(new THREE.Vector3()));
                scene.add(heart);
            },
            undefined,
            (err) => console.warn('heart.glb:', err),
        );

        let animId: number;

        const animate = () => {
            controls.update();
            renderer.render(scene, camera);
            animId = requestAnimationFrame(animate);
        };
        animId = requestAnimationFrame(animate);

        const onResize = () => {
            if (!mount) return;
            camera.aspect = mount.clientWidth / mount.clientHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(mount.clientWidth, mount.clientHeight);
        };
        window.addEventListener('resize', onResize);

        return () => {
            cancelAnimationFrame(animId);
            window.removeEventListener('resize', onResize);
            controls.dispose();
            renderer.dispose();
            if (mount.contains(renderer.domElement)) {
                mount.removeChild(renderer.domElement);
            }
        };
    }, []);

    return <div ref={mountRef} className="w-full h-full" />;
}

// ── Scrolling ECG strip ────────────────────────────────────────────────────
function EcgTracer() {
    const canvasRef = useRef<HTMLCanvasElement>(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const syncSize = () => {
            canvas.width  = canvas.offsetWidth;
            canvas.height = canvas.offsetHeight;
        };
        syncSize();

        const g = (t: number, mu: number, sig: number, h: number) =>
            h * Math.exp(-0.5 * ((t - mu) / sig) ** 2);

        const ecgSample = (phase: number): number => {
            const t = ((phase % 1) + 1) % 1;
            return (
                g(t, 0.14, 0.030,  0.18) +
                g(t, 0.43, 0.013, -0.12) +
                g(t, 0.46, 0.017,  1.00) +
                g(t, 0.50, 0.013, -0.18) +
                g(t, 0.66, 0.050,  0.32)
            );
        };

        const PX_PER_BEAT = 240;
        const BPM         = 63;

        let phase    = 0;
        let lastTime = performance.now();
        let animId:  number;

        const draw = (now: number) => {
            const dt  = Math.min((now - lastTime) / 1000, 0.1);
            lastTime  = now;
            phase    += (BPM / 60) * dt;

            const W   = canvas.width;
            const H   = canvas.height;
            const mid = H * 0.50;
            const amp = H * 0.38;

            ctx.clearRect(0, 0, W, H);

            // Faint ECG-paper grid
            const gridPx = PX_PER_BEAT / 5;
            ctx.lineWidth   = 1;
            ctx.strokeStyle = 'rgba(34, 197, 94, 0.06)';
            for (let x = 0; x < W; x += gridPx) {
                ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
            }
            ctx.beginPath(); ctx.moveTo(0, mid); ctx.lineTo(W, mid); ctx.stroke();

            // Waveform
            ctx.beginPath();
            for (let x = 0; x <= W; x++) {
                const p = phase - (W - x) / PX_PER_BEAT;
                const y = mid - ecgSample(p) * amp;
                x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
            }

            const grad = ctx.createLinearGradient(0, 0, W, 0);
            grad.addColorStop(0.00, 'rgba(52, 211, 153, 0.00)');
            grad.addColorStop(0.18, 'rgba(52, 211, 153, 0.55)');
            grad.addColorStop(0.60, 'rgba(52, 211, 153, 0.80)');
            grad.addColorStop(1.00, 'rgba(52, 211, 153, 0.90)');

            ctx.strokeStyle = grad;
            ctx.lineWidth   = 1.5;
            ctx.lineJoin    = 'round';
            ctx.stroke();

            animId = requestAnimationFrame(draw);
        };

        animId = requestAnimationFrame(draw);
        window.addEventListener('resize', syncSize);

        return () => {
            cancelAnimationFrame(animId);
            window.removeEventListener('resize', syncSize);
        };
    }, []);

    return (
        <canvas
            ref={canvasRef}
            className="absolute bottom-0 inset-x-0 w-full pointer-events-none"
            style={{ height: '72px' }}
        />
    );
}

// ── Main upload screen ────────────────────────────────────────────────────
export default function UploadScreen({ onUploadSuccess }: Props) {
    const [isDragging,    setIsDragging]    = useState(false);
    const [error,         setError]         = useState<string | null>(null);
    const [uploading,     setUploading]     = useState(false);
    const [demoLoading,   setDemoLoading]   = useState<string | null>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);

    // ── File upload logic ─────────────────────────────────────────────────
    const handleFiles = async (files: FileList | null) => {
        if (!files || files.length === 0) return;
        setError(null);
        setUploading(true);

        try {
            const fileArray = Array.from(files);

            const csvFile = fileArray.find(f => f.name.toLowerCase().endsWith('.csv'));
            if (csvFile) {
                const res = await ecgClient.uploadCsv(csvFile);
                onUploadSuccess(res.session_id);
                return;
            }

            const heaFile = fileArray.find(f => f.name.toLowerCase().endsWith('.hea'));
            const datFile = fileArray.find(f => f.name.toLowerCase().endsWith('.dat'));

            if (heaFile && datFile) {
                const res = await ecgClient.uploadWfdb(heaFile, datFile);
                onUploadSuccess(res.session_id);
                return;
            }

            if (fileArray.length === 1 && (heaFile || datFile)) {
                setError('WFDB requires both .hea AND .dat files. Select both together.');
                return;
            }

            setError('Unsupported format. Upload a .csv file or both .hea & .dat files.');
        } catch (err: any) {
            setError(
                err?.status === 0
                    ? err.message
                    : explainFetchError(err, 'Upload failed.'),
            );
        } finally {
            setUploading(false);
        }
    };

    // ── Demo loading ──────────────────────────────────────────────────────
    const handleDemo = async (scenario: 'normal' | 'afib' | 'bbb') => {
        setError(null);
        setDemoLoading(scenario);
        try {
            const { session_id } = await ecgClient.loadDemo(scenario);
            onUploadSuccess(session_id);
        } catch (err: any) {
            setError(
                err?.status === 0
                    ? err.message
                    : explainFetchError(err, 'Could not load demo record.'),
            );
        } finally {
            setDemoLoading(null);
        }
    };

    const onDrop = (e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(false);
        handleFiles(e.dataTransfer.files);
    };

    const busy = uploading || demoLoading !== null;

    return (
        <div className="min-h-screen bg-zinc-950 flex relative">

            {/* ── Left panel — upload UI ──────────────────────────────── */}
            <div className="flex-1 flex flex-col justify-center px-12 py-16 max-w-2xl">

                {isProductionApiMissing && (
                    <div className="mb-6 px-4 py-3 rounded-xl border border-amber-500/40 bg-amber-500/10 text-amber-200 text-sm leading-relaxed">
                        <p className="font-semibold text-amber-100 mb-1">API not configured</p>
                        <p className="text-amber-200/90 text-xs">
                            Add <code className="text-amber-300">VITE_API_BASE_URL</code> in Vercel
                            (e.g. <code className="text-amber-300">https://your-backend…/api/v1</code>),
                            then trigger a new deployment so the build picks it up.
                        </p>
                    </div>
                )}

                {/* Brand */}
                <div className="mb-10">
                    <span className="text-[10px] font-bold tracking-[0.25em] uppercase text-emerald-500/80">
                        Cardiac Digital Twin
                    </span>
                    <h1 className="mt-3 text-5xl font-bold text-zinc-50 tracking-tight leading-[1.1]">
                        See inside every heartbeat.
                    </h1>
                    <p className="mt-4 text-zinc-400 text-base leading-relaxed max-w-sm">
                        Upload a 12-lead ECG and get full explainability.
                    </p>
                </div>

                {/* Divider */}
                <div className="flex items-center gap-3 mb-5">
                    <div className="flex-1 h-px bg-zinc-800" />
                    <span className="text-[10px] text-zinc-600 uppercase tracking-widest">upload your own</span>
                    <div className="flex-1 h-px bg-zinc-800" />
                </div>

                {/* Drop zone */}
                <div
                    onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
                    onDragLeave={(e) => { e.preventDefault(); setIsDragging(false); }}
                    onDrop={onDrop}
                    onClick={() => !busy && fileInputRef.current?.click()}
                    className={`relative border-2 border-dashed rounded-2xl p-8 text-center transition-all duration-200
                        ${busy ? 'opacity-60 cursor-not-allowed' : 'cursor-pointer'}
                        ${isDragging
                            ? 'border-emerald-500 bg-emerald-500/5 scale-[1.01]'
                            : 'border-zinc-800 hover:border-red-500/50 bg-zinc-900/50 hover:bg-zinc-900 hover:shadow-[0_0_22px_rgba(255,51,68,0.22)]'
                        }`}
                >
                    <input
                        type="file"
                        multiple
                        className="hidden"
                        ref={fileInputRef}
                        onChange={(e) => handleFiles(e.target.files)}
                        accept=".csv,.hea,.dat"
                        disabled={busy}
                    />

                    {uploading ? (
                        <div className="flex flex-col items-center gap-3">
                            <div className="w-8 h-8 border-2 border-zinc-700 border-t-emerald-500 rounded-full animate-spin" />
                            <p className="text-sm text-zinc-400">Uploading and processing…</p>
                        </div>
                    ) : (
                        <div className="flex flex-col items-center gap-2">
                            <svg className="w-8 h-8 text-zinc-600 mb-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                                    d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                            </svg>
                            <p className="text-sm font-medium text-zinc-300">Drop files here or click to browse</p>
                            <div className="flex gap-2 mt-1">
                                <span className="text-[10px] bg-zinc-800 text-zinc-500 px-2 py-0.5 rounded font-mono">.csv</span>
                                <span className="text-[10px] bg-zinc-800 text-zinc-500 px-2 py-0.5 rounded font-mono">.hea + .dat</span>
                            </div>
                        </div>
                    )}
                </div>

                {error && (
                    <div className="mt-4 px-4 py-3 bg-red-500/10 border border-red-500/20 rounded-xl text-red-400 text-sm">
                        {error}
                    </div>
                )}

            </div>

            {/* ── Right panel — 3D heart ──────────────────────────────── */}
            <div className="hidden lg:flex flex-1 relative overflow-hidden">
                <LandingHeart />

                {/* Overlay gradient fade on left edge */}
                <div className="absolute inset-y-0 left-0 w-24 bg-gradient-to-r from-zinc-950 to-transparent pointer-events-none" />

                {/* Stats overlay bottom-right */}
                <div className="absolute bottom-10 right-10 text-right space-y-3 pointer-events-none">
                    {[
                        { label: 'Macro AUC',   value: '0.946' },
                        { label: 'Conditions',  value: '71' },
                        { label: 'ECG Leads',   value: '12' },
                    ].map(({ label, value }) => (
                        <div key={label}>
                            <p className="text-2xl font-light text-zinc-100">{value}</p>
                            <p className="text-[10px] uppercase tracking-widest text-zinc-600">{label}</p>
                        </div>
                    ))}
                </div>
            </div>

            <EcgTracer />
        </div>
    );
}
