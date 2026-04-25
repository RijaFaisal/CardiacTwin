import { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import type { WaveformData } from '../../types/analysis';

interface Props {
    bpm:       number | null;
    severity:  'normal' | 'warning' | 'critical';
    rPeaks?:   number[];
    waveform?: WaveformData;
}

const SEV_LIGHT: Record<string, number> = {
    normal:   0x22c55e,
    warning:  0xeab308,
    critical: 0xff2222,
};

function computeRhythm(rrMs: number[]) {
    if (rrMs.length < 3) return null;
    const mean  = rrMs.reduce((a, b) => a + b, 0) / rrMs.length;
    const sdnn  = Math.round(Math.sqrt(rrMs.reduce((s, v) => s + (v - mean) ** 2, 0) / rrMs.length));
    const diffs = rrMs.slice(1).map((v, i) => (v - rrMs[i]) ** 2);
    const rmssd = Math.round(Math.sqrt(diffs.reduce((a, b) => a + b, 0) / diffs.length));
    // Coefficient of variation (SDNN / mean RR) is a better rhythm regularity marker than
    // raw RMSSD because it scales with heart rate and clearly separates sinus from AFib.
    // CV < 5%: normal sinus  |  5-10%: normal variation  |  10-20%: mildly irregular  |  >20%: irregular
    const cv = mean > 0 ? (sdnn / mean) * 100 : 0;
    let label: string;
    let color: string;
    if      (cv < 5)  { label = 'Regular';          color = 'text-emerald-400'; }
    else if (cv < 10) { label = 'Normal Variation';  color = 'text-emerald-300'; }
    else if (cv < 20) { label = 'Mildly Irregular';  color = 'text-amber-400';   }
    else              { label = 'Irregular';          color = 'text-red-400';     }
    return { label, color, rmssd, sdnn, meanRR: Math.round(mean) };
}

export default function HeartVisualization({ bpm, severity, rPeaks, waveform }: Props) {
    const mountRef   = useRef<HTMLDivElement>(null);
    const ecgRef     = useRef<HTMLCanvasElement>(null);
    const liveRRRef  = useRef<HTMLSpanElement>(null);
    const beatNumRef = useRef<HTMLSpanElement>(null);
    const flashRef   = useRef<HTMLDivElement>(null);

    // Updated every render so both animation loops always see fresh data
    const dataRef   = useRef({ rPeaks, waveform });
    dataRef.current = { rPeaks, waveform };

    // ── Three.js heart + RR-interval beat sequencer ───────────────────────
    useEffect(() => {
        const mount = mountRef.current;
        if (!mount) return;

        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x0a0a0f);

        const camera = new THREE.PerspectiveCamera(45, mount.clientWidth / mount.clientHeight, 0.1, 100);
        camera.position.set(0, 0, 3.5);

        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(mount.clientWidth, mount.clientHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        mount.appendChild(renderer.domElement);

        scene.add(new THREE.AmbientLight(0xffffff, 0.6));
        const dir = new THREE.DirectionalLight(0xffffff, 1.2);
        dir.position.set(3, 5, 3);
        scene.add(dir);

        const accent = new THREE.PointLight(SEV_LIGHT[severity] ?? SEV_LIGHT.normal, 3, 10);
        accent.position.set(-2, 1, 2);
        scene.add(accent);

        const fill = new THREE.PointLight(0xff4444, 0.8, 6);
        fill.position.set(0, -2, 1);
        scene.add(fill);

        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.minDistance   = 1.5;
        controls.maxDistance   = 8;

        let heart: THREE.Object3D | null = null;
        new GLTFLoader().load('/heart.glb', (gltf) => {
            heart = gltf.scene;
            const box = new THREE.Box3().setFromObject(heart);
            heart.position.sub(box.getCenter(new THREE.Vector3()));
            scene.add(heart);
        }, undefined, (e) => console.warn('heart.glb:', e));

        // Beat sequencer — replays the actual recorded RR intervals in a loop.
        // When no ECG data is available it falls back to a fixed BPM.
        let beatIdx    = 0;
        let totalBeats = 0;
        let timeInBeat = 0; // ms elapsed within the current beat
        let last       = performance.now();
        let animId: number;

        const fireBeat = (rrMs: number[], currentRR: number) => {
            totalBeats++;
            // Update live RR display directly — no React re-render cost
            if (liveRRRef.current)  liveRRRef.current.textContent  = `${Math.round(currentRR)} ms`;
            if (beatNumRef.current) beatNumRef.current.textContent = `${totalBeats}`;
            // Flash the ring
            if (flashRef.current) {
                flashRef.current.style.opacity = '1';
                flashRef.current.style.transform = 'scale(1)';
                setTimeout(() => {
                    if (flashRef.current) {
                        flashRef.current.style.opacity = '0';
                        flashRef.current.style.transform = 'scale(1.6)';
                    }
                }, 120);
            }
        };

        const animate = () => {
            const now = performance.now();
            const dt  = now - last;
            last      = now;

            const { rPeaks: rp, waveform: wf } = dataRef.current;

            // Build RR sequence; gate out physiologically impossible values
            const rrMs: number[] = [];
            if (rp && rp.length >= 2 && wf) {
                for (let i = 0; i < rp.length - 1; i++) {
                    const rr = (rp[i + 1] - rp[i]) / wf.sampling_rate * 1000;
                    if (rr > 300 && rr < 2000) rrMs.push(rr);
                }
            }

            const currentRR = rrMs.length > 0
                ? rrMs[beatIdx % rrMs.length]
                : 60_000 / (bpm ?? 72);

            timeInBeat += dt;
            if (timeInBeat >= currentRR) {
                timeInBeat -= currentRR;
                const nextIdx = rrMs.length > 0 ? (beatIdx + 1) % rrMs.length : 0;
                beatIdx = nextIdx;
                fireBeat(rrMs, currentRR);
            }

            // phase 0→1 within the current beat
            const phase = timeInBeat / currentRR;

            if (heart) {
                heart.rotation.y += 0.003;
                const scale = phase < 0.15
                    ? 1 + 0.08 * (phase / 0.15)
                    : phase < 0.35
                        ? 1 + 0.08 * (1 - (phase - 0.15) / 0.20)
                        : 1.0;
                heart.scale.setScalar(scale);
            }

            accent.intensity = phase < 0.15
                ? 3 + 4 * (phase / 0.15)
                : phase < 0.35
                    ? 3 + 4 * (1 - (phase - 0.15) / 0.20)
                    : 3;

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
            if (mount.contains(renderer.domElement)) mount.removeChild(renderer.domElement);
        };
    }, [bpm, severity]);

    // ── ECG waveform overlay ──────────────────────────────────────────────
    useEffect(() => {
        const canvas = ecgRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const syncSize = () => {
            canvas.width  = canvas.offsetWidth;
            canvas.height = canvas.offsetHeight;
        };
        syncSize();
        const ro = new ResizeObserver(syncSize);
        ro.observe(canvas);

        // Start the playback cursor at the first R-peak so the ECG line and
        // the heart contraction fire together from the first beat.
        let sampleF = dataRef.current.rPeaks?.[0] ?? 0;

        // Signal normalization — recomputed lazily when the waveform changes
        let cachedSamples: number[] | null = null;
        let normMid = 0, normScale = 1;

        let last   = performance.now();
        let animId: number;

        const draw = () => {
            const now = performance.now();
            const dt  = (now - last) / 1000; // seconds
            last      = now;

            const { rPeaks: rp, waveform: wf } = dataRef.current;
            const W = canvas.width;
            const H = canvas.height;

            if (!wf || wf.samples.length === 0 || W === 0 || H === 0) {
                animId = requestAnimationFrame(draw);
                return;
            }

            const { samples, sampling_rate: sr } = wf;
            const n = samples.length;

            // Advance at the recording's own sample rate → real-time playback
            sampleF = (sampleF + dt * sr) % n;

            // Recompute normalization only when the waveform array changes
            if (samples !== cachedSamples) {
                cachedSamples = samples;
                const mean = samples.reduce((a, b) => a + b, 0) / n;
                const std  = Math.sqrt(samples.reduce((s, v) => s + (v - mean) ** 2, 0) / n);
                normMid   = mean;
                normScale = std > 1e-6 ? std * 3 : 1; // ±3σ → ±1 normalised
            }

            ctx.clearRect(0, 0, W, H);

            // Background — transparent at top blends into the 3D scene
            const bg = ctx.createLinearGradient(0, 0, 0, H);
            bg.addColorStop(0,   'rgba(10,10,15,0)');
            bg.addColorStop(0.3, 'rgba(10,10,15,0.65)');
            bg.addColorStop(1,   'rgba(10,10,15,0.96)');
            ctx.fillStyle = bg;
            ctx.fillRect(0, 0, W, H);

            // Time-division grid
            ctx.strokeStyle = 'rgba(63,63,70,0.35)';
            ctx.lineWidth   = 0.5;
            for (let x = W % 40; x < W; x += 40) {
                ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
            }

            const baseline  = H * 0.62;
            const amplitude = H * 0.30;
            const cursorX   = W * 0.42; // "now" marker sits at 42% from left

            // Isoelectric baseline
            ctx.beginPath();
            ctx.strokeStyle = 'rgba(63,63,70,0.25)';
            ctx.moveTo(0, baseline); ctx.lineTo(W, baseline);
            ctx.stroke();

            // 3-second view window centred on the cursor
            const winSamples = sr * 3;

            ctx.beginPath();
            ctx.strokeStyle = '#10b981';
            ctx.lineWidth   = 1.6;
            ctx.shadowColor = '#10b981';
            ctx.shadowBlur  = 5;
            ctx.lineJoin    = 'round';

            for (let px = 0; px <= W; px++) {
                const offset = ((px - cursorX) / W) * winSamples;
                const idx    = Math.round(((sampleF + offset) % n + n) % n);
                const norm   = (samples[idx] - normMid) / normScale;
                const y      = baseline - norm * amplitude;
                px === 0 ? ctx.moveTo(px, y) : ctx.lineTo(px, y);
            }
            ctx.stroke();
            ctx.shadowBlur = 0;

            // R-peak markers — red dots that land on each R wave crest
            if (rp && rp.length > 0) {
                ctx.fillStyle = 'rgba(239,68,68,0.9)';
                for (const r of rp) {
                    const offset = r - sampleF;
                    const px     = cursorX + (offset / winSamples) * W;
                    if (px < 0 || px > W) continue;
                    const idx  = Math.round((r % n + n) % n);
                    const norm = (samples[idx] - normMid) / normScale;
                    const y    = baseline - norm * amplitude;
                    ctx.beginPath();
                    ctx.arc(px, y, 3, 0, Math.PI * 2);
                    ctx.fill();
                }
            }

            // Dashed "now" cursor line
            ctx.beginPath();
            ctx.strokeStyle = 'rgba(255,255,255,0.10)';
            ctx.lineWidth   = 1;
            ctx.setLineDash([3, 5]);
            ctx.moveTo(cursorX, 0); ctx.lineTo(cursorX, H);
            ctx.stroke();
            ctx.setLineDash([]);

            // Left + right edge fades
            const leftG = ctx.createLinearGradient(0, 0, W * 0.09, 0);
            leftG.addColorStop(0, 'rgba(10,10,15,1)');
            leftG.addColorStop(1, 'rgba(10,10,15,0)');
            ctx.fillStyle = leftG;
            ctx.fillRect(0, 0, W * 0.09, H);

            const rightG = ctx.createLinearGradient(W * 0.91, 0, W, 0);
            rightG.addColorStop(0, 'rgba(10,10,15,0)');
            rightG.addColorStop(1, 'rgba(10,10,15,1)');
            ctx.fillStyle = rightG;
            ctx.fillRect(W * 0.91, 0, W * 0.09, H);

            animId = requestAnimationFrame(draw);
        };
        animId = requestAnimationFrame(draw);

        return () => { ro.disconnect(); cancelAnimationFrame(animId); };
    }, []);

    // HRV values derived from props for the stats overlay
    const rrMs: number[] = [];
    if (rPeaks && rPeaks.length >= 2 && waveform) {
        for (let i = 0; i < rPeaks.length - 1; i++) {
            const rr = (rPeaks[i + 1] - rPeaks[i]) / waveform.sampling_rate * 1000;
            if (rr > 300 && rr < 2000) rrMs.push(rr);
        }
    }
    const rhythm = computeRhythm(rrMs);

    return (
        <div className="w-full h-full flex flex-col bg-[#0a0a0f]">

            {/* Header */}
            <div className="shrink-0 px-4 pt-3 pb-1.5 flex items-center justify-between border-b border-zinc-800/40">
                <span className="text-[10px] font-semibold tracking-widest uppercase text-zinc-500">
                    Cardiac Digital Twin
                </span>
                <div className="flex items-center gap-4 text-xs">
                    {rhythm && (
                        <span className={`font-medium ${rhythm.color}`}>{rhythm.label}</span>
                    )}
                    {bpm !== null && (
                        <span className="font-mono text-zinc-300">
                            {Math.round(bpm)} <span className="text-zinc-500">bpm</span>
                        </span>
                    )}
                    <span className="font-mono text-zinc-500 text-[10px]">
                        RR&nbsp;<span ref={liveRRRef} className="text-emerald-400">
                            {rhythm ? `${rhythm.meanRR} ms` : '—'}
                        </span>
                    </span>
                </div>
            </div>

            {/* Three.js viewport */}
            <div ref={mountRef} className="flex-1 w-full min-h-0 relative">
                {/* Beat flash ring — fires on each R-peak, animated via direct DOM */}
                <div
                    ref={flashRef}
                    className="pointer-events-none absolute inset-0 flex items-center justify-center"
                    style={{ opacity: 0, transition: 'opacity 120ms ease-out, transform 300ms ease-out', transform: 'scale(1)' }}
                >
                    <div className="w-32 h-32 rounded-full border-2 border-emerald-400/60" />
                </div>
            </div>

            {/* ECG strip */}
            <div className="shrink-0 relative h-28 border-t border-zinc-800/30">
                <canvas ref={ecgRef} className="absolute inset-0 w-full h-full" />

                {/* Lead + sample rate label */}
                {waveform && (
                    <div className="absolute top-1.5 left-3 text-[9px] font-mono text-zinc-600 pointer-events-none z-10">
                        {waveform.lead_name} · {waveform.sampling_rate} Hz
                    </div>
                )}

                {/* HRV stats */}
                {rhythm && (
                    <div className="absolute bottom-2.5 left-3 flex gap-5 pointer-events-none z-10">
                        {[
                            { label: 'RMSSD', value: `${rhythm.rmssd} ms`,       title: 'Root mean square of successive RR differences' },
                            { label: 'SDNN',  value: `${rhythm.sdnn} ms`,        title: 'Standard deviation of RR intervals'            },
                            { label: 'Beats', value: `${rPeaks?.length ?? '—'}`, title: 'Detected R-peaks'                              },
                        ].map(({ label, value, title }) => (
                            <div key={label} className="flex flex-col gap-0.5" title={title}>
                                <span className="text-[8px] uppercase tracking-widest text-zinc-600">{label}</span>
                                <span className="text-[11px] font-mono text-zinc-400">{value}</span>
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
}
