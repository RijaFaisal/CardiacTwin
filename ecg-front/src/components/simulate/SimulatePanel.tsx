import { useState, useEffect } from 'react';
import { ecgClient, type SimulateResult } from '../../api/ecgClient';

const PATHOLOGIES = [
    { value: 'none',        label: 'Normal Sinus Rhythm' },
    { value: 'af',          label: 'Atrial Fibrillation' },
    { value: 'mi',          label: 'Myocardial Infarction' },
    { value: 'vt',          label: 'Ventricular Tachycardia' },
    { value: 'hypertrophy', label: 'Ventricular Hypertrophy' },
];

const TREATMENTS = [
    { value: '',           label: '— No treatment —' },
    { value: 'pacemaker',  label: 'Pacemaker Implantation' },
    { value: 'medication', label: 'Antiarrhythmic Medication' },
    { value: 'ablation',   label: 'Catheter Ablation' },
];

const CLASS_COLORS: Record<string, string> = {
    Normal:               'bg-emerald-500',
    'Bundle Branch Block':'bg-blue-500',
    Ventricular:          'bg-red-500',
    Atrial:               'bg-amber-500',
    Other:                'bg-zinc-500',
};

const SEVERITY: Record<string, string> = {
    Normal:               'text-emerald-400',
    'Bundle Branch Block':'text-blue-400',
    Ventricular:          'text-red-400',
    Atrial:               'text-amber-400',
    Other:                'text-zinc-400',
};

function BeatDistribution({ dist }: { dist: Record<string, number> }) {
    return (
        <div className="flex flex-col gap-1.5 mt-3">
            {Object.entries(dist).map(([cls, pct]) => (
                <div key={cls} className="flex items-center gap-2">
                    <span className="text-[10px] text-zinc-500 w-24 shrink-0">{cls}</span>
                    <div className="flex-1 h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                        <div
                            className={`h-full rounded-full ${CLASS_COLORS[cls] ?? 'bg-zinc-500'}`}
                            style={{ width: `${pct}%` }}
                        />
                    </div>
                    <span className="text-[10px] font-mono text-zinc-400 w-8 text-right">{pct}%</span>
                </div>
            ))}
        </div>
    );
}

function MetricPair({ label, value, unit, highlight }: {
    label: string; value: string | number; unit?: string; highlight?: boolean;
}) {
    return (
        <div className="flex flex-col">
            <span className="text-[10px] text-zinc-500 uppercase tracking-wider">{label}</span>
            <div className="flex items-baseline gap-1 mt-0.5">
                <span className={`text-lg font-light ${highlight ? 'text-amber-400' : 'text-zinc-100'}`}>
                    {value}
                </span>
                {unit && <span className="text-[10px] text-zinc-600">{unit}</span>}
            </div>
        </div>
    );
}

function ResultCard({ result, title }: { result: SimulateResult; title: string }) {
    const isIrregular = result.rr_irregular;
    return (
        <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-4 flex flex-col gap-3">
            <div className="flex items-center justify-between">
                <span className="text-xs text-zinc-500 uppercase tracking-wider">{title}</span>
                {isIrregular && (
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-amber-900/40 text-amber-400">
                        irregular rhythm
                    </span>
                )}
            </div>

            <div>
                <h3 className="text-base font-semibold text-zinc-100">{result.display_name}</h3>
                {result.pathology_display && (
                    <p className="text-xs text-zinc-500 mt-0.5">Treating: {result.pathology_display}</p>
                )}
            </div>

            <div className="grid grid-cols-3 gap-3">
                <MetricPair label="Heart Rate" value={Math.round(result.heart_rate)} unit="bpm"
                    highlight={result.heart_rate > 100 || result.heart_rate < 50} />
                <MetricPair label="HRV SDNN"  value={(result.hrv_sdnn  * 1000).toFixed(0)} unit="ms" />
                <MetricPair label="HRV RMSSD" value={(result.hrv_rmssd * 1000).toFixed(0)} unit="ms" />
            </div>

            <div>
                <div className="flex items-center gap-2">
                    <span className="text-[10px] text-zinc-500 uppercase tracking-wider">Dominant</span>
                    <span className={`text-xs font-medium ${SEVERITY[result.dominant_class] ?? 'text-zinc-300'}`}>
                        {result.dominant_class}
                    </span>
                </div>
                <BeatDistribution dist={result.beat_distribution} />
            </div>

            <p className="text-xs text-zinc-400 leading-relaxed border-t border-zinc-800 pt-3">
                {result.description}
            </p>

            {result.efficacy && (
                <div className="bg-emerald-950/40 border border-emerald-800/30 rounded p-2">
                    <p className="text-xs text-emerald-400">{result.efficacy}</p>
                </div>
            )}

            <span className="text-[10px] text-zinc-600">{result.demographic_note}</span>
        </div>
    );
}

export default function SimulatePanel() {
    const [pathology,  setPathology]  = useState('none');
    const [treatment,  setTreatment]  = useState('');
    const [age,        setAge]        = useState(45);
    const [gender,     setGender]     = useState<'M' | 'F'>('M');

    const [beforeResult, setBeforeResult] = useState<SimulateResult | null>(null);
    const [afterResult,  setAfterResult]  = useState<SimulateResult | null>(null);
    const [loading,      setLoading]      = useState(false);
    const [error,        setError]        = useState<string | null>(null);

    // Fetch pathology result whenever inputs change
    useEffect(() => {
        let cancelled = false;
        setLoading(true);
        setError(null);

        ecgClient.simulatePathology(pathology, age, gender)
            .then(res => { if (!cancelled) setBeforeResult(res); })
            .catch(err => { if (!cancelled) setError(err.message); })
            .finally(() => { if (!cancelled) setLoading(false); });

        return () => { cancelled = true; };
    }, [pathology, age, gender]);

    // Fetch treatment result when treatment selected
    useEffect(() => {
        if (!treatment) { setAfterResult(null); return; }
        let cancelled = false;

        ecgClient.simulateTreatment(treatment, pathology, age, gender)
            .then(res => { if (!cancelled) setAfterResult(res); })
            .catch(err => { if (!cancelled) setError(err.message); });

        return () => { cancelled = true; };
    }, [treatment, pathology, age, gender]);

    // HR change arrow for the after card header
    const hrDelta = beforeResult && afterResult
        ? afterResult.heart_rate - beforeResult.heart_rate
        : null;

    return (
        <div className="flex flex-col gap-4">

            {/* Section header */}
            <div className="flex items-center justify-between">
                <div>
                    <h2 className="text-lg font-semibold text-zinc-100">Digital Twin Simulator</h2>
                    <p className="text-xs text-zinc-500 mt-0.5">
                        Simulate cardiac parameters for different pathologies and treatments
                    </p>
                </div>
                <span className="text-xs text-zinc-600 uppercase tracking-widest">What-if Analysis</span>
            </div>

            {/* Controls bar */}
            <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-3 flex flex-wrap items-center gap-4">

                <div className="flex flex-col gap-1">
                    <label className="text-[10px] text-zinc-500 uppercase tracking-wider">Pathology</label>
                    <select
                        value={pathology}
                        onChange={e => { setPathology(e.target.value); setTreatment(''); }}
                        className="bg-zinc-950 border border-zinc-700 rounded px-2 py-1.5 text-xs text-zinc-200 outline-none focus:border-zinc-500"
                    >
                        {PATHOLOGIES.map(p => (
                            <option key={p.value} value={p.value}>{p.label}</option>
                        ))}
                    </select>
                </div>

                <div className="flex flex-col gap-1">
                    <label className="text-[10px] text-zinc-500 uppercase tracking-wider">Treatment</label>
                    <select
                        value={treatment}
                        onChange={e => setTreatment(e.target.value)}
                        className="bg-zinc-950 border border-zinc-700 rounded px-2 py-1.5 text-xs text-zinc-200 outline-none focus:border-zinc-500"
                    >
                        {TREATMENTS.map(t => (
                            <option key={t.value} value={t.value}>{t.label}</option>
                        ))}
                    </select>
                </div>

                <div className="flex flex-col gap-1">
                    <label className="text-[10px] text-zinc-500 uppercase tracking-wider">Age</label>
                    <div className="flex items-center gap-2">
                        <input
                            type="range" min={18} max={90} value={age}
                            onChange={e => setAge(Number(e.target.value))}
                            className="w-24 accent-zinc-500"
                        />
                        <span className="text-xs font-mono text-zinc-300 w-6">{age}</span>
                    </div>
                </div>

                <div className="flex flex-col gap-1">
                    <label className="text-[10px] text-zinc-500 uppercase tracking-wider">Gender</label>
                    <div className="flex rounded overflow-hidden border border-zinc-700">
                        {(['M', 'F'] as const).map(g => (
                            <button
                                key={g}
                                onClick={() => setGender(g)}
                                className={`px-3 py-1.5 text-xs transition-colors ${
                                    gender === g
                                        ? 'bg-zinc-700 text-zinc-100'
                                        : 'bg-zinc-950 text-zinc-500 hover:text-zinc-300'
                                }`}
                            >
                                {g === 'M' ? 'Male' : 'Female'}
                            </button>
                        ))}
                    </div>
                </div>

                {loading && (
                    <span className="text-xs text-zinc-600 animate-pulse ml-auto">Simulating...</span>
                )}
                {error && (
                    <span className="text-xs text-red-400 ml-auto">{error}</span>
                )}
            </div>

            {/* Before / After cards */}
            <div className="grid grid-cols-2 gap-4">

                {beforeResult ? (
                    <ResultCard result={beforeResult} title="Before — Pathology State" />
                ) : (
                    <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-4 flex items-center justify-center h-48">
                        <span className="text-xs text-zinc-600">Loading...</span>
                    </div>
                )}

                {afterResult ? (
                    <div className="relative">
                        {hrDelta !== null && (
                            <div className={`absolute -top-3 left-1/2 -translate-x-1/2 z-10 px-2 py-0.5 rounded-full text-[10px] font-mono border ${
                                hrDelta < 0
                                    ? 'bg-emerald-950 border-emerald-800 text-emerald-400'
                                    : 'bg-red-950 border-red-800 text-red-400'
                            }`}>
                                HR {hrDelta > 0 ? '+' : ''}{hrDelta.toFixed(1)} bpm
                            </div>
                        )}
                        <ResultCard result={afterResult} title="After — Post-Treatment State" />
                    </div>
                ) : (
                    <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-4 flex flex-col items-center justify-center h-48 gap-2">
                        <span className="text-xs text-zinc-500">Select a treatment to see post-intervention state</span>
                        <span className="text-[10px] text-zinc-700">Pacemaker · Medication · Ablation</span>
                    </div>
                )}

            </div>
        </div>
    );
}
