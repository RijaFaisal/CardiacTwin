import { useState } from 'react';
import type { AnalysisResponse, GradcamResult } from '../../types/analysis';
import type { Tab } from '../../types/ui';
import MiniWaveform from './MiniWaveform';
import { ecgClient } from '../../api/ecgClient';

interface Props {
    data: AnalysisResponse;
    onSwitchTab: (tab: Tab) => void;
}

const METRIC_CONFIG: Record<string, { label: string; low: number; high: number }> = {
    heart_rate_bpm:  { label: 'Heart Rate',   low: 60,  high: 100 },
    pr_interval_ms:  { label: 'PR Interval',  low: 120, high: 200 },
    qrs_duration_ms: { label: 'QRS Duration', low: 70,  high: 110 },
    qtc_ms:          { label: 'QTc',          low: 350, high: 440 },
};

const SEV_CARD: Record<string, string> = {
    normal:   'border-emerald-500/20 bg-emerald-500/5',
    warning:  'border-amber-500/20  bg-amber-500/5',
    critical: 'border-red-500/20    bg-red-500/5',
};
const SEV_TEXT: Record<string, string> = {
    normal:   'text-emerald-400',
    warning:  'text-amber-400',
    critical: 'text-red-400',
};
const STATUS_C: Record<string, { zone: string; dot: string; badge: string; value: string }> = {
    normal:  { zone: 'bg-emerald-500/25', dot: 'bg-emerald-400', badge: 'bg-emerald-500/10 text-emerald-400', value: 'text-zinc-100' },
    high:    { zone: 'bg-amber-500/20',   dot: 'bg-amber-400',   badge: 'bg-amber-500/10  text-amber-400',   value: 'text-amber-300' },
    low:     { zone: 'bg-blue-500/20',    dot: 'bg-blue-400',    badge: 'bg-blue-500/10   text-blue-400',    value: 'text-blue-300'  },
    unknown: { zone: 'bg-zinc-700',       dot: 'bg-zinc-500',    badge: 'bg-zinc-800       text-zinc-500',   value: 'text-zinc-100'  },
    info:    { zone: 'bg-zinc-700',       dot: 'bg-zinc-500',    badge: 'bg-zinc-800       text-zinc-500',   value: 'text-zinc-100'  },
};

function RangeBar({ value, low, high, status }: { value: number | null; low: number; high: number; status: string }) {
    const pad      = (high - low) * 0.5;
    const extLow   = low  - pad;
    const ext      = high + pad - extLow;
    const normLeft  = ((low  - extLow) / ext) * 100;
    const normWidth = ((high - low)    / ext) * 100;
    const markerPct = value !== null
        ? Math.max(1, Math.min(99, ((value - extLow) / ext) * 100))
        : null;
    const c = STATUS_C[status] ?? STATUS_C.unknown;

    return (
        <div className="mt-3">
            <div className="relative h-1 bg-zinc-800 rounded-full">
                <div
                    className={`absolute h-full rounded-full ${c.zone}`}
                    style={{ left: `${normLeft}%`, width: `${normWidth}%` }}
                />
                {markerPct !== null && (
                    <div
                        className={`absolute w-2.5 h-2.5 rounded-full -top-[5px] border-2 border-zinc-950 shadow-sm ${c.dot}`}
                        style={{ left: `calc(${markerPct}% - 5px)` }}
                    />
                )}
            </div>
            <p className="text-[9px] text-zinc-700 mt-1.5">Normal {low}–{high}</p>
        </div>
    );
}

export default function AboveFold({ data, onSwitchTab }: Props) {
    const { verdict, metrics, ai_analysis, session_id, waveforms, peaks } = data;
    const [leadIndex,     setLeadIndex]     = useState(1); // Lead II default
    const [showMarkers,   setShowMarkers]   = useState(true);
    const [showSaliency,  setShowSaliency]  = useState(false);
    const [gradcam,       setGradcam]       = useState<GradcamResult | null>(data.gradcam ?? null);
    const [loadingGradcam, setLoadingGradcam] = useState(false);

    const handleSaliencyToggle = async () => {
        if (!showSaliency && !gradcam) {
            setLoadingGradcam(true);
            try {
                const res = await ecgClient.analyzeSession(session_id, true);
                setGradcam(res.gradcam ?? null);
            } catch (e) {
                console.error('Grad-CAM fetch failed', e);
            } finally {
                setLoadingGradcam(false);
            }
        }
        setShowSaliency(prev => !prev);
    };

    return (
        <div className="space-y-4 max-w-6xl">

            {/* Row 1 — Verdict + Top Findings */}
            <div className="grid grid-cols-3 gap-4">

                <div className={`col-span-2 rounded-xl border p-5 flex flex-col gap-4 ${SEV_CARD[verdict.severity]}`}>
                    <div className="flex items-start justify-between">
                        <span className="text-[9px] font-semibold uppercase tracking-widest text-zinc-500">Primary Finding</span>
                        <span className={`text-[9px] font-bold uppercase tracking-widest px-2 py-0.5 rounded-full border ${SEV_CARD[verdict.severity]} ${SEV_TEXT[verdict.severity]}`}>
                            {verdict.severity}
                        </span>
                    </div>
                    <div>
                        <h1 className={`text-3xl font-bold tracking-tight ${SEV_TEXT[verdict.severity]}`}>
                            {verdict.display_name}
                        </h1>
                        <p className="text-zinc-400 text-sm mt-2 leading-relaxed">{verdict.definition}</p>
                    </div>
                    <div className="flex items-center gap-8 pt-3 border-t border-zinc-800/40">
                        <div>
                            <p className="text-[9px] text-zinc-600 uppercase tracking-wider mb-1">AI Confidence</p>
                            <p className="text-2xl font-light text-zinc-100">
                                {verdict.percentage}<span className="text-sm text-zinc-500 ml-0.5">%</span>
                            </p>
                        </div>
                        <div>
                            <p className="text-[9px] text-zinc-600 uppercase tracking-wider mb-1">Signal Quality</p>
                            <p className="text-2xl font-light text-zinc-100">
                                {verdict.quality.score_pct}<span className="text-sm text-zinc-500 ml-0.5">%</span>
                            </p>
                        </div>
                        <div className="ml-auto text-right">
                            <p className="text-[9px] text-zinc-600 uppercase tracking-wider mb-1">Session</p>
                            <p className="font-mono text-xs text-zinc-500">{session_id.slice(0, 8).toUpperCase()}</p>
                        </div>
                    </div>
                </div>

                <div className="rounded-xl border border-zinc-800 bg-zinc-900 p-4 flex flex-col">
                    <div className="flex items-center justify-between mb-3">
                        <span className="text-[9px] font-semibold uppercase tracking-widest text-zinc-500">Top Findings</span>
                        <span className="text-[9px] text-zinc-700 uppercase tracking-wider">AI · FCN-Wang</span>
                    </div>
                    <div className="flex flex-col gap-3 flex-1">
                        {ai_analysis.top_predictions.slice(0, 4).map(pred => (
                            <div key={pred.code}>
                                <div className="flex items-center justify-between mb-1">
                                    <span className="text-xs text-zinc-300 truncate pr-2 leading-tight">{pred.display_name}</span>
                                    <span className="text-[10px] font-mono text-zinc-500 shrink-0">{pred.percentage}%</span>
                                </div>
                                <div className="h-1 bg-zinc-800 rounded-full overflow-hidden">
                                    <div
                                        className={`h-full rounded-full transition-all ${
                                            pred.severity === 'critical' ? 'bg-red-500' :
                                            pred.severity === 'warning'  ? 'bg-amber-500' :
                                            'bg-emerald-500'
                                        }`}
                                        style={{ width: `${pred.percentage}%` }}
                                    />
                                </div>
                            </div>
                        ))}
                    </div>
                    <button
                        onClick={() => onSwitchTab('ecg')}
                        className="text-[10px] text-indigo-400 hover:text-indigo-300 mt-3 self-start flex items-center gap-1 transition-colors"
                    >
                        View all findings →
                    </button>
                </div>
            </div>

            {/* Row 2 — Metric Cards */}
            <div className="grid grid-cols-4 gap-4">
                {(['heart_rate_bpm', 'pr_interval_ms', 'qrs_duration_ms', 'qtc_ms'] as const).map(key => {
                    const m   = metrics[key];
                    const cfg = METRIC_CONFIG[key];
                    if (!m || !cfg) return null;
                    const c = STATUS_C[m.status] ?? STATUS_C.unknown;
                    return (
                        <div key={key} className="bg-zinc-900 border border-zinc-800 rounded-xl p-4">
                            <div className="flex items-center justify-between mb-2">
                                <span className="text-[9px] font-semibold uppercase tracking-widest text-zinc-500">
                                    {cfg.label}
                                </span>
                                {m.status !== 'unknown' && m.status !== 'info' && (
                                    <span className={`text-[9px] px-1.5 py-0.5 rounded font-bold uppercase tracking-wider ${c.badge}`}>
                                        {m.status}
                                    </span>
                                )}
                            </div>
                            <div className="flex items-baseline gap-1">
                                <span className={`text-3xl font-light ${c.value}`}>{m.value ?? '—'}</span>
                                <span className="text-xs text-zinc-600">{m.unit}</span>
                            </div>
                            <RangeBar value={m.value} low={cfg.low} high={cfg.high} status={m.status} />
                        </div>
                    );
                })}
            </div>

            {/* Row 3 — Waveform */}
            <div className="bg-zinc-900 border border-zinc-800 rounded-xl overflow-hidden">
                <div className="px-4 py-2.5 border-b border-zinc-800 flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        <select
                            value={leadIndex}
                            onChange={e => setLeadIndex(Number(e.target.value))}
                            className="bg-zinc-950 border border-zinc-800 rounded px-2 py-1 text-xs text-zinc-300 outline-none focus:border-zinc-600 cursor-pointer"
                        >
                            {waveforms.map((wf, idx) => (
                                <option key={idx} value={idx}>{wf.lead_name}</option>
                            ))}
                        </select>
                        <span className="text-[10px] text-zinc-600">
                            {waveforms[leadIndex]?.sampling_rate} Hz
                            {waveforms[leadIndex]?.duration_s ? ` · ${waveforms[leadIndex].duration_s.toFixed(1)}s` : ''}
                        </span>
                    </div>
                    <div className="flex items-center gap-3">
                        <label className="flex items-center gap-1.5 text-xs text-zinc-400 hover:text-zinc-200 transition-colors cursor-pointer">
                            <input
                                type="checkbox"
                                checked={showMarkers}
                                onChange={e => setShowMarkers(e.target.checked)}
                                className="accent-indigo-500 w-3 h-3"
                            />
                            PQRST
                        </label>
                        <button
                            onClick={handleSaliencyToggle}
                            disabled={loadingGradcam}
                            className={`flex items-center gap-1.5 px-2.5 py-1 rounded text-xs border transition-colors ${
                                showSaliency
                                    ? 'border-amber-600/50 bg-amber-950/40 text-amber-400'
                                    : 'border-zinc-700 text-zinc-400 hover:text-zinc-200 hover:border-zinc-600'
                            } disabled:opacity-50 disabled:cursor-wait`}
                        >
                            <span className="w-2 h-2 rounded-sm bg-gradient-to-r from-emerald-500 via-amber-400 to-red-500 shrink-0" />
                            {loadingGradcam ? 'Loading…' : 'AI Attention'}
                        </button>
                        {showSaliency && gradcam && (
                            <div className="flex items-center gap-1.5 text-[9px] text-zinc-600">
                                <span className="w-8 h-1 rounded-full bg-gradient-to-r from-emerald-500 via-amber-400 to-red-500" />
                                low → high
                            </div>
                        )}
                    </div>
                </div>
                <div className="h-52">
                    {waveforms.length > 0 && (
                        <MiniWaveform
                            waveform={waveforms[leadIndex]}
                            peaks={peaks}
                            showMarkers={showMarkers}
                            saliency={showSaliency ? gradcam : null}
                        />
                    )}
                </div>
            </div>
        </div>
    );
}
