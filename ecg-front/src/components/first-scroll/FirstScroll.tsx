// src/components/first-scroll/FirstScroll.tsx
import type { AnalysisResponse } from '../../types/analysis';
import MiniWaveform from '../above-fold/MiniWaveform';
import TwelveLeadGrid from './TwelveLeadGrid';

export default function FirstScroll({ data }: { data: AnalysisResponse }) {
    const { metrics, ai_analysis } = data;

    const REFERENCE_RANGES: Record<string, string> = {
        "heart_rate_bpm": "60 - 100",
        "pr_interval_ms": "120 - 200",
        "qrs_duration_ms": "70 - 110",
        "qtc_ms": "350 - 440",
    };

    return (
        <div id="all-predictions" className="flex flex-col gap-6 max-w-7xl mx-auto">
            
            {/* 12-Lead Grid */}
            <TwelveLeadGrid waveforms={data.waveforms} peaks={data.peaks} />

            {/* Expanded Rhythm Strip */}
            <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-4">
                <div className="flex justify-between items-center mb-4">
                    <h2 className="text-lg font-semibold text-zinc-100">Full Rhythm Analysis</h2>
                    <div className="flex gap-4 text-sm">
                        <button className="text-zinc-400 hover:text-white transition-colors">Toggle Calipers</button>
                        <div className="flex items-center gap-2 text-emerald-400">
                            <span className="w-2 h-2 rounded-full bg-emerald-400"></span>
                            Signal Quality: {data.verdict.quality.score_pct}%
                        </div>
                    </div>
                </div>
                {/* Temporarily reusing MiniWaveform until full 12-lead component is built */}
                <div className="h-[400px] bg-zinc-950 rounded border border-zinc-800 flex flex-col relative">
                    {data.waveforms?.length > 1 && (
                        <MiniWaveform 
                            waveform={data.waveforms[1]} // Lead II is the clinical standard for rhythm analysis
                            peaks={data.peaks} 
                            showMarkers={true} 
                        />
                    )}
                </div>
            </div>

            <div className="grid grid-cols-2 gap-6">
                {/* Full Interval Metrics */}
                <div className="flex flex-col gap-4">
                    <h2 className="text-lg font-semibold text-zinc-100">Interval Measurements</h2>
                    <div className="grid grid-cols-2 gap-3">
                        {Object.entries(metrics).filter(([k]) => k !== 'qt_interval_ms').map(([key, m]) => (
                            <div key={key} className="bg-zinc-900 border border-zinc-800 rounded p-3 flex flex-col justify-between">
                                <div className="flex justify-between items-start mb-2">
                                    <div className="text-xs text-zinc-500 uppercase tracking-wider">{key.replace(/_/g, ' ')}</div>
                                    {m.status && m.status !== 'info' && (
                                        <span className={`text-[9px] px-1.5 py-0.5 rounded font-semibold uppercase tracking-widest ${
                                            m.status === 'normal' ? 'bg-emerald-500/10 text-emerald-400' :
                                            m.status === 'high' ? 'bg-amber-500/10 text-amber-400' :
                                            'bg-blue-500/10 text-blue-400'
                                        }`}>
                                            {m.status}
                                        </span>
                                    )}
                                </div>
                                <div>
                                    <div className="flex items-baseline gap-1">
                                        <span className={`text-2xl font-semibold ${
                                            m.status === 'high' ? 'text-amber-400' : 
                                            m.status === 'low' ? 'text-blue-400' : 
                                            'text-zinc-100'
                                        }`}>
                                            {m.value ?? '--'}
                                        </span>
                                        <span className="text-xs text-zinc-500">{m.unit}</span>
                                    </div>
                                    {REFERENCE_RANGES[key] && (
                                        <div className="mt-1 text-[10px] text-zinc-600 font-medium">
                                            Normal: {REFERENCE_RANGES[key]} {m.unit}
                                        </div>
                                    )}
                                </div>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Full AI Predictions List (> 10%) */}
                <div className="flex flex-col gap-4">
                    <div className="flex justify-between items-center">
                        <h2 className="text-lg font-semibold text-zinc-100">All Detected Patterns (&gt;10%)</h2>
                    </div>
                    <div className="bg-zinc-900 border border-zinc-800 rounded p-4 flex flex-col gap-3 h-full">
                        {ai_analysis.top_predictions.filter(p => p.percentage > 10).map((pred) => (
                            <div key={pred.code} className="grid grid-cols-12 items-center gap-4 border-b border-zinc-800 pb-2 last:border-0 last:pb-0">
                                <div className="col-span-6 flex flex-col">
                                    <div className="flex items-center gap-2">
                                        <span className="text-sm text-zinc-200">{pred.display_name}</span>
                                        {data.verdict.code === pred.code && (
                                            <span className="text-[10px] text-emerald-400 bg-emerald-400/10 px-1.5 py-0.5 rounded uppercase tracking-wider">
                                                (Corroborated by NK2)
                                            </span>
                                        )}
                                    </div>
                                    <span className="text-xs text-zinc-500 truncate" title={pred.definition}>{pred.definition}</span>
                                </div>
                                <div className="col-span-4 w-full h-2 bg-zinc-950 rounded-full overflow-hidden">
                                    <div
                                        className={`h-full ${pred.severity === 'critical' ? 'bg-red-500' : pred.severity === 'warning' ? 'bg-amber-500' : 'bg-emerald-500'}`}
                                        style={{ width: `${pred.percentage}%` }}
                                    />
                                </div>
                                <div className="col-span-2 text-right text-sm font-mono text-zinc-400">
                                    {pred.percentage}%
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
}