import { useState } from 'react';
import type { AnalysisResponse } from '../../types/analysis';
import MiniWaveform from './MiniWaveform';

export default function AboveFold({ data }: { data: AnalysisResponse }) {
    const { verdict, metrics, ai_analysis, session_id, waveforms, peaks } = data;
    const [selectedLeadIndex, setSelectedLeadIndex] = useState(0);
    const [showMarkers, setShowMarkers] = useState(true);

    return (
        <div className="flex flex-col h-full gap-4">

            {/* 1. Header bar */}
            <header className="flex justify-between items-center bg-zinc-900 rounded-lg p-3 border border-zinc-800 shrink-0">
                <div className="flex gap-4 text-sm text-zinc-400">
                    <span className="font-mono text-zinc-300">ID: {session_id}</span>
                    <span>10s Recording</span>
                    <span>12 Lead</span>
                    <span>WFDB</span>
                </div>
                <div className={`px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider
          ${verdict.severity === 'critical' ? 'bg-red-500/20 text-red-400 border border-red-500/30' :
                        verdict.severity === 'warning' ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30' :
                            'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'}`}>
                    {verdict.display_name}
                </div>
            </header>

            {/* Grid for Verdict and AI Predictions */}
            <div className="grid grid-cols-3 gap-4 shrink-0">

                {/* 2. Verdict card */}
                <div className="col-span-2 bg-zinc-900 border border-zinc-800 rounded-lg p-5 flex flex-col justify-between">
                    <div>
                        <h1 className="text-4xl font-bold tracking-tight text-zinc-100 mb-2">
                            {verdict.display_name}
                        </h1>
                        <p className="text-zinc-400 text-sm">{verdict.definition}</p>
                    </div>

                    <div className="mt-4 flex items-center gap-6">
                        <div className="flex items-baseline gap-1">
                            <span className="text-3xl font-light">{verdict.percentage}%</span>
                            <span className="text-xs text-zinc-500 uppercase">Confidence</span>
                        </div>
                    </div>
                </div>

                {/* 5. Top 3 AI predictions */}
                <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-4 flex flex-col">
                    <h3 className="text-xs font-semibold text-zinc-500 uppercase tracking-wider mb-3">Top AI Findings</h3>
                    <div className="flex flex-col gap-3 flex-1">
                        {ai_analysis.top_predictions.slice(0, 3).map((pred) => (
                            <div key={pred.code} className="flex items-center justify-between">
                                <div className="flex flex-col">
                                    <span className="text-sm font-medium text-zinc-200">{pred.display_name}</span>
                                    <div className="w-32 h-1.5 bg-zinc-800 rounded-full mt-1 overflow-hidden">
                                        <div
                                            className={`h-full ${pred.severity === 'critical' ? 'bg-red-500' : 'bg-zinc-500'}`}
                                            style={{ width: `${pred.percentage}%` }}
                                        />
                                    </div>
                                </div>
                                <span className="text-xs font-mono text-zinc-400">{pred.percentage}%</span>
                            </div>
                        ))}
                    </div>
                    <button 
                        onClick={() => document.getElementById('all-predictions')?.scrollIntoView({ behavior: 'smooth' })}
                        className="text-xs text-blue-400 hover:text-blue-300 self-start mt-2"
                    >
                        See all predictions ↓
                    </button>
                </div>
            </div>

            {/* 3. The four most important interval metrics */}
            <div className="grid grid-cols-4 gap-4 shrink-0">
                {['heart_rate_bpm', 'pr_interval_ms', 'qrs_duration_ms', 'qtc_ms'].map((key) => {
                    const m = metrics[key] || { value: null, unit: '', status: 'unknown' };
                    return (
                        <div key={key} className="bg-zinc-900 border border-zinc-800 rounded-lg p-3 flex justify-between items-end">
                            <div className="flex flex-col">
                                <span className="text-xs text-zinc-500 uppercase tracking-wider">{key.split('_')[0]}</span>
                                <div className="flex items-baseline gap-1 mt-1">
                                    <span className={`text-2xl font-light ${m.status === 'high' ? 'text-amber-400' : 'text-zinc-100'}`}>
                                        {m.value ?? '--'}
                                    </span>
                                    <span className="text-xs text-zinc-600">{m.unit}</span>
                                </div>
                            </div>
                            {/* Mock Range Bar */}
                            <div className="w-1/3 h-1 bg-zinc-800 rounded-full mb-2">
                                <div className="h-full bg-zinc-500 w-1/2 ml-1/4 rounded-full" />
                            </div>
                        </div>
                    );
                })}
            </div>

            {/* 4. Waveform (Constrained 220px) */}
            <div className="bg-zinc-900 border border-zinc-800 rounded-lg flex-1 min-h-[220px] max-h-[260px] relative overflow-hidden flex flex-col">
                <div className="p-2 border-b border-zinc-800 flex justify-between items-center text-xs text-zinc-400">
                    <div className="flex gap-2">
                        <select 
                            className="bg-zinc-950 border border-zinc-800 rounded px-2 py-1 outline-none focus:border-zinc-600 text-zinc-300"
                            value={selectedLeadIndex}
                            onChange={(e) => setSelectedLeadIndex(Number(e.target.value))}
                        >
                            {waveforms.map((wf, idx) => (
                                <option key={idx} value={idx}>{wf.lead_name}</option>
                            ))}
                        </select>
                    </div>
                    <div className="flex gap-3">
                        <label className="flex items-center gap-1 cursor-pointer">
                            <input 
                                type="checkbox" 
                                checked={showMarkers}
                                onChange={(e) => setShowMarkers(e.target.checked)}
                                className="accent-zinc-500" 
                            /> PQRST Markers
                        </label>
                        <label className="flex items-center gap-1 cursor-pointer"><input type="checkbox" defaultChecked className="accent-zinc-500" /> Interval Bands</label>
                    </div>
                </div>
                <div className="flex-1 w-full relative">
                    {waveforms.length > 0 && (
                        <MiniWaveform 
                            waveform={waveforms[selectedLeadIndex]} 
                            peaks={peaks}
                            showMarkers={showMarkers}
                        />
                    )}
                </div>
            </div>

        </div>
    );
}