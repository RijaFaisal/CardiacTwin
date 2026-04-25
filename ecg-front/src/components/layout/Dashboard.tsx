import { useState, useEffect } from 'react';
import type { AnalysisResponse } from '../../types/analysis';
import type { Tab } from '../../types/ui';
import { ecgClient } from '../../api/ecgClient';
import Sidebar from './Sidebar';
import TopBar from './TopBar';
import AboveFold from '../above-fold/AboveFold';
import FirstScroll from '../first-scroll/FirstScroll';
import HeartVisualization from '../heart/HeartVisualization';
import SimulatePanel from '../simulate/SimulatePanel';
import ThirdScroll from '../third-scroll/ThirdScroll';

interface Props {
    sessionId: string;
    onReset: () => void;
}

function Skeleton() {
    return (
        <div className="p-6 space-y-4 max-w-6xl animate-pulse">
            <div className="grid grid-cols-3 gap-4">
                <div className="col-span-2 h-40 bg-zinc-800/50 rounded-xl" />
                <div className="h-40 bg-zinc-800/50 rounded-xl" />
            </div>
            <div className="grid grid-cols-4 gap-4">
                {[0, 1, 2, 3].map(i => <div key={i} className="h-28 bg-zinc-800/50 rounded-xl" />)}
            </div>
            <div className="h-56 bg-zinc-800/50 rounded-xl" />
        </div>
    );
}

function ErrorState({ message, onReset }: { message: string; onReset: () => void }) {
    return (
        <div className="flex flex-col items-center justify-center h-full gap-5 text-center p-8">
            <div className="w-14 h-14 rounded-full bg-red-500/10 flex items-center justify-center">
                <svg className="w-7 h-7 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                        d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z" />
                </svg>
            </div>
            <div>
                <p className="text-zinc-200 font-medium mb-1">Analysis failed</p>
                <p className="text-zinc-500 text-sm max-w-sm">{message}</p>
            </div>
            <button
                onClick={onReset}
                className="bg-zinc-100 text-zinc-900 text-sm font-semibold px-5 py-2 rounded-lg hover:bg-white transition-colors"
            >
                Try another file
            </button>
        </div>
    );
}

export default function Dashboard({ sessionId, onReset }: Props) {
    const [data, setData] = useState<AnalysisResponse | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [tab, setTab] = useState<Tab>('overview');
    const [exporting, setExporting] = useState(false);
    const [exportErr, setExportErr] = useState<string | null>(null);

    useEffect(() => {
        setData(null);
        setError(null);
        ecgClient.analyzeSession(sessionId)
            .then(setData)
            .catch(err => setError(err.message ?? 'Analysis failed. Please try again.'));
    }, [sessionId]);

    const handleExport = async () => {
        if (!data) return;
        setExporting(true);
        setExportErr(null);
        try {
            const blob = await ecgClient.exportReport(sessionId);
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `ecg_report_${sessionId.slice(0, 8)}.pdf`;
            a.click();
            URL.revokeObjectURL(url);
        } catch (err: any) {
            setExportErr(err.message ?? 'Export failed');
        } finally {
            setExporting(false);
        }
    };

    const criticalFindings = data?.ai_analysis.top_predictions.filter(
        p => p.severity === 'critical' && p.percentage > 20
    ) ?? [];

    return (
        <div className="h-screen bg-zinc-950 flex flex-col overflow-hidden">

            {data && <TopBar data={data} onExport={handleExport} exporting={exporting} />}

            <div className="flex flex-1 overflow-hidden">
                <Sidebar tab={tab} onTabChange={setTab} onReset={onReset} />

                <main className="flex-1 flex flex-col overflow-hidden bg-zinc-950">

                    {/* Inline alerts */}
                    {criticalFindings.length > 0 && (
                        <div className="shrink-0 bg-red-950/80 border-b border-red-900/60 px-6 py-2 flex items-center gap-3">
                            <span className="relative flex h-2.5 w-2.5 shrink-0">
                                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75" />
                                <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-red-500" />
                            </span>
                            <span className="text-xs font-medium text-red-200">
                                Critical: {criticalFindings.map(f => `${f.display_name} (${f.percentage}%)`).join(' · ')}
                            </span>
                        </div>
                    )}
                    {exportErr && (
                        <div className="shrink-0 bg-red-500/10 border-b border-red-500/20 px-6 py-2 text-xs text-red-400">
                            {exportErr}
                        </div>
                    )}

                    {/* Content area */}
                    {error ? (
                        <ErrorState message={error} onReset={onReset} />
                    ) : !data ? (
                        <div className="flex-1 overflow-y-auto"><Skeleton /></div>
                    ) : tab === 'twin' ? (
                        <div className="flex-1 overflow-hidden">
                            <HeartVisualization
                                bpm={data.metrics?.heart_rate_bpm?.value ?? null}
                                severity={data.verdict.severity}
                            />
                        </div>
                    ) : (
                        <div className="flex-1 overflow-y-auto">
                            <div className="p-6">
                                {tab === 'overview' && <AboveFold data={data} onSwitchTab={setTab} />}
                                {tab === 'ecg' && <FirstScroll data={data} />}
                                {tab === 'simulator' && <SimulatePanel />}
                                {tab === 'metrics' && <ThirdScroll />}
                            </div>
                        </div>
                    )}
                </main>
            </div>
        </div>
    );
}
