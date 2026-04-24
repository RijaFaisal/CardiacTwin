import type { AnalysisResponse } from '../../types/analysis';

interface Props {
    data: AnalysisResponse;
    onExport: () => void;
    exporting: boolean;
}

const SEV_BADGE: Record<string, string> = {
    normal:   'bg-emerald-500/15 text-emerald-400 border border-emerald-500/30',
    warning:  'bg-amber-500/15  text-amber-400  border border-amber-500/30',
    critical: 'bg-red-500/15    text-red-400    border border-red-500/30',
};

export default function TopBar({ data, onExport, exporting }: Props) {
    const shortId  = data.session_id.slice(0, 8).toUpperCase();
    const leads    = data.waveforms?.length ?? 0;
    const dur      = data.waveforms?.[0]?.duration_s;
    const duration = dur ? `${Math.round(dur)}s` : null;
    const qual     = data.verdict.quality;

    return (
        <header className="h-14 shrink-0 bg-zinc-900 border-b border-zinc-800 flex items-center px-5 gap-4">

            {/* Logo — aligns with sidebar width */}
            <div className="w-[220px] shrink-0 flex items-center gap-2.5">
                <div className="w-7 h-7 rounded-md bg-indigo-500/20 flex items-center justify-center">
                    <svg className="w-4 h-4 text-indigo-400" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M12 21.593c-5.63-5.539-11-10.297-11-14.402 0-3.791 3.068-5.191 5.281-5.191 1.312 0 4.151.501 5.719 4.457 1.59-3.968 4.464-4.447 5.726-4.447 2.54 0 5.274 1.621 5.274 5.181 0 4.069-5.136 8.625-11 14.402z" />
                    </svg>
                </div>
                <div className="leading-none">
                    <p className="text-sm font-semibold text-zinc-100 tracking-tight">CardiacTwin</p>
                    <p className="text-[9px] text-zinc-600 tracking-wider uppercase mt-0.5">ECG Analysis System</p>
                </div>
            </div>

            <div className="w-px h-6 bg-zinc-800 shrink-0" />

            {/* Session metadata */}
            <div className="flex items-center gap-3 flex-1 text-xs">
                <div className="flex items-center gap-1.5">
                    <span className="text-zinc-600">Session</span>
                    <span className="font-mono text-zinc-300">{shortId}</span>
                </div>
                {leads > 0 && (
                    <span className="bg-zinc-800 text-zinc-400 px-2 py-0.5 rounded text-[10px] font-medium">
                        {leads}-lead
                    </span>
                )}
                {duration && (
                    <span className="bg-zinc-800 text-zinc-400 px-2 py-0.5 rounded text-[10px] font-medium">
                        {duration}
                    </span>
                )}
                <div className="flex items-center gap-1.5 text-[10px] text-zinc-500">
                    <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${qual.status === 'normal' ? 'bg-emerald-400' : 'bg-amber-400'}`} />
                    Signal {qual.score_pct}%
                </div>
            </div>

            {/* Right: verdict badge + export */}
            <div className="flex items-center gap-3 shrink-0">
                <span className={`px-2.5 py-1 rounded-full text-[10px] font-bold uppercase tracking-wider ${SEV_BADGE[data.verdict.severity]}`}>
                    {data.verdict.display_name}
                </span>
                <button
                    onClick={onExport}
                    disabled={exporting}
                    className="flex items-center gap-1.5 bg-zinc-100 text-zinc-900 text-xs font-semibold px-3 py-1.5 rounded-md hover:bg-white transition-colors disabled:opacity-50 disabled:cursor-wait"
                >
                    <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                            d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5M16.5 12L12 16.5m0 0L7.5 12m4.5 4.5V3" />
                    </svg>
                    {exporting ? 'Generating…' : 'Export PDF'}
                </button>
            </div>
        </header>
    );
}
