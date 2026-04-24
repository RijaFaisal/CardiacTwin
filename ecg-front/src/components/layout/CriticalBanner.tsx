// The Golden Rule pinned banner
// src/components/layout/CriticalBanner.tsx
import type { PredictionEntry } from '../../types/analysis';

interface CriticalBannerProps {
    findings: PredictionEntry[];
}

export default function CriticalBanner({ findings }: CriticalBannerProps) {
    if (!findings || findings.length === 0) return null;

    return (
        <div className="fixed top-0 left-0 right-0 z-50 w-full bg-red-950 border-b border-red-900 shadow-lg shadow-red-900/20">
            <div className="flex items-center justify-center gap-4 py-2 px-4">
                <span className="flex h-3 w-3 relative">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
                    <span className="relative inline-flex rounded-full h-3 w-3 bg-red-500"></span>
                </span>
                <div className="flex gap-6 text-sm font-medium text-red-200">
                    {findings.map((finding) => (
                        <span key={finding.code} className="flex items-center gap-2">
                            <strong className="text-white">{finding.display_name}</strong>
                            <span className="bg-red-900 px-2 py-0.5 rounded text-xs font-mono">
                                {finding.percentage}%
                            </span>
                        </span>
                    ))}
                </div>
            </div>
        </div>
    );
}