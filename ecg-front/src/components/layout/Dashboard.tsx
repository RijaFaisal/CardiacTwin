// Main orchestrator
// src/components/layout/Dashboard.tsx
import { useState, useEffect } from 'react';
import CriticalBanner from './CriticalBanner';
import AboveFold from '../above-fold/AboveFold';
import FirstScroll from '../first-scroll/FirstScroll';
// import SecondScroll from '../second-scroll/SecondScroll';
// import ThirdScroll from '../third-scroll/ThirdScroll';
import type { AnalysisResponse } from '../../types/analysis';
import { ecgClient } from '../../api/ecgClient';

export default function Dashboard({ sessionId }: { sessionId: string }) {
    const [data, setData] = useState<AnalysisResponse | null>(null);

    useEffect(() => {
        ecgClient.analyzeSession(sessionId)
            .then(setData)
            .catch((err) => {
                console.error("Clinical Analysis Error:", err);
                // Handle error state here if needed
            });
    }, [sessionId]);

    if (!data) return <div className="min-h-screen bg-zinc-950 text-zinc-400 flex items-center justify-center">Loading Clinical Data...</div>;

    // The Golden Rule: Check for any critical finding > 20%
    const criticalFindings = data.ai_analysis.top_predictions.filter(
        (p) => p.severity === 'critical' && p.percentage > 20
    );

    return (
        <div className="min-h-screen bg-zinc-950 text-zinc-100 font-sans relative pb-20">

            {/* The Golden Rule: Pinned Banner */}
            {criticalFindings.length > 0 && (
                <CriticalBanner findings={criticalFindings} />
            )}

            {/* Above the Fold - Forced to 100vh minus banner */}
            <div className="h-screen flex flex-col p-4 space-y-4 overflow-hidden pt-12">
                <AboveFold data={data} />
            </div>

            {/* First Scroll: Clinically useful but not urgent */}
            <div className="min-h-screen p-4 border-t border-zinc-800 bg-zinc-950/50">
                <FirstScroll data={data} />
            </div>

            {/* Second Scroll: Supporting detail */}
            <div className="min-h-screen p-4 border-t border-zinc-800">
                {/* <SecondScroll data={data} /> */}
            </div>

            {/* Third Scroll: Technical */}
            <div className="p-4 border-t border-zinc-800">
                {/* <ThirdScroll data={data} /> */}
            </div>

            {/* Sticky Footer for Full Report */}
            <div className="fixed bottom-0 left-0 right-0 bg-zinc-900 border-t border-zinc-800 p-3 flex justify-end z-40">
                <button className="bg-zinc-100 text-zinc-900 px-6 py-2 rounded font-medium hover:bg-white transition-colors">
                    Generate Full Clinical Report
                </button>
            </div>
        </div>
    );
}