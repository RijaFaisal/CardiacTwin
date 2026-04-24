// Fixed 220px, Uplot wrapper
// src/components/above-fold/MiniWaveform.tsx
import { useEffect, useRef } from 'react';
import uPlot from 'uplot';
import 'uplot/dist/uPlot.min.css';
import type { WaveformData, PeakMarkers } from '../../types/analysis';

interface MiniWaveformProps {
    waveform: WaveformData;
    peaks?: PeakMarkers;
    showMarkers?: boolean;
    minimal?: boolean;
}

export default function MiniWaveform({ waveform, peaks, showMarkers = false, minimal = false }: MiniWaveformProps) {
    const containerRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (!containerRef.current || !waveform.samples.length) return;

        // 1. Generate X-axis (time in seconds) based on the sampling rate
        const xData = waveform.samples.map((_: number, i: number) => i / waveform.sampling_rate);
        const yData = waveform.samples;

        const data: uPlot.AlignedData = [xData, yData];
        const series: uPlot.Series[] = [
            {}, // X-axis series (Time)
            {
                stroke: '#10b981', // Emerald-400 for that medical monitor vibe
                width: 2,
                points: { show: false }, // Hide dots, just show the line
            },
        ];

        if (showMarkers && peaks) {
            const createPeakSeries = (indices: number[], color: string, label: string) => {
                if (!indices) return;
                const peakData = new Array(waveform.samples.length).fill(null);

                indices.forEach(idx => {
                    const validIdx = Math.round(idx);
                    if (validIdx >= 0 && validIdx < waveform.samples.length && !isNaN(validIdx)) {
                        peakData[validIdx] = waveform.samples[validIdx];
                    }
                });
                data.push(peakData);
                series.push({
                    label,
                    stroke: color,
                    width: 0,
                    points: { show: true, size: 8, fill: color, stroke: color },
                });
            };

            createPeakSeries(peaks.p_peaks, '#3b82f6', 'P'); // blue
            createPeakSeries(peaks.q_peaks, '#8b5cf6', 'Q'); // purple
            createPeakSeries(peaks.r_peaks, '#ef4444', 'R'); // red
            createPeakSeries(peaks.s_peaks, '#f59e0b', 'S'); // amber
            createPeakSeries(peaks.t_peaks, '#ec4899', 'T'); // pink
        }

        // 2. Configure a sleek, dark-themed ECG plot
        const opts: uPlot.Options = {
            width: containerRef.current.clientWidth,
            height: containerRef.current.clientHeight,
            cursor: minimal ? { show: false } : { drag: { x: true, y: false } },
            scales: {
                x: { time: false },
                y: { auto: true }, // Or hardcode [-2, 2] for strict mV scaling
            },
            series,
            axes: minimal ? [
                { show: false },
                { show: false }
            ] : [
                {
                    stroke: '#71717a', // zinc-500 text
                    grid: { stroke: '#27272a', width: 1 }, // zinc-800 grid lines
                    space: 40,
                },
                {
                    stroke: '#71717a',
                    grid: { stroke: '#27272a', width: 1 },
                },
            ],
        };

        // 3. Initialize uPlot
        const u = new uPlot(opts, data, containerRef.current);

        // 4. Handle resize fluidly without complex hooks
        const resizeObserver = new ResizeObserver((entries) => {
            for (let entry of entries) {
                const { width, height } = entry.contentRect;
                u.setSize({ width, height });
            }
        });
        resizeObserver.observe(containerRef.current);

        // 5. Cleanup: Destroy instance and observer on unmount or data change
        return () => {
            resizeObserver.disconnect();
            u.destroy();
        };
    }, [waveform, peaks, showMarkers]);

    return (
        <div className={`w-full h-full ${minimal ? 'min-h-[120px]' : 'min-h-[220px]'} bg-zinc-950 rounded relative`}>
            {/* Lead Name Badge */}
            <div className={`absolute top-2 left-2 z-10 font-bold text-zinc-300 bg-zinc-900/80 px-2 py-0.5 rounded border border-zinc-700/50 backdrop-blur-sm pointer-events-none ${minimal ? 'text-xs' : 'text-sm text-emerald-400 border-emerald-500/30'}`}>
                {waveform.lead_name}
            </div>

            {/* Explicit HTML Axis Labels for guaranteed visibility */}
            {!minimal && (
                <>
                    <div className="absolute bottom-1 left-1/2 -translate-x-1/2 text-[10px] font-semibold text-zinc-500 tracking-wider z-10 pointer-events-none bg-zinc-950/80 px-2 rounded">
                        TIME (s)
                    </div>
                    <div className="absolute top-1/2 left-1 -translate-y-1/2 -rotate-90 origin-left text-[10px] font-semibold text-zinc-500 tracking-wider z-10 pointer-events-none bg-zinc-950/80 px-2 rounded">
                        VOLTAGE (mV)
                    </div>
                </>
            )}

            <div
                ref={containerRef}
                className="absolute inset-0"
            />
            
            {showMarkers && peaks && !minimal && (
                <div className="absolute top-2 right-2 flex gap-2 bg-zinc-900/80 px-2 py-1 rounded border border-zinc-800 backdrop-blur-sm pointer-events-none z-10 shadow-sm">
                    <div className="flex items-center gap-1 text-[9px] font-medium text-zinc-400 uppercase tracking-widest"><span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: '#3b82f6' }}></span>P</div>
                    <div className="flex items-center gap-1 text-[9px] font-medium text-zinc-400 uppercase tracking-widest"><span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: '#8b5cf6' }}></span>Q</div>
                    <div className="flex items-center gap-1 text-[9px] font-medium text-zinc-400 uppercase tracking-widest"><span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: '#ef4444' }}></span>R</div>
                    <div className="flex items-center gap-1 text-[9px] font-medium text-zinc-400 uppercase tracking-widest"><span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: '#f59e0b' }}></span>S</div>
                    <div className="flex items-center gap-1 text-[9px] font-medium text-zinc-400 uppercase tracking-widest"><span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: '#ec4899' }}></span>T</div>
                </div>
            )}
        </div>
    );
}