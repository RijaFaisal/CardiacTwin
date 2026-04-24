// src/components/first-scroll/TwelveLeadGrid.tsx
import type { WaveformData, PeakMarkers } from '../../types/analysis';
import MiniWaveform from '../above-fold/MiniWaveform';

interface TwelveLeadGridProps {
    waveforms: WaveformData[];
    peaks?: PeakMarkers;
}

export default function TwelveLeadGrid({ waveforms, peaks }: TwelveLeadGridProps) {
    if (!waveforms || waveforms.length < 12) return null;

    // Standard 4x3 Clinical Grid Layout
    // Row 1: I, aVR, V1, V4
    // Row 2: II, aVL, V2, V5
    // Row 3: III, aVF, V3, V6
    const gridLayout = [
        [0, 3, 6, 9],
        [1, 4, 7, 10],
        [2, 5, 8, 11]
    ];

    return (
        <div className="flex flex-col gap-4">
            <div className="flex justify-between items-center">
                <h2 className="text-lg font-semibold text-zinc-100">Standard 12-Lead ECG</h2>
                <div className="text-xs text-zinc-500 uppercase tracking-wider">Simultaneous 10s recording</div>
            </div>
            
            <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-4">
                <div className="grid grid-cols-4 grid-rows-3 gap-3 h-[600px]">
                    {gridLayout.map((row, rowIndex) => (
                        row.map((leadIndex, colIndex) => (
                            <div key={`${rowIndex}-${colIndex}`} className="w-full h-full">
                                <MiniWaveform 
                                    waveform={waveforms[leadIndex]} 
                                    minimal={true}
                                />
                            </div>
                        ))
                    ))}
                </div>
            </div>
        </div>
    );
}
