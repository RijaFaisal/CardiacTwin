import { useState, useRef } from 'react';
import { ecgClient } from '../../api/ecgClient';

export default function UploadScreen({ onUploadSuccess }: { onUploadSuccess: (id: string) => void }) {
    const [isDragging, setIsDragging] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [uploading, setUploading] = useState(false);
    const fileInputRef = useRef<HTMLInputElement>(null);

    const handleFiles = async (files: FileList | null) => {
        if (!files || files.length === 0) return;
        setError(null);
        setUploading(true);

        try {
            const fileArray = Array.from(files);
            
            // Check if it's a CSV
            const csvFile = fileArray.find(f => f.name.toLowerCase().endsWith('.csv'));
            if (csvFile) {
                const res = await ecgClient.uploadCsv(csvFile);
                onUploadSuccess(res.session_id);
                return;
            }

            // Check if it's a WFDB pair
            const heaFile = fileArray.find(f => f.name.toLowerCase().endsWith('.hea'));
            const datFile = fileArray.find(f => f.name.toLowerCase().endsWith('.dat'));

            if (heaFile && datFile) {
                const res = await ecgClient.uploadWfdb(heaFile, datFile);
                onUploadSuccess(res.session_id);
                return;
            }

            if (fileArray.length === 1 && (heaFile || datFile)) {
                setError("WFDB requires both the .hea AND .dat files together. Please select both files.");
                return;
            }

            setError("Unsupported file format. Please upload a .csv file or both .hea & .dat files.");
        } catch (err: any) {
            setError(err.message || "An error occurred during upload.");
        } finally {
            setUploading(false);
        }
    };

    const onDrop = (e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(false);
        handleFiles(e.dataTransfer.files);
    };

    return (
        <div className="flex flex-col items-center justify-center min-h-screen p-4">
            <div className="max-w-xl w-full">
                <div className="text-center mb-10">
                    <h1 className="text-4xl font-bold text-zinc-100 mb-4 tracking-tight">Clinical ECG Analysis</h1>
                    <p className="text-zinc-400">Upload a 12-lead ECG recording to automatically generate a full clinical AI report.</p>
                </div>

                <div 
                    onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
                    onDragLeave={(e) => { e.preventDefault(); setIsDragging(false); }}
                    onDrop={onDrop}
                    onClick={() => fileInputRef.current?.click()}
                    className={`border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition-colors duration-200
                        ${isDragging ? 'border-emerald-500 bg-emerald-500/5' : 'border-zinc-700 hover:border-zinc-500 bg-zinc-900'}
                        ${uploading ? 'opacity-50 pointer-events-none' : ''}`}
                >
                    <input 
                        type="file" 
                        multiple 
                        className="hidden" 
                        ref={fileInputRef}
                        onChange={(e) => handleFiles(e.target.files)}
                        accept=".csv,.hea,.dat"
                    />

                    {uploading ? (
                        <div className="flex flex-col items-center">
                            <div className="w-12 h-12 border-4 border-zinc-700 border-t-emerald-500 rounded-full animate-spin mb-4"></div>
                            <p className="text-zinc-300 font-medium">Uploading and Processing...</p>
                        </div>
                    ) : (
                        <div className="flex flex-col items-center">
                            <svg className="w-16 h-16 text-zinc-600 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                            </svg>
                            <h3 className="text-lg font-medium text-zinc-200 mb-1">Click to browse or drag and drop</h3>
                            <p className="text-sm text-zinc-500 mb-6">Supports .CSV files or WFDB pairs (.hea + .dat)</p>
                            
                            <div className="flex gap-4">
                                <span className="bg-zinc-800 text-zinc-400 text-xs px-3 py-1 rounded">CSV</span>
                                <span className="bg-zinc-800 text-zinc-400 text-xs px-3 py-1 rounded">WFDB (.dat + .hea)</span>
                            </div>
                        </div>
                    )}
                </div>

                {error && (
                    <div className="mt-6 p-4 bg-red-500/10 border border-red-500/20 rounded-lg text-red-400 text-sm text-center">
                        {error}
                    </div>
                )}
            </div>
        </div>
    );
}
