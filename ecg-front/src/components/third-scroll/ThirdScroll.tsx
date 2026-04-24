// src/components/third-scroll/ThirdScroll.tsx
import { useState, useEffect } from 'react';
import { ecgClient } from '../../api/ecgClient';

interface PerClass {
    precision: number;
    recall:    number;
    f1:        number;
    support:   number;
}

interface ModelMetrics {
    accuracy:  number;
    macro_f1:  number;
    macro_auc?: number;
    macro_f1_ci?: { low: number; high: number };
    per_class: Record<string, PerClass>;
}

interface MetricsResponse {
    cnn:       ModelMetrics | null;
    baselines: { logistic_regression: ModelMetrics; svm: ModelMetrics; transformer?: ModelMetrics } | null;
}

const MODELS = [
    { key: 'cnn',         label: 'CNN (MIT-BIH)',   color: '#6366f1' },
    { key: 'transformer', label: 'Transformer',     color: '#ec4899' },
    { key: 'svm',         label: 'SVM',             color: '#10b981' },
    { key: 'lr',          label: 'Logistic Reg.',   color: '#f59e0b' },
];

const CLASS_COLORS: Record<string, string> = {
    Normal:               '#10b981',
    'Bundle Branch Block':'#6366f1',
    Ventricular:          '#ef4444',
    Atrial:               '#f59e0b',
    Other:                '#8b5cf6',
};

export default function ThirdScroll() {
    const [metrics, setMetrics]   = useState<MetricsResponse | null>(null);
    const [loading, setLoading]   = useState(true);
    const [error,   setError]     = useState<string | null>(null);
    const [tab,     setTab]       = useState<'overview' | 'perclass'>('overview');

    useEffect(() => {
        ecgClient.getModelMetrics()
            .then(setMetrics)
            .catch((e) => setError(e.message))
            .finally(() => setLoading(false));
    }, []);

    if (loading) return (
        <div className="text-zinc-500 text-sm p-4">Loading model metrics…</div>
    );
    if (error) return (
        <div className="text-red-400 text-sm p-4">Metrics unavailable: {error}</div>
    );
    if (!metrics) return null;

    const cnn  = metrics.cnn;
    const svm  = metrics.baselines?.svm;
    const lr   = metrics.baselines?.logistic_regression;
    const trf  = metrics.baselines?.transformer;

    const overviewRows = [
        { ...MODELS[0], accuracy: cnn?.accuracy,  macro_f1: cnn?.macro_f1,  ci: cnn?.macro_f1_ci },
        ...(trf ? [{ ...MODELS[1], accuracy: trf.accuracy, macro_f1: trf.macro_f1, ci: undefined as undefined }] : []),
        { ...MODELS[2], accuracy: svm?.accuracy,  macro_f1: svm?.macro_f1,  ci: undefined as undefined },
        { ...MODELS[3], accuracy: lr?.accuracy,   macro_f1: lr?.macro_f1,   ci: undefined as undefined },
    ];

    const maxF1 = Math.max(...overviewRows.map(r => r.macro_f1 ?? 0));

    const classes = cnn ? Object.keys(cnn.per_class) : [];

    return (
        <div className="max-w-4xl mx-auto py-8 px-4 space-y-6">

            {/* Title */}
            <div>
                <h2 className="text-lg font-semibold text-zinc-100 tracking-tight">
                    Model Performance Comparison
                </h2>
                <p className="text-[10px] text-zinc-600 mt-0.5">
                    Custom models trained from scratch on MIT-BIH (5-class, single-lead).
                    Production analysis uses a separate pre-trained FCN-Wang (71 SNOMED classes, 12-lead).
                </p>
                <p className="text-xs text-zinc-500 mt-0.5">
                    Inter-patient split · MIT-BIH Arrhythmia Database · {cnn?.per_class
                        ? Object.values(cnn.per_class).reduce((s, c) => s + c.support, 0)
                        : '—'} test beats
                </p>
            </div>

            {/* Tabs */}
            <div className="flex gap-2 border-b border-zinc-800 pb-0">
                {(['overview', 'perclass'] as const).map(t => (
                    <button
                        key={t}
                        onClick={() => setTab(t)}
                        className={`text-xs px-3 py-1.5 rounded-t border-b-2 transition-colors ${
                            tab === t
                                ? 'border-indigo-500 text-indigo-300'
                                : 'border-transparent text-zinc-500 hover:text-zinc-300'
                        }`}
                    >
                        {t === 'overview' ? 'Overview' : 'Per-Class F1'}
                    </button>
                ))}
            </div>

            {tab === 'overview' && (
                <div className="space-y-4">
                    {overviewRows.map(row => (
                        <div key={row.key} className="space-y-1">
                            <div className="flex items-center justify-between text-xs">
                                <span className="text-zinc-300 font-medium">{row.label}</span>
                                <div className="flex items-center gap-3 text-zinc-400">
                                    {row.ci && (
                                        <span className="text-zinc-600">
                                            95% CI [{row.ci.low.toFixed(1)}, {row.ci.high.toFixed(1)}]
                                        </span>
                                    )}
                                    <span>Acc {row.accuracy?.toFixed(1) ?? '—'}%</span>
                                    <span className="font-semibold text-zinc-200">
                                        F1 {row.macro_f1?.toFixed(1) ?? '—'}
                                    </span>
                                </div>
                            </div>
                            <div className="h-6 bg-zinc-800 rounded overflow-hidden">
                                <div
                                    className="h-full rounded transition-all duration-700"
                                    style={{
                                        width:      `${((row.macro_f1 ?? 0) / 100) * 100}%`,
                                        background: row.color,
                                        opacity:    row.macro_f1 === maxF1 ? 1 : 0.6,
                                    }}
                                />
                            </div>
                        </div>
                    ))}

                    {/* CNN CI callout */}
                    {cnn?.macro_f1_ci && (
                        <p className="text-xs text-zinc-600 pt-1">
                            CNN macro-F1 bootstrap 95% CI (n=1 000 iterations):&nbsp;
                            <span className="text-zinc-400">
                                [{cnn.macro_f1_ci.low.toFixed(1)}, {cnn.macro_f1_ci.high.toFixed(1)}]
                            </span>
                            &nbsp;— honest inter-patient generalisation estimate.
                        </p>
                    )}
                </div>
            )}

            {tab === 'perclass' && cnn && (
                <div className="space-y-3">
                    {classes.map(cls => {
                        const cnnF1 = cnn.per_class[cls]?.f1  ?? 0;
                        const svmF1 = svm?.per_class[cls]?.f1 ?? 0;
                        const lrF1  = lr?.per_class[cls]?.f1  ?? 0;
                        const sup   = cnn.per_class[cls]?.support ?? 0;
                        const color = CLASS_COLORS[cls] ?? '#6366f1';

                        return (
                            <div key={cls} className="bg-zinc-900 rounded-lg p-3 space-y-2">
                                <div className="flex items-center justify-between">
                                    <span className="text-xs font-medium text-zinc-200">{cls}</span>
                                    <span className="text-xs text-zinc-600">{sup} test beats</span>
                                </div>
                                {[
                                    { label: 'CNN', f1: cnnF1, color },
                                    ...(trf ? [{ label: 'TRF', f1: trf.per_class[cls]?.f1 ?? 0, color: '#ec4899' }] : []),
                                    { label: 'SVM', f1: svmF1, color: '#10b981' },
                                    { label: 'LR',  f1: lrF1,  color: '#f59e0b' },
                                ].map(m => (
                                    <div key={m.label} className="flex items-center gap-2">
                                        <span className="text-xs text-zinc-600 w-8 shrink-0">{m.label}</span>
                                        <div className="flex-1 h-3 bg-zinc-800 rounded overflow-hidden">
                                            <div
                                                className="h-full rounded"
                                                style={{
                                                    width:      `${m.f1}%`,
                                                    background: m.color,
                                                    opacity:    0.8,
                                                }}
                                            />
                                        </div>
                                        <span className="text-xs text-zinc-400 w-10 text-right">
                                            {m.f1.toFixed(1)}
                                        </span>
                                    </div>
                                ))}
                            </div>
                        );
                    })}
                </div>
            )}
        </div>
    );
}
