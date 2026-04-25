import type { Tab } from '../../types/ui';

const NAV: { id: Tab; label: string; sub: string; icon: React.ReactNode }[] = [
    {
        id: 'overview',
        label: 'Overview',
        sub: 'Verdict & metrics',
        icon: (
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                    d="M3.75 6A2.25 2.25 0 016 3.75h2.25A2.25 2.25 0 0110.5 6v2.25a2.25 2.25 0 01-2.25 2.25H6a2.25 2.25 0 01-2.25-2.25V6zM3.75 15.75A2.25 2.25 0 016 13.5h2.25a2.25 2.25 0 012.25 2.25V18a2.25 2.25 0 01-2.25 2.25H6A2.25 2.25 0 013.75 18v-2.25zM13.5 6a2.25 2.25 0 012.25-2.25H18A2.25 2.25 0 0120.25 6v2.25A2.25 2.25 0 0118 10.5h-2.25a2.25 2.25 0 01-2.25-2.25V6zM13.5 15.75a2.25 2.25 0 012.25-2.25H18a2.25 2.25 0 012.25 2.25V18A2.25 2.25 0 0118 20.25h-2.25A2.25 2.25 0 0113.5 18v-2.25z" />
            </svg>
        ),
    },
    {
        id: 'ecg',
        label: 'ECG Viewer',
        sub: '12-lead · Grad-CAM',
        icon: (
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                    d="M3 12h3l2-7 4 14 3-10 2 3h4" />
            </svg>
        ),
    },
    {
        id: 'twin',
        label: 'Digital Twin',
        sub: '3D heart model',
        icon: (
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                    d="M21 8.25c0-2.485-2.099-4.5-4.688-4.5-1.935 0-3.597 1.126-4.312 2.733-.715-1.607-2.377-2.733-4.313-2.733C5.1 3.75 3 5.765 3 8.25c0 7.22 9 12 9 12s9-4.78 9-12z" />
            </svg>
        ),
    },
    {
        id: 'simulator',
        label: 'Simulator',
        sub: 'What-if analysis',
        icon: (
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                    d="M9.75 3.104v5.714a2.25 2.25 0 01-.659 1.591L5 14.5M9.75 3.104c-.251.023-.501.05-.75.082m.75-.082a24.301 24.301 0 014.5 0m0 0v5.714c0 .597.237 1.17.659 1.591L19.8 15.3M14.25 3.104c.251.023.501.05.75.082M19.8 15.3l-1.57.393A9.065 9.065 0 0112 15a9.065 9.065 0 00-6.23-.693L5 14.5m14.8.8l1.402 1.402c1.232 1.232.65 3.318-1.067 3.611A48.309 48.309 0 0112 21c-2.773 0-5.491-.235-8.135-.687-1.718-.293-2.3-2.379-1.067-3.61L5 14.5" />
            </svg>
        ),
    },
];

interface Props {
    tab: Tab;
    onTabChange: (t: Tab) => void;
    onReset: () => void;
}

export default function Sidebar({ tab, onTabChange, onReset }: Props) {
    return (
        <aside className="w-[220px] shrink-0 h-full bg-zinc-900 border-r border-zinc-800 flex flex-col overflow-hidden">

            <nav className="flex-1 p-2 pt-3 space-y-0.5 overflow-y-auto">
                {NAV.map(item => (
                    <button
                        key={item.id}
                        onClick={() => onTabChange(item.id)}
                        className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-left transition-all group ${
                            tab === item.id
                                ? 'bg-zinc-800 text-zinc-100'
                                : 'text-zinc-500 hover:text-zinc-200 hover:bg-zinc-800/60'
                        }`}
                    >
                        <span className={`shrink-0 transition-colors ${
                            tab === item.id ? 'text-indigo-400' : 'text-zinc-600 group-hover:text-zinc-400'
                        }`}>
                            {item.icon}
                        </span>
                        <div className="flex flex-col min-w-0 flex-1">
                            <span className="text-xs font-medium truncate leading-none mb-0.5">{item.label}</span>
                            <span className={`text-[10px] truncate leading-none ${
                                tab === item.id ? 'text-zinc-500' : 'text-zinc-700'
                            }`}>{item.sub}</span>
                        </div>
                        {tab === item.id && (
                            <div className="w-0.5 h-5 rounded-full bg-indigo-500 shrink-0" />
                        )}
                    </button>
                ))}
            </nav>

            <div className="p-2 border-t border-zinc-800 shrink-0">
                <button
                    onClick={onReset}
                    className="w-full flex items-center gap-2.5 px-3 py-2.5 rounded-lg text-xs text-zinc-500 hover:text-zinc-200 hover:bg-zinc-800/60 transition-colors"
                >
                    <svg className="w-4 h-4 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                            d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
                    </svg>
                    <span className="font-medium">New Analysis</span>
                </button>
                <p className="text-[9px] text-zinc-700 text-center mt-2 px-2">Research &amp; educational use only</p>
            </div>
        </aside>
    );
}
