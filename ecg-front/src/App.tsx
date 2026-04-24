import { useState } from 'react';
import Dashboard from './components/layout/Dashboard';
import UploadScreen from './components/upload/UploadScreen';

export default function App() {
    const [sessionId, setSessionId] = useState<string | null>(null);

    return (
        <div className="bg-zinc-950 min-h-screen">
            {sessionId ? (
                <Dashboard sessionId={sessionId} />
            ) : (
                <UploadScreen onUploadSuccess={setSessionId} />
            )}
        </div>
    );
}