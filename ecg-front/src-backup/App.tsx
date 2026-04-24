import React from 'react';
import Dashboard from './components/layout/Dashboard';

export default function App() {
    // Plug in your actual session ID from FastAPI Swagger here
    const testSessionId = "080b91bb-425d-45f4-aa97-5d66d32b6d61";

    return (
        <div className="bg-zinc-950 min-h-screen">
            <Dashboard sessionId={testSessionId} />
        </div>
    );
}