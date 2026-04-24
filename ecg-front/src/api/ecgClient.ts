// src/api/ecgClient.ts
import type { AnalysisResponse } from '../types/analysis';

/** Production: set in Vercel → Settings → Environment Variables as VITE_API_BASE_URL */
const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000/api/v1'
).replace(/\/$/, '');

export class APIError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
    this.name = 'APIError';
  }
}

export const ecgClient = {
  async uploadWfdb(heaFile: File, datFile: File): Promise<{ session_id: string }> {
    const formData = new FormData();
    formData.append('header_file', heaFile);
    formData.append('data_file', datFile);

    const response = await fetch(`${API_BASE_URL}/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new APIError(response.status, errorData.detail || 'WFDB Upload failed');
    }
    return response.json();
  },

  async uploadCsv(csvFile: File): Promise<{ session_id: string }> {
    const formData = new FormData();
    formData.append('file', csvFile);

    const response = await fetch(`${API_BASE_URL}/upload/csv`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new APIError(response.status, errorData.detail || 'CSV Upload failed');
    }
    return response.json();
  },

  async analyzeSession(sessionId: string, explain = false): Promise<AnalysisResponse> {
    const url = explain
      ? `${API_BASE_URL}/analyze/${sessionId}?explain=true`
      : `${API_BASE_URL}/analyze/${sessionId}`;
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new APIError(
        response.status,
        errorData.detail || `Analysis failed with status ${response.status}`
      );
    }

    return response.json();
  },

  async checkSignalQuality(sessionId: string): Promise<{ session_id: string; signal_quality: any }> {
    const response = await fetch(`${API_BASE_URL}/analyze/${sessionId}/signal-quality`, {
      method: 'GET',
      headers: { 'Accept': 'application/json' },
    });
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new APIError(response.status, errorData.detail || `Signal quality check failed with status ${response.status}`);
    }
    return response.json();
  },

  async simulatePathology(pathology: string, age: number, gender: string): Promise<SimulateResult> {
    const response = await fetch(`${API_BASE_URL}/simulate/pathology`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ pathology, age, gender }),
    });
    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new APIError(response.status, err.detail || 'Simulation failed');
    }
    return response.json();
  },

  async simulateTreatment(treatment: string, pathology: string, age: number, gender: string): Promise<SimulateResult> {
    const response = await fetch(`${API_BASE_URL}/simulate/treatment`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ treatment, pathology, age, gender }),
    });
    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new APIError(response.status, err.detail || 'Treatment simulation failed');
    }
    return response.json();
  },

  async getModelMetrics(): Promise<{ cnn: any; baselines: any }> {
    const response = await fetch(`${API_BASE_URL}/simulate/metrics`);
    if (!response.ok) throw new APIError(response.status, 'Could not load model metrics');
    return response.json();
  },

  async loadDemo(scenario: 'normal' | 'afib' | 'bbb'): Promise<{ session_id: string }> {
    const response = await fetch(`${API_BASE_URL}/demo/${scenario}`);
    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new APIError(response.status, err.detail || 'Demo load failed');
    }
    return response.json();
  },

  async exportReport(sessionId: string): Promise<Blob> {
    const response = await fetch(`${API_BASE_URL}/export/${sessionId}/pdf`, {
      method: 'POST',
    });
    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new APIError(response.status, err.detail || 'Report export failed');
    }
    return response.blob();
  },
};

export interface SimulateResult {
  display_name:      string;
  heart_rate:        number;
  hrv_sdnn:          number;
  hrv_rmssd:         number;
  rr_irregular:      boolean;
  dominant_class:    string;
  beat_distribution: Record<string, number>;
  description:       string;
  demographic_note:  string;
  efficacy?:         string;
  pathology_display?: string;
}