// src/api/ecgClient.ts
import type { AnalysisResponse } from '../types/analysis';

const API_BASE_URL = 'http://localhost:8000/api/v1';

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

  async analyzeSession(sessionId: string): Promise<AnalysisResponse> {
    const response = await fetch(`${API_BASE_URL}/analyze/${sessionId}`, {
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
      headers: {
        'Accept': 'application/json',
      },
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new APIError(
        response.status,
        errorData.detail || `Signal quality check failed with status ${response.status}`
      );
    }

    return response.json();
  }
};