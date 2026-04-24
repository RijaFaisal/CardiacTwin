// src/types/analysis.ts

export interface MetricCard {
  value: number | null;
  status: 'normal' | 'low' | 'high' | 'unknown' | 'info';
  unit: string;
}

export interface WaveAmplitudes {
  p: number | null;
  q: number | null;
  r: number | null;
  s: number | null;
  t: number | null;
}

export interface PeakMarkers {
  r_peaks: number[];
  p_peaks: number[];
  q_peaks: number[];
  s_peaks: number[];
  t_peaks: number[];
  p_onsets: number[];
  p_offsets: number[];
  t_offsets: number[];
  amplitudes: WaveAmplitudes;
}

export interface WaveformData {
  samples: number[];
  sampling_rate: number;
  duration_s: number;
  lead_index: number;
  lead_name: string;
}

export interface QualityBadge {
  score_pct: number;
  raw_mean: number;
  status: 'normal' | 'low';
}

export interface PredictionEntry {
  code: string;
  display_name: string;
  definition: string;
  probability: number;
  percentage: number;
  severity: 'normal' | 'warning' | 'critical';
}

export interface AIAnalysis {
  top_predictions: PredictionEntry[];
  raw_probabilities: Record<string, number>;
}

export interface Verdict {
  code: string;
  display_name: string;
  definition: string;
  probability: number;
  percentage: number;
  severity: 'normal' | 'warning' | 'critical';
  above_threshold: boolean;
  quality: QualityBadge;
}

export interface GradcamBeat {
  r_peak:     number;     // sample index at 360 Hz
  class:      string;
  confidence: number;
  saliency:   number[];   // 180 values, 0–1
  segment:    number[];   // 180 raw normalised samples
}

export interface GradcamResult {
  beats:          GradcamBeat[];
  dominant_class: string | null;
  cnn_available:  boolean;
}

export interface AnalysisResponse {
  session_id: string;
  status: string;
  verdict: Verdict;
  metrics: Record<string, MetricCard>;
  waveforms: WaveformData[];
  peaks: PeakMarkers;
  ai_analysis: AIAnalysis;
  processing_notes: string[];
  gradcam?: GradcamResult;
}