"""
core/inference/gradcam.py

Loads the CardiacDigitalTwin 1D CNN and runs Grad-CAM explainability
on a single-lead ECG signal.

Used by the analyze route when ?explain=true is passed.
The CNN was trained on MIT-BIH lead II at 360 Hz, 180-sample beat windows.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import scipy.signal

logger = logging.getLogger(__name__)

_MODEL_PATH = Path("models/arrhythmia_cnn_best.h5")
_TARGET_SR  = 360
_WIN        = 180           # samples per beat window
_HALF       = _WIN // 2

CLASS_NAMES = ["Normal", "Bundle Branch Block", "Ventricular", "Atrial", "Other"]


class _GradCamRunner:
    def __init__(self) -> None:
        self._model      = None
        self._grad_model = None
        self._loaded     = False

    def _load(self) -> None:
        if self._loaded:
            return
        if not _MODEL_PATH.exists():
            logger.warning("CNN model not found at %s — Grad-CAM disabled", _MODEL_PATH)
            return
        try:
            import tensorflow as tf
            from ml.explain import build_grad_model

            self._model      = tf.keras.models.load_model(str(_MODEL_PATH))
            self._grad_model = build_grad_model(self._model)
            self._loaded     = True
            logger.info("Grad-CAM CNN loaded from %s", _MODEL_PATH)
        except Exception as exc:
            logger.warning("Could not load CNN for Grad-CAM: %s", exc)

    def run(
        self,
        signal_lead2: np.ndarray,
        native_sr: int,
        r_peaks_native: list[int],
    ) -> dict:
        """
        Run beat-level CNN classification + Grad-CAM saliency.

        Parameters
        ----------
        signal_lead2    : 1-D array, lead II (or best available lead), native SR
        native_sr       : sampling rate of signal_lead2
        r_peaks_native  : R-peak indices in the native sampling rate

        Returns
        -------
        dict with keys:
            beats          : list of per-beat dicts (class, confidence, saliency)
            dominant_class : str, most common predicted class
            cnn_available  : bool
        """
        self._load()

        if not self._loaded:
            return {"beats": [], "dominant_class": None, "cnn_available": False}

        from ml.explain import compute_gradcam

        # Resample to 360 Hz if needed
        if native_sr != _TARGET_SR:
            n_out = int(len(signal_lead2) * _TARGET_SR / native_sr)
            sig   = scipy.signal.resample(signal_lead2, n_out)
            scale = _TARGET_SR / native_sr
            peaks = [int(p * scale) for p in r_peaks_native]
        else:
            sig   = signal_lead2.astype(np.float32)
            peaks = list(r_peaks_native)

        # Z-score normalise
        std = sig.std()
        if std > 1e-8:
            sig = (sig - sig.mean()) / std

        beats = []
        class_votes: dict[str, int] = {}

        import tensorflow as tf

        for r in peaks:
            lo, hi = r - _HALF, r + _HALF
            if lo < 0 or hi > len(sig):
                continue

            segment = sig[lo:hi].astype(np.float32)

            # CNN inference
            x     = segment.reshape(1, _WIN, 1)
            probs = self._model.predict(x, verbose=0)[0]
            idx   = int(np.argmax(probs))
            label = CLASS_NAMES[idx]
            conf  = float(probs[idx])

            # Grad-CAM saliency
            saliency = compute_gradcam(self._grad_model, segment, idx, _WIN)

            beats.append({
                "r_peak":     r,
                "class":      label,
                "confidence": round(conf, 4),
                "saliency":   [round(v, 4) for v in saliency],
                "segment":    segment.tolist(),
            })

            class_votes[label] = class_votes.get(label, 0) + 1

        dominant = max(class_votes, key=class_votes.get) if class_votes else None

        return {
            "beats":          beats,
            "dominant_class": dominant,
            "cnn_available":  True,
        }


# Module-level singleton — loaded lazily on first use
gradcam_runner = _GradCamRunner()
