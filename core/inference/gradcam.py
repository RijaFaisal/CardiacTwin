"""
core/inference/gradcam.py

Explainability for the ECG pipeline.  Two complementary methods:

1. FCN Wang Grad-CAM (PRIMARY)
   - Model: FCN Wang trained on PTB-XL (71 classes, 12-lead)
   - Output: (1000,) temporal heatmap — which time points in the full
     10-second ECG drove the top predicted diagnosis.
   - Explains the same model that produced the diagnosis shown to the user.

2. MIT-BIH CNN beat Grad-CAM (SECONDARY)
   - Model: 1D CNN trained on MIT-BIH (5 beat classes, Lead II)
   - Output: per-beat (180,) saliency segments.
   - Provides fine-grained beat morphology detail as a comparison baseline.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import scipy.signal

logger = logging.getLogger(__name__)

# ── MIT-BIH CNN constants ────────────────────────────────────────────────
_MITBIH_MODEL_PATH = Path("models/arrhythmia_cnn_best.h5")
_TARGET_SR         = 360
_WIN               = 180
_HALF              = _WIN // 2
_BEAT_CLASSES      = ["Normal", "Bundle Branch Block", "Ventricular", "Atrial", "Other"]


class _GradCamRunner:
    """Singleton that runs both FCN Wang and MIT-BIH CNN Grad-CAM."""

    def __init__(self) -> None:
        self._mitbih_model      = None
        self._mitbih_grad_model = None
        self._loaded            = False

    # ── MIT-BIH CNN lazy loader ──────────────────────────────────────────

    def _load_mitbih(self) -> None:
        if self._loaded:
            return
        if not _MITBIH_MODEL_PATH.exists():
            logger.warning("MIT-BIH CNN not found at %s", _MITBIH_MODEL_PATH)
            return
        try:
            import tensorflow as tf
            from ml.explain import build_grad_model
            self._mitbih_model      = tf.keras.models.load_model(str(_MITBIH_MODEL_PATH))
            self._mitbih_grad_model = build_grad_model(self._mitbih_model)
            self._loaded            = True
            logger.info("MIT-BIH CNN loaded for beat-level Grad-CAM")
        except Exception as exc:
            logger.warning("Could not load MIT-BIH CNN: %s", exc)

    # ── FCN Wang Grad-CAM ────────────────────────────────────────────────

    def _run_fcn_wang_gradcam(
        self,
        ecg_scaled: np.ndarray,
        probabilities: dict[str, float],
    ) -> dict:
        """
        Temporal Grad-CAM on the FCN Wang model.

        Parameters
        ----------
        ecg_scaled    : (1000, 12) preprocessed ECG fed to FCN Wang
        probabilities : 71-class probability dict from the FCN Wang prediction

        Returns
        -------
        dict with keys: saliency (list[float]), explained_class (str),
                        explained_class_prob (float), model (str)
        """
        try:
            from core.inference.fcn_wang_runner import ml_pipeline
            from core.inference.fcn_wang_gradcam import compute_fcn_wang_gradcam

            if ml_pipeline is None or not probabilities:
                return {"available": False}

            # Explain the top predicted class
            top_code, top_prob = max(probabilities.items(), key=lambda x: x[1])
            classes             = list(ml_pipeline.classes)
            if top_code not in classes:
                return {"available": False}
            class_idx = classes.index(top_code)

            saliency = compute_fcn_wang_gradcam(
                ml_pipeline.model, ecg_scaled, class_idx
            )

            return {
                "available":           True,
                "model":               "FCN Wang (PTB-XL, 71-class)",
                "explained_class":     top_code,
                "explained_class_prob": round(top_prob, 4),
                "saliency":            [round(v, 4) for v in saliency],
            }

        except Exception as exc:
            logger.warning("FCN Wang Grad-CAM failed: %s", exc)
            return {"available": False}

    # ── MIT-BIH CNN beat-level Grad-CAM ─────────────────────────────────

    def _run_mitbih_gradcam(
        self,
        signal_lead2: np.ndarray,
        native_sr: int,
        r_peaks_native: list[int],
    ) -> dict:
        self._load_mitbih()

        if not self._loaded:
            return {"beats": [], "dominant_class": None, "available": False}

        from ml.explain import compute_gradcam

        if native_sr != _TARGET_SR:
            n_out = int(len(signal_lead2) * _TARGET_SR / native_sr)
            sig   = scipy.signal.resample(signal_lead2, n_out)
            scale = _TARGET_SR / native_sr
            peaks = [int(p * scale) for p in r_peaks_native]
        else:
            sig   = signal_lead2.astype(np.float32)
            peaks = list(r_peaks_native)

        std = sig.std()
        if std > 1e-8:
            sig = (sig - sig.mean()) / std

        beats: list[dict] = []
        class_votes: dict[str, int] = {}

        for r in peaks:
            lo, hi = r - _HALF, r + _HALF
            if lo < 0 or hi > len(sig):
                continue

            segment  = sig[lo:hi].astype(np.float32)
            x        = segment.reshape(1, _WIN, 1)
            probs    = self._mitbih_model.predict(x, verbose=0)[0]
            idx      = int(np.argmax(probs))
            label    = _BEAT_CLASSES[idx]
            conf     = float(probs[idx])
            saliency = compute_gradcam(self._mitbih_grad_model, segment, idx, _WIN)

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
            "available":       True,
            "model":           "MIT-BIH CNN (beat-level baseline)",
            "beats":           beats,
            "dominant_class":  dominant,
        }

    # ── Public entry point ───────────────────────────────────────────────

    def run(
        self,
        signal_lead2: np.ndarray,
        native_sr: int,
        r_peaks_native: list[int],
        ecg_scaled: np.ndarray | None = None,
        probabilities: dict[str, float] | None = None,
    ) -> dict:
        """
        Run Grad-CAM explainability.

        FCN Wang temporal saliency is the primary output (explains the
        diagnosis shown to the user).  MIT-BIH CNN beat detail is secondary
        (included when available as a morphology reference).

        Parameters
        ----------
        signal_lead2    : Lead II signal at native sampling rate
        native_sr       : sampling rate of signal_lead2
        r_peaks_native  : R-peak indices at native SR (for beat-level detail)
        ecg_scaled      : (1000, 12) scaled ECG — required for FCN Wang Grad-CAM
        probabilities   : FCN Wang 71-class output — required for FCN Wang Grad-CAM
        """
        result: dict = {}

        # Primary: FCN Wang temporal Grad-CAM
        if ecg_scaled is not None and probabilities:
            result["fcn_wang"] = self._run_fcn_wang_gradcam(ecg_scaled, probabilities)
        else:
            result["fcn_wang"] = {"available": False}

        # Secondary: MIT-BIH CNN beat-level Grad-CAM
        result["mitbih_cnn"] = self._run_mitbih_gradcam(
            signal_lead2, native_sr, r_peaks_native
        )

        return result


gradcam_runner = _GradCamRunner()
