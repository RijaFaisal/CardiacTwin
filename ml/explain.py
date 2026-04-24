"""
Grad-CAM explainability for the 1D ECG CNN.

Grad-CAM (Gradient-weighted Class Activation Mapping) answers:
"Which part of this beat waveform drove the model's prediction?"

For a 1D CNN the algorithm is:
  1. Forward pass — capture the last Conv1D layer's output (feature maps).
  2. Backward pass — compute the gradient of the predicted class score
     w.r.t. those feature maps.
  3. Global-average-pool the gradients across the filter dimension →
     one weight per filter.
  4. Weight each filter's activation map and sum → 1-D heatmap.
  5. ReLU (keep only positive influence), normalise to [0, 1].
  6. Upsample from Conv layer resolution (~22 time-steps) back to 180
     using linear interpolation.

Reference: Selvaraju et al. "Grad-CAM" (ICCV 2017).
"""

import numpy as np
import tensorflow as tf
from scipy.ndimage import zoom


def build_grad_model(model: tf.keras.Model) -> tf.keras.Model:
    """
    Build a sub-model that outputs (last_conv1d_activations, softmax_probs).

    Calling this once at startup is cheap; the returned model shares
    weights with the original and requires no re-training.

    Raises ValueError if the model has no Conv1D layers.
    """
    last_conv = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv1D):
            last_conv = layer
            break

    if last_conv is None:
        raise ValueError("No Conv1D layer found in the model.")

    return tf.keras.Model(
        inputs=model.inputs,
        outputs=[last_conv.output, model.output],
        name="grad_cam_model"
    )


def compute_gradcam(
    grad_model: tf.keras.Model,
    segment_1d: np.ndarray,
    class_idx: int,
    target_length: int = 180,
) -> list[float]:
    """
    Compute a 1-D Grad-CAM heatmap for a single beat segment.

    Parameters
    ----------
    grad_model : tf.keras.Model
        Built by ``build_grad_model()``.
    segment_1d : np.ndarray, shape (180,)
        Pre-processed, normalised beat window.
    class_idx : int
        The predicted class index (0–4) to explain.
    target_length : int
        Output heatmap length — should match segment window (180).

    Returns
    -------
    list[float]
        Heatmap values in [0, 1], length == target_length.
        Values closer to 1 indicate regions that most influenced
        the model's decision.
    """
    x = tf.convert_to_tensor(
        segment_1d.reshape(1, target_length, 1), dtype=tf.float32
    )

    with tf.GradientTape() as tape:
        # Watch the intermediate tensor (not a tf.Variable)
        tape.watch(x)
        conv_outputs, predictions = grad_model(x, training=False)
        tape.watch(conv_outputs)
        loss = predictions[:, class_idx]

    # Gradients of the class score w.r.t. the last Conv1D output
    grads = tape.gradient(loss, conv_outputs)   # (1, T, F)

    if grads is None:
        return [0.0] * target_length

    # Global-average-pool gradients across filters → one weight per filter
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))   # (F,)

    # Weighted sum of feature maps
    heatmap = tf.reduce_sum(
        pooled_grads * conv_outputs[0], axis=-1          # (T,)
    )

    # ReLU: keep only positive contributions
    heatmap = tf.nn.relu(heatmap).numpy()

    # Upsample to target_length using linear interpolation
    conv_len = len(heatmap)
    if conv_len != target_length:
        heatmap = zoom(heatmap, target_length / conv_len)
        heatmap = heatmap[:target_length]   # guard against rounding

    # Clip to remove cubic-interpolation artefacts (<0) then normalise to [0,1]
    heatmap = np.clip(heatmap, 0, None)
    max_val = float(heatmap.max())
    if max_val > 1e-8:
        heatmap = heatmap / max_val

    return heatmap.tolist()


def flag_rr_anomalies(
    rr_intervals_sec: list[float],
    threshold_std: float = 2.0,
) -> list[bool]:
    """
    Flag RR intervals that deviate significantly from the mean.

    Parameters
    ----------
    rr_intervals_sec : list[float]
        RR intervals in seconds between consecutive beats.
    threshold_std : float
        Number of standard deviations to use as threshold.

    Returns
    -------
    list[bool]
        True where the interval is anomalous (possible arrhythmia),
        aligned with rr_intervals_sec.
    """
    if len(rr_intervals_sec) < 3:
        return [False] * len(rr_intervals_sec)

    arr  = np.array(rr_intervals_sec)
    mean = arr.mean()
    std  = arr.std()

    if std < 1e-8:
        return [False] * len(rr_intervals_sec)

    return [bool(abs(rr - mean) > threshold_std * std) for rr in arr]
