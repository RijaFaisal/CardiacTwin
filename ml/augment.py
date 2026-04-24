"""
ml/augment.py

ECG beat augmentation for minority class oversampling.

Three transforms applied to a single 180-sample beat window:
  1. Time warp   — resample ±5% then back to original length
  2. Amplitude   — scale by ±15%
  3. Noise       — additive Gaussian at ~30 dB SNR

Reference: Um et al. (2017) arXiv:1706.00527 — time warping and
magnitude warping are the two highest-impact augmentations for
1-D biosignal time series.
"""

import numpy as np
import scipy.signal


def augment_beat(segment: np.ndarray) -> np.ndarray:
    """
    Apply random time warp + amplitude scale + Gaussian noise to one beat.

    Parameters
    ----------
    segment : np.ndarray, shape (180,)
        Pre-processed, normalised beat window.

    Returns
    -------
    np.ndarray, shape (180,)  — augmented copy, same length as input.
    """
    out = segment.copy().astype(np.float32)
    n   = len(out)

    # 1. Time warp ±5%
    factor  = np.random.uniform(0.95, 1.05)
    warped  = scipy.signal.resample(out, max(1, int(n * factor)))
    out     = scipy.signal.resample(warped, n).astype(np.float32)

    # 2. Amplitude scale ±15%
    out *= np.random.uniform(0.85, 1.15)

    # 3. Additive Gaussian noise (~30 dB SNR)
    std = out.std()
    if std > 1e-8:
        out += np.random.normal(0, 0.01 * std, out.shape).astype(np.float32)

    return out


def oversample_minority(
    X: np.ndarray,
    y: np.ndarray,
    target_per_class: int = 500,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Augment minority classes until each has at least target_per_class samples.
    Majority classes that already exceed the target are left untouched.

    Parameters
    ----------
    X                : (N, 180) beat segments
    y                : (N,)     integer class labels
    target_per_class : minimum samples per class after augmentation
    seed             : random seed for reproducibility

    Returns
    -------
    X_out, y_out — shuffled arrays with balanced class counts.
    """
    np.random.seed(seed)

    X_out = list(X)
    y_out = list(y)

    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        n_have  = len(cls_idx)

        if n_have >= target_per_class:
            continue  # already enough

        n_needed = target_per_class - n_have
        for _ in range(n_needed):
            src = X[np.random.choice(cls_idx)]
            X_out.append(augment_beat(src))
            y_out.append(cls)

    X_out = np.array(X_out, dtype=np.float32)
    y_out = np.array(y_out, dtype=np.int64)

    # Shuffle so augmented samples are interleaved
    perm  = np.random.permutation(len(X_out))
    return X_out[perm], y_out[perm]


if __name__ == '__main__':
    # Quick smoke-test
    X = np.load('ml/data/X_train.npy')
    y = np.load('ml/data/y_train.npy')

    print("Before augmentation:")
    for cls in sorted(np.unique(y)):
        print(f"  Class {cls}: {np.sum(y == cls)}")

    X_aug, y_aug = oversample_minority(X, y, target_per_class=500)

    print("\nAfter augmentation (target=500 per class):")
    for cls in sorted(np.unique(y_aug)):
        print(f"  Class {cls}: {np.sum(y_aug == cls)}")

    print(f"\nTotal beats: {len(y)} → {len(y_aug)}")
