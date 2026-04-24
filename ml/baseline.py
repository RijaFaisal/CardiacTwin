"""
Baseline classifiers (SVM + Logistic Regression) trained on the same
180-sample beat segments as the CNN, using hand-crafted statistical
features instead of raw waveform input.

Run from repo root:
    python ml/baseline.py

Outputs:
    ml/results/baseline_metrics.json
"""
import sys
import os
sys.path.append(os.path.abspath('.'))

import json
import time
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

CLASS_NAMES = ["Normal", "Bundle Branch Block", "Ventricular", "Atrial", "Other"]

# ── Load data ─────────────────────────────────────────────────────
print("Loading data...")
X_train_raw = np.load('ml/data/X_train.npy')   # (N, 180)
y_train     = np.load('ml/data/y_train.npy')
X_test_raw  = np.load('ml/data/X_test.npy')
y_test      = np.load('ml/data/y_test.npy')

print(f"Train: {X_train_raw.shape}  Test: {X_test_raw.shape}")

# ── Feature extraction: 12 statistical descriptors per beat ───────
def beat_features(X):
    feats = np.stack([
        X.mean(axis=1),
        X.std(axis=1),
        X.max(axis=1),
        X.min(axis=1),
        X.max(axis=1) - X.min(axis=1),          # peak-to-peak amplitude
        np.percentile(X, 25, axis=1),
        np.percentile(X, 75, axis=1),
        np.percentile(X, 75, axis=1) - np.percentile(X, 25, axis=1),  # IQR
        np.abs(np.diff(X, axis=1)).mean(axis=1), # mean absolute diff (slope)
        np.sum(X ** 2, axis=1),                  # signal energy
        np.argmax(X, axis=1).astype(float),      # R-peak position
        np.abs(X).mean(axis=1),                  # mean absolute value
    ], axis=1)
    return feats

print("Extracting features...")
X_train = beat_features(X_train_raw)
X_test  = beat_features(X_test_raw)

# ── Standardise ───────────────────────────────────────────────────
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

results = {}

# ── Logistic Regression ───────────────────────────────────────────
print("\nTraining Logistic Regression...")
t0 = time.time()
lr = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    solver='lbfgs',
    random_state=42,
    n_jobs=-1
)
lr.fit(X_train, y_train)
lr_time = time.time() - t0

y_pred_lr = lr.predict(X_test)
acc_lr    = accuracy_score(y_test, y_pred_lr)
report_lr = classification_report(y_test, y_pred_lr, target_names=CLASS_NAMES,
                                   output_dict=True, zero_division=0)

print(f"  Accuracy: {acc_lr*100:.2f}%  ({lr_time:.1f}s)")
print(classification_report(y_test, y_pred_lr, target_names=CLASS_NAMES, zero_division=0))

results['logistic_regression'] = {
    'accuracy': round(acc_lr * 100, 2),
    'train_time_s': round(lr_time, 1),
    'per_class': {
        name: {
            'precision': round(report_lr[name]['precision'] * 100, 1),
            'recall':    round(report_lr[name]['recall']    * 100, 1),
            'f1':        round(report_lr[name]['f1-score']  * 100, 1),
            'support':   int(report_lr[name]['support']),
        }
        for name in CLASS_NAMES
    },
    'macro_f1': round(report_lr['macro avg']['f1-score'] * 100, 2),
}

# ── SVM ───────────────────────────────────────────────────────────
print("\nTraining SVM (RBF kernel)...")
t0 = time.time()

# Use a 30k-sample cap for speed; MIT-BIH train set is typically ~65k
cap = 30_000
if len(X_train) > cap:
    idx = np.random.RandomState(42).choice(len(X_train), cap, replace=False)
    Xs, ys = X_train[idx], y_train[idx]
    print(f"  (using {cap:,} / {len(X_train):,} samples for SVM speed)")
else:
    Xs, ys = X_train, y_train

svm = SVC(kernel='rbf', class_weight='balanced', random_state=42, probability=False)
svm.fit(Xs, ys)
svm_time = time.time() - t0

y_pred_svm = svm.predict(X_test)
acc_svm    = accuracy_score(y_test, y_pred_svm)
report_svm = classification_report(y_test, y_pred_svm, target_names=CLASS_NAMES,
                                    output_dict=True, zero_division=0)

print(f"  Accuracy: {acc_svm*100:.2f}%  ({svm_time:.1f}s)")
print(classification_report(y_test, y_pred_svm, target_names=CLASS_NAMES, zero_division=0))

results['svm'] = {
    'accuracy': round(acc_svm * 100, 2),
    'train_time_s': round(svm_time, 1),
    'note': f'trained on {min(cap, len(X_train)):,} of {len(X_train):,} samples',
    'per_class': {
        name: {
            'precision': round(report_svm[name]['precision'] * 100, 1),
            'recall':    round(report_svm[name]['recall']    * 100, 1),
            'f1':        round(report_svm[name]['f1-score']  * 100, 1),
            'support':   int(report_svm[name]['support']),
        }
        for name in CLASS_NAMES
    },
    'macro_f1': round(report_svm['macro avg']['f1-score'] * 100, 2),
}

# ── Save SVM predictions for McNemar's test in evaluate.py ───────
np.save('ml/results/svm_predictions.npy', y_pred_svm)

# ── Save (preserve keys added by other scripts, e.g. transformer) ─
os.makedirs('ml/results', exist_ok=True)
out_path = 'ml/results/baseline_metrics.json'
existing = {}
if os.path.exists(out_path):
    try:
        with open(out_path) as f:
            existing = json.load(f)
    except Exception:
        pass
existing.update(results)
with open(out_path, 'w') as f:
    json.dump(existing, f, indent=2)

print(f"\nBaseline metrics saved to {out_path}")
print(f"SVM predictions saved to ml/results/svm_predictions.npy")
print(f"\nSummary:")
print(f"  Logistic Regression : {acc_lr*100:.2f}% accuracy, {results['logistic_regression']['macro_f1']:.2f}% macro-F1")
print(f"  SVM (RBF)           : {acc_svm*100:.2f}% accuracy, {results['svm']['macro_f1']:.2f}% macro-F1")
