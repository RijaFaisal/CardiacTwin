import sys
import os
sys.path.append(os.path.abspath('.'))

import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from sklearn.utils import resample as bootstrap_resample
import matplotlib.pyplot as plt
import seaborn as sns

# ── Class names ───────────────────────────────────────────────────
CLASS_NAMES = [
    "Normal",
    "Bundle Branch Block",
    "Ventricular",
    "Atrial",
    "Other"
]

# ── Load test data ────────────────────────────────────────────────
print("Loading test data...")
X_test = np.load('ml/data/X_test.npy')
y_test = np.load('ml/data/y_test.npy')

print(f"Test set shape : {X_test.shape}")
print(f"Test labels    : {y_test.shape}")

# Reshape for CNN
X_test = X_test[..., np.newaxis]

# ── Load best model ───────────────────────────────────────────────
print("\nLoading model...")
model = keras.models.load_model('models/arrhythmia_cnn_best.h5')
print("Model loaded")

# ── Run inference ─────────────────────────────────────────────────
print("\nRunning inference on test set...")
y_prob = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_prob, axis=1)

# ── Overall accuracy ──────────────────────────────────────────────
accuracy = accuracy_score(y_test, y_pred)
print(f"\n--- Test Results ---")
print(f"Overall Accuracy : {accuracy * 100:.2f}%")
print(f"Target           : > 90%")
print(f"Status           : {'TARGET MET ✅' if accuracy >= 0.90 else 'BELOW TARGET ❌'}")

# ── Per-class report ──────────────────────────────────────────────
print(f"\n--- Per-Class Report ---")
print(classification_report(
    y_test, y_pred,
    target_names=CLASS_NAMES,
    zero_division=0
))

# ── Confusion matrix ──────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Reds',
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)
plt.title(f'Confusion Matrix — Test Accuracy: {accuracy * 100:.2f}%')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

os.makedirs('ml/results', exist_ok=True)
plt.savefig('ml/results/confusion_matrix.png', dpi=150)
plt.show()
print("\nConfusion matrix saved to ml/results/confusion_matrix.png")

# ── Class distribution in test set ───────────────────────────────
print("\n--- Test Set Class Distribution ---")
for i, name in enumerate(CLASS_NAMES):
    count = np.sum(y_test == i)
    correct = np.sum((y_test == i) & (y_pred == i))
    if count > 0:
        print(f"  {name:25s}: {correct}/{count} correct ({100*correct/count:.1f}%)")
    else:
        print(f"  {name:25s}: no samples in test set")

# ── AUC-ROC — only for classes present in the test set ───────────
num_classes     = len(CLASS_NAMES)
present_classes = sorted(np.unique(y_test).tolist())
try:
    y_test_bin = label_binarize(y_test, classes=list(range(num_classes)))
    macro_auc  = roc_auc_score(
        y_test_bin[:, present_classes],
        y_prob[:, present_classes],
        multi_class='ovr',
        average='macro',
    )
    print(f"\nMacro AUC-ROC : {macro_auc:.4f}  (classes present: {[CLASS_NAMES[c] for c in present_classes]})")
except ValueError as e:
    macro_auc = None
    print(f"\nMacro AUC-ROC : N/A ({e})")

# ── Bootstrap 95% CI on macro-F1 (1 000 iterations) ─────────────
print("\nComputing bootstrap CI for macro-F1 (1000 iterations)...")
N_BOOT = 1000
boot_f1s = []
idx_all  = np.arange(len(y_test))
for _ in range(N_BOOT):
    idx_b     = bootstrap_resample(idx_all, replace=True)
    boot_f1s.append(f1_score(y_test[idx_b], y_pred[idx_b],
                              average='macro', zero_division=0))
ci_low, ci_high = np.percentile(boot_f1s, [2.5, 97.5])
macro_f1        = f1_score(y_test, y_pred, average='macro', zero_division=0)
print(f"Macro F1 : {macro_f1 * 100:.2f}%  (95% CI: {ci_low*100:.2f}% – {ci_high*100:.2f}%)")

# ── McNemar's test vs SVM baseline ───────────────────────────────
baseline_path = 'ml/results/baseline_metrics.json'
if os.path.exists(baseline_path):
    try:
        from statsmodels.stats.contingency_tables import mcnemar
        # Reload SVM predictions if saved, otherwise skip
        svm_pred_path = 'ml/results/svm_predictions.npy'
        if os.path.exists(svm_pred_path):
            y_pred_svm     = np.load(svm_pred_path)
            cnn_correct    = (y_pred     == y_test)
            svm_correct    = (y_pred_svm == y_test)
            both_right     = int(np.sum( cnn_correct &  svm_correct))
            cnn_only       = int(np.sum( cnn_correct & ~svm_correct))
            svm_only       = int(np.sum(~cnn_correct &  svm_correct))
            both_wrong     = int(np.sum(~cnn_correct & ~svm_correct))
            result         = mcnemar([[both_right, cnn_only], [svm_only, both_wrong]], exact=True)
            print(f"\nMcNemar CNN vs SVM: p = {result.pvalue:.4f} "
                  f"({'CNN significantly better' if result.pvalue < 0.05 else 'not significant'})")
        else:
            print("\nMcNemar test: run baseline.py first to generate svm_predictions.npy")
    except ImportError:
        print("\nMcNemar test: install statsmodels  →  pip install statsmodels")

# ── Save CNN metrics as JSON (for API / frontend comparison) ──────
report_dict = classification_report(
    y_test, y_pred,
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0
)
cnn_metrics = {
    'accuracy':    round(accuracy * 100, 2),
    'macro_f1':    round(macro_f1 * 100, 2),
    'macro_auc':   round(macro_auc, 4) if macro_auc is not None else None,
    'macro_f1_ci': {
        'low':  round(ci_low  * 100, 2),
        'high': round(ci_high * 100, 2),
    },
    'per_class': {
        name: {
            'precision': round(report_dict[name]['precision'] * 100, 1),
            'recall':    round(report_dict[name]['recall']    * 100, 1),
            'f1':        round(report_dict[name]['f1-score']  * 100, 1),
            'support':   int(report_dict[name]['support']),
        }
        for name in CLASS_NAMES
    },
    'test_samples': int(len(y_test)),
}
metrics_path = 'ml/results/cnn_metrics.json'
with open(metrics_path, 'w') as f:
    json.dump(cnn_metrics, f, indent=2)
print(f"\nCNN metrics saved to {metrics_path}")