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
    accuracy_score
)
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

# ── Save CNN metrics as JSON (for API / frontend comparison) ──────
report_dict = classification_report(
    y_test, y_pred,
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0
)
cnn_metrics = {
    'accuracy':  round(accuracy * 100, 2),
    'macro_f1':  round(report_dict['macro avg']['f1-score'] * 100, 2),
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