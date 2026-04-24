"""
ml/evaluate_fcn_wang.py

Evaluate the FCN Wang model (trained on PTB-XL) on the official PTB-XL
held-out test set (strat_fold == 10, 2198 records).

This is the production model used in the analysis pipeline. These are
the metrics that matter — the MIT-BIH CNN metrics are a secondary
beat-level baseline, not the headline result.

Outputs
-------
ml/results/fcn_wang_metrics.json  — per-class AUC, macro AUC, accuracy,
                                     confidence intervals, sample count

Run from repo root:
    python ml/evaluate_fcn_wang.py
"""

import ast
import json
import os
import sys

sys.path.append(os.path.abspath("."))

import joblib
import numpy as np
import torch
import wfdb
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils import resample as bootstrap_resample

import pandas as pd
import scipy.signal

from config import settings
from core.inference.fcn import fcn_wang

# ── Paths ─────────────────────────────────────────────────────────────────

PTBXL_ROOT = "data/raw/ptbxl/physionet.org/files/ptb-xl/1.0.3"
PTBXL_CSV  = f"{PTBXL_ROOT}/ptbxl_database.csv"
TARGET_LEN  = 1000   # 10s at 100 Hz

# ── Load model and artifacts ───────────────────────────────────────────────

print("Loading FCN Wang model and artifacts...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

scaler = joblib.load(str(settings.SCALER_PATH))
mlb    = joblib.load(str(settings.MLB_PATH))
CLASSES: list[str] = list(mlb.classes_)
N_CLASSES = len(CLASSES)

model = fcn_wang(num_classes=N_CLASSES, input_channels=12, lin_ftrs_head=[128])
checkpoint = torch.load(
    str(settings.MODEL_WEIGHTS_PATH), map_location=device, weights_only=False
)
state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
clean_sd = {
    (k.replace("model.", "") if k.startswith("model.") else k): v
    for k, v in state_dict.items()
}
model.load_state_dict(clean_sd)
model.to(device)
model.eval()
print(f"Model ready on {device}  |  {N_CLASSES} classes")

# ── Load PTB-XL metadata ───────────────────────────────────────────────────

print(f"\nLoading PTB-XL metadata from {PTBXL_CSV}...")
df = pd.read_csv(PTBXL_CSV)
test_df = df[df["strat_fold"] == 10].reset_index(drop=True)
print(f"Test set: {len(test_df)} records (strat_fold=10)")


def _parse_scp(raw: str) -> list[str]:
    """Parse scp_codes string to list of codes with confidence > 0."""
    try:
        codes = ast.literal_eval(raw)
        return [k for k, v in codes.items() if float(v) > 0]
    except Exception:
        return []


# ── Build ground-truth matrix ──────────────────────────────────────────────

print("Building ground-truth label matrix...")
gt_codes = [_parse_scp(row) for row in test_df["scp_codes"]]

# Binarize using the same MLB the model was trained with
y_true = np.zeros((len(test_df), N_CLASSES), dtype=np.float32)
for i, codes in enumerate(gt_codes):
    for code in codes:
        if code in CLASSES:
            y_true[i, CLASSES.index(code)] = 1.0

print(f"Label matrix: {y_true.shape}  |  "
      f"{int(y_true.sum())} total positive labels")

# ── Run inference ──────────────────────────────────────────────────────────

print("\nRunning inference on test set...")
y_prob         = np.zeros((len(test_df), N_CLASSES), dtype=np.float32)
processed_mask = np.zeros(len(test_df), dtype=bool)
skipped        = 0

for rel_idx, row in enumerate(test_df.itertuples(index=False)):
    filename  = row.filename_lr
    full_path = os.path.join(PTBXL_ROOT, filename)

    try:
        record = wfdb.rdsamp(full_path)
        signal = record[0].astype(np.float32)   # (n_samples, 12)

        if signal.shape[0] != TARGET_LEN:
            signal = scipy.signal.resample(signal, TARGET_LEN)

        flat   = signal.reshape(-1, 1)
        scaled = scaler.transform(flat).reshape(signal.shape)

        x = torch.tensor(scaled.T[np.newaxis], dtype=torch.float32).to(device)

        with torch.no_grad():
            logits = model(x)
            probs  = torch.sigmoid(logits).cpu().numpy()[0]

        y_prob[rel_idx]         = probs
        processed_mask[rel_idx] = True

    except Exception as exc:
        skipped += 1
        if skipped <= 5:
            print(f"  Skipped {filename}: {exc}")

print(f"Inference complete. Skipped {skipped}/{len(test_df)} records.")

y_true_v = y_true[processed_mask]
y_prob_v = y_prob[processed_mask]
n_valid  = int(processed_mask.sum())
print(f"Valid records for metrics: {n_valid}")

# ── Per-class AUC ─────────────────────────────────────────────────────────

print("\nComputing per-class AUC-ROC...")
per_class: dict[str, dict] = {}
aucs: list[float] = []

for j, code in enumerate(CLASSES):
    pos = int(y_true_v[:, j].sum())
    if pos == 0:
        per_class[code] = {"auc": None, "support": 0, "note": "no positive samples"}
        continue

    try:
        auc = float(roc_auc_score(y_true_v[:, j], y_prob_v[:, j]))
        ap  = float(average_precision_score(y_true_v[:, j], y_prob_v[:, j]))
        per_class[code] = {
            "auc":     round(auc, 4),
            "avg_precision": round(ap, 4),
            "support": pos,
        }
        aucs.append(auc)
    except Exception as exc:
        per_class[code] = {"auc": None, "support": pos, "note": str(exc)}

macro_auc = float(np.mean(aucs)) if aucs else None
print(f"Macro AUC-ROC: {macro_auc:.4f}  ({len(aucs)} classes with positive samples)")

# ── Threshold-based accuracy and F1 ───────────────────────────────────────

THRESHOLD = 0.5
y_pred_bin = (y_prob_v >= THRESHOLD).astype(int)

# Exact-match (subset) accuracy
exact_acc = float(accuracy_score(y_true_v, y_pred_bin))

# Micro / macro F1 across all classes
from sklearn.metrics import f1_score
micro_f1 = float(f1_score(y_true_v, y_pred_bin, average="micro",  zero_division=0))
macro_f1 = float(f1_score(y_true_v, y_pred_bin, average="macro",  zero_division=0))

print(f"Exact-match accuracy : {exact_acc * 100:.2f}%")
print(f"Micro F1             : {micro_f1 * 100:.2f}%")
print(f"Macro F1             : {macro_f1 * 100:.2f}%")

# ── Bootstrap 95% CI on macro AUC ─────────────────────────────────────────

print("\nBootstrap CI on macro AUC (500 iterations)...")
N_BOOT = 500
boot_aucs: list[float] = []
idx_all = np.arange(n_valid)

for _ in range(N_BOOT):
    idx_b     = bootstrap_resample(idx_all, replace=True)
    fold_aucs = []
    for j in range(N_CLASSES):
        pos = int(y_true_v[idx_b, j].sum())
        if pos == 0 or pos == len(idx_b):
            continue
        try:
            fold_aucs.append(roc_auc_score(y_true_v[idx_b, j], y_prob_v[idx_b, j]))
        except Exception:
            pass
    if fold_aucs:
        boot_aucs.append(float(np.mean(fold_aucs)))

ci_low  = float(np.percentile(boot_aucs, 2.5))  if boot_aucs else None
ci_high = float(np.percentile(boot_aucs, 97.5)) if boot_aucs else None
print(f"Macro AUC 95% CI: [{ci_low:.4f}, {ci_high:.4f}]")

# ── Top-10 classes by support ──────────────────────────────────────────────

print("\n--- Top 10 classes by support ---")
top10 = sorted(
    [(c, d) for c, d in per_class.items() if d.get("auc") is not None],
    key=lambda x: -x[1]["support"]
)[:10]
for code, d in top10:
    print(f"  {code:<10} support={d['support']:4d}  AUC={d['auc']:.3f}")

# ── Save results ───────────────────────────────────────────────────────────

os.makedirs("ml/results", exist_ok=True)
metrics = {
    "model":           "FCN Wang (PTB-XL, 71-class, 12-lead)",
    "dataset":         "PTB-XL strat_fold=10 (held-out test set)",
    "n_records":       n_valid,
    "n_records_total": len(test_df),
    "n_classes":       N_CLASSES,
    "primary_metric":  "macro_auc",
    "macro_auc":       round(macro_auc, 4) if macro_auc else None,
    "macro_auc_ci": {
        "low":  round(ci_low,  4) if ci_low  is not None else None,
        "high": round(ci_high, 4) if ci_high is not None else None,
    },
    "threshold":       THRESHOLD,
    "macro_f1":        round(macro_f1 * 100, 2),
    "micro_f1":        round(micro_f1 * 100, 2),
    "exact_match_acc": round(exact_acc * 100, 2),
    "note": (
        "Primary metric is macro AUC (threshold-free discrimination). "
        "Macro F1 is low due to class imbalance across 71 labels at a fixed 0.5 threshold "
        "— this is expected for multi-label classification and does not indicate poor discrimination."
    ),
    "per_class":       per_class,
}

out_path = "ml/results/fcn_wang_metrics.json"
with open(out_path, "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\nSaved to {out_path}")
print(f"\n=== FCN Wang Test Results ===")
print(f"  Macro AUC : {macro_auc:.4f}  (95% CI: {ci_low:.4f}–{ci_high:.4f})")
print(f"  Macro F1  : {macro_f1 * 100:.2f}%")
print(f"  Micro F1  : {micro_f1 * 100:.2f}%")
print(f"  Records   : {n_valid}")
