"""
ml/download_ptbxl_missing.py

Downloads only the PTB-XL test-set records (strat_fold=10) that are
missing from the local dataset — specifically the records100 subfolders
09000 through 21000.

PhysioNet does not require login for PTB-XL (open-access).

Run from repo root:
    python ml/download_ptbxl_missing.py

After completion, re-run ml/evaluate_fcn_wang.py to get full test metrics.
"""

import ast
import os
import sys
import time

import pandas as pd
import requests

PTBXL_ROOT  = "data/raw/ptbxl/physionet.org/files/ptb-xl/1.0.3"
PTBXL_CSV   = f"{PTBXL_ROOT}/ptbxl_database.csv"
BASE_URL     = "https://physionet.org/files/ptb-xl/1.0.3"
EXTENSIONS   = [".hea", ".dat"]

df      = pd.read_csv(PTBXL_CSV)
test_df = df[df["strat_fold"] == 10].reset_index(drop=True)

# Collect only missing files
to_download: list[tuple[str, str]] = []  # (local_path, url)

for row in test_df.itertuples(index=False):
    base = f"{PTBXL_ROOT}/{row.filename_lr}"
    for ext in EXTENSIONS:
        local = base + ext
        if not os.path.exists(local):
            url = f"{BASE_URL}/{row.filename_lr}{ext}"
            to_download.append((local, url))

if not to_download:
    print("All test-set records already present. Nothing to download.")
    sys.exit(0)

print(f"Files to download: {len(to_download)}")
print(f"(approx. {len(to_download) // 2} records)\n")

session = requests.Session()
session.headers["User-Agent"] = "cardiac-fyp/1.0"

ok = 0
fail = 0

for i, (local_path, url) in enumerate(to_download, 1):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    try:
        resp = session.get(url, timeout=30)
        if resp.status_code == 200:
            with open(local_path, "wb") as f:
                f.write(resp.content)
            ok += 1
        else:
            print(f"  [{i}/{len(to_download)}] HTTP {resp.status_code} — {url}")
            fail += 1
    except Exception as exc:
        print(f"  [{i}/{len(to_download)}] ERROR {exc} — {url}")
        fail += 1

    if i % 100 == 0:
        print(f"  Progress: {i}/{len(to_download)}  ({ok} ok, {fail} failed)")
        time.sleep(0.5)  # be polite to PhysioNet

print(f"\nDone. {ok} downloaded, {fail} failed.")
print("Now run:  python ml/evaluate_fcn_wang.py")
