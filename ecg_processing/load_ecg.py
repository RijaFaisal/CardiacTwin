"""
WFDB helpers: download MITDB and load local records.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import wfdb
from wfdb import Annotation

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MITDB_DIR = PROJECT_ROOT / "data" / "raw" / "mitdb"
DEFAULT_MITBIH_DIR = PROJECT_ROOT / "data" / "raw" / "mitbih"

PathLike = Union[str, Path]


def download_mitdb(
    dl_dir: PathLike | None = None,
    *,
    records: str | list[str] = "all",
    annotators: str | list[str] | None = "all",
    keep_subdirs: bool = False,
    overwrite: bool = False,
) -> Path:
    root = Path(dl_dir).resolve() if dl_dir is not None else DEFAULT_MITDB_DIR
    root.mkdir(parents=True, exist_ok=True)
    wfdb.dl_database(
        "mitdb",
        str(root),
        records=records,
        annotators=annotators,
        keep_subdirs=keep_subdirs,
        overwrite=overwrite,
    )
    return root


def load_record(record_path: PathLike) -> tuple[np.ndarray, float, Annotation]:
    """
    Load one WFDB record from a **path stem** (no ``.hea`` / ``.dat`` suffix).

    Examples
    --------
    ``load_record("data/raw/mitdb/100")``
    ``load_record(PROJECT_ROOT / "data" / "raw" / "mitbih" / "100")``

    Returns
    -------
    signal:
        ``(n_samples, n_channels)`` — typically mV in ``p_signal``.
    fs:
        Sampling rate in Hz.
    annotations:
        Beat annotations from ``.atr`` (MIT-BIH reference).
    """
    stem = Path(record_path).expanduser()
    if not stem.is_absolute():
        stem = (Path.cwd() / stem).resolve()
    else:
        stem = stem.resolve()

    hea = stem.with_suffix(".hea")
    if not hea.exists():
        raise FileNotFoundError(
            f"No header at {hea}. Check path (cwd={Path.cwd()}) and record id."
        )

    record = wfdb.rdrecord(str(stem))
    if record.p_signal is not None:
        signal = np.asarray(record.p_signal, dtype=np.float64)
    elif record.d_signal is not None:
        signal = np.asarray(record.d_signal, dtype=np.float64)
    else:
        raise RuntimeError(f"No signal array for record stem {stem}")

    fs = float(record.fs)
    try:
        ann = wfdb.rdann(str(stem), "atr")
    except FileNotFoundError:
        ann = None
    return signal, fs, ann


def load_mitdb_record(
    record_name: str,
    *,
    data_dir: PathLike | None = None,
) -> tuple[np.ndarray, float, Annotation]:
    base = Path(data_dir).resolve() if data_dir is not None else DEFAULT_MITDB_DIR
    return load_record(base / record_name)


if __name__ == "__main__":
    target = DEFAULT_MITDB_DIR
    if not (target / "100.hea").exists():
        print(f"Downloading record 100 into {target} ...")
        download_mitdb(target, records=["100"])
    sig, fs, _annotations = load_record(target / "100")
    print(f"fs: {int(fs)}")
    print(f"signal shape: {sig.shape}")
