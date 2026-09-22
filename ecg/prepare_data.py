#!/usr/bin/env python3
"""Convert MIT-BIH WFDB records to R-peak-centred ECG tensors.

Defaults use the de Chazal DS1/DS2 partition and 256-sample beats. AAMI
labels and record metadata are retained alongside the reconstruction inputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

try:
    import wfdb
except ImportError as e:
    raise SystemExit("This script requires wfdb. Install it with: pip install wfdb") from e


# -----------------------------------------------------------------------------
# Common record sets
# -----------------------------------------------------------------------------
ALL_RECORDS = [
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119", "121",
    "122", "123", "124", "200", "201", "202", "203", "205", "207", "208",
    "209", "210", "212", "213", "214", "215", "217", "219", "220", "221",
    "222", "223", "228", "230", "231", "232", "233", "234",
]
PACED_RECORDS = {"102", "104", "107", "217"}
NON_PACED_RECORDS = [r for r in ALL_RECORDS if r not in PACED_RECORDS]

DE_CHAZAL_DS1 = [
    "101", "106", "108", "109", "112", "114", "115", "116", "118", "119", "122",
    "124", "201", "203", "205", "207", "208", "209", "215", "220", "223", "230",
]
DE_CHAZAL_DS2 = [
    "100", "103", "105", "111", "113", "117", "121", "123", "200", "202", "210",
    "212", "213", "214", "219", "221", "222", "228", "231", "232", "233", "234",
]


# -----------------------------------------------------------------------------
# AAMI mapping kept as optional metadata
# -----------------------------------------------------------------------------
AAMI_MAP: Dict[str, str] = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    "A": "S", "a": "S", "J": "S", "S": "S",
    "V": "V", "E": "V",
    "F": "F",
    "/": "Q", "f": "Q", "Q": "Q", "?": "Q",
}
AAMI_CLASS_TO_INT = {"N": 0, "S": 1, "V": 2, "F": 3, "Q": 4}


@dataclass
class BeatMeta:
    record: str
    sample: int
    symbol: str
    aami_class: str
    label: int
    lead_name: str
    fs: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare MIT-BIH into beat-level .pt files")
    parser.add_argument("--db-name", type=str, default="mitdb", help="PhysioNet database name")
    parser.add_argument("--download", action="store_true", help="Download WFDB files into --raw-dir")
    parser.add_argument("--raw-dir", type=str, default=str(Path(__file__).resolve().parent / "data/mitdb_raw"), help="Raw WFDB directory")
    parser.add_argument("--out-dir", type=str, default=str(Path(__file__).resolve().parent / "data/mitdb_pt"), help="Output directory")

    parser.add_argument(
        "--split",
        type=str,
        default="de_chazal",
        choices=["de_chazal", "all_as_train", "custom"],
        help="Patient split convention",
    )
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--train-records", type=str, default="", help="Comma-separated custom train records")
    parser.add_argument("--test-records", type=str, default="", help="Comma-separated custom test records")

    parser.add_argument("--lead", type=str, default="MLII", help="Preferred extraction lead")
    parser.add_argument(
        "--allow-fallback-lead",
        action="store_true",
        help="Use another available lead when the preferred lead is absent",
    )

    parser.add_argument("--window-left", type=int, default=99, help="Samples before the R peak")
    parser.add_argument("--window-right", type=int, default=160, help="Samples after the R peak")
    parser.add_argument(
        "--output-len",
        type=int,
        default=256,
        help="Final beat length (default: 256). Use <=0 to keep the raw extraction length.",
    )

    parser.add_argument(
        "--normalize",
        type=str,
        default="none",
        choices=["none", "record_zscore", "beat_zscore", "beat_minmax"],
        help="Normalization policy",
    )
    parser.add_argument("--drop-q-class", action="store_true", help="Drop AAMI Q beats")
    parser.add_argument(
        "--save-per-record-json",
        action="store_true",
        help="Also save a compact JSON summary per split",
    )
    return parser.parse_args()


def maybe_download_database(db_name: str, raw_dir: str) -> None:
    os.makedirs(raw_dir, exist_ok=True)
    has_header = any(name.endswith(".hea") for name in os.listdir(raw_dir))
    if has_header:
        print(f"[Info] Raw directory already contains WFDB files: {raw_dir}")
        return
    print(f"[Info] Downloading {db_name} into {raw_dir} ...")
    wfdb.dl_database(db_name, dl_dir=raw_dir)
    print("[Info] Download complete.")


def validate_local_records(records: Sequence[str], raw_dir: str) -> None:
    """Require a complete local WFDB triplet for every requested record."""
    missing: List[str] = []
    for record_name in records:
        for suffix in ("hea", "dat", "atr"):
            path = os.path.join(raw_dir, f"{record_name}.{suffix}")
            if not os.path.isfile(path):
                missing.append(path)

    if missing:
        preview = "\n  ".join(missing[:20])
        remainder = len(missing) - min(len(missing), 20)
        extra = f"\n  ... and {remainder} more" if remainder else ""
        raise FileNotFoundError(
            "The local MIT-BIH dataset is incomplete. Each record requires .hea, .dat, "
            f"and .atr files. Missing:\n  {preview}{extra}\n"
            "Use --download to fetch the complete database, or choose a custom split "
            "containing only complete records."
        )


def get_split_records(args: argparse.Namespace) -> Tuple[List[str], List[str]]:
    if args.split == "de_chazal":
        return DE_CHAZAL_DS1.copy(), DE_CHAZAL_DS2.copy()
    if args.split == "all_as_train":
        return NON_PACED_RECORDS.copy(), []
    if args.split == "custom":
        train_records = [r.strip() for r in args.train_records.split(",") if r.strip()]
        test_records = [r.strip() for r in args.test_records.split(",") if r.strip()]
        if not train_records:
            raise ValueError("--split custom requires --train-records")
        return train_records, test_records
    raise ValueError(f"Unknown split: {args.split}")


def choose_lead(sig_names: Sequence[str], preferred: str, allow_fallback: bool) -> int:
    sig_names = list(sig_names)
    if preferred in sig_names:
        return sig_names.index(preferred)

    alternatives = ["MLII", "II", "V1", "V5", "V2", "ECG1", "ECG2"]
    for alt in alternatives:
        if alt in sig_names:
            if allow_fallback:
                return sig_names.index(alt)
            raise ValueError(
                f"Preferred lead '{preferred}' not found. Available leads: {sig_names}. "
                f"Pass --allow-fallback-lead to use '{alt}'."
            )

    if allow_fallback and len(sig_names) > 0:
        return 0
    raise ValueError(
        f"Preferred lead '{preferred}' not found and no fallback allowed. Available leads: {sig_names}."
    )


def normalize_signal(x: np.ndarray, mode: str) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    if mode == "none":
        return x
    if mode == "beat_zscore":
        mu = float(x.mean())
        sd = float(x.std())
        return (x - mu) / (sd + 1e-8)
    if mode == "beat_minmax":
        lo = float(x.min())
        hi = float(x.max())
        return 2.0 * (x - lo) / (hi - lo + 1e-8) - 1.0
    raise ValueError(f"normalize_signal only supports beat-level modes here, got {mode}")


def resample_1d(x: np.ndarray, out_len: int) -> np.ndarray:
    if out_len <= 0 or len(x) == out_len:
        return x.astype(np.float32, copy=False)
    old_grid = np.linspace(0.0, 1.0, num=len(x), endpoint=True)
    new_grid = np.linspace(0.0, 1.0, num=out_len, endpoint=True)
    return np.interp(new_grid, old_grid, x).astype(np.float32)


def extract_beats_from_record(record_name: str, args: argparse.Namespace):
    record_path = os.path.join(args.raw_dir, record_name)
    record = wfdb.rdrecord(record_path)
    ann = wfdb.rdann(record_path, "atr")

    lead_idx = choose_lead(record.sig_name, args.lead, args.allow_fallback_lead)
    lead_name = record.sig_name[lead_idx]
    signal = record.p_signal[:, lead_idx].astype(np.float32)
    fs = float(record.fs)

    if args.normalize == "record_zscore":
        signal = (signal - signal.mean()) / (signal.std() + 1e-8)

    beats: List[np.ndarray] = []
    labels: List[int] = []
    metas: List[BeatMeta] = []
    class_counter = {k: 0 for k in ["N", "S", "V", "F", "Q"]}

    left = int(args.window_left)
    right = int(args.window_right)
    raw_len = left + right
    final_len = raw_len if int(args.output_len) <= 0 else int(args.output_len)

    for sample_idx, symbol in zip(ann.sample, ann.symbol):
        if symbol not in AAMI_MAP:
            continue
        aami = AAMI_MAP[symbol]
        if args.drop_q_class and aami == "Q":
            continue

        start = int(sample_idx) - left
        end = int(sample_idx) + right
        if start < 0 or end > len(signal):
            continue

        beat = signal[start:end].copy()
        if len(beat) != raw_len:
            continue

        beat = normalize_signal(beat, args.normalize if args.normalize != "record_zscore" else "none")
        beat = resample_1d(beat, final_len)

        label = AAMI_CLASS_TO_INT[aami]
        beats.append(beat)
        labels.append(label)
        metas.append(
            BeatMeta(
                record=record_name,
                sample=int(sample_idx),
                symbol=symbol,
                aami_class=aami,
                label=label,
                lead_name=lead_name,
                fs=fs,
            )
        )
        class_counter[aami] += 1

    return beats, labels, metas, class_counter, final_len


def process_record_list(records: Sequence[str], args: argparse.Namespace) -> Dict[str, object]:
    all_beats: List[np.ndarray] = []
    all_labels: List[int] = []
    all_meta: List[BeatMeta] = []
    per_record_summary: Dict[str, Dict[str, int]] = {}
    final_len = None

    for rec in records:
        beats, labels, metas, counter, rec_final_len = extract_beats_from_record(rec, args)
        if final_len is None:
            final_len = rec_final_len
        all_beats.extend(beats)
        all_labels.extend(labels)
        all_meta.extend(metas)
        per_record_summary[rec] = counter
        print(
            f"[Record {rec}] beats={len(beats)} | "
            + ", ".join(f"{k}={counter[k]}" for k in ["N", "S", "V", "F", "Q"])
        )

    if not all_beats:
        raise RuntimeError("No beats extracted. Check lead selection / split / normalization settings.")

    data = torch.from_numpy(np.stack(all_beats, axis=0)).to(torch.float32)
    labels_tensor = torch.tensor(all_labels, dtype=torch.long)
    if data.ndim != 2 or data.shape[0] != labels_tensor.numel():
        raise RuntimeError(
            f"Invalid extracted payload: data shape={tuple(data.shape)}, "
            f"labels shape={tuple(labels_tensor.shape)}"
        )
    if not torch.isfinite(data).all():
        raise RuntimeError("Extracted beats contain NaN or Inf values.")

    active_class_to_int = {
        name: index for name, index in AAMI_CLASS_TO_INT.items()
        if not (args.drop_q_class and name == "Q")
    }
    class_counts = {
        name: int((labels_tensor == index).sum().item())
        for name, index in active_class_to_int.items()
    }

    payload: Dict[str, object] = {
        "data": data,
        "records": [m.record for m in all_meta],
        "samples": torch.tensor([m.sample for m in all_meta], dtype=torch.int64),
        "lead_names": [m.lead_name for m in all_meta],
        "fs": torch.tensor([m.fs for m in all_meta], dtype=torch.float32),
        "config": {
            "db_name": args.db_name,
            "split": args.split,
            "lead": args.lead,
            "allow_fallback_lead": bool(args.allow_fallback_lead),
            "window_left": int(args.window_left),
            "window_right": int(args.window_right),
            "raw_window_len": int(args.window_left + args.window_right),
            "output_len": int(final_len),
            "normalize": args.normalize,
            "drop_q_class": bool(args.drop_q_class),
            "save_labels": True,
            "task": "beat_reconstruction_and_aami_classification",
        },
        "per_record_summary": per_record_summary,
        "labels": labels_tensor,
        "symbols": [m.symbol for m in all_meta],
        "aami_classes": [m.aami_class for m in all_meta],
        "class_to_int": active_class_to_int,
        "class_names": list(active_class_to_int.keys()),
        "num_classes": len(active_class_to_int),
        "class_counts": class_counts,
    }

    return payload


def save_payload(payload: Dict[str, object], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp_path = out_path + ".tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, out_path)
    print(f"[Saved] {out_path}")


def save_summary_json(payload: Dict[str, object], out_path: str) -> None:
    summary = {
        "num_beats": int(payload["data"].shape[0]),
        "signal_len": int(payload["data"].shape[1]),
        "config": payload["config"],
        "per_record_summary": payload["per_record_summary"],
    }

    summary["class_counts"] = payload["class_counts"]

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[Saved] {out_path}")


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.download:
        maybe_download_database(args.db_name, args.raw_dir)

    train_records, test_records = get_split_records(args)
    import random
    if not 0 < args.val_fraction < 1:
        raise ValueError("val-fraction must be between zero and one")
    groups = {}
    for rec in train_records:
        groups.setdefault("201_202" if rec in ("201", "202") else rec, []).append(rec)
    keys = sorted(groups)
    random.Random(args.split_seed).shuffle(keys)
    count = max(1, round(len(keys) * args.val_fraction))
    if count >= len(keys):
        raise ValueError("Too few records for train/validation split")
    val_records = sorted(rec for key in keys[:count] for rec in groups[key])
    train_records = [rec for rec in train_records if rec not in val_records]
    overlap = sorted(set(train_records + val_records) & set(test_records))
    if overlap:
        raise ValueError(f"Train/test record leakage detected: {overlap}")
    requested_records = list(dict.fromkeys(train_records + val_records + test_records))
    validate_local_records(requested_records, args.raw_dir)

    manifest = {
        "db_name": args.db_name,
        "train_records": train_records,
        "val_records": val_records,
        "test_records": test_records,
        "split_seed": args.split_seed,
        "validation_group_fraction": args.val_fraction,
        "known_cross_partition_subject": "201/202 retained from DS1/DS2",
        "paced_records_excluded": sorted(list(PACED_RECORDS)),
        "aami_class_to_int": AAMI_CLASS_TO_INT,
        "save_labels": True,
        "output_len": int(args.output_len),
        "task": "beat_reconstruction_and_aami_classification",
    }
    manifest_path = os.path.join(args.out_dir, "split_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[Saved] {manifest_path}")

    train_payload = process_record_list(train_records, args)
    save_payload(train_payload, os.path.join(args.out_dir, "train_data.pt"))
    if args.save_per_record_json:
        save_summary_json(train_payload, os.path.join(args.out_dir, "train_summary.json"))

    val_payload = process_record_list(val_records, args)
    save_payload(val_payload, os.path.join(args.out_dir, "val_data.pt"))
    print("Train/val beats:", len(train_payload["data"]), len(val_payload["data"]))
    if test_records:
        test_payload = process_record_list(test_records, args)
        save_payload(test_payload, os.path.join(args.out_dir, "test_data.pt"))
        if args.save_per_record_json:
            save_summary_json(test_payload, os.path.join(args.out_dir, "test_summary.json"))

    manifest["sample_counts"] = {"train": len(train_payload["data"]), "val": len(val_payload["data"]), "test": len(test_payload["data"]) if test_records else 0}
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print("Sample counts:", manifest["sample_counts"])
    print("[Done] Beat-level PT preparation finished.")


if __name__ == "__main__":
    main()
