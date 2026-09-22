import os
import h5py
import numpy as np
import torch
from tqdm import tqdm

from config import Config


def convert_split(h5_path, split):
    print(f"Processing {split}: {h5_path}")
    print(f" -> T={Config.TIME_STEPS}; bin width={Config.BIN_DT_MS:.4f} ms")

    with h5py.File(h5_path, "r") as handle:
        times_ds = handle["spikes"]["times"]
        units_ds = handle["spikes"]["units"]
        labels_ds = handle["labels"]

        sample_count = len(labels_ds)
        x_all = torch.zeros(
            sample_count,
            Config.TIME_STEPS,
            Config.INPUT_DIM,
            dtype=torch.float32,
        )
        y_all = torch.zeros(sample_count, dtype=torch.long)

        total_events = 0
        dropped_events = 0

        for index in tqdm(range(sample_count), desc=f"Rasterizing {split}"):
            times = np.asarray(times_ds[index], dtype=np.float32)
            units = np.asarray(units_ds[index], dtype=np.int64)

            if len(times) > 0 and float(np.max(times)) <= 10.0:
                times = times * 1000.0

            bins = np.floor(times / Config.BIN_DT_MS).astype(np.int64)
            valid = (
                (bins >= 0)
                & (bins < Config.TIME_STEPS)
                & (units >= 0)
                & (units < Config.INPUT_DIM)
            )

            total_events += len(times)
            dropped_events += int((~valid).sum())

            bins = bins[valid]
            units = units[valid]

            if len(bins) > 0:
                flat_indices = torch.from_numpy(
                    bins * Config.INPUT_DIM + units
                ).long()
                counts = torch.bincount(
                    flat_indices,
                    minlength=Config.TIME_STEPS * Config.INPUT_DIM,
                ).reshape(Config.TIME_STEPS, Config.INPUT_DIM)
                x_all[index] = counts.float()

            y_all[index] = int(labels_ds[index])

    drop_ratio = dropped_events / max(1, total_events)
    print(f"{split}: shape={tuple(x_all.shape)}")
    print(f"  mean count={x_all.mean().item():.6f}")
    print(f"  max count={x_all.max().item():.1f}")
    print(f"  nonzero ratio={(x_all > 0).float().mean().item():.4%}")
    print(f"  multi-spike ratio={(x_all > 1).float().mean().item():.4%}")
    print(f"  dropped events={dropped_events}/{total_events} ({drop_ratio:.4%})")

    return {
        "x": x_all,
        "y": y_all,
        "meta": {
            "dataset": "SHD",
            "split": split,
            "time_steps": Config.TIME_STEPS,
            "bin_size_ms": Config.BIN_DT_MS,
            "input_dim": Config.INPUT_DIM,
            "aggregation": "raw spike count",
            "log1p": False,
            "normalized": False,
            "scaled": False,
            "clipped": False,
            "drop_ratio": drop_ratio,
        },
    }


def main():
    os.makedirs(Config.PROCESSED_DIR, exist_ok=True)

    train = convert_split(
        os.path.join(Config.RAW_DIR, "shd_train.h5"),
        "train",
    )
    test = convert_split(
        os.path.join(Config.RAW_DIR, "shd_test.h5"),
        "test",
    )

    train_path = os.path.join(Config.PROCESSED_DIR, Config.TRAIN_FILE)
    test_path = os.path.join(Config.PROCESSED_DIR, Config.TEST_FILE)
    torch.save(train, train_path)
    torch.save(test, test_path)

    print(f"Saved: {train_path}")
    print(f"Saved: {test_path}")


if __name__ == "__main__":
    main()
