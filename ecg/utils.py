"""ECG tensor loading, random-state control, and evaluation metrics."""
import os
import random
import numpy as np
import torch
import torch.utils.data as data
from config import Config

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

class deterministic_eval:
    def __init__(self, seed=42):
        self.seed = seed
        
    def __enter__(self):
        self.cpu_rng_state = torch.get_rng_state()
        if torch.cuda.is_available():
            self.gpu_rng_states = torch.cuda.get_rng_state_all()
        self.np_state = np.random.get_state()
        self.py_state = random.getstate()
        
        set_seed(self.seed)
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.set_rng_state(self.cpu_rng_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(self.gpu_rng_states)
        np.random.set_state(self.np_state)
        random.setstate(self.py_state)

def calculate_nmse_db(pred: torch.Tensor, target: torch.Tensor) -> float:
    error_energy = torch.sum((target - pred) ** 2)
    target_energy = torch.sum(target ** 2) + 1e-12
    return (10 * torch.log10(error_energy / target_energy)).item()

class MITBIH_Dataset(data.Dataset):
    def __init__(self, split: str = "train"):
        filename = {"train": Config.TRAIN_FILE, "val": Config.VAL_FILE, "test": Config.TEST_FILE}[split]
        file_path = os.path.join(Config.DATA_DIR, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Could not find ECG PT file: {file_path}")
        print(f"Loading {split} data from: {file_path}")
        raw = torch.load(file_path, weights_only=True)
        if isinstance(raw, dict):
            if "data" in raw:
                raw = raw["data"]
            else:
                first_tensor = next((v for v in raw.values() if torch.is_tensor(v)), None)
                if first_tensor is None:
                    raise ValueError("Unsupported dict format in ECG pt file.")
                raw = first_tensor
        elif isinstance(raw, (list, tuple)):
            raw = raw[0]
        if raw.dim() != 2:
            raise ValueError(f"Expected static ECG tensor [num_beats, {Config.N}], got {tuple(raw.shape)}")
        if raw.size(1) != Config.N:
            raise ValueError(f"Expected beat length {Config.N}, got {raw.size(1)}")
        self.data = raw.float()
        print(f"Loaded {split} set: {tuple(self.data.shape)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], torch.tensor(0.0)

def get_dataloaders(eval_split="val"):
    use_pin = torch.cuda.is_available()
    train_dataset = MITBIH_Dataset("train")
    val_dataset = MITBIH_Dataset(eval_split)
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=use_pin,
    )
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=use_pin,
    )
    return train_loader, val_loader


def atomic_torch_save(payload, path):
    temporary = str(path) + ".tmp"
    torch.save(payload, temporary)
    os.replace(temporary, path)
