"""DVS-Gesture frames (16 x 2 x 128 x 128), with the preprocessing of baselines/utils.py.

The file order is the unsorted glob order used there, so that the evaluation
batches (and hence the evaluation noise) are the same for every method.
"""

import glob
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data

import config as C


class DVSGestureDataset(data.Dataset):
    def __init__(self, split, root=None):
        self.H, self.W = C.Config.IMG_H, C.Config.IMG_W
        target_dir = os.path.join(str(root or C.DATA_DIR), split)
        if not os.path.isdir(target_dir):
            raise FileNotFoundError(target_dir)
        self.file_list, self.labels = [], []
        for class_name in sorted(os.listdir(target_dir)):
            class_dir = os.path.join(target_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            try:
                label = int(class_name)
            except ValueError:
                continue
            for path in glob.glob(os.path.join(class_dir, "*.pt")):
                self.file_list.append(path)
                self.labels.append(label)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        try:
            frames = torch.load(self.file_list[index], weights_only=True)
        except Exception:
            frames = torch.load(self.file_list[index])
        if isinstance(frames, tuple):
            frames = frames[0]
        frames = frames.float()
        if frames.dim() == 3:
            frames = frames.unsqueeze(1)
        if frames.shape[-2:] != (self.H, self.W):
            frames = F.interpolate(frames, size=(self.H, self.W), mode="bilinear", align_corners=False)
        frames = torch.log1p(frames)
        peak = frames.max()
        if peak > 0:
            frames = frames / peak
        return frames, self.labels[index]


def _worker_init(seed):
    def init(worker_id):
        worker_seed = (int(seed) + int(worker_id)) % (2 ** 32)
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    return init


def make_loaders(seed, batch_size=C.BATCH_SIZE, num_workers=C.NUM_WORKERS, root=None):
    """(train_loader, test_loader).  The official test split is the evaluation split."""
    train_set = DVSGestureDataset("train", root)
    test_set = DVSGestureDataset("test", root)
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    init = _worker_init(seed)
    train = data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=True,
                            worker_init_fn=init, generator=generator)
    test = data.DataLoader(test_set, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True, worker_init_fn=init)
    return train, test
