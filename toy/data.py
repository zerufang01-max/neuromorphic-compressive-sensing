"""Deterministic Gaussian sensing and online exact-sparse signals."""

import math
import random
import numpy as np
import torch

from config import NUMERICAL_EPS


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generator(seed, device):
    rng = torch.Generator(device=device)
    rng.manual_seed(int(seed))
    return rng


def measurement_matrix(n, m, max_m, seed, device):
    """Take the first m rows of one shared max_m-by-n Gaussian operator."""
    rng = generator(seed, torch.device("cpu"))
    master = torch.randn(max_m, n, generator=rng) / math.sqrt(max_m)
    matrix = master[:m]
    matrix = matrix / matrix.norm(
        dim=0, keepdim=True).clamp_min(NUMERICAL_EPS)
    return matrix.to(device)


def sparse_batch(size, sparsity, matrix, rng, amplitude_min, amplitude_max):
    n = matrix.shape[1]
    support = torch.rand(
        size, n, generator=rng, device=matrix.device
    ).topk(sparsity, dim=1).indices
    signs = 2 * torch.randint(
        0, 2, (size, sparsity), generator=rng,
        device=matrix.device, dtype=torch.int64
    ).float() - 1
    magnitudes = amplitude_min + (amplitude_max - amplitude_min) * torch.rand(
        size, sparsity, generator=rng, device=matrix.device)
    target = torch.zeros(size, n, device=matrix.device)
    target.scatter_(1, support, signs * magnitudes)
    return target, target @ matrix.t()
