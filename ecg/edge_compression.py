"""Linear ECG measurement operator."""
import math
import torch
import torch.nn as nn
from config import Config

class EdgeSensor(nn.Module):
    def __init__(self):
        super().__init__()
        self.N = Config.N
        self.M = Config.M
        self.device = Config.DEVICE
        self.mode = "shared"
        self.Phi = nn.Parameter(self._generate_matrix(seed=Config.SEED))

    def _generate_matrix(self, seed: int) -> torch.Tensor:
        g_cpu = torch.Generator()
        g_cpu.manual_seed(seed)
        phi = torch.randn(self.M, self.N, generator=g_cpu) * (1.0 / math.sqrt(self.M))
        return phi.to(self.device)

    def active_named_parameters(self, mode: str):
        if Config.CS_MODE == "fixed_wavelet":
            return []
        return [("Phi", self.Phi)]

    def effective_phi(self) -> torch.Tensor:
        return self.Phi

    def compute_measurement(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.size(1) != self.N:
            raise ValueError(f"EdgeSensor expects [B, {self.N}] static ECG beats, got {tuple(x.shape)}")
        return torch.matmul(x, self.effective_phi().t())

    def forward(self, x: torch.Tensor, state=None, mode: str | None = None):
        y_raw = self.compute_measurement(x)
        return y_raw, None, {}
