"""Transmitter (EdgeSensor of baselines/edge_compression.py) with explicit settings.

The module structure, parameter and buffer names are the same, so the baseline
checkpoints load with strict=True.  The forward computation is the same; the
per-step diagnostic statistics are left out (they do not affect the outputs).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as C


class SurrogateSpike(torch.autograd.Function):
    """Heaviside(x) with a rectangular surrogate of half-width 0.5."""

    @staticmethod
    def forward(ctx, inputs):
        ctx.save_for_backward(inputs)
        return (inputs >= 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (inputs,) = ctx.saved_tensors
        return grad_output.clone() * (inputs.abs() < 0.5).float()


class ProbabilityUniformQuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, bits):
        levels = (1 << int(bits)) - 1
        return torch.round(torch.clamp(inputs, 0.0, 1.0) * levels) / levels

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class LIFEncoder(nn.Module):
    def __init__(self, threshold):
        super().__init__()
        self.register_buffer("thresh", torch.tensor(float(threshold)))
        self.register_buffer("decay", torch.tensor(0.0))
        self.act = SurrogateSpike.apply

    def forward(self, x, state=None):
        v = torch.zeros_like(x) if state is None else state
        v = v * self.decay + x
        spike = self.act(v - self.thresh)
        v = v - spike * self.thresh
        return spike, v


class EdgeSensor(nn.Module):
    """Conv16(s2) -> LIF -> Conv32 -> LIF -> Conv_C(s2) -> sigmoid -> output LIF.

    ``tx_threshold`` is the output LIF threshold (theta_tx of Eq. (1));
    ``internal_threshold`` sets the two internal LIF layers (default: the same
    value, as for the baselines, where ENCODER_THRESHOLD sets all three).  All thresholds are
    buffers, so a loaded checkpoint carries its own values.
    """

    def __init__(self, meas_channels=4, tx_threshold=0.25, tx_rho=0.1, quant_bits=8, internal_threshold=None):
        super().__init__()
        self.mode = "ann"
        self.tx_rho = float(tx_rho)
        self.quant_bits = int(quant_bits)

        def conv(in_c, out_c, stride):
            return nn.utils.weight_norm(nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False))

        inner = tx_threshold if internal_threshold is None else internal_threshold
        self.encoder = nn.Sequential(
            conv(C.Config.INPUT_CHANNELS, 16, 2), nn.BatchNorm2d(16), LIFEncoder(inner),
            conv(16, 32, 1), nn.BatchNorm2d(32), LIFEncoder(inner),
            conv(32, int(meas_channels), 2),
        )
        for m in self.encoder.modules():
            if isinstance(m, nn.Conv2d):
                weight = m.weight_v if hasattr(m, "weight_v") else m.weight
                nn.init.kaiming_normal_(weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.signal_gain = 1.0
        self.lif_out = LIFEncoder(tx_threshold)

    @property
    def threshold(self):
        return float(self.lif_out.thresh)

    def forward(self, x_img, state=None, mode="ann", tau=None):
        s_lif1, s_lif2, s_final = state if state is not None else (None, None, None)
        feat = x_img
        lif_index = 0
        next1 = next2 = None
        for layer in self.encoder:
            if isinstance(layer, LIFEncoder):
                if mode == "ann":
                    feat = F.relu(feat)
                elif lif_index == 0:
                    feat, next1 = layer(feat, s_lif1)
                else:
                    feat, next2 = layer(feat, s_lif2)
                lif_index += 1
            else:
                feat = layer(feat)
        feat_sig = torch.sigmoid(feat * self.signal_gain)

        if mode == "ann":
            y_out = ProbabilityUniformQuantizeSTE.apply(feat_sig.view(feat_sig.size(0), -1), self.quant_bits)
            return y_out, (None, None, None), {"tx_prob": None, "tx_hard": y_out}

        prev_v = torch.zeros_like(feat_sig) if s_final is None else s_final
        v = prev_v * self.tx_rho + feat_sig
        theta = self.lif_out.thresh
        tau = C.SLISTA_TRAINING["tau_final"] if tau is None else tau
        p = torch.sigmoid((v - theta) / tau)
        s_hard = (v >= theta).float()
        s_st = s_hard - p.detach() + p
        v_next = v - s_hard * theta
        flat = s_hard.view(s_hard.size(0), -1)
        aux = {"tx_prob": p.view(p.size(0), -1), "tx_hard": flat}
        return s_st.view(s_st.size(0), -1), (next1, next2, v_next), aux
