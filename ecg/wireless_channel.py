import math

import torch
import torch.nn as nn

from config import Config


class SignedUniformQuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, num_bits, clip_value):
        ctx.save_for_backward(inputs)
        ctx.clip_value = clip_value
        x = torch.clamp(inputs, -clip_value, clip_value)
        levels = (1 << num_bits) - 1
        q = torch.round((x / clip_value + 1.0) * 0.5 * levels)
        return (q / levels * 2.0 - 1.0) * clip_value

    @staticmethod
    def backward(ctx, grad_output):
        (inputs,) = ctx.saved_tensors
        grad = grad_output.clone()
        grad[(inputs < -ctx.clip_value) | (inputs > ctx.clip_value)] = 0.0
        return grad, None, None


class SignedUniformBitAWGNHARDSTE(torch.autograd.Function):
    """Uniform quantization and uncoded BPSK over real AWGN.

    Forward uses hard bit decisions. Backward is straight-through inside the
    quantizer clipping interval. SNR denotes Es/N0 per transmitted BPSK bit.
    """

    @staticmethod
    def forward(ctx, inputs, snr_db, num_bits, clip_value):
        ctx.save_for_backward(inputs)
        ctx.clip_value = clip_value
        x = torch.clamp(inputs, -clip_value, clip_value)
        levels = (1 << num_bits) - 1
        q = torch.round((x / clip_value + 1.0) * 0.5 * levels).long()
        q = torch.clamp(q, 0, levels)

        shifts = torch.arange(num_bits, device=inputs.device, dtype=torch.long)
        bits = ((q.unsqueeze(-1) >> shifts) & 1).to(inputs.dtype)
        symbols = bits.mul(2.0).sub(1.0)
        snr_linear = 10.0 ** (float(snr_db) / 10.0)
        sigma = math.sqrt(1.0 / (2.0 * snr_linear))
        received = symbols + sigma * torch.randn_like(symbols)
        decided = (received >= 0.0).long()
        q_received = torch.sum(decided << shifts, dim=-1)
        return (q_received.to(inputs.dtype) / levels * 2.0 - 1.0) * clip_value

    @staticmethod
    def backward(ctx, grad_output):
        (inputs,) = ctx.saved_tensors
        grad = grad_output.clone()
        grad[(inputs < -ctx.clip_value) | (inputs > ctx.clip_value)] = 0.0
        return grad, None, None, None


class WirelessChannel(nn.Module):
    def __init__(self, snr_db=None):
        super().__init__()
        self.snr_db = Config.SNR_DB if snr_db is None else float(snr_db)
        self._print_stats()

    def _print_stats(self):
        print("\n" + "-" * 40)
        print(" >>> Digital Channel: Dense Quantized BPSK-AWGN for ECG")
        print(f" >>> Train Channel Mode: {Config.TRAIN_CHANNEL_MODE}")
        print(f" >>> Eval Channel Mode: {Config.EVAL_CHANNEL_MODE}")
        print(f" >>> SNR: {self.snr_db:g} dB")
        print(f" >>> Quant bits: {Config.QUANT_BITS}")
        print(f" >>> Clip value: {Config.QUANT_CLIP_VALUE}")
        print(f" >>> Bits/sample: {Config.BITS_PER_SAMPLE}")
        print("-" * 40 + "\n")

    def forward(self, x, state=None, mode="shared", channel_mode=None):
        if not Config.USE_WIRELESS:
            return x, state
        if channel_mode is None:
            channel_mode = Config.EVAL_CHANNEL_MODE
        if channel_mode == "clean":
            return x, state
        if channel_mode == "quant_only":
            return SignedUniformQuantizeSTE.apply(
                x, Config.QUANT_BITS, Config.QUANT_CLIP_VALUE
            ), state
        if channel_mode in ("bit_awgn_hard", "sampled_awgn"):
            return SignedUniformBitAWGNHARDSTE.apply(
                x, self.snr_db, Config.QUANT_BITS, Config.QUANT_CLIP_VALUE
            ), state
        raise ValueError(f"Unknown channel_mode: {channel_mode}")
