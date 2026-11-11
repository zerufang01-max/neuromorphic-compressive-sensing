"""Bit channels used for training and evaluation (single implementation).

AWGNChannel follows baselines/wireless_channel.WirelessChannel, including its
random-number consumption; RayleighChannel is the quasi-static fading channel
used for evaluation:
- AWGN uses the global CUDA generator (seeded by trainer.deterministic_eval);
- Rayleigh uses its own generator, reset to the evaluation seed in eval().

Transport ('dense' or 'aer'), the number of AER blocks and the ANN bit width
are attributes of the channel instead of global settings.
"""

import math

import torch
import torch.nn as nn

import config as C

HARD_THRESHOLD = 0.5


def block_geometry(latent_dim, num_blocks, require_power_of_two=False):
    block_size = int(latent_dim) // int(num_blocks)
    if block_size * int(num_blocks) != int(latent_dim):
        raise ValueError("num_blocks must divide the measurement dimension")
    if require_power_of_two and (num_blocks & (num_blocks - 1) or block_size & (block_size - 1)):
        raise ValueError("Exact fading AER routing requires power-of-two dimensions")
    header_bits = math.ceil(math.log2(num_blocks)) if num_blocks > 1 else 0
    local_bits = math.ceil(math.log2(block_size)) if block_size > 1 else 0
    return int(num_blocks), block_size, header_bits, local_bits


def block_aer_bits(hard, num_blocks):
    """AER payload (block headers + local addresses) per sample for one frame."""
    nb, bs, header_bits, local_bits = block_geometry(hard.shape[1], num_blocks)
    blocks = (hard > 0.5).reshape(hard.shape[0], nb, bs)
    return (blocks.any(dim=-1).sum(dim=-1).to(hard.dtype) * float(header_bits)
            + blocks.sum(dim=(1, 2)).to(hard.dtype) * float(local_bits))


def expected_block_bits(prob, num_blocks):
    """Differentiable expected AER payload per sample (rate regulariser)."""
    nb, bs, header_bits, local_bits = block_geometry(prob.shape[1], num_blocks)
    blocks = prob.clamp(0.0, 1.0).reshape(prob.shape[0], nb, bs)
    active = 1.0 - torch.prod(1.0 - blocks, dim=-1)
    return active.sum(dim=1) * header_bits + blocks.sum(dim=(1, 2)) * local_bits


class DenseQuantizedAWGNSTE(torch.autograd.Function):
    """Bit-input AWGN and hard demodulation for an unsigned q-bit ANN code."""

    @staticmethod
    def forward(ctx, inputs, sigma, bits):
        levels = (1 << int(bits)) - 1
        codes = torch.round(torch.clamp(inputs, 0.0, 1.0) * levels).long()
        decoded = torch.zeros_like(codes)
        for bit_index in range(int(bits)):
            bit = ((codes >> bit_index) & 1).to(inputs.dtype)
            noisy = bit + float(sigma) * torch.randn_like(bit)
            decoded |= (noisy >= HARD_THRESHOLD).long() << bit_index
        return decoded.to(inputs.dtype) / float(levels)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


class AWGNChannel(nn.Module):
    kind = "awgn"

    def __init__(self, snr_db=C.TRAIN_SNR_DB, transport="dense", num_blocks=C.NUM_BLOCKS, quant_bits=8):
        super().__init__()
        self.snr_db = float(snr_db)
        self.transport = transport
        self.num_blocks = int(num_blocks)
        self.quant_bits = int(quant_bits)
        self._routing = None

    @property
    def sigma(self):
        return C.awgn_sigma(self.snr_db)

    @property
    def hard_ber(self):
        return 0.5 * math.erfc(0.5 / (math.sqrt(2.0) * self.sigma))

    def set_snr_db(self, snr_db):
        self.snr_db = float(snr_db)
        self._routing = None

    def reset_rng(self, seed):  # AWGN uses the global generator (see deterministic_eval)
        return None

    def _hard_awgn(self, bits):
        noisy = bits + self.sigma * torch.randn_like(bits)
        return (noisy >= HARD_THRESHOLD).to(bits.dtype)

    def _soft_received(self, prob):
        error = prob.new_tensor(self.hard_ber)
        return error + (1.0 - 2.0 * error) * prob

    @staticmethod
    def _routing_matrix(dim, num_bits, p, device, dtype):
        matrix = torch.zeros((dim, dim), device=device, dtype=dtype)
        if p <= 0.0:
            matrix.fill_diagonal_(1.0)
            return matrix
        for source in range(dim):
            for mask in range(1 << num_bits):
                d = int(mask).bit_count()
                matrix[source, (source ^ mask) % dim] += p ** d * (1.0 - p) ** (num_bits - d)
        return matrix

    def _block_expected(self, prob):
        nb, bs, hb, lb = block_geometry(prob.shape[1], self.num_blocks)
        key = (self.hard_ber, prob.device, prob.dtype, nb, bs)
        if self._routing is None or self._routing[0] != key:
            self._routing = (key,
                             self._routing_matrix(nb, hb, self.hard_ber, prob.device, prob.dtype),
                             self._routing_matrix(bs, lb, self.hard_ber, prob.device, prob.dtype))
        _, block_k, local_k = self._routing
        blocks = prob.clamp(0.0, 1.0).reshape(prob.shape[0], nb, bs)
        out = torch.einsum("bsi,sd,ij->bdj", blocks, block_k, local_k)
        return out.clamp(0.0, 1.0).reshape(prob.shape[0], -1)

    def _decode_integer(self, values, num_bits):
        decoded = torch.zeros_like(values)
        for bit_index in range(num_bits):
            bit = ((values >> bit_index) & 1).to(torch.float32)
            noisy = bit + self.sigma * torch.randn_like(bit)
            decoded |= (noisy >= HARD_THRESHOLD).long() << bit_index
        return decoded

    def _block_hard(self, hard):
        nb, bs, hb, lb = block_geometry(hard.shape[1], self.num_blocks)
        batch = hard.shape[0]
        blocks = (hard > 0.5).to(hard.dtype).reshape(batch, nb, bs)
        active = torch.nonzero(blocks.amax(dim=-1) > 0, as_tuple=False)
        output = torch.zeros_like(blocks)
        if active.numel() == 0:
            return output.reshape(batch, -1)
        destination_block = self._decode_integer(active[:, 1], hb).remainder(nb)
        lookup = torch.full((batch, nb), -1, device=hard.device, dtype=torch.long)
        lookup[active[:, 0], active[:, 1]] = destination_block
        events = torch.nonzero(blocks > 0.5, as_tuple=False)
        destination_local = self._decode_integer(events[:, 2], lb).remainder(bs)
        flat = (events[:, 0] * nb + lookup[events[:, 0], events[:, 1]]) * bs + destination_local
        output.reshape(-1).scatter_add_(0, flat, torch.ones_like(flat, dtype=hard.dtype))
        return output.clamp_(0.0, 1.0).reshape(batch, -1)

    def forward(self, x, state=None, mode="ann", tx_prob=None, tx_hard=None):
        if mode == "ann":
            return DenseQuantizedAWGNSTE.apply(x, self.sigma, self.quant_bits), state
        hard_input = tx_hard if tx_hard is not None else (x > 0.5).to(x.dtype)
        relax = torch.is_grad_enabled() and self.training and tx_prob is not None
        if self.transport == "dense":
            y_hard = self._hard_awgn(hard_input)
            if relax:
                y_soft = self._soft_received(tx_prob)
                return y_hard.detach() - y_soft.detach() + y_soft, state
            return y_hard, state
        if self.transport == "aer":
            y_hard = self._block_hard(hard_input)
            if relax:
                y_soft = self._block_expected(tx_prob)
                return y_hard.detach() - y_soft.detach() + y_soft, state
            return y_hard, state
        raise ValueError(self.transport)


class RayleighChannel(nn.Module):
    """SISO BPSK, perfect CSIR, quasi-static Rayleigh fading (one fade per sample)."""

    kind = "rayleigh"

    def __init__(self, snr_db=20.0, transport="dense", num_blocks=C.NUM_BLOCKS, quant_bits=8,
                 eval_seed=C.EVAL_CHANNEL_SEED):
        super().__init__()
        self.snr_db = float(snr_db)
        self.transport = transport
        self.num_blocks = int(num_blocks)
        self.quant_bits = int(quant_bits)
        self.eval_seed = int(eval_seed)
        self._rng = None
        self._rng_device = None
        self._rng_seed = self.eval_seed
        super().train(False)

    def set_snr_db(self, snr_db):
        self.snr_db = float(snr_db)

    def reset_rng(self, seed):
        self._rng_seed = int(seed)
        self._rng = None
        self._rng_device = None

    def train(self, mode=True):
        if mode:
            raise RuntimeError("RayleighChannel is an evaluation channel")
        super().train(False)
        self.reset_rng(self.eval_seed)
        return self

    def _generator(self, reference):
        key = str(reference.device)
        if self._rng is None or self._rng_device != key:
            self._rng = torch.Generator(device=reference.device)
            self._rng.manual_seed(self._rng_seed)
            self._rng_device = key
        return self._rng

    def _new_sequence_state(self, reference):
        batch = int(reference.shape[0])
        generator = self._generator(reference)
        snr_db = torch.full((batch,), self.snr_db, device=reference.device, dtype=reference.dtype)
        real = torch.randn(batch, device=reference.device, dtype=reference.dtype, generator=generator)
        imag = torch.randn(batch, device=reference.device, dtype=reference.dtype, generator=generator)
        fading = torch.complex(real, imag) / math.sqrt(2.0)
        gamma = torch.pow(reference.new_tensor(10.0), snr_db / 10.0)
        gain = fading.abs().square().to(reference.dtype)
        return {"bit_error_probability": 0.5 * torch.erfc(torch.sqrt((gamma * gain).clamp_min(0.0)))}

    @staticmethod
    def _broadcast(values, reference):
        return values.reshape(values.shape[0], *([1] * (reference.ndim - 1)))

    def _hard_bits(self, bits, state, batch_indices=None):
        p = state["bit_error_probability"]
        if batch_indices is not None:
            p = p[batch_indices]
        flips = torch.rand(bits.shape, device=bits.device, dtype=bits.dtype,
                           generator=self._generator(bits)) < self._broadcast(p, bits)
        return torch.logical_xor(bits > 0.5, flips).to(bits.dtype)

    def _decode_integer(self, values, num_bits, batch_indices, state):
        decoded = torch.zeros_like(values)
        for bit_index in range(num_bits):
            bit = ((values >> bit_index) & 1).to(torch.float32)
            decoded |= self._hard_bits(bit, state, batch_indices).long() << bit_index
        return decoded

    def _block_hard(self, hard, state):
        nb, bs, hb, lb = block_geometry(hard.shape[1], self.num_blocks, require_power_of_two=True)
        batch = int(hard.shape[0])
        blocks = (hard > 0.5).to(hard.dtype).reshape(batch, nb, bs)
        active = torch.nonzero(blocks.amax(dim=-1) > 0, as_tuple=False)
        output = torch.zeros_like(blocks)
        if active.numel() == 0:
            return output.reshape(batch, -1)
        destination_block = self._decode_integer(active[:, 1], hb, active[:, 0], state).remainder(nb)
        lookup = torch.full((batch, nb), -1, device=hard.device, dtype=torch.long)
        lookup[active[:, 0], active[:, 1]] = destination_block
        events = torch.nonzero(blocks > 0.5, as_tuple=False)
        destination_local = self._decode_integer(events[:, 2], lb, events[:, 0], state).remainder(bs)
        flat = (events[:, 0] * nb + lookup[events[:, 0], events[:, 1]]) * bs + destination_local
        output.reshape(-1).scatter_add_(0, flat, torch.ones_like(flat, dtype=hard.dtype))
        return output.clamp_(0.0, 1.0).reshape(batch, -1)

    def _quantized_ann(self, inputs, state):
        levels = (1 << self.quant_bits) - 1
        codes = torch.round(inputs.clamp(0.0, 1.0) * levels).long()
        decoded = torch.zeros_like(codes)
        for bit_index in range(self.quant_bits):
            transmitted = ((codes >> bit_index) & 1).to(inputs.dtype)
            decoded |= self._hard_bits(transmitted, state).long() << bit_index
        hard_output = decoded.to(inputs.dtype) / float(levels)
        return inputs + (hard_output - inputs).detach()

    def forward(self, x, state=None, mode="ann", tx_prob=None, tx_hard=None):
        if state is None:
            state = self._new_sequence_state(x)
        if mode == "ann":
            return self._quantized_ann(x, state), state
        hard_input = tx_hard if tx_hard is not None else (x > 0.5).to(x.dtype)
        if self.transport == "dense":
            return self._hard_bits(hard_input, state), state
        if self.transport == "aer":
            return self._block_hard(hard_input, state), state
        raise ValueError(self.transport)


def make_channel(kind, snr_db, transport, quant_bits=8, num_blocks=C.NUM_BLOCKS):
    if kind == "awgn":
        return AWGNChannel(snr_db, transport, num_blocks, quant_bits)
    if kind == "rayleigh":
        return RayleighChannel(snr_db, transport, num_blocks, quant_bits)
    raise ValueError(kind)
