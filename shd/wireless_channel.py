import math

import torch
import torch.nn as nn

from config import Config


def gaussian_cdf(value):
    return 0.5 * (1.0 + torch.erf(value / math.sqrt(2.0)))


def binary_awgn_ber(sigma, reference):
    """Hard-decision BER for binary amplitudes {0,1} at threshold 0.5."""
    sigma_tensor = reference.new_tensor(float(sigma)).clamp_min(1e-12)
    margin = 0.5 * (
        float(Config.CHANNEL_BIT_ONE) - float(Config.CHANNEL_BIT_ZERO)
    )
    return 0.5 * torch.erfc(
        reference.new_tensor(margin)
        / (math.sqrt(2.0) * sigma_tensor)
    )


class DenseQuantizedAWGNSTE(torch.autograd.Function):
    """Bit-wise AWGN for a dense quantized ANN measurement.

    The input has already been uniformly quantized to [0,1]. It is converted
    to an integer code, serialized into binary amplitudes {0,1}, corrupted by
    AWGN, and reconstructed from hard bits or posterior bit probabilities.
    """

    @staticmethod
    def forward(ctx, inputs, sigma, bits, receiver):
        bits = int(bits)
        receiver = str(receiver)
        if bits not in (1, 2, 4, 8):
            raise ValueError(
                f"ANN quantization bits must be one of 1,2,4,8; got {bits}."
            )
        if receiver not in {"hard", "soft"}:
            raise ValueError(
                f"ANN receiver must be 'hard' or 'soft'; got {receiver!r}."
            )

        levels = (1 << bits) - 1
        codes = torch.round(
            torch.clamp(inputs, 0.0, 1.0) * levels
        ).long()
        shifts = torch.arange(
            bits, device=inputs.device, dtype=torch.long
        )
        serialized = torch.bitwise_and(
            torch.bitwise_right_shift(codes.unsqueeze(-1), shifts), 1
        ).to(inputs.dtype)

        sigma = max(float(sigma), 1e-12)
        received = serialized + sigma * torch.randn_like(serialized)
        if receiver == "hard":
            recovered_bits = (
                received >= float(Config.CHANNEL_HARD_THRESHOLD)
            ).to(inputs.dtype)
        else:
            # Equal-prior posterior for amplitudes {0,1} in AWGN.
            recovered_bits = torch.sigmoid(
                (received - float(Config.CHANNEL_HARD_THRESHOLD))
                / (sigma * sigma)
            )

        weights = (2.0 ** shifts.to(inputs.dtype)).view(
            *((1,) * inputs.ndim), bits
        )
        recovered_code = torch.sum(recovered_bits * weights, dim=-1)
        return recovered_code / float(levels)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None


class DenseQuantizedAWGNChannel(nn.Module):
    def __init__(self, snr_db=None, receiver=None):
        super().__init__()
        self.snr_db = float(
            Config.TRAIN_SNR_DB if snr_db is None else snr_db
        )
        self.receiver = str(
            Config.ANN_RX_MODE if receiver is None else receiver
        )

    @property
    def sigma(self):
        return Config.awgn_sigma(self.snr_db)

    def set_snr_db(self, snr_db):
        self.snr_db = float(snr_db)
        return self

    def forward(
        self, x, state=None, mode="ann", tx_prob=None, tx_hard=None,
        tx_aux=None,
    ):
        if not Config.USE_WIRELESS:
            return x, state
        if mode != "ann":
            raise ValueError(
                "DenseQuantizedAWGNChannel is only valid for ANN modes."
            )
        output = DenseQuantizedAWGNSTE.apply(
            x,
            self.sigma,
            int(Config.ANN_QUANT_BITS),
            self.receiver,
        )
        return output, state


class DenseSpikeAWGNChannel(nn.Module):
    """Full binary-spike bitmap over independent binary AWGN channels.

    Spikes use a single {0,1} plane.  Evaluation
    applies AWGN and a threshold-0.5 hard decision to every bitmap bit.  During
    training, the forward path stays hard while the backward path uses the
    exact Gaussian hard-decision expectation.
    """

    def __init__(self, snr_db=None):
        super().__init__()
        self.snr_db = float(
            Config.TRAIN_SNR_DB if snr_db is None else snr_db
        )

    @property
    def sigma(self):
        return Config.awgn_sigma(self.snr_db)

    def set_snr_db(self, snr_db):
        self.snr_db = float(snr_db)
        return self

    def hard_bit_error_probability(self, reference=None):
        if reference is None:
            reference = torch.zeros((), dtype=torch.float64)
        return binary_awgn_ber(self.sigma, reference)

    def _hard_bits(self, bits):
        received = bits.to(torch.float32)
        received = received + self.sigma * torch.randn_like(received)
        return (
            received >= float(Config.CHANNEL_HARD_THRESHOLD)
        ).to(bits.dtype)

    def forward(
        self, x, state=None, mode="spiking", tx_prob=None,
        tx_hard=None, tx_aux=None,
    ):
        if not Config.USE_WIRELESS:
            return x, state
        if mode == "ann":
            raise ValueError(
                "DenseSpikeAWGNChannel is only valid for SNN modes."
            )

        auxiliary = tx_aux or {}
        hard = auxiliary.get("tx_hard", tx_hard)
        if hard is None:
            hard = (x >= 0.5).to(x.dtype)
        hard_output = self._hard_bits(hard)
        if self.training and torch.is_grad_enabled():
            probability = auxiliary.get("tx_prob", tx_prob)
            if probability is None:
                probability = x
            ber = self.hard_bit_error_probability(probability)
            expected = ber + (1.0 - 2.0 * ber) * probability
            return hard_output.detach() - expected.detach() + expected, state
        return hard_output, state

    def audit_dense_binary_planes(self):
        reference = torch.zeros((), dtype=torch.float64)
        ber = float(self.hard_bit_error_probability(reference).item())
        return {
            "binary_planes": 1,
            "bits_per_frame": int(Config.MEAS_DIM),
            "ber": ber,
            "passed": bool(0.0 <= ber <= 0.5),
        }


class BlockAERAWGNChannel(nn.Module):
    """Binary Block-AER whose address bits cross binary AWGN."""

    def __init__(self, snr_db=None, block_size=None):
        super().__init__()
        self.snr_db = float(
            Config.TRAIN_SNR_DB if snr_db is None else snr_db
        )
        self.block_size = int(
            Config.BLOCK_SIZE if block_size is None else block_size
        )
        if Config.MEAS_DIM % self.block_size != 0:
            raise ValueError("block_size must divide MEAS_DIM")
        self.num_blocks = Config.MEAS_DIM // self.block_size
        self.header_bits = (
            math.ceil(math.log2(self.num_blocks))
            if self.num_blocks > 1 else 0
        )
        self.local_bits = (
            math.ceil(math.log2(self.block_size))
            if self.block_size > 1 else 0
        )

    @property
    def sigma(self):
        return Config.awgn_sigma(self.snr_db)

    def set_snr_db(self, snr_db):
        self.snr_db = float(snr_db)
        return self

    def hard_bit_error_probability(self, reference=None):
        if reference is None:
            reference = torch.zeros((), dtype=torch.float64)
        return binary_awgn_ber(self.sigma, reference)

    def _awgn_hard_bits(self, bits):
        received = bits.to(torch.float32)
        received = received + self.sigma * torch.randn_like(received)
        return (
            received >= float(Config.CHANNEL_HARD_THRESHOLD)
        ).long()

    def _transmit_indices(self, indices, num_bits):
        if num_bits <= 0:
            return torch.zeros_like(indices)
        shifts = torch.arange(
            num_bits, device=indices.device, dtype=torch.long
        )
        bits = torch.bitwise_and(
            torch.bitwise_right_shift(indices[:, None], shifts[None, :]), 1
        )
        received_bits = self._awgn_hard_bits(bits)
        return torch.sum(
            torch.bitwise_left_shift(received_bits, shifts[None, :]),
            dim=1,
        )

    def _routing_matrix(self, size, bits, reference):
        if size == 1:
            return reference.new_ones((1, 1))
        indices = torch.arange(
            size, device=reference.device, dtype=torch.long
        )
        xor = torch.bitwise_xor(indices[:, None], indices[None, :])
        hamming = torch.zeros_like(xor)
        work = xor.clone()
        for _ in range(bits):
            hamming += torch.bitwise_and(work, 1)
            work = torch.bitwise_right_shift(work, 1)
        hamming = hamming.to(reference.dtype)
        error_probability = self.hard_bit_error_probability(reference)
        return torch.pow(error_probability, hamming) * torch.pow(
            1.0 - error_probability, bits - hamming
        )

    def _hard_route(self, spike_hard):
        batch_size = spike_hard.shape[0]
        blocks = spike_hard.reshape(
            batch_size, self.num_blocks, self.block_size
        )
        active = torch.nonzero(
            blocks.abs().amax(dim=-1) > 0, as_tuple=False
        )
        output = torch.zeros_like(blocks)
        if active.numel() == 0:
            return output.reshape(batch_size, Config.MEAS_DIM)

        active_batch = active[:, 0]
        source_block = active[:, 1]
        destination_block = self._transmit_indices(
            source_block, self.header_bits
        ) % self.num_blocks
        lookup = torch.full(
            (batch_size, self.num_blocks), -1,
            dtype=torch.long,
            device=spike_hard.device,
        )
        lookup[active_batch, source_block] = destination_block

        events = torch.nonzero(blocks != 0, as_tuple=False)
        event_batch = events[:, 0]
        event_source_block = events[:, 1]
        event_local = events[:, 2]
        destination_local = self._transmit_indices(
            event_local, self.local_bits
        ) % self.block_size
        received_sign = torch.ones_like(event_local, dtype=blocks.dtype)

        event_destination_block = lookup[
            event_batch, event_source_block
        ]
        flat_destination = (
            (event_batch * self.num_blocks + event_destination_block)
            * self.block_size + destination_local
        )
        output_flat = output.reshape(-1)
        output_flat.scatter_add_(0, flat_destination, received_sign)
        return output_flat.clamp_(0.0, 1.0).reshape(
            batch_size, Config.MEAS_DIM
        )

    def _expected_route(self, event_prob):
        batch_size = event_prob.shape[0]
        probability = event_prob.reshape(batch_size, self.num_blocks, self.block_size)
        block_routing = self._routing_matrix(self.num_blocks, self.header_bits, probability)
        local_routing = self._routing_matrix(self.block_size, self.local_bits, probability)
        # A differentiable routing surrogate; hard collisions merge by logical OR.
        expected = torch.einsum("bsi,sd,ij->bdj", probability, block_routing, local_routing)
        return expected.clamp(0.0, 1.0).reshape(batch_size, Config.MEAS_DIM)

    def forward(
        self, x, state=None, mode="spiking", tx_prob=None,
        tx_hard=None, tx_aux=None,
    ):
        if not Config.USE_WIRELESS:
            return x, state
        auxiliary = tx_aux or {}
        hard_input = auxiliary.get("tx_hard", tx_hard)
        if hard_input is None:
            hard_input = (x >= 0.5).to(x.dtype)
        hard_output = self._hard_route(hard_input)

        if self.training and torch.is_grad_enabled():
            event_prob = auxiliary.get("tx_prob", tx_prob)
            if event_prob is not None:
                soft_output = self._expected_route(event_prob)
                return (
                    hard_output.detach()
                    - soft_output.detach()
                    + soft_output,
                    state,
                )
            return hard_output.detach() - x.detach() + x, state
        return hard_output, state

    def expected_rate_bits(self, auxiliary):
        event_prob = auxiliary["tx_prob"]
        blocks = event_prob.clamp(0.0, 1.0).reshape(
            event_prob.shape[0], self.num_blocks, self.block_size
        )
        active_block_prob = 1.0 - torch.prod(1.0 - blocks, dim=-1)
        bits = (
            active_block_prob.sum(dim=1) * self.header_bits
            + blocks.sum(dim=(1, 2)) * self.local_bits
        )
        return bits.mean() / Config.MEAS_DIM

    def block_aer_bits(self, spike_hard):
        blocks = spike_hard.reshape(
            spike_hard.shape[0], self.num_blocks, self.block_size
        )
        active_blocks = (
            blocks.abs().amax(dim=-1) > 0
        ).sum(dim=1)
        events = (blocks != 0).sum(dim=(1, 2))
        return (
            active_blocks.to(spike_hard.dtype) * self.header_bits
            + events.to(spike_hard.dtype) * self.local_bits
        )

    hard_bits = block_aer_bits

    def audit_block_routing_matrix(self, atol=1e-6):
        reference = torch.zeros(
            1, device=Config.DEVICE, dtype=torch.float64
        )
        block = self._routing_matrix(
            self.num_blocks, self.header_bits, reference
        )
        local = self._routing_matrix(
            self.block_size, self.local_bits, reference
        )
        row_error = max(
            float((block.sum(dim=1) - 1.0).abs().max().item()),
            float((local.sum(dim=1) - 1.0).abs().max().item()),
        )
        ber = float(self.hard_bit_error_probability(reference).item())
        return {
            "row_sum_max_error": row_error,
            "theoretical_hard_ber": ber,
            "snr_db": float(self.snr_db),
            "sigma": float(self.sigma),
            "passed": bool(row_error <= atol and 0.0 <= ber <= 0.5),
        }


WirelessChannel = DenseQuantizedAWGNChannel
BlockAERChannel = BlockAERAWGNChannel


class FrozenRayleighChannel(BlockAERAWGNChannel):
    """Evaluation-only coherent BPSK, perfect CSIR, no CSIT, E|h|²=1.

    One complex fade per sample, shared across its full temporal sequence.
    Hard errors have probability erfc(sqrt(Eb/N0 * |h|²))/2.
    Binary bitmaps, ANN bit codes and unsigned AER routing retain
    their original semantics. Fading and bit errors use separate RNGs so
    fades do not depend on how many bits each method transmits.
    """

    def __init__(self, snr_db=20.0, transport=None):
        super().__init__(snr_db=snr_db)
        self.transport = transport or Config.SNN_TRANSPORT
        self.reset_rng()
        self.eval()

    def reset_rng(self, seed=None):
        self._seed = int(Config.EVAL_CHANNEL_SEED if seed is None else seed)
        self._fade_rng = None
        self._bit_rng = None
        self._device_key = None

    def _generators(self, x):
        if self._device_key != str(x.device):
            self._fade_rng = torch.Generator(device=x.device).manual_seed(self._seed)
            self._bit_rng = torch.Generator(device=x.device).manual_seed(self._seed + 1000003)
            self._device_key = str(x.device)

    def _new_state(self, x):
        self._generators(x)
        components = torch.randn(x.shape[0], 2, device=x.device,
                                 dtype=torch.float32, generator=self._fade_rng)
        gain = components.square().sum(dim=1) / 2.0
        gamma = 10.0 ** (self.snr_db / 10.0)
        return {"gain_power": gain,
                "ber": 0.5 * torch.erfc(torch.sqrt(gamma * gain))}

    def _flip(self, bits, state, batch_indices=None):
        self._generators(bits)
        probability = state["ber"]
        if batch_indices is not None:
            probability = probability[batch_indices]
        probability = probability.reshape(-1, *([1] * (bits.ndim - 1)))
        flips = torch.rand(bits.shape, device=bits.device, dtype=torch.float32,
                           generator=self._bit_rng) < probability
        return torch.logical_xor(bits > 0.5, flips).to(bits.dtype)

    def _indices(self, indices, width, batch_indices, state):
        if width == 0:
            return torch.zeros_like(indices)
        shifts = torch.arange(width, device=indices.device, dtype=torch.long)
        bits = (indices[:, None] >> shifts[None, :]) & 1
        received = self._flip(bits, state, batch_indices).long()
        return (received << shifts[None, :]).sum(dim=1)

    def _rayleigh_route(self, hard, state):
        blocks = hard.reshape(hard.shape[0], self.num_blocks, self.block_size)
        output = torch.zeros_like(blocks)
        active = torch.nonzero(blocks.abs().amax(dim=-1) > 0, as_tuple=False)
        if active.numel() == 0:
            return output.reshape_as(hard)
        batches, source_blocks = active[:, 0], active[:, 1]
        destinations = self._indices(source_blocks, self.header_bits, batches, state) % self.num_blocks
        lookup = torch.full((hard.shape[0], self.num_blocks), -1,
                            device=hard.device, dtype=torch.long)
        lookup[batches, source_blocks] = destinations
        events = torch.nonzero(blocks != 0, as_tuple=False)
        b, block, local = events[:, 0], events[:, 1], events[:, 2]
        destination_local = self._indices(local, self.local_bits, b, state) % self.block_size
        recovered_sign = torch.ones_like(local, dtype=hard.dtype)
        destination = (b * self.num_blocks + lookup[b, block]) * self.block_size + destination_local
        output.reshape(-1).scatter_add_(0, destination, recovered_sign)
        return output.clamp_(0, 1).reshape_as(hard)

    def forward(self, x, state=None, mode="ann", tx_prob=None, tx_hard=None, tx_aux=None):
        if self.training:
            raise RuntimeError("FrozenRayleighChannel is for evaluation only")
        if not Config.USE_WIRELESS:
            return x, state
        if state is None:
            state = self._new_state(x)
        if mode == "ann":
            bits = int(Config.ANN_QUANT_BITS)
            levels = (1 << bits) - 1
            codes = torch.round(x.clamp(0, 1) * levels).long()
            shifts = torch.arange(bits, device=x.device, dtype=torch.long)
            serialized = (codes.unsqueeze(-1) >> shifts) & 1
            recovered = self._flip(serialized, state).long()
            return ((recovered << shifts).sum(dim=-1).to(x.dtype) / levels), state
        aux = tx_aux or {}
        hard = aux.get("tx_hard", tx_hard)
        if hard is None:
            hard = (x >= 0.5).to(x.dtype)
        if self.transport == "block_aer_awgn_hard":
            return self._rayleigh_route(hard, state), state
        if self.transport == "dense_spike_awgn_hard":
            return self._flip(hard, state), state
        raise ValueError(f"Unsupported SHD transport: {self.transport}")


def check_frozen_rayleigh_channel():
    """Small CPU checks, run once by the dispatcher before model evaluation."""
    if not Config.USE_WIRELESS:
        raise RuntimeError("Rayleigh evaluation requires USE_WIRELESS=True")
    channel = FrozenRayleighChannel(10, "block_aer_awgn_hard")
    hard = torch.zeros(2, Config.MEAS_DIM)
    hard[0, 0], hard[1, -1] = 1, 1
    clean = {"ber": torch.zeros(2)}
    received, reused = channel(hard, state=clean, mode="spiking", tx_hard=hard)
    assert reused is clean and torch.equal(received, hard), "AER zero-error routing"
    baseline = BlockAERAWGNChannel(10)
    assert torch.equal(channel.block_aer_bits(hard), baseline.block_aer_bits(hard)), "AER bit accounting"
    channel.transport = "dense_spike_awgn_hard"
    received, reused = channel(hard, state=clean, mode="spiking", tx_hard=hard)
    assert reused is clean and torch.equal(received, hard), "Binary dense zero-error transport"
    levels = (1 << int(Config.ANN_QUANT_BITS)) - 1
    x = torch.arange(2*Config.MEAS_DIM).reshape(2, -1).float().remainder(levels+1) / levels
    received, _ = channel(x, state=clean, mode="ann")
    assert torch.equal(received, x), "ANN bit serialization"
    channel.reset_rng(2026)
    reference = torch.zeros(200000, 1)
    state = channel._new_state(reference)
    empirical = channel._flip(reference, state).mean().item()
    expected = 0.5 * (1 - math.sqrt(10/11))
    assert abs(empirical-expected) < 0.002, f"Rayleigh BER mismatch: {empirical} vs {expected}"
    assert abs(state["gain_power"].mean().item()-1) < 0.02, "Fading power normalization"
    channel.reset_rng(2026)
    first = channel._new_state(hard)
    channel._flip(hard, first)
    after_bits = channel._new_state(hard)["gain_power"]
    channel.reset_rng(2026)
    channel._new_state(hard)
    without_bits = channel._new_state(hard)["gain_power"]
    assert torch.equal(after_bits, without_bits), "Fading depends on payload length"
    print(f"[Channel check passed] binary/AER/ANN; Rayleigh BER={empirical:.5f}, expected={expected:.5f}", flush=True)
