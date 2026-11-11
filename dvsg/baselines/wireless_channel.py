"""Bit channel of the DVS-Gesture baselines: BPSK over AWGN with hard decisions.

Spiking transmitters send the binary measurement map either in full or as
block-based AER addresses; ANN transmitters send 8-bit quantized measurements.
During training the backward pass uses the expected received values.
"""
import math

import torch
import torch.nn as nn

from config import Config


class DenseQuantizedAWGNSTE(torch.autograd.Function):
    """Bit-input AWGN and hard demodulation for an unsigned ANN code."""

    @staticmethod
    def forward(ctx, inputs, sigma, bits, threshold):
        bits = int(bits)
        levels = (1 << bits) - 1
        codes = torch.round(
            torch.clamp(inputs, 0.0, 1.0) * levels
        ).long()
        decoded = torch.zeros_like(codes)
        for bit_index in range(bits):
            bit = ((codes >> bit_index) & 1).to(inputs.dtype)
            noisy = bit + float(sigma) * torch.randn_like(bit)
            hard = (noisy >= float(threshold)).long()
            decoded |= hard << bit_index
        return decoded.to(inputs.dtype) / float(levels)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None


class WirelessChannel(nn.Module):
    """0/1 bit-input AWGN channel used by all formal DVSG experiments."""

    def __init__(self, snr_db=None):
        super().__init__()
        self.device = Config.DEVICE
        self.snr_db = (
            float(Config.TRAIN_SNR_DB)
            if snr_db is None else float(snr_db)
        )
        self._block_K = None
        self._local_K = None
        self._routing_ber = None
        self._print_stats()

    @property
    def sigma(self):
        return Config.awgn_sigma(self.snr_db)

    @property
    def hard_ber(self):
        return Config.awgn_hard_ber(self.snr_db)

    def set_snr_db(self, snr_db):
        self.snr_db = float(snr_db)
        self._block_K = None
        self._local_K = None
        self._routing_ber = None

    def _print_stats(self):
        print("\n" + "-" * 40)
        print(
            f" >>> AWGN Channel: SNN={Config.SNN_TRANSPORT} / "
            f"ANN=dense Q{Config.ANN_QUANT_BITS} hard"
        )
        print(f" >>> Eb/N0: {self.snr_db:g} dB")
        print(f" >>> sigma: {self.sigma:.6g}")
        print(f" >>> hard BER: {self.hard_ber:.6g}")
        print("-" * 40 + "\n")

    def _hard_awgn(self, bits):
        noisy = bits + self.sigma * torch.randn_like(bits)
        return (
            noisy >= float(Config.CHANNEL_HARD_THRESHOLD)
        ).to(bits.dtype)

    def _soft_received_probability(self, tx_probability):
        error = tx_probability.new_tensor(self.hard_ber)
        return error + (1.0 - 2.0 * error) * tx_probability

    @staticmethod
    def _get_dim_routing_matrix(
        dim_size, num_bits, error_probability, device, dtype
    ):
        matrix = torch.zeros(
            (dim_size, dim_size), device=device, dtype=dtype
        )
        if error_probability <= 0.0:
            matrix.fill_diagonal_(1.0)
            return matrix
        for source in range(dim_size):
            for error_mask in range(1 << num_bits):
                distance = int(error_mask).bit_count()
                probability = (
                    error_probability ** distance
                    * (1.0 - error_probability)
                    ** (num_bits - distance)
                )
                destination = (source ^ error_mask) % dim_size
                matrix[source, destination] += probability
        return matrix

    @staticmethod
    def _block_geometry():
        num_blocks = int(Config.NUM_BLOCKS)
        block_size = int(Config.BLOCK_SIZE)
        if num_blocks * block_size != int(Config.LATENT_DIM):
            raise ValueError(
                "NUM_BLOCKS * BLOCK_SIZE must equal LATENT_DIM."
            )
        header_bits = (
            math.ceil(math.log2(num_blocks)) if num_blocks > 1 else 0
        )
        local_bits = (
            math.ceil(math.log2(block_size)) if block_size > 1 else 0
        )
        return num_blocks, block_size, header_bits, local_bits

    def block_aer_bits(self, hard):
        num_blocks, block_size, header_bits, local_bits = (
            self._block_geometry()
        )
        blocks = (hard > 0.5).reshape(
            hard.shape[0], num_blocks, block_size
        )
        active_blocks = blocks.any(dim=-1).sum(dim=-1)
        events = blocks.sum(dim=(1, 2))
        return (
            active_blocks.to(hard.dtype) * float(header_bits)
            + events.to(hard.dtype) * float(local_bits)
        )

    def _cached_block_routing(self, reference):
        num_blocks, block_size, header_bits, local_bits = (
            self._block_geometry()
        )
        ber = float(self.hard_ber)
        if (
            self._block_K is None
            or self._local_K is None
            or self._routing_ber != ber
            or self._block_K.device != reference.device
            or self._block_K.dtype != reference.dtype
        ):
            self._block_K = self._get_dim_routing_matrix(
                num_blocks, header_bits, ber,
                reference.device, reference.dtype,
            )
            self._local_K = self._get_dim_routing_matrix(
                block_size, local_bits, ber,
                reference.device, reference.dtype,
            )
            self._routing_ber = ber
        return self._block_K, self._local_K

    def _block_expected(self, probabilities):
        num_blocks, block_size, _, _ = self._block_geometry()
        blocks = probabilities.clamp(0.0, 1.0).reshape(
            probabilities.shape[0], num_blocks, block_size
        )
        block_routing, local_routing = self._cached_block_routing(
            probabilities
        )
        expected = torch.einsum(
            "bsi,sd,ij->bdj",
            blocks, block_routing, local_routing,
        )
        return expected.clamp(0.0, 1.0).reshape(
            probabilities.shape[0], Config.LATENT_DIM
        )

    def _decode_integer_awgn(self, values, num_bits):
        decoded = torch.zeros_like(values)
        for bit_index in range(num_bits):
            bit = ((values >> bit_index) & 1).to(torch.float32)
            noisy = bit + self.sigma * torch.randn_like(bit)
            hard = (
                noisy >= float(Config.CHANNEL_HARD_THRESHOLD)
            ).long()
            decoded |= hard << bit_index
        return decoded

    def _block_hard(self, hard):
        num_blocks, block_size, header_bits, local_bits = (
            self._block_geometry()
        )
        batch_size = hard.shape[0]
        blocks = (hard > 0.5).to(hard.dtype).reshape(
            batch_size, num_blocks, block_size
        )
        active = torch.nonzero(
            blocks.amax(dim=-1) > 0, as_tuple=False
        )
        output = torch.zeros_like(blocks)
        if active.numel() == 0:
            return output.reshape(batch_size, Config.LATENT_DIM)

        active_batch = active[:, 0]
        source_block = active[:, 1]
        destination_block = self._decode_integer_awgn(
            source_block, header_bits
        ).remainder(num_blocks)
        lookup = torch.full(
            (batch_size, num_blocks), -1,
            device=hard.device, dtype=torch.long,
        )
        lookup[active_batch, source_block] = destination_block

        events = torch.nonzero(blocks > 0.5, as_tuple=False)
        event_batch = events[:, 0]
        event_source_block = events[:, 1]
        source_local = events[:, 2]
        destination_local = self._decode_integer_awgn(
            source_local, local_bits
        ).remainder(block_size)
        flat_destination = (
            (
                event_batch * num_blocks
                + lookup[event_batch, event_source_block]
            )
            * block_size
            + destination_local
        )
        output.reshape(-1).scatter_add_(
            0,
            flat_destination,
            torch.ones_like(flat_destination, dtype=hard.dtype),
        )
        return output.clamp_(0.0, 1.0).reshape(
            batch_size, Config.LATENT_DIM
        )

    def audit_block_routing_matrix(self, atol=1e-6):
        num_blocks, block_size, header_bits, local_bits = (
            self._block_geometry()
        )
        dtype = torch.float64
        ber = float(self.hard_ber)
        header = self._get_dim_routing_matrix(
            num_blocks, header_bits, ber, self.device, dtype
        )
        local = self._get_dim_routing_matrix(
            block_size, local_bits, ber, self.device, dtype
        )
        identity_header = self._get_dim_routing_matrix(
            num_blocks, header_bits, 0.0, self.device, dtype
        )
        identity_local = self._get_dim_routing_matrix(
            block_size, local_bits, 0.0, self.device, dtype
        )
        errors = {
            "header_row_sum_max_error": float(
                (header.sum(dim=1) - 1.0).abs().max().item()
            ),
            "local_row_sum_max_error": float(
                (local.sum(dim=1) - 1.0).abs().max().item()
            ),
            "header_p0_identity_max_error": float(
                (identity_header - torch.eye(
                    num_blocks, device=self.device, dtype=dtype
                )).abs().max().item()
            ),
            "local_p0_identity_max_error": float(
                (identity_local - torch.eye(
                    block_size, device=self.device, dtype=dtype
                )).abs().max().item()
            ),
        }
        errors["nonnegative"] = bool(
            (header >= 0).all().item() and (local >= 0).all().item()
        )
        errors["passed"] = bool(
            max(
                errors["header_row_sum_max_error"],
                errors["local_row_sum_max_error"],
                errors["header_p0_identity_max_error"],
                errors["local_p0_identity_max_error"],
            ) <= atol and errors["nonnegative"]
        )
        errors["awgn_hard_ber"] = ber
        return errors

    def forward(
        self, x, state=None, mode="ann", tx_prob=None, tx_hard=None
    ):
        if not Config.USE_WIRELESS:
            return x, state

        if mode == "ann":
            return (
                DenseQuantizedAWGNSTE.apply(
                    x, self.sigma, Config.ANN_QUANT_BITS,
                    Config.CHANNEL_HARD_THRESHOLD,
                ),
                state,
            )

        transport = str(Config.SNN_TRANSPORT)
        hard_input = (
            tx_hard if tx_hard is not None else (x > 0.5).to(x.dtype)
        )
        if transport in ("dense_binary", "dense_binary_awgn_hard"):
            y_hard = self._hard_awgn(hard_input)
            if (
                torch.is_grad_enabled()
                and self.training
                and tx_prob is not None
                and Config.TX_ABLATION_MODE != "hard_spike"
            ):
                y_soft = self._soft_received_probability(tx_prob)
                return y_hard.detach() - y_soft.detach() + y_soft, state
            return y_hard, state

        if transport in ("block_aer", "block_aer_awgn_hard"):
            y_hard = self._block_hard(hard_input)
            if (
                torch.is_grad_enabled()
                and self.training
                and tx_prob is not None
                and Config.TX_ABLATION_MODE != "hard_spike"
                and Config.USE_RELAXED_AER_AWGN
            ):
                y_soft = self._block_expected(tx_prob)
                return y_hard.detach() - y_soft.detach() + y_soft, state
            return y_hard, state

        raise ValueError(f"Unknown SNN_TRANSPORT={transport!r}.")
