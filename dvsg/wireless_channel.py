import torch
import torch.nn as nn
import math
from config import Config

class Dense_BSC_STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, p):
        ctx.save_for_backward(inputs)
        mask = (torch.rand_like(inputs) < p).float()
        return inputs * (1.0 - mask) + (1.0 - inputs) * mask

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class AER_BSC_STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, p):
        ctx.save_for_backward(inputs)
        device = inputs.device
        B, L = inputs.shape
        
        C = Config.MEAS_CHANNELS
        H = Config.IMG_H // Config.MEAS_STRIDE
        W = Config.IMG_W // Config.MEAS_STRIDE
        
        x_spatial = inputs.view(B, C, H, W)
        
        bits_C = max(1, math.ceil(math.log2(C)))
        bits_H = max(1, math.ceil(math.log2(H)))
        bits_W = max(1, math.ceil(math.log2(W)))
        
        spikes = x_spatial.nonzero(as_tuple=False)
        if spikes.numel() == 0:
            return inputs.clone()
            
        b_idx = spikes[:, 0]
        c_idx = spikes[:, 1]
        h_idx = spikes[:, 2]
        w_idx = spikes[:, 3]
        vals = x_spatial[b_idx, c_idx, h_idx, w_idx]
        
        def flip_bits(ints, num_bits, prob):
            if num_bits <= 0: return ints
            mask = (torch.rand((len(ints), num_bits), device=device) < prob).long()
            powers = 2 ** torch.arange(num_bits - 1, -1, -1, device=device)
            flip_ints = (mask * powers).sum(dim=1)
            return ints ^ flip_ints
            
        c_corr = flip_bits(c_idx, bits_C, p) % C
        h_corr = flip_bits(h_idx, bits_H, p) % H
        w_corr = flip_bits(w_idx, bits_W, p) % W
        
        outputs_spatial = torch.zeros_like(x_spatial)
        flat_idx = b_idx * (C * H * W) + c_corr * (H * W) + h_corr * W + w_corr
        outputs_flat = outputs_spatial.view(-1)
        outputs_flat.scatter_add_(0, flat_idx, vals)
        
        outputs_spatial = outputs_flat.view(B, C, H, W).clamp(0, 1)
        
        return outputs_spatial.view(B, L)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class WirelessChannel(nn.Module):
    def __init__(self, p=None):
        super().__init__()
        self.device = Config.DEVICE
        self.p = p if p is not None else Config.BSC_CROSSOVER_PROB
        
        self._K = None
        self._K_p = -1.0 
        
        self._print_stats()

    def _print_stats(self):
        print(f"\n{'-'*40}")
        print(f" >>> Digital Channel: Dual Mode (AER for SNN / Dense for ANN)")
        print(f" >>> USE_RELAXED_AER_BSC: {getattr(Config, 'USE_RELAXED_AER_BSC', False)}")
        print(f"{'-'*40}")
        print(f" Crossover Probability (p): {self.p}")
        print(f"{'-'*40}\n")

    def _get_dim_routing_matrix(self, dim_size, num_bits, p, device, dtype):
        mat = torch.zeros((dim_size, dim_size), device=device, dtype=dtype)
        if p <= 0:
            mat.fill_diagonal_(1.0)
            return mat
            
        for src in range(dim_size):
            for err in range(1 << num_bits):
                flipped_bits = src ^ err
                d = bin(flipped_bits).count('1')
                prob = (p ** d) * ((1 - p) ** (num_bits - d))
                dst = err % dim_size
                mat[src, dst] += prob
        return mat

    def _build_aer_routing_matrix(self, device, dtype):
        if self.p <= 0:
            return None
            
        C = Config.MEAS_CHANNELS
        H = Config.IMG_H // Config.MEAS_STRIDE
        W = Config.IMG_W // Config.MEAS_STRIDE
        
        bits_C = max(1, math.ceil(math.log2(C)))
        bits_H = max(1, math.ceil(math.log2(H)))
        bits_W = max(1, math.ceil(math.log2(W)))
        
        K_C = self._get_dim_routing_matrix(C, bits_C, self.p, device, dtype)
        K_H = self._get_dim_routing_matrix(H, bits_H, self.p, device, dtype)
        K_W = self._get_dim_routing_matrix(W, bits_W, self.p, device, dtype)
        
        K_CH = torch.kron(K_C, K_H)
        K = torch.kron(K_CH, K_W)
        return K

    def _expected_aer_bsc(self, tx_prob):
        if self.p <= 0:
            return tx_prob
            
        if self._K is None or self._K_p != self.p or self._K.device != tx_prob.device:
            self._K = self._build_aer_routing_matrix(tx_prob.device, tx_prob.dtype)
            self._K_p = self.p
            
        y_soft = torch.matmul(tx_prob, self._K)
        return y_soft

    def forward(self, x, state=None, mode='ann', tx_prob=None, tx_hard=None):
        if not Config.USE_WIRELESS or self.p <= 0.0:
            return x, state
            
        if mode == 'ann':
            return Dense_BSC_STE.apply(x, self.p), state
            
        is_ablation = (Config.TX_ABLATION_MODE == 'hard_spike')
        use_relaxed = getattr(Config, 'USE_RELAXED_AER_BSC', False)
        
        if not torch.is_grad_enabled() or not self.training:
            hard_input = tx_hard if tx_hard is not None else x
            return AER_BSC_STE.apply(hard_input, self.p), state
            
        if is_ablation or not use_relaxed or tx_prob is None:
            return AER_BSC_STE.apply(x, self.p), state
            
        hard_input = tx_hard if tx_hard is not None else (x > 0).float()
        y_hard = AER_BSC_STE.apply(hard_input, self.p)
        
        y_soft = self._expected_aer_bsc(tx_prob)
        
        # Forward uses hard AER/BSC; backward uses temperature-annealed expected AER/BSC proxy
        y_rx = y_hard.detach() - y_soft.detach() + y_soft
        
        return y_rx, state