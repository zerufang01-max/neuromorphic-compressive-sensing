import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import Config

class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input >= 0).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        mask = (input.abs() < 0.5).float()
        return grad_input * mask

class ProbabilityUniformQuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, bits):
        bits = int(bits)
        if bits not in (1, 2, 4, 8):
            raise ValueError(f"ANN quantization bits must be one of 1, 2, 4, 8; got {bits}.")
        levels = (1 << bits) - 1
        probabilities = torch.clamp(inputs, 0.0, 1.0)
        return torch.round(probabilities * levels) / levels

    @staticmethod
    def backward(ctx, grad_output):
        # The quantizer is an STE; the preceding sigmoid supplies the bounded
        # derivative used by the original ANN measurement implementation.
        return grad_output, None

class EdgeSensor(nn.Module):
    def __init__(self):
        super().__init__()
        self.mode = 'ann'
        
        self.fc = nn.Linear(Config.INPUT_DIM, Config.MEAS_DIM, bias=True)
        self.register_buffer('thresh', torch.tensor(Config.ENCODER_THRESHOLD), persistent=False)
        self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.fc.weight, gain=0.5)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self, x, state=None, mode='ann', tau=None):
        feat = self.fc(x)
        feat_sig = torch.sigmoid(feat)
        
        if mode == 'ann':
            bits = int(Config.ANN_QUANT_BITS)
            y_out = ProbabilityUniformQuantizeSTE.apply(feat_sig, bits)
            aux_dict = {
                'tx_prob': None,
                'tx_hard': y_out,
                'tau': None,
                'quant_bits': bits,
                'transport': 'dense_quantized',
            }
            return y_out, None, aux_dict
            
        else:
            if state is None:
                prev_v = torch.zeros_like(feat_sig)
            else:
                prev_v = state
                
            rho = Config.TX_RHO
            v = prev_v * rho + feat_sig
            m = v
            theta = self.thresh
            
            if Config.TX_ABLATION_MODE == 'hard_spike':
                s_hard = SurrogateSpike.apply(m - theta)
                s_st = s_hard
                p = None
            else:
                if tau is None:
                    tau = Config.TAU_FINAL
                p = torch.sigmoid((m - theta) / tau)
                s_hard = (m >= theta).float()
                s_st = s_hard - p.detach() + p
            
            if Config.TX_RESET_MODE == 'soft':
                v_next = v - s_hard * theta
            else:
                v_next = v - s_hard * theta
                
            # Diagnostics stay as detached CUDA scalars.  Calling .item() here
            # would force seven host/device synchronizations per time step,
            # even though none of these quantities participates in the loss.
            # Trainer converts the accumulated epoch totals only once.
            collect_tx_stats = bool(
                getattr(Config, 'LOG_TX_STATE_STATS', False)
                or getattr(Config, 'LOG_ANNEAL_STATS', False)
            )
            with torch.no_grad():
                if collect_tx_stats:
                    tx_total_spikes = s_hard.sum()
                    tx_assisted_spikes = (
                        (feat_sig < theta) & (m >= theta)
                    ).sum()
                    tx_counterfactual_reset_spikes = (
                        feat_sig >= theta
                    ).sum()
                    active_mask = (s_hard > 0).float()
                    state_term = torch.clamp(
                        rho * prev_v, min=0.0
                    )
                    tx_state_contribution_sum = (
                        state_term * active_mask
                    ).sum()
                    if p is not None:
                        tx_soft_sum = p.sum()
                        tx_abs_soft_hard_sum = torch.abs(
                            p - s_hard
                        ).sum()
                    else:
                        tx_soft_sum = tx_total_spikes
                        tx_abs_soft_hard_sum = (
                            tx_total_spikes.new_zeros(())
                        )
                else:
                    zero = s_hard.new_zeros(())
                    tx_total_spikes = zero
                    tx_assisted_spikes = zero
                    tx_counterfactual_reset_spikes = zero
                    tx_state_contribution_sum = zero
                    tx_soft_sum = zero
                    tx_abs_soft_hard_sum = zero
            
            aux_dict = {
                'tx_prob': p if p is not None else None,
                'tx_hard': s_hard,
                'tau': tau if Config.TX_ABLATION_MODE != 'hard_spike' else None,
                'tx_total_spikes': tx_total_spikes,
                'tx_assisted_spikes': tx_assisted_spikes,
                'tx_counterfactual_reset_spikes': tx_counterfactual_reset_spikes,
                'tx_state_contribution_sum': tx_state_contribution_sum,
                'tx_threshold': theta.detach(),
                'tx_rho': rho,
                'tx_elements': s_hard.numel(),
                'tx_soft_sum': tx_soft_sum,
                'tx_abs_soft_hard_sum': tx_abs_soft_hard_sum,
                # Detached samples are used only to estimate the threshold-
                # margin density for the Theorem-1 diagnostic plot.
                'tx_threshold_margin': (
                    (m - theta).detach()
                    if getattr(Config, 'LOG_ANNEAL_STATS', False)
                    else None
                )
            }
            
            return s_st, v_next, aux_dict


class BinaryEdgeSensor(nn.Module):
    """Bias-free, row-energy-normalized linear sensing with single-threshold LIF."""

    def __init__(self, threshold, rho=None, normalization="energy"):
        super().__init__()
        self.mode = "spiking"
        self.fc = nn.Linear(
            Config.INPUT_DIM, Config.MEAS_DIM, bias=False
        )
        nn.init.orthogonal_(self.fc.weight)
        self.register_buffer(
            "thresh", torch.tensor(float(threshold)), persistent=False
        )
        self.rho = float(Config.TX_RHO if rho is None else rho)
        if normalization not in {"energy", "max", "none"}:
            raise ValueError(
                "normalization must be 'energy', 'max', or 'none'"
            )
        self.normalization = normalization

    def effective_weight(self):
        weight = self.fc.weight
        if self.normalization == "energy":
            denominator = weight.norm(p=2, dim=1, keepdim=True)
        elif self.normalization == "max":
            denominator = weight.abs().amax(dim=1, keepdim=True)
        else:
            return weight
        return weight / denominator.clamp_min(1e-8)

    def forward(self, x, state=None, mode="spiking", tau=None):
        current = F.linear(x, self.effective_weight(), None)
        previous_v = torch.zeros_like(current) if state is None else state
        membrane = self.rho * previous_v + current
        theta = self.thresh.to(device=current.device, dtype=current.dtype)

        spike_hard = (membrane >= theta).to(current.dtype)
        if Config.TX_ABLATION_MODE == "hard_spike":
            event_prob = SurrogateSpike.apply(membrane - theta)
            spike_st = event_prob
            tau = None
        else:
            if tau is None:
                tau = Config.TAU_FINAL
            tau_tensor = current.new_tensor(max(float(tau), 1e-5))
            event_prob = torch.sigmoid((membrane - theta) / tau_tensor)
            spike_st = spike_hard.detach() - event_prob.detach() + event_prob
        next_v = membrane - theta * spike_hard

        with torch.no_grad():
            hard_event = spike_hard.abs()
            active = hard_event > 0
            total = hard_event.sum()
            assisted = ((current < theta) & active).sum()
            counterfactual = (current >= theta).sum()
            state_contribution = (
                (self.rho * previous_v).clamp_min(0.0) * hard_event
            ).sum()

        aux_dict = {
            "tx_prob": event_prob,
            "tx_hard": spike_hard,
            "signed_tx": False,
            "raw_current": current,
            "tau": tau,
            "tx_total_spikes": total,
            "tx_assisted_spikes": assisted,
            "tx_counterfactual_reset_spikes": counterfactual,
            "tx_state_contribution_sum": state_contribution,
            "tx_threshold": theta.detach(),
            "tx_rho": self.rho,
            "tx_elements": spike_hard.numel(),
            "tx_soft_sum": event_prob.sum(),
            "tx_abs_soft_hard_sum": (
                event_prob - spike_hard.abs()
            ).abs().sum(),
            "tx_threshold_margin": (
                (membrane - theta).detach()
                if getattr(Config, "LOG_ANNEAL_STATS", False)
                else None
            ),
        }
        return spike_st, next_v, aux_dict
