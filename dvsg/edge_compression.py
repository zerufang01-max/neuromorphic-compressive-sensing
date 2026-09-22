import torch
import torch.nn as nn
import torch.nn.functional as F
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

class BinarizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs):
        ctx.save_for_backward(inputs)
        return torch.round(inputs)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class LIFEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('thresh', torch.tensor(Config.ENCODER_THRESHOLD))
        self.register_buffer('decay', torch.tensor(0.0))
        self.act = SurrogateSpike.apply
        
    def forward(self, x, state=None):
        if state is None:
            v = torch.zeros_like(x)
        else:
            v = state
            
        v = v * self.decay + x
        spike = self.act(v - self.thresh)
        v = v - spike * self.thresh
        
        return spike, v

class EdgeSensor(nn.Module):
    def __init__(self):
        super().__init__()
        self.mode = 'ann'
        
        def _build_conv(in_c, out_c, k, s, p):
            conv = nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=False)
            return nn.utils.weight_norm(conv)
        
        self.encoder = nn.Sequential(
            _build_conv(Config.INPUT_CHANNELS, 16, 3, 2, 1),
            nn.BatchNorm2d(16),
            LIFEncoder(),
            _build_conv(16, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            LIFEncoder(),
            _build_conv(32, Config.MEAS_CHANNELS, 3, 2, 1)
        )
        
        self._init_weights()
        self.signal_gain = 1.0 
        self.lif_out = LIFEncoder()

    def _init_weights(self):
        for m in self.encoder.modules():
            if isinstance(m, nn.Conv2d):
                if hasattr(m, 'weight_v'): 
                    nn.init.kaiming_normal_(m.weight_v, mode='fan_out', nonlinearity='relu')
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x_img, state=None, mode='ann', tau=None):
        if state is None:
            state = (None, None, None)
            
        s_lif1, s_lif2, s_final = state
        
        feat = x_img
        lif_idx = 0
        next_s_lif1, next_s_lif2 = None, None
        
        for layer in self.encoder:
            if isinstance(layer, LIFEncoder):
                if mode == 'ann':
                    feat = F.relu(feat)
                else:
                    if lif_idx == 0:
                        feat, next_s_lif1 = layer(feat, s_lif1)
                    elif lif_idx == 1:
                        feat, next_s_lif2 = layer(feat, s_lif2)
                lif_idx += 1
            else:
                feat = layer(feat)
                
        feat = feat * self.signal_gain
        feat_sig = torch.sigmoid(feat)
        
        if mode == 'ann':
            y_flat = feat_sig.view(feat_sig.size(0), -1)
            y_out = BinarizeSTE.apply(y_flat)
            new_state = (None, None, None)
            aux_dict = {
                'tx_prob': None,
                'tx_hard': y_out,
                'tau': None
            }
        else:
            if s_final is None:
                prev_v = torch.zeros_like(feat_sig)
            else:
                prev_v = s_final
                
            rho = Config.TX_RHO
            v = prev_v * rho + feat_sig
            m = v
            theta = self.lif_out.thresh
            
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
                
            with torch.no_grad():
                tx_total_spikes = s_hard.sum().item()
                tx_assisted_spikes = ((feat_sig < theta) & (m >= theta)).sum().item()
                tx_counterfactual_reset_spikes = (feat_sig >= theta).sum().item()
                active_mask = (s_hard > 0).float()
                state_term = torch.clamp(rho * prev_v, min=0.0)
                tx_state_contribution_sum = (state_term * active_mask).sum().item()
            
            y_out = s_st.view(s_st.size(0), -1)
            new_state = (next_s_lif1, next_s_lif2, v_next)
            
            aux_dict = {
                'tx_prob': p.view(p.size(0), -1) if p is not None else None,
                'tx_hard': s_hard.view(s_hard.size(0), -1),
                'tau': tau if Config.TX_ABLATION_MODE != 'hard_spike' else None,
                'tx_total_spikes': tx_total_spikes,
                'tx_assisted_spikes': tx_assisted_spikes,
                'tx_counterfactual_reset_spikes': tx_counterfactual_reset_spikes,
                'tx_state_contribution_sum': tx_state_contribution_sum,
                'tx_threshold': theta.item(),
                'tx_rho': rho,
                'tx_elements': s_hard.numel()
            }
            
        return y_out, new_state, aux_dict