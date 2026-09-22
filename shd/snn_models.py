import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import Config

SHD_FULL_NO_TEMPERATURE_VERSION = 1

class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold):
        if not isinstance(threshold, torch.Tensor):
            threshold = torch.tensor(threshold, device=input.device, dtype=input.dtype)
        ctx.save_for_backward(input, threshold)
        return (input >= threshold).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        input, threshold = ctx.saved_tensors
        grad_input = grad_output.clone()
        mask = ((input - threshold).abs() < 0.5).float()
        return grad_input * mask, None


def _orient_sequence(x, input_dim):
    if x.dim() != 3:
        raise ValueError(
            f"Expected [B, T, F] or [B, F, T], got {x.shape}."
        )
    if x.shape[-1] == int(input_dim):
        return x.float()
    if x.shape[1] == int(input_dim):
        return x.transpose(1, 2).float()
    raise ValueError(
        f"Cannot infer sequence layout from {x.shape}; "
        f"expected feature dimension {input_dim}."
    )


class AttentiveTemporalPool(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.score = nn.Linear(int(feature_dim), 1)

    def forward(self, sequence):
        weights = torch.softmax(
            self.score(sequence).squeeze(-1), dim=1
        )
        attended = torch.sum(sequence * weights.unsqueeze(-1), dim=1)
        temporal_mean = sequence.mean(dim=1)
        temporal_max = sequence.max(dim=1).values
        return torch.cat(
            [attended, temporal_mean, temporal_max], dim=-1
        )


class TemporalResidualBlock(nn.Module):
    def __init__(self, channels, dilation, dropout, norm_groups=None):
        super().__init__()
        channels = int(channels)
        dilation = int(dilation)
        self.block = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                groups=channels,
                bias=False,
            ),
            (nn.BatchNorm1d(channels) if norm_groups is None else nn.GroupNorm(norm_groups, channels)),
            nn.GELU(),
            nn.Conv1d(
                channels, channels, kernel_size=1, bias=False
            ),
            (nn.BatchNorm1d(channels) if norm_groups is None else nn.GroupNorm(norm_groups, channels)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )

    def forward(self, x):
        return x + self.block(x)


class TemporalConvClassifier(nn.Module):
    """Selected SHD TCN used for both semantic supervision and final probing."""
    def __init__(
        self,
        input_dim,
        num_classes,
        frame_dim=384,
        dropout=0.20,
        dilations=(1, 2, 4),
        norm_groups=32,
        **kwargs,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.frame_projection = nn.Linear(
            self.input_dim, int(frame_dim)
        )
        self.temporal = nn.Sequential(*[
            TemporalResidualBlock(frame_dim, dilation, dropout, norm_groups=norm_groups)
            for dilation in dilations
        ])
        self.pool = AttentiveTemporalPool(int(frame_dim))
        self.head = nn.Sequential(
            nn.LayerNorm(int(frame_dim) * 3),
            nn.Dropout(float(dropout)),
            nn.Linear(int(frame_dim) * 3, 256),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(256, int(num_classes)),
        )

    def forward(self, x):
        x = _orient_sequence(x, self.input_dim)
        features = F.gelu(self.frame_projection(x))
        features = self.temporal(
            features.transpose(1, 2)
        ).transpose(1, 2)
        return self.head(self.pool(features))


class HybridLISTA(nn.Module):
    def __init__(self):
        super().__init__()
        self.K = Config.NUM_LAYERS
        self.mode = 'rate'
        self.meas_dim = Config.MEAS_DIM
        self.sparse_dim = Config.SPARSE_DIM

        self.W_e = nn.Linear(self.meas_dim, self.sparse_dim, bias=False)
        self.S_k = nn.ModuleList([nn.Linear(self.sparse_dim, self.sparse_dim, bias=False) for _ in range(self.K)])
        initial_theta = max(float(Config.THETA_LISTA_ANN), 1e-4)
        self.theta_ann = nn.ParameterList([
            nn.Parameter(torch.tensor(initial_theta)) for _ in range(self.K)
        ])

        self.P_snn = nn.Linear(self.meas_dim, self.sparse_dim, bias=False)
        self.PD_snn_k = nn.ModuleList([
            nn.Linear(self.sparse_dim, self.sparse_dim, bias=False) 
            for _ in range(self.K)
        ])

        for k in range(self.K):
            self.register_buffer(f'theta_snn_{k}', torch.zeros(1, self.sparse_dim) + Config.THETA_LISTA_SNN, persistent=False)

        self.synthesis_dict = nn.Linear(self.sparse_dim, Config.INPUT_DIM, bias=False)
        self.semantic_head = TemporalConvClassifier(
            input_dim=self.sparse_dim,
            num_classes=Config.NUM_CLASSES,
            frame_dim=Config.SEMANTIC_TCN_DIM,
            norm_groups=Config.SEMANTIC_NORM_GROUPS,
            dropout=Config.SEMANTIC_DROPOUT,
        )

        self._init_weights()

    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if 'PD_snn_k' in name:
                    nn.init.normal_(m.weight, mean=0.0, std=1e-4)
                else:
                    nn.init.orthogonal_(m.weight, gain=0.5)
                
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def get_optimizer_groups(self, hp):
        current_mode = getattr(self, 'mode', 'spiking')
        lr_algo = hp.get('lr_lista_ann', 7e-4) if current_mode in ['rate', 'ann'] else hp.get('lr_lista_snn', 3e-4)
        lr_dict = hp.get('lr_dict', 5e-4)

        algo_params = []
        p_snn_params = []
        dict_params = []
        threshold_params = []
        semantic_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith('synthesis_dict.'):
                dict_params.append(param)
            elif name.startswith('theta_ann.'):
                threshold_params.append(param)
            elif name.startswith('semantic_head.'):
                semantic_params.append(param)
            elif (
                current_mode not in ['rate', 'ann']
                and name.startswith('P_snn.')
            ):
                p_snn_params.append(param)
            else:
                algo_params.append(param)

        return [
            {'params': algo_params, 'lr': lr_algo},
            {
                'params': p_snn_params,
                'lr': lr_algo,
                'weight_decay': float(
                    getattr(Config, 'SLISTA_P_WEIGHT_DECAY', 0.0)
                ),
            },
            {'params': dict_params, 'lr': lr_dict},
            {'params': threshold_params, 'lr': hp.get('lr_theta_ann', 5e-4)},
            {
                'params': semantic_params,
                'lr': hp.get('lr_semantic', 3e-4),
                'weight_decay': getattr(
                    Config, 'SEMANTIC_WEIGHT_DECAY', 1e-4
                ),
            },
        ]

    def get_threshold_value(self):
        if getattr(self, 'mode', 'spiking') in ('ann', 'rate'):
            return float(
                self.theta_ann[0].detach().clamp_min(1e-4).item()
            )
        return float(getattr(self, 'theta_snn_0').reshape(-1)[0].item())

    def get_ann_threshold_values(self):
        return [
            float(value.detach().clamp_min(1e-4).item())
            for value in self.theta_ann
        ]

    def project_ann_thresholds(self):
        """Projected-gradient constraint theta_k >= 1e-4."""
        with torch.no_grad():
            for value in self.theta_ann:
                value.clamp_(min=1e-4)

    def forward(self, y_frame, states=None):
        B = y_frame.shape[0]

        if self.mode == 'rate':
            z = torch.zeros(B, self.sparse_dim, device=y_frame.device)
            We_y = self.W_e(y_frame)
            layers_out = []

            for k in range(self.K):
                z_in = We_y + self.S_k[k](z)
                # Thresholds are optimized directly and projected to positive values.
                theta_k = self.theta_ann[k].clamp_min(1e-4)
                z = torch.sign(z_in) * F.relu(torch.abs(z_in) - theta_k)
                layers_out.append(z)

            out_rate = self.synthesis_dict(z)
            out_rate = F.relu(out_rate)
            x_recon = out_rate if getattr(Config, 'DECODER_OUTPUT', 'sigmoid') == 'linear' else torch.sigmoid(out_rate)
            return x_recon, layers_out, None, {}

        I_0 = self.P_snn(y_frame)

        if states is None:
            states = torch.zeros(self.K, B, self.sparse_dim, device=y_frame.device)

        frame_decay = y_frame.new_tensor(Config.DECODER_TAU)

        z_k = torch.zeros_like(I_0)
        next_states = torch.zeros_like(states)
        layers_out = []
        # Accumulate diagnostic spike counts on the model device.
        total_spikes = y_frame.new_zeros(())

        for k in range(self.K):
            theta_k = getattr(self, f'theta_snn_{k}')
            
            if k == 0:
                I_k = I_0
            else:
                I_k = I_0 - self.PD_snn_k[k](z_k)

            u_k = frame_decay * states[k] + I_k

            hard_pos = (u_k >= theta_k).to(u_k.dtype)
            hard_neg = (-u_k >= theta_k).to(u_k.dtype)
            # Apply the existing TX ablation switch to the reconstruction
            # surrogate as well. Hard forward spikes/reset stay identical.
            hard_spikes = hard_pos - hard_neg
            if getattr(Config, 'TX_ABLATION_MODE', 'none') == 'hard_spike':
                s_k = (
                    SurrogateSpike.apply(u_k, theta_k)
                    - SurrogateSpike.apply(-u_k, theta_k)
                )
            else:
                surrogate_tau = y_frame.new_tensor(
                    float(getattr(Config, 'SLISTA_SURROGATE_TAU', 0.3))
                ).clamp_min(1e-4)
                soft_pos = torch.sigmoid(
                    (u_k - theta_k) / surrogate_tau
                )
                soft_neg = torch.sigmoid(
                    (-u_k - theta_k) / surrogate_tau
                )
                soft_spikes = soft_pos - soft_neg
                s_k = (
                    hard_spikes.detach()
                    - soft_spikes.detach()
                    + soft_spikes
                )

            z_inc = s_k
            
            u_k = u_k - theta_k * hard_spikes
            next_states[k] = u_k

            z_k = z_k + z_inc
            layers_out.append(z_k)
            
            with torch.no_grad():
                total_spikes += (
                    hard_pos.detach().sum() + hard_neg.detach().sum()
                )

        # The discrete spike accumulator defines the sparse support.  Convert
        # the residual membrane into spike-count units, but use it only to
        # refine coefficients that were activated by at least one spike.
        # This prevents a continuous membrane residual from creating new
        # nonzero coefficients outside the spike-defined support.
        theta_last = getattr(self, f'theta_snn_{self.K - 1}')
        membrane_readout = getattr(
            Config, 'SLISTA_MEMBRANE_READOUT', 'support_gated'
        )
        if membrane_readout == 'support_gated':
            support_mask = (z_k != 0).to(dtype=z_k.dtype)
            z_final_raw = (
                z_k
                + support_mask * (next_states[-1] / theta_last)
            )
        elif membrane_readout == 'spike_only':
            z_final_raw = z_k
        else:
            raise ValueError(
                'SLISTA_MEMBRANE_READOUT must be '
                "'support_gated' or 'spike_only'; "
                f'got {membrane_readout!r}.'
            )
        z_final = F.softshrink(z_final_raw, Config.THETA_LISTA_ANN)
        if self.K > 0:
            layers_out[-1] = z_final

        out_spk = self.synthesis_dict(z_final)
        
        out_spk = F.relu(out_spk)
        x_recon = out_spk if getattr(Config, 'DECODER_OUTPUT', 'sigmoid') == 'linear' else torch.sigmoid(out_spk)
        
        internal_stats = {
            'fired_rate': (
                total_spikes
                / max(
                    1,
                    2 * B * self.sparse_dim * max(self.K, 1),
                )
            )
        }
        return x_recon, layers_out, next_states, internal_stats

class LSTMRecon(nn.Module):
    def __init__(self):
        super().__init__()
        self.mode = 'ann'
        self.hidden_dim = 512
        self.lstm = nn.LSTMCell(Config.MEAS_DIM, self.hidden_dim)
        self.fc = nn.Linear(self.hidden_dim, Config.INPUT_DIM)
        
        self.semantic_head = TemporalConvClassifier(
            input_dim=self.hidden_dim,
            num_classes=Config.NUM_CLASSES,
            frame_dim=Config.SEMANTIC_TCN_DIM,
            norm_groups=Config.SEMANTIC_NORM_GROUPS,
            dropout=Config.SEMANTIC_DROPOUT,
        )

    def get_optimizer_groups(self, hp):
        reconstruction_params = []
        semantic_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith('semantic_head.'):
                semantic_params.append(param)
            else:
                reconstruction_params.append(param)
        return [
            {
                'params': reconstruction_params,
                'lr': hp.get('lr_lstm', 7e-4),
            },
            {
                'params': semantic_params,
                'lr': hp.get('lr_semantic', 3e-4),
                'weight_decay': Config.SEMANTIC_WEIGHT_DECAY,
            },
        ]

    def forward(self, y_frame, states=None):
        B = y_frame.shape[0]
        if states is None:
            h = torch.zeros(B, self.hidden_dim, device=y_frame.device)
            c = torch.zeros(B, self.hidden_dim, device=y_frame.device)
            states = (h, c)
            
        h, c = states
        h_next, c_next = self.lstm(y_frame, (h, c))
        out = self.fc(h_next)
        
        x_recon = out 
        return x_recon, [h_next], (h_next, c_next), {}
        
class RegressionGLIF(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dt = Config.BIN_DT_SEC 
        self.register_buffer(
            'log_tau_fast', torch.tensor(float(Config.SCNN_LEAK))
        )
        self.register_buffer(
            'surrogate_tau',
            torch.tensor(float(Config.SCNN_SURROGATE_TAU)),
            persistent=False,
        )
        
    def forward(self, x_in, threshold, init_v=None, mode='spiking'):
        if mode == 'rate':
            pos_rate = torch.relu(x_in - threshold)
            neg_rate = torch.relu(-x_in - threshold)
            return pos_rate - neg_rate, None
        else:
            decay = self.log_tau_fast
            V = init_v if init_v is not None else torch.zeros_like(x_in)
            V = V * decay + x_in
            
            if not isinstance(threshold, torch.Tensor):
                threshold = torch.tensor(threshold, device=x_in.device, dtype=x_in.dtype)
                
            hard_pos = (V >= threshold).to(V.dtype)
            hard_neg = (-V >= threshold).to(V.dtype)
            tau = self.surrogate_tau.to(V).clamp_min(1e-4)
            soft_pos = torch.sigmoid((V - threshold) / tau)
            soft_neg = torch.sigmoid((-V - threshold) / tau)
            soft_spikes = soft_pos - soft_neg
            hard_spikes = hard_pos - hard_neg
            spikes = (
                hard_spikes.detach()
                - soft_spikes.detach()
                + soft_spikes
            )
            
            V = V - hard_spikes * threshold
            return spikes, V

class SpikingResBlockFC(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels, bias=True)
        self.lif1 = RegressionGLIF(channels)
        
        self.fc2 = nn.Linear(channels, channels, bias=True)
        self.lif2 = RegressionGLIF(channels)

    def forward(self, x, states, thresh, mode='spiking'):
        identity = x
        out = self.fc1(x)
        out, s1 = self.lif1(out, thresh, init_v=states[0], mode=mode)
        
        out = self.fc2(out)
        out = out + identity
        out, s2 = self.lif2(out, thresh, init_v=states[1], mode=mode)
        return out, [s1, s2]

class SpikingDenseRecon(nn.Module):
    def __init__(self):
        super().__init__()
        self.mode = 'spiking'
        self.meas_dim = Config.MEAS_DIM
        
        self.feat_dim = 256
        self.sparse_dim = 700

        self.head_1 = nn.Linear(self.meas_dim, self.feat_dim, bias=False)
        self.lif_head_1 = RegressionGLIF(self.feat_dim)
        self.head_2 = nn.Linear(self.feat_dim, self.sparse_dim, bias=False)
        self.lif_head_2 = RegressionGLIF(self.sparse_dim)

        # Three event-driven residual blocks recover depth without the former
        # 700->1700->2600 dense synthesis expansion.  Only the final dictionary
        # projection uses MACs; all preceding synapses consume sparse ACs.
        self.res_blocks = nn.ModuleList([
            SpikingResBlockFC(self.sparse_dim) for _ in range(3)
        ])
        self.tail = nn.Linear(self.sparse_dim, Config.INPUT_DIM, bias=False)
        
        self._init_weights()
        
        self.register_buffer(
            'thresh',
            torch.tensor(Config.SCNN_NEURON_THRESHOLD),
            persistent=False,
        )
        self.semantic_head = TemporalConvClassifier(
            input_dim=self.sparse_dim,
            num_classes=Config.NUM_CLASSES,
            frame_dim=Config.SEMANTIC_TCN_DIM,
            norm_groups=Config.SEMANTIC_NORM_GROUPS,
            dropout=Config.SEMANTIC_DROPOUT,
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def get_optimizer_groups(self, hp):
        lr_algo = hp.get('lr_spiking_dense', hp.get('lr_lista_snn', 3e-4)) \
            if self.mode == 'spiking' else hp.get('lr_lista_ann', 7e-4)
        lr_dict = hp.get('lr_dict', 5e-4)

        algo_params = []
        dict_params = []
        semantic_params = []
        dictionary_prefixes = ('tail.',)

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith('semantic_head.'):
                semantic_params.append(param)
            elif name.startswith(dictionary_prefixes):
                dict_params.append(param)
            else:
                algo_params.append(param)

        return [
            {'params': algo_params, 'lr': lr_algo},
            {'params': dict_params, 'lr': lr_dict},
            {
                'params': semantic_params,
                'lr': hp.get('lr_semantic', 3e-4),
                'weight_decay': Config.SEMANTIC_WEIGHT_DECAY,
            },
        ]

    def forward(self, y_flat, states=None):
        B = y_flat.shape[0]
        x = y_flat
        
        if states is None:
            states = [None, None] + [
                [None, None] for _ in self.res_blocks
            ]
        elif len(states) != 2 + len(self.res_blocks):
            raise ValueError(
                f"Expected {2 + len(self.res_blocks)} SCNN state groups, "
                f"but received {len(states)}."
            )
        next_states = []

        x = self.head_1(x)
        x, state_head_1 = self.lif_head_1(
            x, self.thresh, init_v=states[0], mode=self.mode
        )
        next_states.append(state_head_1)

        x = self.head_2(x)
        x, state_head_2 = self.lif_head_2(
            x, self.thresh, init_v=states[1], mode=self.mode
        )
        next_states.append(state_head_2)

        for block_index, block in enumerate(self.res_blocks):
            x, block_states = block(
                x,
                states[2 + block_index],
                self.thresh,
                mode=self.mode,
            )
            next_states.append(block_states)

        latent_code = x
        out_raw = self.tail(x)
         
        out_img = F.relu(out_raw)
        return out_img, [latent_code], next_states, {}


class FixedSignedLIF(nn.Module):
    """Signed LIF with hard forward spikes and sigmoid surrogate gradients."""

    def __init__(self, threshold=0.5, leak=0.1, surrogate_tau=0.3):
        super().__init__()
        self.threshold = float(threshold)
        self.leak = float(leak)
        self.surrogate_tau = float(surrogate_tau)

    def forward(self, current, state=None):
        voltage = torch.zeros_like(current) if state is None else state
        voltage = self.leak * voltage + current
        theta = current.new_tensor(self.threshold)
        tau = current.new_tensor(max(self.surrogate_tau, 1e-4))
        hard = (
            (voltage >= theta).to(current.dtype)
            - (voltage <= -theta).to(current.dtype)
        )
        soft = (
            torch.sigmoid((voltage - theta) / tau)
            - torch.sigmoid((-voltage - theta) / tau)
        )
        spikes = hard.detach() - soft.detach() + soft
        return spikes, voltage - theta * hard


class GenericSNNRecon(nn.Module):
    """Eight-layer residual FC-SNN baseline without LISTA iterations."""

    def __init__(self, hidden_dim=700, blocks=3, neuron_threshold=0.5,
                 leak=0.1, surrogate_tau=0.3):
        super().__init__()
        self.mode = "spiking"
        self.feat_dim = 256
        self.hidden_dim = int(hidden_dim)
        self.blocks = int(blocks)
        self.neuron_threshold = float(neuron_threshold)
        self.leak = float(leak)
        self.surrogate_tau = float(surrogate_tau)
        self.head_1 = nn.Linear(
            Config.MEAS_DIM, self.feat_dim, bias=False
        )
        self.input_lif = FixedSignedLIF(
            self.neuron_threshold, self.leak, self.surrogate_tau
        )
        self.head_2 = nn.Linear(
            self.feat_dim, self.hidden_dim, bias=False
        )
        self.latent_lif = FixedSignedLIF(
            self.neuron_threshold, self.leak, self.surrogate_tau
        )
        self.res_blocks = nn.ModuleList([
            SpikingResBlockFC(self.hidden_dim)
            for _ in range(self.blocks)
        ])
        # This learned synthesis map is the baseline dictionary Psi.  The
        # spiking hidden representation is its latent code z.
        self.synthesis_dict = nn.Linear(
            self.hidden_dim, Config.INPUT_DIM, bias=False
        )
        self.semantic_head = TemporalConvClassifier(
            input_dim=self.hidden_dim,
            num_classes=Config.NUM_CLASSES,
            frame_dim=Config.SEMANTIC_TCN_DIM,
            norm_groups=Config.SEMANTIC_NORM_GROUPS,
            dropout=Config.SEMANTIC_DROPOUT,
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )

    def get_threshold_value(self):
        return self.neuron_threshold

    def get_optimizer_groups(self, hp):
        algorithm = []
        dictionary = []
        semantic = []
        for name, parameter in self.named_parameters():
            if name.startswith("semantic_head."):
                semantic.append(parameter)
            elif name.startswith("synthesis_dict."):
                dictionary.append(parameter)
            else:
                algorithm.append(parameter)
        return [
            {
                "params": algorithm,
                "lr": hp.get("lr_generic_snn", 6e-4),
            },
            {"params": dictionary, "lr": hp.get("lr_dict", 5e-4)},
            {
                "params": semantic,
                "lr": hp.get("lr_semantic", 3e-4),
                "weight_decay": Config.SEMANTIC_WEIGHT_DECAY,
            },
        ]

    def forward(self, y_frame, states=None):
        if states is None:
            states = [None, None] + [
                [None, None] for _ in range(self.blocks)
            ]
        hidden, head_1_state = self.input_lif(
            self.head_1(y_frame), states[0]
        )
        hidden, head_2_state = self.latent_lif(
            self.head_2(hidden), states[1]
        )
        next_states = [head_1_state, head_2_state]
        for index in range(self.blocks):
            hidden, block_states = self.res_blocks[index](
                hidden,
                states[index + 2],
                hidden.new_tensor(self.neuron_threshold),
                mode=self.mode,
            )
            next_states.append(block_states)

        # Standard membrane readout: recover the final layer's continuous
        # pre-reset membrane while retaining hard spikes in the recurrent
        # state.  This is the 700-dimensional latent code z.
        final_post_reset = next_states[-1][1]
        latent_code = (
            final_post_reset
            + hidden * hidden.new_tensor(self.neuron_threshold)
        )
        reconstruction = F.relu(self.synthesis_dict(latent_code))
        return reconstruction, [latent_code], next_states, {}
