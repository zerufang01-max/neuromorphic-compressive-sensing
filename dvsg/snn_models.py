import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config
import math

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
        mask = ((input - threshold).abs() < Config.SURROGATE_WIDTH).float()
        return grad_input * mask, None

class RegressionGLIF(nn.Module):
    def __init__(self, channels):
        super().__init__()
       
        self.register_buffer('decay', torch.tensor(Config.DECODER_TAU)) 
        
    def forward(self, x_in, threshold, init_v=None, mode='spiking'):
        if mode == 'rate':
            return torch.relu(x_in - threshold), None
        else:
            V = init_v if init_v is not None else torch.zeros_like(x_in)
           
            V = V * self.decay + x_in
            
            if not isinstance(threshold, torch.Tensor):
                threshold = torch.tensor(threshold, device=x_in.device, dtype=x_in.dtype)
                
            spikes = SurrogateSpike.apply(V, threshold)
            V = V - spikes * threshold
            return spikes, V

class BiasFreeBatchNorm(nn.BatchNorm2d):
    def forward(self, input):
        if self.affine:
             with torch.no_grad():
                 self.bias.zero_()
        return super().forward(input)

class VideoClassifier(nn.Module):
    def __init__(self, num_classes, time_steps=16):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(Config.INPUT_CHANNELS, 64, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(128), nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1)
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.50), nn.Linear(256, num_classes)
        )

    def forward(self, x_video):
        if isinstance(x_video, list): x = torch.stack(x_video, dim=1)
        else: x = x_video
        x = x.permute(0, 2, 1, 3, 4)
        x = self.features(x)
        return self.head(x)

class LatentVideoClassifier(nn.Module):
    def __init__(self, in_channels, num_classes, time_steps=16):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(128), nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1)
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.50), nn.Linear(256, num_classes)
        )

    def forward(self, x_video):
        if isinstance(x_video, list): x = torch.stack(x_video, dim=1)
        else: x = x_video
        x = x.permute(0, 2, 1, 3, 4)
        x = self.features(x)
        return self.head(x)

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1) 
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i); f = torch.sigmoid(cc_f); o = torch.sigmoid(cc_o); g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

class ConvLSTMRecon(nn.Module):
    def __init__(self):
        super().__init__()
        self.h_dim = Config.IMG_H // Config.MEAS_STRIDE 
        self.meas_ch = Config.MEAS_CHANNELS
        self.hidden_dim = Config.SPARSE_DIM
        self.lstm_cell = ConvLSTMCell(self.meas_ch, self.hidden_dim, 3, True)
        
        self.tail = nn.Sequential(
            nn.ConvTranspose2d(self.hidden_dim, 128, 4, 2, 1),
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, Config.INPUT_CHANNELS, 3, 1, 1),
            nn.Sigmoid()
        )
        self._init_weights()
        
  
        self.classifier_image = VideoClassifier(Config.NUM_CLASSES, Config.TIME_STEPS)
        self.classifier_latent = LatentVideoClassifier(self.hidden_dim, Config.NUM_CLASSES, Config.TIME_STEPS)
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def get_optimizer_groups(self, hp):
        lr_algo = hp.get('lr_lstm', 7e-4)
        lr_dict = hp.get('lr_dict', 5e-4)
        
        algo_params = []
        dict_params = []
        
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            # 'tail' decodes the hidden state back to image space
            if 'tail' in name:
                dict_params.append(param)
            else:
                algo_params.append(param)

        return [
            {'params': algo_params, 'lr': lr_algo},
            {'params': dict_params, 'lr': lr_dict}
        ]

    def forward(self, y_flat, states=None):
        B = y_flat.shape[0]
        y_spatial = y_flat.view(B, self.meas_ch, self.h_dim, self.h_dim)
        if states is None:
            h = torch.zeros(B, self.hidden_dim, self.h_dim, self.h_dim, device=y_flat.device)
            c = torch.zeros(B, self.hidden_dim, self.h_dim, self.h_dim, device=y_flat.device)
            states = (h, c)
        h, c = states
        h_next, c_next = self.lstm_cell(y_spatial, (h, c))
        x_recon = self.tail(h_next)
        return x_recon, [h_next], (h_next, c_next), {}


class ConvolutionalLISTA(nn.Module):
    def __init__(self):
        super().__init__()
        self.K = Config.NUM_LAYERS
        self.mode = 'rate'
        self.h_dim = Config.IMG_H // Config.MEAS_STRIDE
        self.meas_ch = Config.MEAS_CHANNELS
        self.sparse_dim = Config.SPARSE_DIM

        self.W_e = nn.Conv2d(self.meas_ch, self.sparse_dim, 3, 1, 1, bias=False)
        self.S_k = nn.ModuleList([nn.Conv2d(self.sparse_dim, self.sparse_dim, 3, 1, 1, bias=False) for _ in range(self.K)])
        self.theta_ann = nn.ParameterList([nn.Parameter(torch.tensor(Config.THETA_LISTA_ANN)) for _ in range(self.K)])

        self.P_snn = nn.Conv2d(self.meas_ch, self.sparse_dim, 3, 1, 1, bias=False)
        self.PD_snn = nn.ModuleList([nn.Conv2d(self.sparse_dim, self.sparse_dim, 3, 1, 1, bias=False) for _ in range(self.K)])
        self.register_buffer('fixed_theta_snn', torch.zeros(1, self.sparse_dim, 1, 1) + Config.THETA_LISTA_SNN)

        self.tail = nn.Sequential(
            nn.ConvTranspose2d(self.sparse_dim, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, Config.INPUT_CHANNELS, 3, 1, 1),
        )
        self._init_weights()
        
     
        self.classifier_image = VideoClassifier(Config.NUM_CLASSES, Config.TIME_STEPS)
        self.classifier_latent = LatentVideoClassifier(self.sparse_dim, Config.NUM_CLASSES, Config.TIME_STEPS)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=0.5)
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def get_optimizer_groups(self, hp):
        current_mode = getattr(self, 'mode', 'spiking')
        lr_algo = hp.get('lr_lista_ann', 7e-4) if current_mode in ['rate', 'ann'] else hp.get('lr_lista_snn', 7e-4)
        lr_dict = hp.get('lr_dict', 5e-4)

        algo_params = []
        dict_params = []
        
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            # 'tail' acts as the synthesis dictionary D in x = Dz
            if 'tail' in name:
                dict_params.append(param)
            else:
                algo_params.append(param)

        return [
            {'params': algo_params, 'lr': lr_algo},
            {'params': dict_params, 'lr': lr_dict}
        ]

    def get_threshold_value(self):
        return float(self.fixed_theta_snn.reshape(-1)[0].item())

    def forward(self, y_flat, states=None):
        B = y_flat.shape[0]
        y_meas = y_flat.view(B, self.meas_ch, self.h_dim, self.h_dim)

        if self.mode == 'rate':
            z = torch.zeros(B, self.sparse_dim, self.h_dim, self.h_dim, device=y_flat.device)
            We_y = self.W_e(y_meas)
            layers_out = []
            for k in range(self.K):
                z_in = We_y + self.S_k[k](z)
                z = F.softshrink(z_in, self.theta_ann[k].item())
                layers_out.append(z)
            return self.tail(z), layers_out, None, {}

        theta = self.fixed_theta_snn
        I_0 = self.P_snn(y_meas)
        
        if states is None:
            states = torch.zeros(
                self.K,
                B,
                self.sparse_dim,
                self.h_dim,
                self.h_dim,
                device=y_flat.device,
            )
        
        frame_decay = y_meas.new_tensor(Config.DECODER_TAU)
        
        # z_k is maintained directly in coefficient-amplitude units.
        # Each signed spike contributes +/- theta to the sparse coefficient.
        z_k = torch.zeros_like(I_0)
        next_states = torch.zeros_like(states)
        layers_out = []
        total_spikes = 0.0
        
        for k in range(self.K):
            # The feedback convolution now receives coefficients in the same
            # amplitude units used by the membrane potential and final output.
            I_k = I_0 if k == 0 else I_0 - self.PD_snn[k](z_k)
        
            u_k = frame_decay * states[k] + I_k
        
            spk_pos = SurrogateSpike.apply(u_k, theta)
            spk_neg = SurrogateSpike.apply(-u_k, theta)
            s_k = spk_pos - spk_neg
        
            z_inc = theta * s_k
            z_k = z_k + z_inc
        
            # Soft reset uses exactly the same coefficient increment.
            u_k = u_k - z_inc
        
            next_states[k] = u_k
            layers_out.append(z_k)
        
            total_spikes += (
                spk_pos.sum().item()
                + spk_neg.sum().item()
            )
        
        # Add the final sub-threshold residual potential to the accumulated
        # spike-domain coefficient estimate.
        z_final_raw = z_k + next_states[-1]
        
        # Keep the existing shared threshold. Setting INIT_THRESHOLD=0.1 in
        # config.py also sets THETA_LISTA_ANN and THETA_LISTA_SNN to 0.1.
        z_final = F.softshrink(
            z_final_raw,
            Config.THETA_LISTA_ANN,
        )
        
        if self.K > 0:
            layers_out[-1] = z_final

        internal_stats = {
            'fired_rate': total_spikes / max(1, 2 * B * self.sparse_dim * self.h_dim * self.h_dim * max(self.K, 1)),
            'theta': float(theta.mean().item()),
            'frame_decay': float(frame_decay.item()),
        }
        return self.tail(z_final), layers_out, next_states, internal_stats

class ConvolutionalLISTA_ImageSpace(ConvolutionalLISTA):
    def __init__(self):
        super().__init__()

class SpikingResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)  
        self.lif1 = RegressionGLIF(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)  
        self.lif2 = RegressionGLIF(channels)

    def forward(self, x, states, thresh, mode='spiking'):
        identity = x
        out = self.bn1(self.conv1(x))
        out, s1 = self.lif1(out, thresh, init_v=states[0], mode=mode)
        out = self.bn2(self.conv2(out)) + identity
        out, s2 = self.lif2(out, thresh, init_v=states[1], mode=mode)
        return out, [s1, s2]

class SpikingCNNRecon(nn.Module):
    def __init__(self):
        super().__init__()
        self.mode = 'spiking'
        self.h_dim = Config.IMG_H // Config.MEAS_STRIDE
        self.meas_ch = Config.MEAS_CHANNELS

        # Match the active S-LISTA reconstruction capacity.
        self.feat_dim = Config.FEATURE_DIM
        self.sparse_dim = Config.SPARSE_DIM

        self.head_1 = nn.Conv2d(
            self.meas_ch,
            self.feat_dim,
            3,
            1,
            1,
            bias=False,
        )
        self.bn_head_1 = BiasFreeBatchNorm(self.feat_dim, affine=True)
        self.act_head_1 = nn.ReLU()

        self.head_2 = nn.Conv2d(
            self.feat_dim,
            self.sparse_dim,
            3,
            1,
            1,
            bias=False,
        )
        self.bn_head_2 = nn.BatchNorm2d(self.sparse_dim)
        self.lif_head = RegressionGLIF(self.sparse_dim)

        # One residual block gives approximately the same active
        # reconstruction parameter count as S-LISTA.
        self.res1 = SpikingResBlock(self.sparse_dim)

        self.conv_up1 = nn.ConvTranspose2d(
            self.sparse_dim,
            256,
            4,
            2,
            1,
        )
        self.bn_up1 = nn.BatchNorm2d(256)
        self.act_up1 = nn.ReLU(True)

        self.conv_up2 = nn.ConvTranspose2d(
            256,
            128,
            4,
            2,
            1,
        )
        self.bn_up2 = nn.BatchNorm2d(128)
        self.act_up2 = nn.ReLU(True)

        self.tail = nn.Sequential(
            nn.Conv2d(128, Config.INPUT_CHANNELS, 3, 1, 1)
        )

        self._init_weights()
        self.thresh = nn.Parameter(torch.tensor(Config.DECODER_THRESHOLD))

        self.classifier_image = VideoClassifier(
            Config.NUM_CLASSES,
            Config.TIME_STEPS,
        )
        self.classifier_latent = LatentVideoClassifier(
            self.sparse_dim,
            Config.NUM_CLASSES,
            Config.TIME_STEPS,
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode='fan_out',
                    nonlinearity='relu',
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.affine:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def get_optimizer_groups(self, hp):
        lr_algo = hp.get('lr_spiking_cnn', 1e-3)
        lr_dict = hp.get('lr_dict', 5e-4)
        
        algo_params = []
        dict_params = []
        
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            # All upsampling layers and the final tail act as the dictionary
            if 'conv_up' in name or 'bn_up' in name or 'tail' in name:
                dict_params.append(param)
            else:
                algo_params.append(param)

        return [
            {'params': algo_params, 'lr': lr_algo},
            {'params': dict_params, 'lr': lr_dict}
        ]

    def forward(self, y_flat, states=None):
        B = y_flat.shape[0]
        x = y_flat.view(B, self.meas_ch, self.h_dim, self.h_dim)

        if states is None:
            states = [None, [None, None]]

        next_states = []

        x = self.act_head_1(self.bn_head_1(self.head_1(x)))
        x, s_head = self.lif_head(
            self.bn_head_2(self.head_2(x)),
            self.thresh,
            init_v=states[0],
            mode=self.mode,
        )
        next_states.append(s_head)

        x, s_res1 = self.res1(
            x,
            states[1],
            self.thresh,
            mode=self.mode,
        )
        next_states.append(s_res1)

        latent_code = x

        x = self.act_up1(self.bn_up1(self.conv_up1(x)))
        x = self.act_up2(self.bn_up2(self.conv_up2(x)))
        x_recon = self.tail(x)

        return x_recon, [latent_code], next_states, {}

