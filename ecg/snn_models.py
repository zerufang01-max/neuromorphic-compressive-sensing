"""Unfolded ECG recovery models with fixed or learned synthesis dictionaries."""
import math

import numpy as np
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, threshold):
        if not isinstance(threshold, torch.Tensor):
            threshold = torch.tensor(
                threshold, device=input_tensor.device, dtype=input_tensor.dtype
            )
        ctx.save_for_backward(input_tensor, threshold)
        return (input_tensor >= threshold).to(input_tensor.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, threshold = ctx.saved_tensors
        mask = ((input_tensor - threshold).abs() < Config.SURROGATE_WIDTH).to(
            input_tensor.dtype
        )
        return grad_output * mask, None


def get_wavelet_matrix(n=256, wavelet_name="sym4"):
    dummy = np.zeros(n, dtype=np.float64)
    coeffs = pywt.wavedec(dummy, wavelet_name, mode="periodization")
    lengths = [len(c) for c in coeffs]
    offsets = np.cumsum([0] + lengths)

    matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        flat = np.zeros(n, dtype=np.float64)
        flat[i] = 1.0
        coeff_list = [
            flat[offsets[j] : offsets[j + 1]] for j in range(len(lengths))
        ]
        matrix[i] = pywt.waverec(
            coeff_list, wavelet_name, mode="periodization"
        )[:n]
    return torch.tensor(matrix, dtype=torch.float32)


def soft_threshold(x, threshold):
    return torch.sign(x) * F.relu(torch.abs(x) - threshold)


def inv_softplus(value):
    return math.log(math.expm1(max(float(value), 1e-6)))


def fixed_operator(phi):
    dictionary = get_wavelet_matrix(
        Config.N, Config.WAVELET_NAME
    ).to(device=phi.device, dtype=phi.dtype)
    operator = phi @ dictionary.t()
    return dictionary, operator


class LearnableDictionaryMixin:
    """Synthesis dictionary shared by the unfolded recovery models."""

    def _init_learnable_dictionary(self, phi):
        if Config.CS_MODE == "fixed_wavelet":
            dictionary = get_wavelet_matrix(
                self.N, Config.WAVELET_NAME
            ).to(device=phi.device, dtype=phi.dtype)
            self.register_buffer("D", dictionary)
        else:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(Config.SEED) + 1009)
            gaussian = torch.randn(
                self.N, self.N, generator=generator, dtype=torch.float32
            )
            dictionary = (gaussian / math.sqrt(self.N)).to(
                device=phi.device, dtype=phi.dtype
            )
            self.D = nn.Parameter(dictionary)
        operator = phi @ dictionary.t()
        self.register_buffer("Phi", phi.detach().clone())
        return operator

    def _dictionary_named_parameters(self):
        if isinstance(self.D, nn.Parameter):
            return [("D", self.D)]
        return []

    @property
    def A(self):
        # D stores one synthesis atom per row because reconstruction is z @ D.
        return self.Phi @ self.D.t()

    def current_operator(self, phi_override=None):
        phi = self.Phi if phi_override is None else phi_override
        return phi @ self.D.t()

    def synchronize_dictionary_operator(self, phi_override=None):
        """Refresh algorithm-specific buffers after a dictionary update."""
        return None


class HybridLISTA(LearnableDictionaryMixin, nn.Module):
    def __init__(self, phi=None):
        super().__init__()
        self.K = Config.NUM_LAYERS
        self.mode = "spiking"
        self.M = Config.M
        self.N = Config.N

        if phi is None:
            phi = torch.randn(self.M, self.N) / math.sqrt(self.M)
        operator = self._init_learnable_dictionary(phi)

        eta = 1.0 / torch.linalg.matrix_norm(operator, ord=2).square().clamp_min(1e-8)

        self.W_e = nn.Linear(self.M, self.N, bias=False)
        self.S_k = nn.ModuleList(
            [nn.Linear(self.N, self.N, bias=False) for _ in range(max(self.K - 1, 0))]
        )
        self.theta_ann_raw = nn.ParameterList(
            [
                nn.Parameter(torch.tensor(inv_softplus(Config.THETA_LISTA_ANN)))
                for _ in range(self.K)
            ]
        )

        self.P_snn = nn.Linear(self.M, self.N, bias=False)
        self.PD_snn_k = nn.ModuleList(
            [nn.Linear(self.N, self.N, bias=False) for _ in range(max(self.K - 1, 0))]
        )
        # One signed-spike threshold is required for every unfolded stage.
        # The recurrent feedback still needs only K - 1 operators.
        for k in range(self.K):
            self.register_buffer(
                f"theta_snn_{k}",
                torch.full((1, self.N), float(Config.THETA_LISTA_SNN)),
                persistent=False,
            )

        with torch.no_grad():
            self.W_e.weight.copy_(eta * operator.t())
            recurrent = torch.eye(self.N, device=operator.device) - eta * operator.t() @ operator
            for layer in self.S_k:
                layer.weight.copy_(recurrent)

            # Standard nn.Linear fan-in initialization.
            nn.init.kaiming_uniform_(self.P_snn.weight, a=math.sqrt(5))
            for layer in self.PD_snn_k:
                nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))

    def set_mode(self, mode):
        self.mode = "spiking" if mode == "snn" else mode

    def get_threshold_value(self):
        if self.mode == "spiking":
            return float(Config.THETA_LISTA_SNN)
        return float(F.softplus(self.theta_ann_raw[0]).detach().item())

    def get_ann_threshold_values(self):
        return [float(F.softplus(value).detach().item()) for value in self.theta_ann_raw]

    def active_named_parameters(self, mode):
        mode = "spiking" if mode == "snn" else mode
        params = self._dictionary_named_parameters()
        if mode == "ann":
            params += list(self.W_e.named_parameters(prefix="W_e"))
            for i, layer in enumerate(self.S_k):
                params += list(layer.named_parameters(prefix=f"S_k.{i}"))
            params += [
                (f"theta_ann_raw.{i}", value)
                for i, value in enumerate(self.theta_ann_raw)
            ]
        elif mode == "spiking":
            params += list(self.P_snn.named_parameters(prefix="P_snn"))
            for i, layer in enumerate(self.PD_snn_k):
                params += list(layer.named_parameters(prefix=f"PD_snn_k.{i}"))
        return [(name, value) for name, value in params if value.requires_grad]

    def forward(self, y_recv, states=None, return_debug=False, phi_override=None):
        batch = y_recv.shape[0]

        if self.mode == "ann":
            wy = self.W_e(y_recv)
            z = soft_threshold(wy, F.softplus(self.theta_ann_raw[0]))
            layers_out = [z]
            for k, layer in enumerate(self.S_k, start=1):
                z = soft_threshold(
                    wy + layer(z), F.softplus(self.theta_ann_raw[k])
                )
                layers_out.append(z)

            x_recon = z @ self.D
            macs = batch * (
                self.M * self.N + max(self.K - 1, 0) * self.N * self.N
            )
            return x_recon, layers_out, None, {
                "MACs": float(macs), "ACs": 0.0, "spike_events": 0.0
            }

        drive0 = self.P_snn(y_recv)
        z = torch.zeros_like(drive0)
        layers_out = []
        total_spikes = y_recv.new_zeros(())
        total_acs = y_recv.new_zeros(())
        layer_spike_events = []
        drives = []
        membranes = []
        spikes = []

        final_membrane = torch.zeros_like(drive0)
        final_theta = drive0.new_tensor(Config.THETA_LISTA_SNN)

        for stage in range(self.K):
            if stage == 0:
                drive = drive0
            else:
                with torch.no_grad():
                    total_acs += z.detach().abs().sum() * self.N
                drive = drive0 - self.PD_snn_k[stage - 1](z)

            theta = getattr(self, f"theta_snn_{stage}")
            spike_pos = SurrogateSpike.apply(drive, theta)
            spike_neg = SurrogateSpike.apply(-drive, theta)
            signed_spike = spike_pos - spike_neg

            # z is a signed spike-count-domain sparse code.  Do not multiply
            # the coefficient increment by the firing threshold.
            z = z + signed_spike

            # Each signed spike resets one threshold in membrane-amplitude
            # units.  The residual after the final unfolded stage provides
            # the continuous within-support amplitude refinement.
            membrane = drive - theta * signed_spike
            final_membrane = membrane
            final_theta = theta

            layers_out.append(z)
            stage_spikes = signed_spike.detach().abs().sum()
            total_spikes += stage_spikes
            layer_spike_events.append(float(stage_spikes.item()))

            if return_debug:
                drives.append(drive[0].detach().cpu().numpy()[None, :])
                membranes.append(
                    membrane[0].detach().cpu().numpy()[None, :]
                )
                spikes.append(signed_spike[0].detach().cpu().numpy()[None, :])

        # Spikes select the sparse support.  The normalized residual membrane
        # refines amplitudes only on that support and cannot create new
        # nonzero coefficients by itself.
        support_mask = (z != 0).to(dtype=z.dtype)
        z_final_raw = (
            z
            + support_mask
            * (final_membrane / final_theta.clamp_min(1e-8))
        )
        z_final = soft_threshold(
            z_final_raw, y_recv.new_tensor(Config.FINAL_THETA_SNN)
        )
        if layers_out:
            layers_out[-1] = z_final
        x_recon = z_final @ self.D

        spike_events = float(total_spikes.detach().item())
        slots = max(1, batch * self.N * max(self.K, 1))
        stats = {
            "fired_rate": spike_events / slots,
            "MACs": float(batch * self.M * self.N),
            "ACs": float(total_acs.detach().item()),
            "spike_events": spike_events,
            "layer_spike_events": layer_spike_events,
        }
        if return_debug:
            stats["debug_viz"] = {
                "q_drive_layers": drives,
                "u_mem_trace": membranes,
                "spike_trace": spikes,
            }
            # Full-batch latent tensors are exposed only for explicit
            # diagnostic calls.  They are detached so plotting cannot retain
            # the training graph or change the optimization path.
            stats["latent_debug"] = {
                "spike_only": z.detach(),
                "corrected": z_final.detach(),
            }
        return x_recon, layers_out, None, stats


class LAMP(LearnableDictionaryMixin, nn.Module):
    def __init__(self, phi=None):
        super().__init__()
        self.K = Config.NUM_LAYERS
        self.mode = "ann"
        self.M = Config.M
        self.N = Config.N

        if phi is None:
            phi = torch.randn(self.M, self.N) / math.sqrt(self.M)
        operator = self._init_learnable_dictionary(phi)

        self.B_k = nn.ModuleList(
            [nn.Linear(self.M, self.N, bias=False) for _ in range(self.K)]
        )
        self.theta_lamp_raw = nn.ParameterList(
            [
                nn.Parameter(torch.tensor(inv_softplus(Config.THETA_LAMP)))
                for _ in range(self.K)
            ]
        )
        with torch.no_grad():
            for layer in self.B_k:
                layer.weight.copy_(operator.t())

    def set_mode(self, mode):
        self.mode = "ann"

    def get_threshold_value(self):
        return float(F.softplus(self.theta_lamp_raw[0]).detach().item())

    def active_named_parameters(self, mode):
        params = self._dictionary_named_parameters()
        for i, layer in enumerate(self.B_k):
            params += list(layer.named_parameters(prefix=f"B_k.{i}"))
        params += [
            (f"theta_lamp_raw.{i}", value)
            for i, value in enumerate(self.theta_lamp_raw)
        ]
        return params

    def forward(self, y_recv, states=None, return_debug=False, phi_override=None):
        batch = y_recv.shape[0]
        z = y_recv.new_zeros(batch, self.N)
        residual = y_recv
        layers_out = []
        operator = self.current_operator(phi_override)

        for k in range(self.K):
            pseudo = z + self.B_k[k](residual)
            scale = residual.norm(p=2, dim=1, keepdim=True) / math.sqrt(self.M)
            threshold = F.softplus(self.theta_lamp_raw[k]) * scale.clamp_min(1e-8)
            z = soft_threshold(pseudo, threshold)
            layers_out.append(z)

            if k < self.K - 1:
                with torch.no_grad():
                    divergence = (pseudo.abs() > threshold).float().sum(
                        dim=1, keepdim=True
                    ) / self.M
                residual = y_recv - z @ operator.t() + divergence * residual

        x_recon = z @ self.D
        macs = batch * (2 * self.K - 1) * self.M * self.N
        return x_recon, layers_out, None, {
            "MACs": float(macs), "ACs": 0.0, "spike_events": 0.0
        }


class FISTA(nn.Module):
    def __init__(self, phi=None):
        super().__init__()
        self.K = Config.FISTA_ITERATIONS
        self.mode = "ann"
        self.M = Config.M
        self.N = Config.N

        if phi is None:
            phi = torch.randn(self.M, self.N) / math.sqrt(self.M)
        dictionary, operator = fixed_operator(phi)
        self.register_buffer("D", dictionary)
        self.register_buffer("A", operator)
        self.register_buffer("theta", torch.tensor(Config.THETA_FISTA))
        self.register_buffer(
            "lipschitz", torch.linalg.matrix_norm(operator, ord=2).square()
        )

    def set_mode(self, mode):
        self.mode = "ann"

    def get_threshold_value(self):
        return float(self.theta.item())

    def forward(self, y_recv, states=None, return_debug=False, phi_override=None):
        batch = y_recv.shape[0]
        z = y_recv.new_zeros(batch, self.N)
        momentum_z = z.clone()
        t = 1.0
        step = 1.0 / self.lipschitz.clamp_min(1e-8)

        for _ in range(self.K):
            old_z = z
            gradient = (momentum_z @ self.A.t() - y_recv) @ self.A
            z = soft_threshold(momentum_z - step * gradient, self.theta * step)
            next_t = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
            momentum_z = z + ((t - 1.0) / next_t) * (z - old_z)
            t = next_t

        x_recon = z @ self.D
        macs = batch * (2 * self.K - 1) * self.M * self.N
        stats = {"MACs": float(macs), "ACs": 0.0, "spike_events": 0.0}
        if return_debug:
            stats["debug_fista"] = {
                "L": float(self.lipschitz.item()),
                "lambda_eff": float((self.theta * step).item()),
                "measurement_mse": float(((z @ self.A.t()) - y_recv).pow(2).mean().item()),
                "z_l1_mean": float(z.abs().mean().item()),
                "z_exact_zero_ratio": float((z == 0).float().mean().item()),
                "z_near_zero_ratio": float((z.abs() < 1e-3).float().mean().item()),
                "z_max_abs": float(z.abs().max().item()),
                "A_shape": tuple(self.A.shape),
                "z_shape": tuple(z.shape),
                "x_recon_shape": tuple(x_recon.shape),
            }
        return x_recon, [z], None, stats


class ALISTA(LearnableDictionaryMixin, nn.Module):
    def __init__(self, phi=None):
        super().__init__()
        self.K = Config.NUM_LAYERS
        self.mode = "ann"
        self.M = Config.M
        self.N = Config.N

        if phi is None:
            phi = torch.randn(self.M, self.N) / math.sqrt(self.M)
        operator = self._init_learnable_dictionary(phi)

        self.register_buffer(
            "W_t", torch.empty(self.N, self.M, device=operator.device)
        )
        self.synchronize_dictionary_operator()

        self.gamma_raw = nn.ParameterList(
            [nn.Parameter(torch.tensor(inv_softplus(1.0))) for _ in range(self.K)]
        )
        self.theta_alista_raw = nn.ParameterList(
            [
                nn.Parameter(torch.tensor(inv_softplus(Config.THETA_ALISTA)))
                for _ in range(self.K)
            ]
        )

    def set_mode(self, mode):
        self.mode = "ann"

    def get_threshold_value(self):
        return float(F.softplus(self.theta_alista_raw[0]).detach().item())

    def active_named_parameters(self, mode):
        params = self._dictionary_named_parameters()
        params += [
            (f"gamma_raw.{i}", value) for i, value in enumerate(self.gamma_raw)
        ]
        params += [
            (f"theta_alista_raw.{i}", value)
            for i, value in enumerate(self.theta_alista_raw)
        ]
        return params

    @torch.no_grad()
    def synchronize_dictionary_operator(self, phi_override=None):
        """Keep ALISTA's analytic W consistent with A = Phi D^T.

        This is called once per training epoch, giving a stable alternating
        update without computing a pseudoinverse in every mini-batch.
        """
        operator = self.current_operator(phi_override).detach()
        gram = operator @ operator.t()
        ridge = 1e-5 * torch.eye(
            self.M, device=operator.device, dtype=operator.dtype
        )
        weight_t = torch.linalg.solve(gram + ridge, operator).t()
        diagonal = torch.diag(weight_t @ operator).clamp_min(1e-6)
        self.W_t.copy_(weight_t / diagonal.unsqueeze(1))

    def forward(self, y_recv, states=None, return_debug=False, phi_override=None):
        batch = y_recv.shape[0]
        z = y_recv.new_zeros(batch, self.N)
        layers_out = []
        operator = self.current_operator(phi_override)

        for k in range(self.K):
            residual = z @ operator.t() - y_recv
            step = residual @ self.W_t.t()
            z = soft_threshold(
                z - F.softplus(self.gamma_raw[k]) * step,
                F.softplus(self.theta_alista_raw[k]),
            )
            layers_out.append(z)

        x_recon = z @ self.D
        macs = batch * (2 * self.K - 1) * self.M * self.N
        return x_recon, layers_out, None, {
            "MACs": float(macs), "ACs": 0.0, "spike_events": 0.0
        }
