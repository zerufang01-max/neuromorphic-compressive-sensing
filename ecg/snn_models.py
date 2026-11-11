"""Unfolded ECG recovery models with fixed or learned synthesis dictionaries."""
import math

import numpy as np
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


def inverse_softplus(value):
    value = float(value)
    return value + math.log(-math.expm1(-value))


class SurrogateSpike(torch.autograd.Function):
    """Heaviside H(x - theta) with a boxcar surrogate of half-width w.

    The surrogate derivative is passed to both x and theta (d/dtheta = -d/dx),
    so the thresholds are trainable.  Identical to toy/models.py."""

    @staticmethod
    def forward(ctx, x, theta, width):
        ctx.save_for_backward(x, theta, width)
        return (x >= theta).to(x.dtype)

    @staticmethod
    def backward(ctx, grad):
        x, theta, width = ctx.saved_tensors
        surrogate = grad * ((x - theta).abs() < width).to(x.dtype)
        return surrogate, -surrogate.sum().reshape(theta.shape), None


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


class LISTA(LearnableDictionaryMixin, nn.Module):
    """ANN LISTA baseline."""

    def __init__(self, phi=None):
        super().__init__()
        self.K = Config.NUM_LAYERS
        self.mode = "ann"
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

        with torch.no_grad():
            self.W_e.weight.copy_(eta * operator.t())
            recurrent = torch.eye(self.N, device=operator.device) - eta * operator.t() @ operator
            for layer in self.S_k:
                layer.weight.copy_(recurrent)

    def set_mode(self, mode):
        self.mode = "ann"

    def get_threshold_value(self):
        return float(F.softplus(self.theta_ann_raw[0]).detach().item())

    def get_ann_threshold_values(self):
        return [float(F.softplus(value).detach().item()) for value in self.theta_ann_raw]

    def active_named_parameters(self, mode):
        params = self._dictionary_named_parameters()
        params += list(self.W_e.named_parameters(prefix="W_e"))
        for i, layer in enumerate(self.S_k):
            params += list(layer.named_parameters(prefix=f"S_k.{i}"))
        params += [
            (f"theta_ann_raw.{i}", value)
            for i, value in enumerate(self.theta_ann_raw)
        ]
        return [(name, value) for name, value in params if value.requires_grad]

    def forward(self, y_recv, states=None, return_debug=False, phi_override=None):
        batch = y_recv.shape[0]
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


class SpikingLISTA(LearnableDictionaryMixin, nn.Module):
    """S-LISTA, single frame (paper Eqs. (5)-(7); same algorithm as
    toy/models.py SpikingLISTA), followed by the synthesis dictionary:

        I^(1) = P y,                    z^(0) = 0
        for l = 1 .. L-1  (spiking layers)
            xi^(l)  = H(I^(l) - theta_l) - H(-I^(l) - theta_l)  in {-1, 0, 1}
            z^(l)   = z^(l-1) + xi^(l)
            I^(l+1) = I^(1) - G^(l) z^(l)
        readout layer L (non-spiking, memoryless)
            v = I^(L),  m = 1[z^(L-1) != 0]                   (event gating)
            z = z^(L-1) + eta * m * v
        x_hat = z D

    T = 1, so the membrane leak rho is inactive.  theta_l = floor +
    softplus(r_l) and eta = softplus(r_eta) are trained; the surrogate of
    layer l is a boxcar of half-width alpha * theta_l.  ``trace`` is the only
    forward implementation (training, evaluation, figures).
    """

    def __init__(self, phi=None):
        super().__init__()
        self.K = Config.NUM_LAYERS
        if self.K < 2:
            raise ValueError("S-LISTA needs at least one spiking layer and the readout layer")
        self.mode = "spiking"
        self.M = Config.M
        self.N = Config.N
        self.depth = self.K
        self.spiking_layers = self.K - 1

        if phi is None:
            phi = torch.randn(self.M, self.N) / math.sqrt(self.M)
        self._init_learnable_dictionary(phi)

        self.kappa = (math.sqrt(2.0 * math.log(self.N))
                      if Config.SLISTA_KAPPA == "universal" else float(Config.SLISTA_KAPPA))
        self.beta = float(Config.SLISTA_BETA)
        self.alpha = float(Config.SLISTA_ALPHA)
        device = phi.device
        self.register_buffer(
            "threshold_floor",
            torch.tensor(float(Config.SLISTA_THRESHOLD_FLOOR), device=device))
        self.P = nn.Linear(self.M, self.N, bias=False)
        self.G = nn.ModuleList(nn.Linear(self.N, self.N, bias=False)
                               for _ in range(self.spiking_layers))
        # Placeholders; calibrate() sets the actual initial values.
        self.raw_theta = nn.Parameter(torch.zeros(self.spiking_layers, device=device))
        self.raw_eta = nn.Parameter(torch.zeros((), device=device))
        # Calibration record (saved in checkpoints, inherited by warm starts).
        self.register_buffer("calibration_sigma", torch.tensor(float("nan"), device=device))
        self.register_buffer("calibration_mad", torch.tensor(float("nan"), device=device))
        self.register_buffer("theta_init", torch.full((self.spiking_layers,), float("nan"), device=device))
        self.register_buffer("eta_init", torch.tensor(float("nan"), device=device))
        self.register_buffer("calibration_layer1_fraction", torch.tensor(float("nan"), device=device))
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.P.weight, a=math.sqrt(5))
            for layer in self.G:
                nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))

    # ------------------------------------------------------------ parameters
    @property
    def theta(self):
        return self.threshold_floor + F.softplus(self.raw_theta)

    @property
    def eta(self):
        return F.softplus(self.raw_eta)

    def set_mode(self, mode):
        self.mode = "spiking"

    def threshold_values(self):
        return [float(v) for v in self.theta.detach().cpu()]

    def active_named_parameters(self, mode):
        params = self._dictionary_named_parameters()
        params += list(self.P.named_parameters(prefix="P"))
        for i, layer in enumerate(self.G):
            params += list(layer.named_parameters(prefix=f"G.{i}"))
        params += [("raw_theta", self.raw_theta), ("raw_eta", self.raw_eta)]
        return [(name, value) for name, value in params if value.requires_grad]

    @torch.no_grad()
    def calibrate(self, measurements):
        """theta_1 = kappa * sigma, theta_{L-1} = beta * sigma, geometric in
        between (one spiking layer uses kappa * sigma); eta = 1 / theta_{L-1}.

        sigma is the standard deviation of the initial current I^(1) = P y over
        all coordinates of one training batch.  With P at its random
        initialisation every coordinate of P y has the same distribution, so
        this equals the off-support standard deviation used in the synthetic
        experiments.  The MAD estimate is recorded for reference only."""
        current = self.P(measurements)
        sigma = float(current.std())
        if not sigma > 0:
            raise RuntimeError("Calibration batch gives zero initial current")
        flat = current.reshape(-1)
        mad = float((flat - flat.median()).abs().median()) / 0.6745
        first, last = self.kappa * sigma, self.beta * sigma
        if self.spiking_layers == 1:
            values = [first]
        else:
            ratio = (last / first) ** (1.0 / (self.spiking_layers - 1))
            values = [first * ratio ** k for k in range(self.spiking_layers)]
        floor = float(self.threshold_floor)
        self.raw_theta.copy_(torch.tensor([inverse_softplus(v - floor) for v in values]))
        self.raw_eta.copy_(torch.tensor(inverse_softplus(1.0 / values[-1])))
        self.calibration_sigma.fill_(sigma)
        self.calibration_mad.fill_(mad)
        self.theta_init.copy_(torch.tensor(values))
        self.eta_init.fill_(1.0 / values[-1])
        # Fraction of first-layer currents above theta_1 at initialisation
        # (about 0.1 % if the current were Gaussian).
        fraction = float((current.abs() >= first).float().mean())
        self.calibration_layer1_fraction.fill_(fraction)
        return {"sigma": sigma, "mad_sigma": mad, "kappa": self.kappa,
                "beta": self.beta, "alpha": self.alpha,
                "theta_init": values, "eta_init": 1.0 / values[-1],
                "initial_layer1_fraction": fraction,
                "batch": int(measurements.shape[0])}

    # --------------------------------------------------------------- forward
    def trace(self, measurements):
        """Run S-LISTA and return every intermediate quantity.

        spikes[l] = xi^(l+1), codes[l] = z^(l+1) for the L-1 spiking layers;
        currents[l] = I^(l+1) for all L layers; v = I^(L); mask; output = z."""
        theta = self.theta
        width = self.alpha * theta.detach()
        initial = self.P(measurements)
        code = torch.zeros_like(initial)
        spikes, codes, currents = [], [], []
        current = initial
        for layer in range(self.spiking_layers):
            currents.append(current)
            positive = SurrogateSpike.apply(current, theta[layer], width[layer])
            negative = SurrogateSpike.apply(-current, theta[layer], width[layer])
            spike = positive - negative
            code = code + spike
            spikes.append(spike)
            codes.append(code)
            current = initial - self.G[layer](code)
        currents.append(current)
        mask = (code != 0).to(code.dtype)
        output = code + self.eta * mask * current
        return {"spikes": spikes, "codes": codes, "currents": currents,
                "v": current, "mask": mask, "spike_code": code,
                "correction": output - code, "output": output}

    def forward(self, y_recv, states=None, return_debug=False, phi_override=None):
        trace = self.trace(y_recv)
        x_recon = trace["output"] @ self.D
        counts = self.operation_counts(trace, self.N, self.M)
        stats = {
            "fired_rate": counts["spikes"] / max(counts["firing_rate_denominator"], 1.0),
            "MACs": counts["mac"],
            "ACs": counts["ac"],
            "spike_events": counts["spikes"],
            "firing_rate_denominator": counts["firing_rate_denominator"],
            "support": counts["support"],
            "layer_spike_events": [float(s.detach().abs().sum()) for s in trace["spikes"]],
            # Kept for the spike penalty (needs the graph) and diagnostics.
            "trace": trace,
        }
        if return_debug:
            stats["latent_debug"] = {
                "spike_only": trace["spike_code"].detach(),
                "corrected": trace["output"].detach(),
            }
        # Layer outputs: the L-1 spike codes followed by the final code z.
        layers_out = list(trace["codes"]) + [trace["output"]]
        return x_recon, layers_out, None, stats

    # -------------------------------------------------- loss and op counting
    @staticmethod
    def spike_penalty(trace):
        """(1/(L-1)) * sum_l ||z^(l)||_1 / n, averaged over the batch; the
        trainer multiplies it by lambda_s.  sum_l ||z^(l)||_1 * n is exactly
        the synaptic AC count, so the penalty is proportional to the
        estimated recovery energy."""
        n = trace["output"].shape[1]
        l1 = torch.stack([code.abs().sum(1) for code in trace["codes"]]).mean(0)
        return (l1 / n).mean()

    @staticmethod
    @torch.no_grad()
    def operation_counts(trace, n, m):
        """Per-batch operation counts (summed over the batch).

        MAC: P y (m n per sample; the received measurement is real valued)
             + readout gain eta * v on the gated support (||m||_0).
        AC : G^(l) z^(l) for l = 1 .. L-1; an integer magnitude k costs k
             additions per connection, i.e. n * ||z^(l)||_1.
        The dictionary readout z D is excluded, as for the ANN baselines."""
        batch = trace["output"].shape[0]
        ac = sum(float(code.abs().sum()) for code in trace["codes"]) * n
        mac = batch * m * n + float(trace["mask"].sum())
        spikes = sum(float(spike.abs().sum()) for spike in trace["spikes"])
        layers = len(trace["spikes"])
        return {"mac": mac, "ac": ac, "spikes": spikes,
                "support": float(trace["mask"].sum()),
                # Signed spikes counted on two channels.
                "firing_rate_numerator": spikes,
                "firing_rate_denominator": 2.0 * batch * n * layers}


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
