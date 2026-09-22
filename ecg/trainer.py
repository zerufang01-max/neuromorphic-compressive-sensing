"""ECG reconstruction training, model selection, and analytical operation counts."""
import os
import random
from pathlib import Path
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from config import Config
import utils

class Trainer:
    def __init__(self, model, edge_sensor, wireless_channel, train_loader, val_loader, custom_cfg=None):
        self.model = model
        self.sensor = edge_sensor
        self.channel = wireless_channel
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = Config.DEVICE
        self.criterion_mse = nn.MSELoss()
        self.custom_cfg = custom_cfg
        self.best_nmse = float("inf")

    def _canonical_mode(self, mode: str):
        return "spiking" if mode == "snn" else mode

    def _call_model(self, y_recv, model_states=None, return_debug=False):
        kwargs = {
            "states": model_states,
            "return_debug": return_debug,
            "phi_override": self.sensor.effective_phi(),
        }
        return self.model(y_recv, **kwargs)

    def _run_sample(self, x_batch, mode="ann", return_debug=False, channel_mode=None):
        current_mode = self._canonical_mode(mode)
        if x_batch.dim() != 2 or x_batch.size(1) != Config.N:
            raise ValueError(f"Expected static ECG batch [B, {Config.N}], got {tuple(x_batch.shape)}")

        y_raw, sensor_state, aux_dict = self.sensor(x_batch, mode="shared")
        
        y_recv, channel_state = self.channel(
            y_raw,
            state=None,
            mode="shared",
            channel_mode=channel_mode,
        )

        if hasattr(self.model, "set_mode"):
            self.model.set_mode(current_mode)
        else:
            self.model.mode = current_mode

        x_rec, layers, _, stats = self._call_model(y_recv, model_states=None, return_debug=return_debug)

        if return_debug and isinstance(stats, dict):
            stats.setdefault("debug", {})
            stats["debug"]["channel_type"] = "shared_dense_quantized_bpsk_awgn"
            stats["debug"]["channel_mode"] = channel_mode
            stats["debug"]["snr_db"] = Config.SNR_DB
            stats["debug"]["quant_bits"] = Config.QUANT_BITS
            stats["debug"]["quant_clip_value"] = Config.QUANT_CLIP_VALUE
            stats["debug"]["bits_per_sample"] = Config.BITS_PER_SAMPLE
            stats["debug"]["measurement_clip_ratio"] = (y_raw.abs() > Config.QUANT_CLIP_VALUE).float().mean().item()

        return {
            "target": x_batch,
            "recon": x_rec,
            "layers": layers,
            "encoded": [y_raw],
            "received": y_recv,
            "stats": stats,
            "recon_trace": [x_rec],
            "aux_dict": aux_dict,
            "sensor_state": sensor_state,
            "channel_state": channel_state,
        }

    def _build_optimizer(self, cfg):
        if hasattr(self.model, "active_named_parameters"):
            named_model_params = self.model.active_named_parameters(self.model.mode)
        else:
            named_model_params = [
                (name, param) for name, param in self.model.named_parameters()
                if param.requires_grad
            ]

        sensor_params = [
            param for _, param in self.sensor.active_named_parameters(
                self.sensor.mode
            ) if param.requires_grad
        ]
        dictionary_params = []
        p_params = []
        theta_params = []
        weight_params = []

        for name, param in named_model_params:
            if name == "D" or name.endswith(".D"):
                dictionary_params.append(param)
            elif name == "P_snn.weight" or name.startswith("P_snn."):
                # Only the initial S-LISTA measurement-to-code projection P
                # receives weight decay.  Applying it to the recurrent PD
                # operators or the synthesis dictionary would change the
                # recovery dynamics and introduce an unwanted scale bias.
                p_params.append(param)
            elif any(key in name for key in ["theta", "step", "eta", "gamma"]):
                theta_params.append(param)
            else:
                weight_params.append(param)

        param_groups = []
        if sensor_params:
            param_groups.append({
                "params": sensor_params,
                "lr": cfg.get("lr_phi", 5e-4),
                "weight_decay": 0.0,
                "group_name": "measurement",
            })
        if dictionary_params:
            param_groups.append({
                "params": dictionary_params,
                "lr": cfg.get("lr_dictionary", Config.DICTIONARY_LR),
                "weight_decay": 0.0,
                "group_name": "dictionary",
            })
        if p_params:
            param_groups.append({
                "params": p_params,
                "lr": cfg.get("lr_p", cfg.get("lr_model", 5e-4)),
                "weight_decay": cfg.get("p_weight_decay", 0.0),
                "group_name": "P_snn",
            })
        if weight_params:
            param_groups.append({
                "params": weight_params,
                "lr": cfg.get("lr_pd", cfg.get("lr_model", 5e-4)),
                "weight_decay": cfg.get("weight_decay", 0.0),
                "group_name": "PD_snn" if self.model.mode == "spiking" else "model",
            })
        if theta_params:
            param_groups.append({
                "params": theta_params,
                "lr": cfg.get("lr_theta", cfg.get("lr_model", 5e-4)),
                "weight_decay": 0.0,
                "group_name": "threshold",
            })
        
        optimizer = optim.Adam(param_groups)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.get("epochs", 100), eta_min=1e-6)
        return optimizer, scheduler, param_groups

    @torch.no_grad()
    def _dictionary_stats(self):
        if not hasattr(self.model, "D"):
            return None
        dictionary = self.model.D.detach().to(dtype=torch.float64)
        row_norms = dictionary.norm(p=2, dim=1)
        normalized = dictionary / row_norms.clamp_min(1e-12).unsqueeze(1)
        gram = (normalized @ normalized.t()).abs()
        gram.fill_diagonal_(0.0)
        singular_values = torch.linalg.svdvals(dictionary)
        sigma_max = singular_values.max().item()
        sigma_min = singular_values.min().item()
        condition = sigma_max / sigma_min if sigma_min > 0.0 else float("inf")
        return {
            "norm_mean": row_norms.mean().item(),
            "norm_std": row_norms.std(unbiased=False).item(),
            "coherence": gram.max().item(),
            "sigma_min": sigma_min,
            "condition": condition,
        }

    @torch.no_grad()
    def _measurement_stats(self):
        phi = self.sensor.effective_phi().detach().to(dtype=torch.float64)
        column_norms = phi.norm(p=2, dim=0)
        row_norms = phi.norm(p=2, dim=1)
        singular_values = torch.linalg.svdvals(phi)
        sigma_max = singular_values.max().item()
        sigma_min = singular_values.min().item()
        condition = sigma_max / sigma_min if sigma_min > 0.0 else float("inf")
        return {
            "column_norm_mean": column_norms.mean().item(),
            "column_norm_std": column_norms.std(unbiased=False).item(),
            "row_norm_mean": row_norms.mean().item(),
            "row_norm_std": row_norms.std(unbiased=False).item(),
            "sigma_min": sigma_min,
            "condition": condition,
        }

    def train_dynamic_recon(self, hp=None, model_name="model", mode="ann", save_path=None):
        current_mode = self._canonical_mode(mode)
        cfg = hp if hp is not None else self.custom_cfg
        if cfg is None:
            cfg = {"epochs": Config.FORMAL_TRAIN_EPOCHS, "lr_model": 5e-4}
        
        if hasattr(self.model, "set_mode"):
            self.model.set_mode(current_mode)
        else:
            self.model.mode = current_mode
        self.sensor.mode = "shared"
        
        if save_path is None:
            channel_tag = (
                f"snr{Config.SNR_DB:g}"
                f"_train{Config.TRAIN_CHANNEL_MODE}"
                f"_eval{Config.EVAL_CHANNEL_MODE}"
                f"_q{Config.QUANT_BITS}"
                f"_clip{Config.QUANT_CLIP_VALUE:g}"
                f"_T{Config.TIME_STEPS}"
            )
            if current_mode == "spiking":
                channel_tag += (
                    f"_fth{Config.FINAL_THETA_SNN:g}"
                    f"_pwd{cfg.get('p_weight_decay', 0.0):.1e}"
                )
            threshold_suffix = f"_th{self.model.get_threshold_value():.2f}" if hasattr(self.model, "get_threshold_value") else ""
            save_path = os.path.join(
                Config.RESULTS_DIR,
                f"{Config.CODE_VERSION}_M{Config.M}_K{Config.NUM_LAYERS}_dynamic_recon_"
                f"{current_mode}_{model_name.lower()}{threshold_suffix}_{channel_tag}.pth",
            )

        if os.path.exists(save_path) and Path(save_path + ".complete").exists():
            checkpoint = torch.load(save_path, map_location=self.device, weights_only=False)
            if checkpoint.get("code_version") != Config.CODE_VERSION:
                raise RuntimeError(
                    f"Incompatible checkpoint code version: "
                    f"{checkpoint.get('code_version', 'unknown')} "
                    f"(expected {Config.CODE_VERSION})."
                )
            self.model.load_state_dict(checkpoint["model"], strict=True)
            self.sensor.load_state_dict(checkpoint["sensor"], strict=True)
            return self.validate_recon(mode=current_mode, channel_mode=Config.EVAL_CHANNEL_MODE)

        optimizer, scheduler, param_groups = self._build_optimizer(cfg)
        self.best_nmse = float("inf")

        if hasattr(self.model, "synchronize_dictionary_operator"):
            self.model.synchronize_dictionary_operator(
                self.sensor.effective_phi()
            )

        alpha_l1 = (
            Config.ALPHA_L1_SNN if current_mode == "spiking"
            else Config.ALPHA_L1_ANN
        )

        def save_current_checkpoint():
            utils.atomic_torch_save({
                "model": self.model.state_dict(),
                "sensor": self.sensor.state_dict(),
                "code_version": Config.CODE_VERSION,
                "num_layers": Config.NUM_LAYERS,
                "slista_init": Config.SLISTA_INIT,
                "dictionary_init": Config.DICTIONARY_INIT,
                "cs_mode": Config.CS_MODE,
                "train_snr_db": Config.TRAIN_SNR_DB,
                "quant_bits": Config.QUANT_BITS,
                "dictionary_lr": cfg.get(
                    "lr_dictionary", Config.DICTIONARY_LR
                ),
                "measurement_init": Config.MEASUREMENT_INIT,
                "measurement_normalization": Config.MEASUREMENT_NORMALIZATION,
                "measurement_lr": cfg.get("lr_phi", 0.0),
                "lr_p": cfg.get("lr_p", cfg.get("lr_model", 0.0)),
                "lr_pd": cfg.get("lr_pd", cfg.get("lr_model", 0.0)),
                "lr_scale": Config.LR_SCALE,
                "p_weight_decay": cfg.get("p_weight_decay", 0.0),
                "warm_start_checkpoint": Config.WARM_START_CHECKPOINT,
            }, save_path)

        resume_path = save_path + ".resume"
        start_epoch = 0
        if Config.WARM_START_CHECKPOINT and not os.path.exists(resume_path):
            self.best_nmse = self.validate_recon(
                mode=current_mode, channel_mode=Config.EVAL_CHANNEL_MODE
            )
            save_current_checkpoint()
            print(
                f"Ep 000 | Warm-start Val NMSE: {self.best_nmse:.2f} dB [*Saved*]"
            )

        if os.path.exists(resume_path):
            state = torch.load(resume_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(state['model'])
            self.sensor.load_state_dict(state['sensor'])
            optimizer.load_state_dict(state['optimizer'])
            scheduler.load_state_dict(state['scheduler'])
            self.best_nmse = state['best_nmse']
            start_epoch = state['epoch']
            random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng'])
            torch.set_rng_state(state['torch_rng'].cpu())
            if torch.cuda.is_available():
                torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda_rng']])
            print(f"Resuming after epoch {start_epoch}", flush=True)
        for epoch in range(start_epoch, cfg.get("epochs", 100)):
            self.model.train()
            self.sensor.train()
            ep_loss = ep_mse = ep_l1 = ep_l1_raw = 0.0
            latent_zeros = 0
            latent_count = 0
            latent_abs_sum = 0.0
            spike_events_sum = 0.0
            spike_slots = 0

            for x_batch, _ in self.train_loader:
                x_batch = x_batch.to(self.device)
                
                run = self._run_sample(
                    x_batch, 
                    mode=current_mode,
                    channel_mode=Config.TRAIN_CHANNEL_MODE
                )
                
                x_rec = run["recon"]
                spike_events_sum += float(
                    run.get("stats", {}).get("spike_events", 0.0)
                )
                spike_slots += (
                    x_batch.size(0) * Config.N * Config.NUM_LAYERS
                )
                loss_mse = Config.ALPHA_MSE * self.criterion_mse(x_rec, x_batch)
                
                loss_l1 = x_rec.new_tensor(0.0)
                if run["layers"]:
                    l1_norm_sum = sum(layer.abs().mean() for layer in run["layers"]) / len(run["layers"])
                    loss_l1 = alpha_l1 * l1_norm_sum
                    final_latent = run["layers"][-1].detach()
                    latent_zeros += int((final_latent == 0.0).sum().item())
                    latent_count += final_latent.numel()
                    latent_abs_sum += float(final_latent.abs().sum().item())
                    
                loss = loss_mse + loss_l1
                
                optimizer.zero_grad()
                loss.backward()
                all_params = [p for group in param_groups for p in group["params"]]
                torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                optimizer.step()
                if hasattr(self.model, "project_thresholds"):
                    self.model.project_thresholds()
                
                ep_loss += loss.item()
                ep_mse += loss_mse.item()
                ep_l1 += loss_l1.item()
                if run["layers"]:
                    ep_l1_raw += l1_norm_sum.item()

            scheduler.step()
            # Refresh analytic matrices (notably ALISTA's W) after D has
            # changed.  This implements an alternating dictionary/operator
            # update and avoids using the operator from the initial D forever.
            if hasattr(self.model, "synchronize_dictionary_operator"):
                self.model.synchronize_dictionary_operator(
                    self.sensor.effective_phi()
                )
            if (epoch + 1) % 10 == 0 or epoch + 1 == cfg.get("epochs", 100):
                val_nmse = self.validate_recon(mode=current_mode, channel_mode=Config.EVAL_CHANNEL_MODE)
                batches = max(len(self.train_loader), 1)
                saved_msg = ""
                if val_nmse < self.best_nmse:
                    self.best_nmse = val_nmse
                    save_current_checkpoint()
                    saved_msg = " [*Saved*]"
                sparsity = latent_zeros / max(latent_count, 1)
                latent_mean_abs = latent_abs_sum / max(latent_count, 1)
                dictionary_stats = self._dictionary_stats()
                measurement_stats = self._measurement_stats()
                dictionary_text = ""
                if dictionary_stats is not None:
                    dictionary_text = (
                        f" | D norm: {dictionary_stats['norm_mean']:.4f}"
                        f"±{dictionary_stats['norm_std']:.2e}"
                        f", mu: {dictionary_stats['coherence']:.4f}"
                        f", sigma_min: {dictionary_stats['sigma_min']:.2e}"
                        f", cond: {dictionary_stats['condition']:.2e}"
                    )
                measurement_text = (
                    f" | Phi col norm: "
                    f"{measurement_stats['column_norm_mean']:.4f}"
                    f"±{measurement_stats['column_norm_std']:.2e}"
                    f", row norm: {measurement_stats['row_norm_mean']:.4f}"
                    f"±{measurement_stats['row_norm_std']:.2e}"
                    f", sigma_min: {measurement_stats['sigma_min']:.2e}"
                    f", cond: {measurement_stats['condition']:.2e}"
                )
                print(
                    f"Ep {epoch + 1:03d} | Loss: {ep_loss / batches:.4f} "
                    f"(MSE: {ep_mse / batches:.4f}, "
                    f"L1: {ep_l1 / batches:.6f}, "
                    f"raw: {ep_l1_raw / batches:.4f}, "
                    f"alpha: {alpha_l1:.1e}) "
                    f"| Sparsity: {sparsity:.2%}, |z|: {latent_mean_abs:.4f} "
                    f"| FR: {spike_events_sum / max(spike_slots, 1):.2%} "
                    f"| Val NMSE: {val_nmse:.2f} dB"
                    f"{dictionary_text}{measurement_text}{saved_msg}"
                )
            utils.atomic_torch_save({
                'model': self.model.state_dict(), 'sensor': self.sensor.state_dict(),
                'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                'best_nmse': self.best_nmse, 'epoch': epoch + 1,
                'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
                'torch_rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            }, resume_path)
        if not os.path.exists(save_path):
            raise RuntimeError("No finite validation checkpoint was selected")
        Path(save_path + '.complete').write_text('completed\n')
        if os.path.exists(resume_path): os.unlink(resume_path)
        return self.best_nmse

    def validate_recon(self, mode="ann", channel_mode=None, loader=None, channel_seed=None):
        current_mode = self._canonical_mode(mode)
        if channel_mode is None:
            channel_mode = Config.EVAL_CHANNEL_MODE
            
        self.model.eval()
        self.sensor.eval()
        if hasattr(self.model, "set_mode"):
            self.model.set_mode(current_mode)
        else:
            self.model.mode = current_mode
        self.sensor.mode = "shared"
        
        error_energy = 0.0
        target_energy = 0.0
        
        with utils.deterministic_eval(Config.EVAL_CHANNEL_SEED if channel_seed is None else channel_seed):
            with torch.no_grad():
                for x_batch, _ in (self.val_loader if loader is None else loader):
                    x_batch = x_batch.to(self.device)
                    run = self._run_sample(x_batch, mode=current_mode, channel_mode=channel_mode)
                    error_energy += float(torch.sum((run["recon"] - x_batch) ** 2).item())
                    target_energy += float(torch.sum(x_batch ** 2).item())
        return 10.0 * math.log10(max(error_energy, 1e-12) / max(target_energy, 1e-12))

    def evaluate(self, mode="ann", loader=None):
        if loader is None:
            loader = getattr(self, "test_loader", None)
        if loader is None:
            raise ValueError("Explicit test loader required for final evaluation")

        current_mode = self._canonical_mode(mode)
        nmse = self.validate_recon(mode=current_mode, channel_mode=Config.EVAL_CHANNEL_MODE, loader=loader, channel_seed=Config.TEST_CHANNEL_SEED)
        
        self.model.eval()
        self.sensor.eval()
        
        if hasattr(self.model, "set_mode"):
            self.model.set_mode(current_mode)
        else:
            self.model.mode = current_mode
            
        self.sensor.mode = "shared"
        
        total_macs = 0.0
        total_acs = 0.0
        total_spikes = 0.0
        total_samples = 0
        latent_zeros = 0
        latent_near_zeros = 0
        latent_count = 0
        latent_abs_sum = 0.0
        layer_spikes = np.zeros(Config.NUM_LAYERS, dtype=np.float64)
        
        with utils.deterministic_eval(Config.TEST_CHANNEL_SEED):
            with torch.no_grad():
                for x_batch, _ in loader:
                    x_batch = x_batch.to(self.device)
                    batch_size = x_batch.size(0)
                    
                    run = self._run_sample(x_batch, mode=current_mode, channel_mode=Config.EVAL_CHANNEL_MODE)
                    stats = run["stats"]
                    
                    total_macs += stats.get("MACs", 0.0)
                    total_acs += stats.get("ACs", 0.0)
                    total_spikes += stats.get("spike_events", 0.0)
                    total_samples += batch_size
                    if run["layers"]:
                        final_latent = run["layers"][-1].detach()
                        latent_zeros += int((final_latent == 0.0).sum().item())
                        latent_near_zeros += int(
                            (final_latent.abs() < 1e-3).sum().item()
                        )
                        latent_count += final_latent.numel()
                        latent_abs_sum += float(final_latent.abs().sum().item())
                    current_layer_spikes = stats.get("layer_spike_events", [])
                    if len(current_layer_spikes) == Config.NUM_LAYERS:
                        layer_spikes += np.asarray(
                            current_layer_spikes, dtype=np.float64
                        )
                
        avg_macs = total_macs / max(total_samples, 1)
        avg_acs = total_acs / max(total_samples, 1)
        avg_spikes = total_spikes / max(total_samples, 1)
        firing_rate = avg_spikes / max(Config.N * Config.NUM_LAYERS, 1)
        layer_firing_rates = (
            layer_spikes / max(total_samples * Config.N, 1)
        ).tolist()
        
       
        energy_uj = avg_macs * Config.E_MAC_UJ + avg_acs * Config.E_AC_UJ
        
        return {
            "nmse": nmse,
            "energy_uj": energy_uj,
            "dense_macs": avg_macs,
            "synaptic_acs": avg_acs,
            "spike_events": avg_spikes,
            "firing_rate": firing_rate,
            "layer_firing_rates": layer_firing_rates,
            "latent_zero_fraction": latent_zeros / max(latent_count, 1),
            "latent_near_zero_fraction": (
                latent_near_zeros / max(latent_count, 1)
            ),
            "latent_mean_abs": latent_abs_sum / max(latent_count, 1),
            "avg_bits_per_sample": float(Config.BITS_PER_SAMPLE),
        }
