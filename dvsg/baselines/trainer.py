"""Training and evaluation of the DVS-Gesture baselines.

Trainer.train_dynamic_recon trains the sensor and the reconstruction model
jointly through the channel; Trainer.train_downstream_classifier trains the
classifier on reconstructed video with both of them frozen.  Checkpoints are
selected on the test split at the training SNR.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
import random
import csv
import math
import torchvision.transforms.functional as TF
from config import Config
import utils

class Trainer:
    """Joint training of sensor and reconstruction model, downstream classifier and evaluation."""
    def __init__(self, model, edge_sensor, wireless_channel, train_loader, val_loader):
        self.model = model
        self.sensor = edge_sensor
        self.channel = wireless_channel
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = Config.DEVICE
        
        self.criterion_mse = nn.MSELoss()
        self.criterion_semantic = nn.CrossEntropyLoss()
        self.downstream_classifier = None
        
        self.best_nmse = 1000.0
        self.best_acc = 0.0

    @staticmethod
    def _augment_semantic_sequence(z_sequence):
        """Apply the original Phase-4 sequence-consistent latent shift.

        z_sequence has shape [B, T, C, H, W].  For each sample, one spatial
        translation is shared by every time step and channel.  Cropped values
        are discarded and newly exposed locations are zero-filled.
        """
        if z_sequence.ndim != 5:
            raise ValueError(
                "Semantic latent augmentation expects [B, T, C, H, W], "
                f"got shape {tuple(z_sequence.shape)}."
            )

        ratio = float(getattr(Config, 'SEMANTIC_SHIFT_RATIO', 0.0))
        probability = float(
            getattr(Config, 'SEMANTIC_SHIFT_PROB', 0.0)
        )
        if ratio <= 0.0 or probability <= 0.0:
            return z_sequence
        if ratio >= 1.0:
            raise ValueError("SEMANTIC_SHIFT_RATIO must be in [0, 1).")
        if not 0.0 <= probability <= 1.0:
            raise ValueError("SEMANTIC_SHIFT_PROB must be in [0, 1].")

        batch_size, _, _, height, width = z_sequence.shape
        max_dx = int(ratio * width)
        max_dy = int(ratio * height)
        if max_dx == 0 and max_dy == 0:
            return z_sequence

        augmented = torch.zeros_like(z_sequence)
        for batch_index in range(batch_size):
            if random.random() > probability:
                augmented[batch_index] = z_sequence[batch_index]
                continue

            dx = random.randint(-max_dx, max_dx) if max_dx > 0 else 0
            dy = random.randint(-max_dy, max_dy) if max_dy > 0 else 0

            src_y1 = max(0, -dy)
            src_y2 = min(height, height - dy)
            dst_y1 = max(0, dy)
            dst_y2 = min(height, height + dy)

            src_x1 = max(0, -dx)
            src_x2 = min(width, width - dx)
            dst_x1 = max(0, dx)
            dst_x2 = min(width, width + dx)

            augmented[
                batch_index,
                :,
                :,
                dst_y1:dst_y2,
                dst_x1:dst_x2,
            ] = z_sequence[
                batch_index,
                :,
                :,
                src_y1:src_y2,
                src_x1:src_x2,
            ]

        return augmented

    def _process_reconstruction(self, x_input, model_states=None, sensor_state=None, channel_state=None, mode='ann', tau=None):
        y_encoded, next_sensor_state, aux_dict = self.sensor(x_input, state=sensor_state, mode=mode, tau=tau)
        
        tx_prob = aux_dict.get('tx_prob') if aux_dict else None
        tx_hard = aux_dict.get('tx_hard') if aux_dict else None
        
        y_noisy, next_channel_state = self.channel(
            y_encoded, 
            state=channel_state, 
            mode=mode,
            tx_prob=tx_prob,
            tx_hard=tx_hard
        )
        
        prev_mode = getattr(self.model, 'mode', None)
        if hasattr(self.model, 'mode'):
            self.model.mode = 'rate' if mode == 'ann' else 'spiking'
        
        x_rec, layers, next_model_states, internal_stats = self.model(y_noisy, states=model_states)
        
        if hasattr(self.model, 'mode') and prev_mode is not None:
            self.model.mode = prev_mode 
            
        return x_rec, layers, next_model_states, next_sensor_state, next_channel_state, y_encoded, internal_stats, aux_dict

    def _unfreeze_all_params(self, mode):
        is_lista = self.model.__class__.__name__ in (
            'ConvolutionalLISTA',
            'ConvolutionalLISTA_ImageSpace',
        )
        for name, param in self.model.named_parameters():
            if (
                is_lista
                and mode == 'spiking'
                and name.startswith(('W_e.', 'S_k.', 'theta_ann.'))
            ):
                param.requires_grad = False
            elif (
                is_lista
                and mode == 'ann'
                and name.startswith(('P_snn.', 'PD_snn.'))
            ):
                param.requires_grad = False
            elif name.startswith('semantic_head.'):
                param.requires_grad = Config.ALPHA_SEMANTIC > 0
            elif name.startswith('theta_ann.'):
                param.requires_grad = mode == 'ann'
            elif 'theta' in name or 'thresh' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        for param in self.sensor.parameters():
            param.requires_grad = True

    def evaluate_bits(self, mode='spiking'):
        self.sensor.eval()
        total_samples = 0
        total_bits = 0.0
        
        total_events = 0
        total_elements = 0
        
        eval_stats = {'total': 0, 'assisted': 0, 'cf': 0, 'state_contrib_sum': 0.0, 'elements': 0, 'thresh': 0.2}
        total_iou_sum = 0.0
        total_iou_count = 0
        
        with torch.no_grad():
            for x_batch, _ in self.val_loader:
                x_batch = x_batch.to(self.device)
                B, T, C, H, W = x_batch.shape
                sensor_state = None
                
                if mode == 'ann':
                    bits_per_sample = (
                        Config.LATENT_DIM * T
                        * int(Config.ANN_QUANT_BITS)
                    )
                    total_bits += bits_per_sample * B
                    total_samples += B
                else:
                    batch_events = 0
                    batch_packets = 0
                    prev_tx_hard = None
                    
                    for t in range(T):
                        x_frame = x_batch[:, t]
                        _, sensor_state, aux_dict = self.sensor(x_frame, state=sensor_state, mode=mode)
                        tx_hard = aux_dict['tx_hard']
                        batch_events += tx_hard.sum().item()
                        if Config.SNN_TRANSPORT in (
                            'block_aer', 'block_aer_awgn_hard'
                        ):
                            blocks = tx_hard.reshape(
                                B, Config.NUM_BLOCKS, Config.BLOCK_SIZE
                            )
                            batch_packets += (
                                blocks.amax(dim=-1) > 0
                            ).sum().item()
                        
                        if prev_tx_hard is not None:
                            intersection = (tx_hard * prev_tx_hard).sum(dim=1)
                            union = ((tx_hard + prev_tx_hard) > 0).float().sum(dim=1)
                            valid_mask = union > 0
                            if valid_mask.any():
                                ious = intersection[valid_mask] / union[valid_mask]
                                total_iou_sum += ious.sum().item()
                                total_iou_count += valid_mask.sum().item()
                        prev_tx_hard = tx_hard
                        
                        if Config.LOG_TX_STATE_STATS:
                            eval_stats['total'] += aux_dict.get('tx_total_spikes', 0)
                            eval_stats['assisted'] += aux_dict.get('tx_assisted_spikes', 0)
                            eval_stats['cf'] += aux_dict.get('tx_counterfactual_reset_spikes', 0)
                            eval_stats['state_contrib_sum'] += aux_dict.get('tx_state_contribution_sum', 0.0)
                            eval_stats['elements'] += aux_dict.get('tx_elements', 0)
                            eval_stats['thresh'] = aux_dict.get('tx_threshold', 0.2)
                    
                    total_events += batch_events
                    total_elements += B * T * Config.LATENT_DIM
                    if Config.SNN_TRANSPORT in (
                        'dense_binary', 'dense_binary_awgn_hard'
                    ):
                        total_bits += B * T * Config.LATENT_DIM
                    elif Config.SNN_TRANSPORT in (
                        'block_aer', 'block_aer_awgn_hard'
                    ):
                        header_bits = math.ceil(
                            math.log2(Config.NUM_BLOCKS)
                        )
                        local_bits = math.ceil(
                            math.log2(Config.BLOCK_SIZE)
                        )
                        total_bits += (
                            batch_packets * header_bits
                            + batch_events * local_bits
                        )
                    else:
                        total_bits += (
                            batch_events * Config.RATE_BITS_PER_EVENT
                        )
                    total_samples += B
        
        avg_firing_rate = 0.0            
        if mode != 'ann':
            avg_firing_rate = total_events / max(1, total_elements)
            print(f" -> TX firing rate: {avg_firing_rate * 100:.2f}%")
                
        return total_bits / max(1, total_samples), avg_firing_rate

    def train_dynamic_recon(
        self, hp=None, model_name="model", mode='spiking'
    ):
        print(f"\n>>> DYNAMIC RECONSTRUCTION TRAINING ({model_name} | Mode: {mode.upper()})")
        self._unfreeze_all_params(mode)
        
        if hp is None: hp = Config.PHASE3_CONFIG
        
        self.sensor.mode = mode
        if hasattr(self.model, 'mode'):
            self.model.mode = 'rate' if mode == 'ann' else 'spiking'

        th_val = (
            self.model.get_threshold_value()
            if hasattr(self.model, 'get_threshold_value') else None
        )
        current_alpha_rate = (
            Config.ALPHA_RATE_SCNN
            if self.model.__class__.__name__ == 'SpikingCNNRecon'
            else Config.ALPHA_RATE_LISTA
        )
        save_path = os.path.join(
            Config.RESULTS_DIR,
            Config.get_recon_ckpt_name(
                mode, model_name, th_val, current_alpha_rate
            ),
        )
        anneal_log_path = os.path.join(
            Config.RESULTS_DIR,
            f"annealing_stats_{model_name.lower()}.csv"
        )
        anneal_history = []
        joint_history_path = os.path.join(
            Config.RESULTS_DIR,
            f"joint_training_history_{model_name.lower()}.csv",
        )
        joint_history = []
        
        param_groups = self.model.get_optimizer_groups(hp)
        
        lr_sensor = hp.get(
            'lr_sensor', hp.get('lr_lista_snn', 7e-4)
        )
        sensor_params = [
            p for p in self.sensor.parameters() if p.requires_grad
        ]
        if sensor_params:
            param_groups.append({
                'params': sensor_params, 'lr': lr_sensor
            })
        
        param_groups = [g for g in param_groups if len(list(g['params'])) > 0]
        
        optimizer = optim.Adam(param_groups, weight_decay=hp['weight_decay'])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hp['epochs'], eta_min=hp['lr_min'])
        
        self.best_nmse = 1000.0
        best_semantic_acc = float('-inf')
        best_semantic_epoch = 0
        
        for epoch in range(hp['epochs']):
            current_tau = Config.get_tau(epoch, hp['epochs'])
            
            self.model.train()
            # validate_joint_metrics() switches the channel to eval mode.
            # Restore training mode every epoch so Block-AER uses its
            # differentiable expected-routing backward path.
            self.channel.train()
            self.sensor.train()

            if not (
                self.model.training
                and self.channel.training
                and self.sensor.training
            ):
                raise RuntimeError(
                    "Training-mode restoration failed for model, channel, "
                    "or sensor."
                )
            
            ep_loss = 0
            ep_recon_objective = 0.0
            ep_semantic_ce = 0.0
            ep_semantic_correct = 0
            semantic_samples = 0
            epoch_z_zero_count = torch.zeros(
                (), dtype=torch.long, device=self.device
            )
            epoch_z_elements = 0
            
            epoch_tx_stats = {
                'total': 0, 'assisted': 0, 'cf': 0,
                'state_contrib_sum': 0.0, 'elements': 0,
                'soft_sum': 0.0, 'abs_soft_hard_sum': 0.0,
                'threshold': 0.2
            }
            threshold_margin_samples = []
            sampled_margins = 0
            
            for x_batch, labels in self.train_loader:
                x_batch = x_batch.to(self.device)
                labels = labels.to(self.device)
                B, T, C, H, W = x_batch.shape
                
                model_states = None
                sensor_state = None
                channel_state = None
                loss_total_step = 0
                semantic_z_sequence = []
                
                for t in range(T):
                    x_frame = x_batch[:, t]
                    
                    x_rec, layers, model_states, sensor_state, channel_state, y_spikes, stats, aux_dict = \
                        self._process_reconstruction(x_frame, model_states, sensor_state, channel_state, mode=mode, tau=current_tau)
                    
                    if mode == 'spiking' and aux_dict:
                        epoch_tx_stats['total'] += aux_dict.get('tx_total_spikes', 0)
                        epoch_tx_stats['elements'] += aux_dict.get('tx_elements', 0)
                        epoch_tx_stats['threshold'] = aux_dict.get('tx_threshold', 0.2)
                        if Config.LOG_TX_STATE_STATS:
                            epoch_tx_stats['assisted'] += aux_dict.get('tx_assisted_spikes', 0)
                            epoch_tx_stats['cf'] += aux_dict.get('tx_counterfactual_reset_spikes', 0)
                            epoch_tx_stats['state_contrib_sum'] += aux_dict.get('tx_state_contribution_sum', 0.0)

                    if mode == 'spiking' and getattr(Config, 'LOG_ANNEAL_STATS', False) and aux_dict:
                        epoch_tx_stats['soft_sum'] += aux_dict.get('tx_soft_sum', 0.0)
                        epoch_tx_stats['abs_soft_hard_sum'] += aux_dict.get('tx_abs_soft_hard_sum', 0.0)
                        margin = aux_dict.get('tx_threshold_margin')
                        max_samples = int(getattr(Config, 'ANNEAL_DENSITY_MAX_SAMPLES', 100000))
                        if margin is not None and sampled_margins < max_samples:
                            flat_margin = margin.reshape(-1)
                            remaining = max_samples - sampled_margins
                            take = min(remaining, 2048, flat_margin.numel())
                            if take > 0:
                                if take == flat_margin.numel():
                                    sample = flat_margin
                                else:
                                    idx = torch.linspace(
                                        0, flat_margin.numel() - 1, steps=take,
                                        device=flat_margin.device
                                    ).long()
                                    sample = flat_margin[idx]
                                threshold_margin_samples.append(sample.detach().cpu().numpy())
                                sampled_margins += int(take)
                    
                    step_mse_loss = 0
                    if x_rec is not None:
                        step_mse_loss = Config.ALPHA_MSE * self.criterion_mse(x_rec, x_frame)
                    
                    step_sparsity_loss = 0
                    if len(layers) > 0:
                        l1_norm_sum = 0
                        for l in layers:
                            l1_norm_sum += l.abs().view(B, -1).sum(dim=1).mean()
                        l1_norm_sum /= len(layers)

                        model_cls_name = self.model.__class__.__name__
                        if model_cls_name == 'ConvolutionalLISTA_ImageSpace':
                            alpha_l1 = Config.ALPHA_L1_LISTA_SNN
                        elif model_cls_name == 'ConvolutionalLISTA':
                            alpha_l1 = Config.ALPHA_L1_LISTA_ANN
                        elif model_cls_name == 'ConvLSTMRecon':
                            alpha_l1 = Config.ALPHA_L1_LSTM
                        elif model_cls_name == 'SpikingCNNRecon':
                            alpha_l1 = Config.ALPHA_L1_SCNN
                        else:
                            alpha_l1 = 0

                        if alpha_l1 > 0:
                            step_sparsity_loss = alpha_l1 * l1_norm_sum
                        final_z = layers[-1]
                        epoch_z_zero_count += (
                            final_z.detach().abs() <= 1e-8
                        ).sum()
                        epoch_z_elements += final_z.numel()
                        semantic_z_sequence.append(final_z)
                    
                    step_rate_loss = 0
                    if (
                        mode == 'spiking'
                        and Config.SNN_TRANSPORT in (
                            'block_aer', 'block_aer_awgn_hard'
                        )
                    ):
                        tx_prob = aux_dict.get('tx_prob')
                        tx_hard = aux_dict.get('tx_hard')
                        
                        if self.model.__class__.__name__ == 'SpikingCNNRecon':
                            alpha_rate = Config.ALPHA_RATE_SCNN
                        else:
                            alpha_rate = Config.ALPHA_RATE_LISTA
                            
                        rate_source = (
                            tx_prob if tx_prob is not None else tx_hard
                        )
                        if rate_source is not None:
                            if Config.SNN_TRANSPORT in (
                                'block_aer', 'block_aer_awgn_hard'
                            ):
                                blocks = rate_source.clamp(
                                    0.0, 1.0
                                ).reshape(
                                    B,
                                    Config.NUM_BLOCKS,
                                    Config.BLOCK_SIZE,
                                )
                                active_prob = 1.0 - torch.prod(
                                    1.0 - blocks, dim=-1
                                )
                                header_bits = math.ceil(
                                    math.log2(Config.NUM_BLOCKS)
                                )
                                local_bits = math.ceil(
                                    math.log2(Config.BLOCK_SIZE)
                                )
                                expected_bits = (
                                    active_prob.sum(dim=1) * header_bits
                                    + blocks.sum(dim=(1, 2)) * local_bits
                                )
                                bits_per_measurement = (
                                    expected_bits.mean() / Config.LATENT_DIM
                                )
                                step_rate_loss = (
                                    alpha_rate * bits_per_measurement
                                )
                            else:
                                step_rate_loss = (
                                    alpha_rate
                                    * Config.RATE_BITS_PER_EVENT
                                    * torch.mean(rate_source)
                                )

                    loss_total_step += (step_mse_loss + step_sparsity_loss + step_rate_loss)

                recon_objective = loss_total_step / T
                semantic_ce = None
                semantic_term = recon_objective.new_zeros(())
                if (
                    Config.ALPHA_SEMANTIC > 0
                    and hasattr(self.model, 'semantic_head')
                    and len(semantic_z_sequence) == T
                ):
                    semantic_z_stack = torch.stack(
                        semantic_z_sequence, dim=1
                    )
                    semantic_z_stack = (
                        self._augment_semantic_sequence(
                            semantic_z_stack
                        )
                    )
                    semantic_logits = self.model.semantic_head(
                        semantic_z_stack
                    )
                    semantic_ce = self.criterion_semantic(
                        semantic_logits, labels
                    )
                    semantic_term = (
                        Config.ALPHA_SEMANTIC
                        * semantic_ce
                        / math.log(Config.NUM_CLASSES)
                    )
                loss_final = recon_objective + semantic_term
                
                optimizer.zero_grad()
                loss_final.backward()
                
                all_params = []
                for g in param_groups:
                    all_params.extend(g['params'])
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                
                optimizer.step()
                if (
                    mode == 'ann'
                    and hasattr(self.model, 'project_ann_thresholds')
                ):
                    self.model.project_ann_thresholds()
                ep_loss += loss_final.item()
                ep_recon_objective += recon_objective.item()
                if semantic_ce is not None:
                    batch_size = int(labels.size(0))
                    ep_semantic_ce += semantic_ce.item() * batch_size
                    ep_semantic_correct += int(
                        semantic_logits.detach().argmax(dim=1)
                        .eq(labels).sum().item()
                    )
                    semantic_samples += batch_size
                    
            scheduler.step()
            
            joint_val = self.validate_joint_metrics(mode=mode)
            val_nmse = joint_val['Temporal-Mean NMSE (dB)']
            z_sparsity = (
                float(epoch_z_zero_count.item())
                / max(1, epoch_z_elements)
            )
            if mode == 'spiking':
                tx_total = epoch_tx_stats['total']
                if torch.is_tensor(tx_total):
                    tx_total = float(tx_total.detach().item())
                else:
                    tx_total = float(tx_total)
                actual_tx_rate = (
                    tx_total / max(1, epoch_tx_stats['elements'])
                )
            else:
                actual_tx_rate = float('nan')
            joint_history.append({
                'Epoch': epoch + 1,
                'Seed': int(Config.SEED),
                'Model Name': model_name,
                'Training Loss': float(
                    ep_loss / max(1, len(self.train_loader))
                ),
                'Reconstruction Objective': float(
                    ep_recon_objective
                    / max(1, len(self.train_loader))
                ),
                'Training Semantic CE': float(
                    ep_semantic_ce / semantic_samples
                    if semantic_samples > 0 else float('nan')
                ),
                'Training Semantic Accuracy (%)': float(
                    100.0 * ep_semantic_correct / semantic_samples
                    if semantic_samples > 0 else float('nan')
                ),
                'Temporal-Mean NMSE (dB)': float(
                    joint_val['Temporal-Mean NMSE (dB)']
                ),
                'Full-Sequence NMSE (dB)': float(
                    joint_val['Full-Sequence NMSE (dB)']
                ),
                'Framewise Mean NMSE (dB)': float(
                    joint_val['Framewise Mean NMSE (dB)']
                ),
                'Validation Semantic CE': float(
                    joint_val['Semantic CE']
                ),
                'Validation Semantic Accuracy (%)': float(
                    joint_val['Joint Accuracy']
                ),
                'TX Hard Firing Rate (%)': float(
                    100.0 * actual_tx_rate
                ),
                'Final Z Sparsity (%)': float(100.0 * z_sparsity),
                'Tau': float(current_tau),
                'Semantic Weight': float(Config.ALPHA_SEMANTIC),
                'Semantic LR': float(hp['lr_semantic']),
                'Semantic Weight Decay': float(
                    Config.SEMANTIC_WEIGHT_DECAY
                ),
                'Semantic Dropout': float(Config.SEMANTIC_DROPOUT),
                'Semantic Shift Ratio': float(
                    Config.SEMANTIC_SHIFT_RATIO
                ),
                'Semantic Shift Probability': float(
                    Config.SEMANTIC_SHIFT_PROB
                ),
            })
            with open(joint_history_path, 'w', newline='') as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=list(joint_history[0].keys())
                )
                writer.writeheader()
                writer.writerows(joint_history)

            anneal_row = None
            if mode == 'spiking' and getattr(Config, 'LOG_ANNEAL_STATS', False):
                elements = max(1, epoch_tx_stats['elements'])
                soft_rate = epoch_tx_stats['soft_sum'] / elements
                hard_rate = epoch_tx_stats['total'] / elements
                rate_mismatch = abs(soft_rate - hard_rate)
                pointwise_mismatch = epoch_tx_stats['abs_soft_hard_sum'] / elements

                density_max = float('nan')
                if threshold_margin_samples:
                    margins = np.concatenate(threshold_margin_samples)
                    if margins.size > 1 and float(np.max(margins)) > float(np.min(margins)):
                        hist, _ = np.histogram(
                            margins,
                            bins=int(getattr(Config, 'ANNEAL_DENSITY_BINS', 200)),
                            density=True,
                        )
                        if hist.size > 0:
                            density_max = float(np.max(hist))

                normalized_bound = (
                    2.0 * density_max * np.log(2.0) * current_tau
                    if np.isfinite(density_max) else float('nan')
                )
                payload_scale = (
                    math.ceil(math.log2(Config.BLOCK_SIZE))
                    * Config.LATENT_DIM
                    if Config.SNN_TRANSPORT in (
                        'block_aer', 'block_aer_awgn_hard'
                    )
                    else Config.RATE_BITS_PER_EVENT * Config.LATENT_DIM
                )
                anneal_row = {
                    'Epoch': epoch + 1,
                    'Seed': int(Config.SEED),
                    'Model Name': model_name,
                    'Tau': float(current_tau),
                    'Soft Firing Rate': float(soft_rate),
                    'Hard Firing Rate': float(hard_rate),
                    'Normalized Payload Mismatch': float(rate_mismatch),
                    'Mean Absolute Soft-Hard Gap': float(pointwise_mismatch),
                    'Payload Mismatch (Bits/Frame)': float(payload_scale * rate_mismatch),
                    'Empirical Density Max': density_max,
                    'Empirical Normalized RHS': float(normalized_bound),
                    'Empirical RHS (Bits/Frame)': float(payload_scale * normalized_bound),
                    'Density Samples': int(sampled_margins),
                }
                anneal_history.append(anneal_row)
                with open(anneal_log_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=list(anneal_row.keys()))
                    writer.writeheader()
                    writer.writerows(anneal_history)
            
            if val_nmse < self.best_nmse:
                self.best_nmse = val_nmse
                torch.save({
                    'model': self.model.state_dict(),
                    'sensor': self.sensor.state_dict(),
                    'epoch': epoch + 1,
                    'val_nmse_db': float(val_nmse),
                    'joint_accuracy': float(
                        joint_val['Joint Accuracy']
                    ),
                    'semantic_ce': float(joint_val['Semantic CE']),
                    'selection_metric': 'temporal_mean_nmse',
                    'semantic_config': {
                        'alpha': float(Config.ALPHA_SEMANTIC),
                        'lr': float(hp['lr_semantic']),
                        'weight_decay': float(
                            Config.SEMANTIC_WEIGHT_DECAY
                        ),
                        'dropout': float(Config.SEMANTIC_DROPOUT),
                        'shift_ratio': float(
                            Config.SEMANTIC_SHIFT_RATIO
                        ),
                        'shift_probability': float(
                            Config.SEMANTIC_SHIFT_PROB
                        ),
                        'head': Config.SEMANTIC_HEAD_TAG,
                    },
                    'transport_config': {
                        'channel': 'bit_input_awgn_hard',
                        'train_snr_db': float(Config.TRAIN_SNR_DB),
                        'snn_transport': Config.SNN_TRANSPORT,
                        'num_blocks': int(Config.NUM_BLOCKS),
                        'block_size': int(Config.BLOCK_SIZE),
                        'ann_quant_bits': int(Config.ANN_QUANT_BITS),
                    },
                }, save_path)

            if joint_val['Joint Accuracy'] > best_semantic_acc:
                best_semantic_acc = joint_val['Joint Accuracy']
                best_semantic_epoch = epoch + 1
            
            if (epoch+1) % 5 == 0 or epoch == 0:
                train_acc = (
                    100.0 * ep_semantic_correct / semantic_samples
                    if semantic_samples > 0 else float('nan')
                )
                message = (
                    f"[Ep {epoch+1:03d}] "
                    f"NMSE: {val_nmse:.2f} dB "
                    f"(Best {self.best_nmse:.2f}) "
                    f"| Acc: train {train_acc:.2f}%, "
                    f"val {joint_val['Joint Accuracy']:.2f}% "
                    f"(Best {best_semantic_acc:.2f}% "
                    f"@ Ep {best_semantic_epoch}) "
                    f"| TX Rate: {actual_tx_rate*100:.2f}% "
                    f"| Z Sparsity: {z_sparsity*100:.2f}%"
                )
                print(message)
        
        if hasattr(self.model, 'semantic_head'):
            print(
                f" -> Best observed joint-head diagnostic: epoch "
                f"{best_semantic_epoch}, Z Val Acc "
                f"{best_semantic_acc:.2f}% "
                f"(diagnostic only; checkpoint remains best NMSE)"
            )
        return self.best_nmse

    def set_downstream_classifier(self, classifier):
        self.downstream_classifier = classifier.to(self.device)

    def collect_z_sequence(self, mode='spiking', max_samples=1200):
        self.model.eval()
        self.sensor.eval()
        self.channel.eval()
        sequences = []
        labels_all = []
        with torch.no_grad():
            for x_batch, labels in self.val_loader:
                x_batch = x_batch.to(self.device)
                model_states = sensor_state = channel_state = None
                frames = []
                for time_index in range(x_batch.shape[1]):
                    (
                        _, layers, model_states, sensor_state,
                        channel_state, _, _, _,
                    ) = self._process_reconstruction(
                        x_batch[:, time_index], model_states,
                        sensor_state, channel_state, mode=mode,
                    )
                    frames.append(layers[-1])
                time_average = torch.stack(frames, dim=1).mean(dim=1)
                pooled = F.adaptive_avg_pool2d(time_average, (4, 4))
                sequences.append(pooled.flatten(1).cpu())
                labels_all.append(labels.cpu())
                if sum(part.shape[0] for part in sequences) >= max_samples:
                    break
        sequence = torch.cat(sequences, dim=0)[:max_samples].numpy()
        labels = torch.cat(labels_all, dim=0)[:max_samples].numpy()
        return sequence, labels

    def _reconstruct_video(self, x_batch, mode):
        model_states = None
        sensor_state = None
        channel_state = None
        reconstructed = []
        for time_index in range(x_batch.shape[1]):
            (
                x_rec,
                _,
                model_states,
                sensor_state,
                channel_state,
                _,
                _,
                _,
            ) = self._process_reconstruction(
                x_batch[:, time_index],
                model_states,
                sensor_state,
                channel_state,
                mode=mode,
            )
            reconstructed.append(x_rec)
        return torch.stack(reconstructed, dim=1)

    def validate_downstream_classifier(
        self, mode='spiking', include_predictions=False
    ):
        if self.downstream_classifier is None:
            raise RuntimeError('Downstream reconstructed-x classifier is unset.')
        utils.deterministic_eval(Config.EVAL_CHANNEL_SEED)
        self.model.eval()
        self.sensor.eval()
        self.channel.eval()
        self.downstream_classifier.eval()
        total_correct = 0
        total_samples = 0
        labels_all = []
        predictions_all = []
        with torch.no_grad():
            for x_batch, labels in self.val_loader:
                x_batch = x_batch.to(self.device)
                labels = labels.to(self.device)
                reconstructed = self._reconstruct_video(x_batch, mode)
                predictions = self.downstream_classifier(
                    reconstructed
                ).argmax(dim=1)
                total_correct += int(predictions.eq(labels).sum().item())
                total_samples += int(labels.numel())
                if include_predictions:
                    labels_all.append(labels.cpu().numpy())
                    predictions_all.append(predictions.cpu().numpy())
        result = {
            'Accuracy (%)': 100.0 * total_correct / max(1, total_samples),
            'Samples': total_samples,
        }
        if include_predictions:
            result['Labels'] = np.concatenate(labels_all)
            result['Predictions'] = np.concatenate(predictions_all)
        return result

    def train_downstream_classifier(self, hp, mode, save_path):
        if self.downstream_classifier is None:
            raise RuntimeError('Downstream reconstructed-x classifier is unset.')
        classifier = self.downstream_classifier
        optimizer = optim.Adam(
            classifier.parameters(),
            lr=float(hp['lr']),
            weight_decay=float(hp.get('weight_decay', 0.0)),
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(hp['epochs']),
            eta_min=float(hp.get('lr_min', 1e-6)),
        )
        best_accuracy = float('-inf')
        self.model.eval()
        self.sensor.eval()
        self.channel.eval()
        for epoch in range(int(hp['epochs'])):
            classifier.train()
            correct = 0
            samples = 0
            for x_batch, labels in self.train_loader:
                x_batch = x_batch.to(self.device)
                labels = labels.to(self.device)
                with torch.no_grad():
                    reconstructed = self._reconstruct_video(x_batch, mode)
                logits = classifier(reconstructed.detach())
                loss = self.criterion_semantic(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                correct += int(logits.argmax(dim=1).eq(labels).sum().item())
                samples += int(labels.numel())
            scheduler.step()
            validation = self.validate_downstream_classifier(mode=mode)
            accuracy = float(validation['Accuracy (%)'])
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save({
                    'classifier': classifier.state_dict(),
                    'epoch': epoch + 1,
                    'val_accuracy': accuracy,
                    'training_snr_db': float(Config.TRAIN_SNR_DB),
                }, save_path)
            print(
                f"Classifier epoch {epoch + 1}/{hp['epochs']} | "
                f"train ACC={100.0 * correct / max(1, samples):.2f}% | "
                f"val ACC={accuracy:.2f}% | best={best_accuracy:.2f}%"
            )
        state = torch.load(
            save_path, map_location=self.device, weights_only=False
        )
        classifier.load_state_dict(state['classifier'], strict=True)
        return best_accuracy

    def validate_joint_metrics(
        self, mode='spiking', include_predictions=False
    ):
        """Evaluate all reconstruction metrics and the deployed z head once."""
        if not hasattr(self.model, 'semantic_head'):
            raise RuntimeError(
                'Joint evaluation requires model.semantic_head.'
            )
        utils.deterministic_eval()
        self.model.eval()
        self.sensor.eval()
        self.channel.eval()
        self.model.semantic_head.eval()

        temporal_error = 0.0
        temporal_target = 0.0
        full_error = 0.0
        full_target = 0.0
        frame_error = None
        frame_target = None
        total_semantic_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for x_batch, labels in self.val_loader:
                x_batch = x_batch.to(self.device)
                labels = labels.to(self.device)
                _, time_steps, _, _, _ = x_batch.shape
                if frame_error is None:
                    frame_error = np.zeros(
                        time_steps, dtype=np.float64
                    )
                    frame_target = np.zeros(
                        time_steps, dtype=np.float64
                    )

                model_states = None
                sensor_state = None
                channel_state = None
                rec_sequence = []
                z_sequence = []
                for t in range(time_steps):
                    (
                        x_rec,
                        layers,
                        model_states,
                        sensor_state,
                        channel_state,
                        _,
                        _,
                        _,
                    ) = self._process_reconstruction(
                        x_batch[:, t],
                        model_states,
                        sensor_state,
                        channel_state,
                        mode=mode,
                    )
                    if x_rec is None or not layers:
                        raise RuntimeError(
                            'Joint evaluation requires reconstruction and z '
                            'at every time step.'
                        )
                    rec_sequence.append(x_rec)
                    z_sequence.append(layers[-1])

                rec_batch = torch.stack(rec_sequence, dim=1)
                logits = self.model.semantic_head(
                    torch.stack(z_sequence, dim=1)
                )
                predictions = logits.argmax(dim=1)
                delta = x_batch - rec_batch
                target_mean = x_batch.mean(dim=1)
                rec_mean = rec_batch.mean(dim=1)
                batch_size = int(labels.size(0))

                full_error += torch.sum(delta ** 2).item()
                full_target += torch.sum(x_batch ** 2).item()
                temporal_error += torch.sum(
                    (target_mean - rec_mean) ** 2
                ).item()
                temporal_target += torch.sum(target_mean ** 2).item()
                for t in range(time_steps):
                    frame_error[t] += torch.sum(
                        delta[:, t] ** 2
                    ).item()
                    frame_target[t] += torch.sum(
                        x_batch[:, t] ** 2
                    ).item()

                semantic_loss = self.criterion_semantic(
                    logits, labels
                )
                total_semantic_loss += (
                    semantic_loss.item() * batch_size
                )
                total_correct += int(
                    predictions.eq(labels).sum().item()
                )
                total_samples += batch_size
                if include_predictions:
                    all_labels.append(labels.cpu().numpy())
                    all_predictions.append(
                        predictions.cpu().numpy()
                    )

        def ratio_db(error, target):
            if target <= 0:
                return float('nan')
            return float(
                10.0 * np.log10(error / target + 1e-10)
            )

        frame_nmse = [
            ratio_db(error, target)
            for error, target in zip(frame_error, frame_target)
            if target > 0
        ] if frame_error is not None else []
        result = {
            'Temporal-Mean NMSE (dB)': ratio_db(
                temporal_error, temporal_target
            ),
            'Full-Sequence NMSE (dB)': ratio_db(
                full_error, full_target
            ),
            'Framewise Mean NMSE (dB)': (
                float(np.mean(frame_nmse))
                if frame_nmse else float('nan')
            ),
            'Semantic CE': (
                total_semantic_loss / max(1, total_samples)
            ),
            'Joint Accuracy': (
                100.0 * total_correct / max(1, total_samples)
            ),
            'Samples': int(total_samples),
        }
        if include_predictions:
            result['Labels'] = (
                np.concatenate(all_labels)
                if all_labels else np.asarray([])
            )
            result['Predictions'] = (
                np.concatenate(all_predictions)
                if all_predictions else np.asarray([])
            )
        return result

    def get_semantic_predictions(self, mode='spiking'):
        metrics = self.validate_joint_metrics(
            mode=mode, include_predictions=True
        )
        return metrics['Labels'], metrics['Predictions']

    def validate_recon_diagnostics(
        self, mode='spiking', eval_seed=None, include_framewise=True
    ):
        """
        Compute all NMSE diagnostics in one validation pass.  The paper's
        existing temporal-mean global NMSE remains the selected training metric.
        """
        seed = int(Config.EVAL_CHANNEL_SEED if eval_seed is None else eval_seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.model.eval()
        self.sensor.eval()

        temporal_error_energy = 0.0
        temporal_target_energy = 0.0
        full_error_energy = 0.0
        full_target_energy = 0.0
        frame_error_energy = None
        frame_target_energy = None
        valid_batches = 0
        total_samples = 0

        eps_den = 1e-12
        eps_log = 1e-10

        with torch.no_grad():
            for x_batch, _ in self.val_loader:
                x_batch = x_batch.to(self.device)
                B, T, C, H, W = x_batch.shape

                model_states = None
                sensor_state = None
                channel_state = None

                x_rec_seq = []
                valid_recon = True

                for t in range(T):
                    x_frame = x_batch[:, t]

                    x_rec, _, model_states, sensor_state, channel_state, _, _, _ = \
                        self._process_reconstruction(
                            x_frame,
                            model_states,
                            sensor_state,
                            channel_state,
                            mode=mode
                        )

                    if x_rec is None:
                        valid_recon = False
                        break

                    x_rec_seq.append(x_rec)

                if not valid_recon:
                    return {
                        'Temporal-Mean NMSE (dB)': float('nan'),
                        'Full-Sequence NMSE (dB)': float('nan'),
                        'Framewise Mean NMSE (dB)': float('nan'),
                        'Framewise Std NMSE (dB)': float('nan'),
                        'Samples': total_samples,
                    }

                x_rec_batch = torch.stack(x_rec_seq, dim=1)

                target_mean = x_batch.mean(dim=1)
                rec_mean = x_rec_batch.mean(dim=1)
                temporal_error_energy += torch.sum((target_mean - rec_mean) ** 2).item()
                temporal_target_energy += torch.sum(target_mean ** 2).item()
                full_error_energy += torch.sum((x_batch - x_rec_batch) ** 2).item()
                full_target_energy += torch.sum(x_batch ** 2).item()

                if include_framewise:
                    batch_frame_error = torch.sum(
                        (x_batch - x_rec_batch) ** 2, dim=(0, 2, 3, 4)
                    ).detach().cpu().numpy()
                    batch_frame_target = torch.sum(
                        x_batch ** 2, dim=(0, 2, 3, 4)
                    ).detach().cpu().numpy()
                    if frame_error_energy is None:
                        frame_error_energy = np.zeros_like(batch_frame_error, dtype=np.float64)
                        frame_target_energy = np.zeros_like(batch_frame_target, dtype=np.float64)
                    frame_error_energy += batch_frame_error
                    frame_target_energy += batch_frame_target
                valid_batches += 1
                total_samples += int(B)

        if valid_batches == 0 or temporal_target_energy <= 0:
            return {
                'Temporal-Mean NMSE (dB)': float('nan'),
                'Full-Sequence NMSE (dB)': float('nan'),
                'Framewise Mean NMSE (dB)': float('nan'),
                'Framewise Std NMSE (dB)': float('nan'),
                'Samples': total_samples,
            }

        temporal_nmse = 10.0 * np.log10(
            temporal_error_energy / max(temporal_target_energy, eps_den) + eps_log
        )
        full_nmse = 10.0 * np.log10(
            full_error_energy / max(full_target_energy, eps_den) + eps_log
        )
        if include_framewise and frame_error_energy is not None:
            frame_nmse = 10.0 * np.log10(
                frame_error_energy / np.maximum(frame_target_energy, eps_den) + eps_log
            )
            frame_mean = float(np.mean(frame_nmse))
            frame_std = float(np.std(frame_nmse, ddof=1)) if len(frame_nmse) > 1 else 0.0
        else:
            frame_mean = float('nan')
            frame_std = float('nan')

        return {
            'Temporal-Mean NMSE (dB)': float(temporal_nmse),
            'Full-Sequence NMSE (dB)': float(full_nmse),
            'Framewise Mean NMSE (dB)': frame_mean,
            'Framewise Std NMSE (dB)': frame_std,
            'Samples': int(total_samples),
            'Eval Channel Seed': seed,
        }

    def validate_recon(self, mode='spiking', nmse_mode='temporal_mean'):
        diagnostics = self.validate_recon_diagnostics(
            mode=mode,
            include_framewise=(nmse_mode == 'framewise_mean')
        )
        if nmse_mode == 'temporal_mean':
            return diagnostics['Temporal-Mean NMSE (dB)']
        if nmse_mode == 'full_sequence':
            return diagnostics['Full-Sequence NMSE (dB)']
        if nmse_mode == 'framewise_mean':
            return diagnostics['Framewise Mean NMSE (dB)']
        raise ValueError(
            f"Unknown nmse_mode={nmse_mode}. Use 'temporal_mean', "
            "'full_sequence', or 'framewise_mean'."
        )
