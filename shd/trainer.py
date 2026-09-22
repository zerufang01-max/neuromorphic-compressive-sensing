import torch
import torch.nn as nn
import torch.optim as optim
import os
import math
import numpy as np
import csv
from config import Config
import utils

class Trainer:
    def __init__(self, model, edge_sensor, wireless_channel, train_loader, val_loader):
        self.model = model
        self.sensor = edge_sensor
        self.channel = wireless_channel
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = Config.DEVICE
        
        self.criterion_mse = nn.MSELoss()
        self.criterion_cls = nn.CrossEntropyLoss(
            label_smoothing=float(
                getattr(Config, 'DOWNSTREAM_LABEL_SMOOTHING', 0.0)
            )
        )
        # The jointly trained task head uses ordinary semantic CE.
        self.criterion_semantic = nn.CrossEntropyLoss()
        
        self.best_nmse = 1000.0
        self.best_acc = 0.0
        self.downstream_classifier = None
        

    def _preprocess_batch(self, x_batch):
        return x_batch

    def _process_reconstruction(self, x_input, model_states=None, sensor_state=None, channel_state=None, mode='ann', tau=None):
        y_encoded, next_sensor_state, aux_dict = self.sensor(x_input, state=sensor_state, mode=mode, tau=tau)
        
        tx_prob = aux_dict.get('tx_prob') if aux_dict else None
        tx_hard = aux_dict.get('tx_hard') if aux_dict else None
        
        y_noisy, next_channel_state = self.channel(
            y_encoded, 
            state=channel_state, 
            mode=mode,
            tx_prob=tx_prob,
            tx_hard=tx_hard,
            tx_aux=aux_dict,
        )
        
        prev_mode = getattr(self.model, 'mode', None)
        if hasattr(self.model, 'mode'):
            self.model.mode = 'rate' if mode == 'ann' else 'spiking'
        
        x_rec, layers, next_model_states, internal_stats = self.model(y_noisy, states=model_states)
        
        if hasattr(self.model, 'mode') and prev_mode is not None:
            self.model.mode = prev_mode 
            
        return x_rec, layers, next_model_states, next_sensor_state, next_channel_state, y_encoded, internal_stats, aux_dict

    def _unfreeze_all_params(self, mode):
        is_hybrid_lista = (
            self.model.__class__.__name__ == 'HybridLISTA'
        )
        for name, param in self.model.named_parameters():
            if name.startswith('classifier.'):
                # Phase 3 never uses the downstream head.  Keeping it frozen
                # avoids needless optimizer bookkeeping; Phase 4 re-enables it.
                param.requires_grad = False
            elif (
                is_hybrid_lista
                and mode == 'spiking'
                and name.startswith(('W_e.', 'S_k.', 'theta_ann.'))
            ):
                # The ANN branch is not executed by S-LISTA.
                param.requires_grad = False
            elif (
                is_hybrid_lista
                and mode == 'ann'
                and name.startswith(('P_snn.', 'PD_snn_k.'))
            ):
                # The SNN branch is not executed by ANN LISTA.
                param.requires_grad = False
            elif name.startswith('semantic_head.'):
                param.requires_grad = (
                    hasattr(self.model, 'semantic_head')
                    and Config.ALPHA_SEMANTIC > 0
                )
            elif name.startswith('theta_ann.'):
                # Only ANN LISTA thresholds are learnable.  SNN thresholds are
                # registered buffers and remain fixed.
                param.requires_grad = (mode == 'ann')
            elif 'theta' in name or 'thresh' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
                
        for name, param in self.sensor.named_parameters():
            if 'thresh' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def evaluate_bits(self, mode='spiking'):
        self.sensor.eval()
        total_samples = 0
        total_bits = 0.0
        total_packets = 0.0
        
        total_events = 0
        total_elements = 0
        
        eval_stats = {'total': 0, 'assisted': 0, 'cf': 0, 'state_contrib_sum': 0.0, 'elements': 0, 'thresh': 0.5}
        total_iou_sum = 0.0
        total_iou_count = 0
        
        with torch.no_grad():
            for x_batch, _ in self.val_loader:
                x_batch = self._preprocess_batch(x_batch)
                
                B, T, D = x_batch.shape
                sensor_state = None
                
                batch_events = 0
                batch_packets = 0
                batch_transport_bits = 0.0
                prev_tx_hard = None
                
                
                for t in range(T):
                    x_frame = x_batch[:, t]
                    _, sensor_state, aux_dict = self.sensor(x_frame, state=sensor_state, mode=mode)
                    
                    tx_hard = aux_dict['tx_hard']
                    signed_tx = bool(aux_dict.get('signed_tx', False))
                   
                    if mode != 'ann':
                        occupancy = tx_hard.abs() if signed_tx else tx_hard
                        batch_events += occupancy.sum().item()
                        if Config.SNN_TRANSPORT == 'block_aer_awgn_hard':
                            block_size = int(Config.BLOCK_SIZE)
                            blocks = occupancy.reshape(
                                B, Config.MEAS_DIM // block_size, block_size
                            )
                            batch_packets += (
                                blocks.amax(dim=-1) > 0
                            ).sum().item()
                        if (
                            Config.SNN_TRANSPORT
                            == 'block_aer_awgn_hard'
                        ):
                            batch_transport_bits += float(
                                self.channel.block_aer_bits(tx_hard)
                                .sum().item()
                            )
                        elif (
                            Config.SNN_TRANSPORT
                            == 'dense_spike_awgn_hard'
                        ):
                            # The full binary bitmap includes zero entries.
                            batch_transport_bits += float(
                                Config.MEAS_DIM * B
                            )
                    
                    if mode != 'ann':
                        if prev_tx_hard is not None:
                            current_occ = tx_hard.abs() if signed_tx else tx_hard
                            previous_occ = (
                                prev_tx_hard.abs()
                                if signed_tx else prev_tx_hard
                            )
                            intersection = (
                                current_occ * previous_occ
                            ).sum(dim=1)
                            union = (
                                (current_occ + previous_occ) > 0
                            ).float().sum(dim=1)
                            valid_mask = union > 0
                            if valid_mask.any():
                                ious = intersection[valid_mask] / union[valid_mask]
                                total_iou_sum += ious.sum().item()
                                total_iou_count += valid_mask.sum().item()
                        prev_tx_hard = tx_hard
                        
                        if Config.LOG_TX_STATE_STATS and aux_dict:
                            eval_stats['total'] += aux_dict.get('tx_total_spikes', 0)
                            eval_stats['assisted'] += aux_dict.get('tx_assisted_spikes', 0)
                            eval_stats['cf'] += aux_dict.get('tx_counterfactual_reset_spikes', 0)
                            eval_stats['state_contrib_sum'] += aux_dict.get('tx_state_contribution_sum', 0.0)
                            eval_stats['elements'] += aux_dict.get('tx_elements', 0)
                            eval_stats['thresh'] = aux_dict.get('tx_threshold', 0.5)
                
                # "Firing rate" always means the fraction of active source
                # measurements, independent of the packetization protocol.
                total_events += batch_events
                total_elements += B * T * Config.MEAS_DIM
                total_samples += B
                
               
                if mode == 'ann':
                    total_bits += (
                        Config.MEAS_DIM * T * int(Config.ANN_QUANT_BITS)
                    ) * B
                elif Config.SNN_TRANSPORT in (
                    'block_aer_awgn_hard',
                    'dense_spike_awgn_hard',
                ):
                    total_bits += batch_transport_bits
                    if Config.SNN_TRANSPORT == 'block_aer_awgn_hard':
                        total_packets += batch_packets
                else:
                    raise ValueError(
                        f"Unsupported AWGN SNN transport: "
                        f"{Config.SNN_TRANSPORT!r}"
                    )
                    
        if mode != 'ann':
            avg_firing_rate = total_events / max(1, total_elements)
            print(f" -> [Info] SNN Comm Bottleneck Firing Rate: {avg_firing_rate * 100:.2f}%")
            if total_iou_count > 0:
                avg_iou = total_iou_sum / total_iou_count
                print(f" -> [Info] Temporal Spike IoU (Consecutive Frames): {avg_iou*100:.2f}%")
            if Config.LOG_TX_STATE_STATS and eval_stats['total'] > 0:
                assist_ratio = eval_stats['assisted'] / eval_stats['total']
                state_contribution = eval_stats['state_contrib_sum'] / (eval_stats['thresh'] * eval_stats['total'])
                cf_rate = eval_stats['cf'] / max(1, eval_stats['elements'])
                print(f" -> [Info] TX Diagnostics: Assist {assist_ratio*100:.1f}% | State Contrib {state_contribution*100:.1f}% | CF Rate {cf_rate*100:.2f}%")
                
        avg_bits = total_bits / max(1, total_samples)
        raw_signal_dim = getattr(Config, 'INPUT_DIM', 700) * getattr(Config, 'TIME_STEPS', 25)
        
       
        avg_measurements = total_events / max(1, total_samples)
            
        measurement_rate = avg_measurements / max(1, raw_signal_dim)
        
        return {
            'bits_per_sample': avg_bits,
            'measurements': avg_measurements,
            'measurement_rate': measurement_rate,
            'active_blocks_per_sample': (
                total_packets / max(1, total_samples)
            ),
            'active_blocks_per_frame': (
                total_packets
                / max(
                    1,
                    total_samples * getattr(Config, 'TIME_STEPS', 25),
                )
            ),
        }

    def train_dynamic_recon(
        self, hp=None, model_name="model", mode='spiking',
        freeze_sensor=False, alpha_rate=None,
    ):
        print(f"\n>>> DYNAMIC RECONSTRUCTION TRAINING ({model_name} | Mode: {mode.upper()})")
        self._unfreeze_all_params(mode)
        
        if freeze_sensor:
            for param in self.sensor.parameters():
                param.requires_grad = False
                
        if hp is None: hp = Config.PHASE3_CONFIG
        
        self.sensor.mode = mode
        if hasattr(self.model, 'mode'):
            self.model.mode = 'rate' if mode == 'ann' else 'spiking'

        th_val = self.model.get_threshold_value() if hasattr(self.model, 'get_threshold_value') else None
        
        model_cls_name = self.model.__class__.__name__
        if alpha_rate is not None:
            current_alpha_rate = float(alpha_rate)
        elif model_cls_name in {'SpikingDenseRecon', 'GenericSNNRecon'}:
            current_alpha_rate = Config.ALPHA_RATE_SCNN
        else:
            current_alpha_rate = Config.ALPHA_RATE_SLISTA
            
        save_name = Config.get_recon_ckpt_name(
            mode, model_name, th_val, current_alpha_rate, hp=hp
        ) if hasattr(Config, 'get_recon_ckpt_name') else f"recon_{model_name}.pth"
        save_path = os.path.join(Config.RESULTS_DIR, save_name)
        semantic_enabled = (
            (
                Config.ALPHA_SEMANTIC > 0
                or getattr(
                    Config, 'SEMANTIC_DETACH_BACKBONE', False
                )
            )
            and hasattr(self.model, 'semantic_head')
        )
        if not semantic_enabled and hasattr(self.model, 'semantic_head'):
            # A true reconstruction-only ablation: the z head is absent from
            # both the loss graph and the optimizer.  This also prevents Adam
            # weight decay from changing an otherwise unused head.
            self.model.semantic_head.eval()
            for param in self.model.semantic_head.parameters():
                param.requires_grad = False
        anneal_log_path = os.path.join(
            Config.RESULTS_DIR, f"annealing_stats_{model_name.lower()}.csv"
        )
        anneal_history = []
        
        param_groups = self.model.get_optimizer_groups(hp)
        
        if not freeze_sensor:
            lr_sensor = hp.get('lr_sensor', hp.get('lr_lista_snn', 7e-4))
            sensor_params = [p for p in self.sensor.parameters() if p.requires_grad]
            if len(sensor_params) > 0:
                param_groups.append({'params': sensor_params, 'lr': lr_sensor})
        
        param_groups = [g for g in param_groups if len(list(g['params'])) > 0]
        
        optimizer = optim.Adam(param_groups, weight_decay=hp['weight_decay'])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hp['epochs'], eta_min=hp['lr_min'])
        all_params = [
            param
            for group in param_groups
            for param in group['params']
            if param.requires_grad
        ]
        
        # Reconstruction is the primary task.  All auxiliary metrics are
        # reported from the checkpoint selected by temporal-mean val NMSE.
        # Accuracy is used only to break an exact NMSE tie.
        self.best_nmse = 1000.0
        selected_semantic_acc = float('-inf')
        selected_semantic_nmse = float('inf')
        selected_semantic_epoch = 0
        
        for epoch in range(hp['epochs']):
            current_tau = Config.get_tau(epoch, hp['epochs'])
            
            self.model.train()
            self.channel.train()
            if freeze_sensor:
                self.sensor.eval()
            else:
                self.sensor.train()
            
            metric_zero = torch.zeros(
                (), device=self.device, dtype=torch.float64
            )
            ep_loss = metric_zero.clone()
            ep_recon_objective = metric_zero.clone()
            ep_semantic_ce = metric_zero.clone()
            ep_semantic_correct = metric_zero.clone()
            semantic_samples = 0
            epoch_activity_accum = metric_zero.clone()
            total_steps = 0
            
            epoch_tx_stats = {
                'total': metric_zero.clone(),
                'assisted': metric_zero.clone(),
                'cf': metric_zero.clone(),
                'state_contrib_sum': metric_zero.clone(),
                'elements': 0,
                'soft_sum': metric_zero.clone(),
                'abs_soft_hard_sum': metric_zero.clone(),
                'threshold': metric_zero.new_tensor(0.5),
            }
            threshold_margin_samples = []
            sampled_margins = 0
            
            for x_batch, labels in self.train_loader:
                x_batch = self._preprocess_batch(x_batch)
                
                B, T, D = x_batch.shape
                
                model_states = None
                sensor_state = None
                channel_state = None
                loss_total_step = 0
                semantic_z_sequence = []
                
                for t in range(T):
                    x_frame = x_batch[:, t]
                    
                    x_rec, layers, model_states, sensor_state, channel_state, y_spikes, stats, aux_dict = \
                        self._process_reconstruction(x_frame, model_states, sensor_state, channel_state, mode=mode, tau=current_tau)
                        
                    if mode == 'spiking' and Config.LOG_TX_STATE_STATS and aux_dict:
                        epoch_tx_stats['total'] += aux_dict[
                            'tx_total_spikes'
                        ].detach().to(torch.float64)
                        epoch_tx_stats['assisted'] += aux_dict[
                            'tx_assisted_spikes'
                        ].detach().to(torch.float64)
                        epoch_tx_stats['cf'] += aux_dict[
                            'tx_counterfactual_reset_spikes'
                        ].detach().to(torch.float64)
                        epoch_tx_stats['state_contrib_sum'] += aux_dict[
                            'tx_state_contribution_sum'
                        ].detach().to(torch.float64)
                        epoch_tx_stats['elements'] += aux_dict.get('tx_elements', 0)
                        epoch_tx_stats['threshold'] = aux_dict[
                            'tx_threshold'
                        ].detach().to(torch.float64)

                    if (
                        mode == 'spiking'
                        and Config.SNN_TRANSPORT == 'block_aer_awgn_hard'
                        and getattr(Config, 'LOG_ANNEAL_STATS', False)
                        and aux_dict
                    ):
                        epoch_tx_stats['soft_sum'] += aux_dict[
                            'tx_soft_sum'
                        ].detach().to(torch.float64)
                        epoch_tx_stats['abs_soft_hard_sum'] += aux_dict[
                            'tx_abs_soft_hard_sum'
                        ].detach().to(torch.float64)
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
                                # Keep the bounded diagnostic reservoir on GPU;
                                # copy it once at epoch end.
                                threshold_margin_samples.append(
                                    sample.detach()
                                )
                                sampled_margins += int(take)
                    
                    step_mse_loss = 0
                    if x_rec is not None:
                        step_mse_loss = Config.ALPHA_MSE * self.criterion_mse(x_rec, x_frame)
                    
                    step_sparsity_loss = 0
                    if len(layers) > 0:
                        with torch.no_grad():
                            current_activity = sum(
                                layer.detach().abs().mean()
                                for layer in layers
                            ) / len(layers)
                            epoch_activity_accum += (
                                current_activity.to(torch.float64)
                            )
                        total_steps += 1
                        
                        l1_norm_sum = 0
                        for l in layers:
                            l1_norm_sum += l.abs().sum(dim=1).mean()
                        l1_norm_sum /= len(layers)

                        if mode == 'ann':
                            step_sparsity_loss = getattr(Config, 'ALPHA_L1_ANN', 1e-7) * l1_norm_sum
                        else:
                            step_sparsity_loss = getattr(Config, 'ALPHA_L1_SNN', 1e-7) * l1_norm_sum

                        if semantic_enabled:
                            semantic_z_sequence.append(layers[-1])
                    
                    step_rate_loss = 0
                    if (
                        mode == 'spiking'
                        and Config.SNN_TRANSPORT
                        == 'block_aer_awgn_hard'
                    ):
                        model_cls_name = self.model.__class__.__name__
                        if model_cls_name in {
                            'SpikingDenseRecon', 'GenericSNNRecon'
                        }:
                            alpha_rate = Config.ALPHA_RATE_SCNN
                        else:
                            alpha_rate = Config.ALPHA_RATE_SLISTA
                            
                        step_rate_loss = (
                            alpha_rate
                            * self.channel.expected_rate_bits(aux_dict)
                        )

                    loss_total_step += (step_mse_loss + step_sparsity_loss + step_rate_loss)

                recon_objective = loss_total_step / T
                semantic_ce = None
                semantic_term = recon_objective.new_zeros(())
                if (
                    semantic_enabled
                    and hasattr(self.model, 'semantic_head')
                    and len(semantic_z_sequence) == T
                ):
                    z_stack = torch.stack(semantic_z_sequence, dim=1)
                    detach_semantic_backbone = bool(
                        getattr(
                            Config,
                            'SEMANTIC_DETACH_BACKBONE',
                            False,
                        )
                    )
                    if detach_semantic_backbone:
                        z_stack = z_stack.detach()
                    semantic_logits = self.model.semantic_head(z_stack)
                    semantic_ce = self.criterion_semantic(semantic_logits, labels)
                    normalized_semantic_ce = semantic_ce / np.log(Config.NUM_CLASSES)
                    if detach_semantic_backbone:
                        # The head remains trained, but its gradient cannot
                        # shape the sensor, S-LISTA iterations, or dictionary.
                        semantic_term = (
                            float(
                                getattr(
                                    Config,
                                    'DETACHED_HEAD_LOSS_WEIGHT',
                                    1.0,
                                )
                            )
                            * normalized_semantic_ce
                        )
                    else:
                        semantic_term = (
                            Config.ALPHA_SEMANTIC
                            * normalized_semantic_ce
                        )

                loss_final = recon_objective + semantic_term
                
                optimizer.zero_grad(set_to_none=True)
                loss_final.backward()
                
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                
                optimizer.step()
                if (
                    mode == 'ann'
                    and hasattr(self.model, 'project_ann_thresholds')
                ):
                    self.model.project_ann_thresholds()
                ep_loss += loss_final.detach().to(torch.float64)
                ep_recon_objective += (
                    recon_objective.detach().to(torch.float64)
                )
                if semantic_ce is not None:
                    batch_size = int(labels.size(0))
                    ep_semantic_ce += (
                        semantic_ce.detach().to(torch.float64)
                        * batch_size
                    )
                    ep_semantic_correct += (
                        semantic_logits.detach().argmax(dim=1)
                        .eq(labels)
                        .sum()
                        .to(torch.float64)
                    )
                    semantic_samples += batch_size
                    
            scheduler.step()

            # Exactly one synchronization for all scalar training diagnostics.
            scalar_names = [
                'ep_loss', 'ep_recon_objective', 'ep_semantic_ce',
                'ep_semantic_correct', 'epoch_activity_accum',
                'tx_total', 'tx_assisted', 'tx_cf',
                'tx_state_contrib_sum', 'tx_soft_sum',
                'tx_abs_soft_hard_sum', 'tx_threshold',
            ]
            scalar_values = torch.stack([
                ep_loss,
                ep_recon_objective,
                ep_semantic_ce,
                ep_semantic_correct,
                epoch_activity_accum,
                epoch_tx_stats['total'],
                epoch_tx_stats['assisted'],
                epoch_tx_stats['cf'],
                epoch_tx_stats['state_contrib_sum'],
                epoch_tx_stats['soft_sum'],
                epoch_tx_stats['abs_soft_hard_sum'],
                epoch_tx_stats['threshold'],
            ]).detach().cpu().tolist()
            scalar_stats = dict(zip(scalar_names, scalar_values))
            ep_loss = scalar_stats['ep_loss']
            ep_recon_objective = scalar_stats[
                'ep_recon_objective'
            ]
            ep_semantic_ce = scalar_stats['ep_semantic_ce']
            ep_semantic_correct = scalar_stats[
                'ep_semantic_correct'
            ]
            epoch_activity_accum = scalar_stats[
                'epoch_activity_accum'
            ]
            epoch_tx_stats.update({
                'total': scalar_stats['tx_total'],
                'assisted': scalar_stats['tx_assisted'],
                'cf': scalar_stats['tx_cf'],
                'state_contrib_sum': scalar_stats[
                    'tx_state_contrib_sum'
                ],
                'soft_sum': scalar_stats['tx_soft_sum'],
                'abs_soft_hard_sum': scalar_stats[
                    'tx_abs_soft_hard_sum'
                ],
                'threshold': scalar_stats['tx_threshold'],
            })

            joint_val = self.validate_joint_metrics(mode=mode)
            val_nmse = joint_val['Temporal-Mean NMSE (dB)']
            avg_epoch_loss = ep_loss / max(1, len(self.train_loader))
            avg_recon_objective = (
                ep_recon_objective / max(1, len(self.train_loader))
            )
            avg_semantic_ce = (
                ep_semantic_ce / semantic_samples
                if semantic_samples > 0 else float('nan')
            )
            avg_semantic_acc = (
                100.0 * ep_semantic_correct / semantic_samples
                if semantic_samples > 0 else float('nan')
            )
            avg_activity = epoch_activity_accum / max(1, total_steps)

            anneal_row = None
            if (
                mode == 'spiking'
                and Config.SNN_TRANSPORT == 'block_aer_awgn_hard'
                and getattr(Config, 'LOG_ANNEAL_STATS', False)
            ):
                elements = max(1, epoch_tx_stats['elements'])
                soft_rate = epoch_tx_stats['soft_sum'] / elements
                hard_rate = epoch_tx_stats['total'] / elements
                rate_mismatch = abs(soft_rate - hard_rate)
                pointwise_mismatch = epoch_tx_stats['abs_soft_hard_sum'] / elements

                density_max = float('nan')
                if threshold_margin_samples:
                    margins = (
                        torch.cat(threshold_margin_samples)
                        .cpu()
                        .numpy()
                    )
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
                block_size = int(Config.BLOCK_SIZE)
                local_bits = (
                    math.ceil(math.log2(block_size))
                    if block_size > 1 else 0
                )
                payload_scale = (
                    local_bits * Config.MEAS_DIM
                )
                anneal_row = {
                    'Epoch': epoch + 1,
                    'Seed': int(Config.SEED),
                    'Model Name': model_name,
                    'Training Loss': float(avg_epoch_loss),
                    'Validation NMSE (dB)': float(val_nmse),
                    'Mean Training Activity': float(avg_activity),
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
            
            semantic_eval_interval = max(
                1, int(getattr(Config, 'SEMANTIC_EVAL_INTERVAL', 5))
            )
            semantic_val = {
                'loss': joint_val['Semantic CE'],
                'accuracy': joint_val['Joint Accuracy'],
            }

            current_semantic_acc = float(semantic_val['accuracy'])
            current_semantic_nmse = float(val_nmse)
            better_nmse = (
                current_semantic_nmse < self.best_nmse
            )
            tied_nmse = np.isclose(
                current_semantic_nmse,
                self.best_nmse,
                rtol=0.0,
                atol=1e-12,
            )
            better_accuracy_tiebreak = (
                semantic_enabled
                and
                tied_nmse
                and current_semantic_acc > selected_semantic_acc
            )

            if better_nmse or better_accuracy_tiebreak:
                self.best_nmse = current_semantic_nmse
                selected_semantic_acc = current_semantic_acc
                selected_semantic_nmse = current_semantic_nmse
                selected_semantic_epoch = epoch + 1
                torch.save({
                    'model': self.model.state_dict(),
                    'sensor': self.sensor.state_dict(),
                    'epoch': epoch + 1,
                    # All reported values belong to this same checkpoint.
                    'val_nmse_db': current_semantic_nmse,
                    'joint_accuracy': (
                        current_semantic_acc
                        if semantic_enabled else float('nan')
                    ),
                    'semantic_ce': (
                        float(semantic_val['loss'])
                        if semantic_enabled else float('nan')
                    ),
                    'semantic_config': {
                        'alpha': float(Config.ALPHA_SEMANTIC),
                        'detach_backbone': bool(
                            getattr(
                                Config,
                                'SEMANTIC_DETACH_BACKBONE',
                                False,
                            )
                        ),
                        'detached_head_loss_weight': float(
                            getattr(
                                Config,
                                'DETACHED_HEAD_LOSS_WEIGHT',
                                1.0,
                            )
                        ),
                        'lr': float(hp['lr_semantic']),
                        'weight_decay': float(
                            Config.SEMANTIC_WEIGHT_DECAY
                        ),
                        'dropout': float(Config.SEMANTIC_DROPOUT),
                        'head_trained': bool(semantic_enabled),
                    },
                    'transport': (
                        Config.SNN_TRANSPORT
                        if mode == 'spiking'
                        else Config.transport_tag('ann')
                    ),
                    'slista_membrane_readout': (
                        str(getattr(
                            Config,
                            'SLISTA_MEMBRANE_READOUT',
                            'support_gated',
                        ))
                        if mode == 'spiking'
                        and self.model.__class__.__name__ == 'HybridLISTA'
                        else None
                    ),
                    'code_version': Config.CODE_VERSION,
                    'selection_metric': 'temporal_mean_nmse',
                    'selection_tiebreaker': (
                        'joint_validation_accuracy'
                        if semantic_enabled else 'none'
                    ),
                    'slista_p_weight_decay': float(
                        getattr(Config, 'SLISTA_P_WEIGHT_DECAY', 0.0)
                    ),
                    'slista_p_regularized_parameter': 'P_snn.weight',
                }, save_path)

            if (epoch + 1) % semantic_eval_interval == 0 or epoch == 0:
                message = (
                    f"[Ep {epoch+1}] Loss: {avg_epoch_loss:.4f} "
                    f"(Recon: {avg_recon_objective:.4f}) | "
                    f"Activity: {avg_activity:.4f} | "
                    f"Val NMSE: {val_nmse:.2f} dB "
                    f"(Best: {self.best_nmse:.2f} dB) | Tau: {current_tau:.4f}"
                )
                if semantic_samples > 0:
                    message += (
                        f" | Sem CE: {avg_semantic_ce:.4f} "
                        f"| Sem Train Acc: {avg_semantic_acc:.2f}% "
                        f"| Sem Val Acc: {semantic_val['accuracy']:.2f}% "
                        f"(Selected: {selected_semantic_acc:.2f}% "
                        f"@ NMSE-best Ep {selected_semantic_epoch})"
                    )
                print(message)
                if mode == 'ann' and hasattr(self.model, 'get_ann_threshold_values'):
                    theta_values = self.model.get_ann_threshold_values()
                    theta_text = ', '.join(f'{value:.4f}' for value in theta_values)
                    print(f" -> ANN LISTA thresholds: [{theta_text}]")
                if mode == 'spiking' and Config.LOG_TX_STATE_STATS and epoch_tx_stats['total'] > 0:
                    assist_ratio = epoch_tx_stats['assisted'] / epoch_tx_stats['total']
                    state_contribution = epoch_tx_stats['state_contrib_sum'] / (epoch_tx_stats['threshold'] * epoch_tx_stats['total'])
                    cf_rate = epoch_tx_stats['cf'] / max(1, epoch_tx_stats['elements'])
                    actual_tx_rate = epoch_tx_stats['total'] / max(1, epoch_tx_stats['elements'])
                    print(f" -> TX Diagnostics: Assist Ratio {assist_ratio*100:.1f}% | State Contrib {state_contribution*100:.1f}% | CF Rate {cf_rate*100:.2f}% | Actual Rate {actual_tx_rate*100:.2f}%")
                if anneal_row is not None:
                    print(
                        " -> Anneal Diagnostics: "
                        f"Soft Rate {anneal_row['Soft Firing Rate']*100:.2f}% | "
                        f"Hard Rate {anneal_row['Hard Firing Rate']*100:.2f}% | "
                        f"Rate Gap {anneal_row['Normalized Payload Mismatch']:.6f} | "
                        f"Mean |soft-hard| {anneal_row['Mean Absolute Soft-Hard Gap']:.6f}"
                    )

        if semantic_enabled:
            print(
                f" -> Selected reconstruction checkpoint: epoch "
                f"{selected_semantic_epoch} | Temporal-Mean NMSE "
                f"{selected_semantic_nmse:.4f} dB | Z Val Acc "
                f"{selected_semantic_acc:.2f}%"
            )

        return selected_semantic_acc

    def validate_joint_metrics(
        self,
        mode='spiking',
        include_predictions=False,
    ):
        """Evaluate fidelity and the deployed joint Z head in one pass.

        The returned NMSE values, semantic CE and accuracy are computed from
        exactly the same checkpoint, samples, recovered Z sequence and AWGN
        realization.  Accuracy/CE are sample weighted; NMSE is accumulated as
        global energy before conversion to dB.
        """
        if not hasattr(self.model, 'semantic_head'):
            raise RuntimeError(
                'Task-driven main evaluation requires semantic_head.'
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

        try:
            with torch.no_grad():
                for x_batch, labels in self.val_loader:
                    x_batch = self._preprocess_batch(x_batch)
                    _, T, _ = x_batch.shape
                    if frame_error is None:
                        frame_error = np.zeros(T, dtype=np.float64)
                        frame_target = np.zeros(T, dtype=np.float64)

                    model_states = None
                    sensor_state = None
                    channel_state = None
                    x_rec_sequence = []
                    z_sequence = []

                    for t in range(T):
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
                                'Joint evaluation requires reconstruction '
                                'and a final LISTA code at every time step.'
                            )
                        x_rec_sequence.append(x_rec)
                        z_sequence.append(layers[-1])

                    x_rec_batch = torch.stack(x_rec_sequence, dim=1)
                    z_batch = torch.stack(z_sequence, dim=1)
                    logits = self.model.semantic_head(z_batch)

                    delta = x_batch - x_rec_batch
                    target_mean = x_batch.mean(dim=1)
                    recon_mean = x_rec_batch.mean(dim=1)
                    batch_size = int(labels.size(0))
                    predictions = logits.argmax(dim=1)

                    # Transfer per-batch scalar reductions in one synchronization.
                    batch_scalars = torch.stack([
                        torch.sum(delta ** 2),
                        torch.sum(x_batch ** 2),
                        torch.sum(
                            (target_mean - recon_mean) ** 2
                        ),
                        torch.sum(target_mean ** 2),
                        self.criterion_semantic(logits, labels),
                        predictions.eq(labels).sum().to(delta.dtype),
                    ])
                    batch_frame_error = torch.stack([
                        torch.sum(delta[:, t] ** 2)
                        for t in range(T)
                    ])
                    batch_frame_target = torch.stack([
                        torch.sum(x_batch[:, t] ** 2)
                        for t in range(T)
                    ])
                    host_values = torch.cat([
                        batch_scalars,
                        batch_frame_error,
                        batch_frame_target,
                    ]).cpu().tolist()

                    full_error += host_values[0]
                    full_target += host_values[1]
                    temporal_error += host_values[2]
                    temporal_target += host_values[3]
                    total_semantic_loss += (
                        host_values[4] * batch_size
                    )
                    total_correct += int(host_values[5])
                    for t in range(T):
                        frame_error[t] += host_values[6 + t]
                        frame_target[t] += host_values[6 + T + t]
                    total_samples += batch_size

                    if include_predictions:
                        all_labels.append(labels.cpu().numpy())
                        all_predictions.append(
                            predictions.cpu().numpy()
                        )
        finally:
            # Leave the channel generator at the end of the validation pass.
            pass

        def ratio_db(error, target):
            if target <= 0:
                return float('nan')
            return float(
                10.0 * np.log10(error / target + 1e-10)
            )

        frame_db = [
            ratio_db(error, target)
            for error, target in zip(frame_error, frame_target)
            if target > 0
        ] if frame_error is not None else []

        metrics = {
            'Temporal-Mean NMSE (dB)': ratio_db(
                temporal_error, temporal_target
            ),
            'Full-Sequence NMSE (dB)': ratio_db(
                full_error, full_target
            ),
            'Framewise Mean NMSE (dB)': (
                float(np.mean(frame_db))
                if frame_db else float('nan')
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
            metrics['Labels'] = (
                np.concatenate(all_labels)
                if all_labels else np.asarray([])
            )
            metrics['Predictions'] = (
                np.concatenate(all_predictions)
                if all_predictions else np.asarray([])
            )
        return metrics

    def validate_semantic_head(self, mode='spiking'):
        metrics = self.validate_joint_metrics(mode=mode)
        return {
            'loss': metrics['Semantic CE'],
            'accuracy': metrics['Joint Accuracy'],
        }

    def get_semantic_predictions(self, mode='spiking'):
        metrics = self.validate_joint_metrics(
            mode=mode,
            include_predictions=True,
        )
        return metrics['Labels'], metrics['Predictions']

    def _make_classifier_input(self, x_batch, mode="spiking", target="recon"):
        B, T, D = x_batch.shape
        input_seq = []
        model_states = None
        sensor_state = None
        channel_state = None
        
        with torch.no_grad():
            for t in range(T):
                x_frame = x_batch[:, t]
                x_rec, layers, model_states, sensor_state, channel_state, _, _, _ = \
                    self._process_reconstruction(x_frame, model_states, sensor_state, channel_state, mode=mode)

                if target == 'z':
                    if not layers:
                        raise RuntimeError(
                            "Z-probe requested, but the reconstruction model "
                            "did not return a latent sparse-code layer."
                        )
                    item = layers[-1]
                elif target == 'recon':
                    item = x_rec if x_rec is not None else x_frame
                else:
                    raise ValueError(
                        f"Unknown classifier target={target!r}; "
                        "expected 'z' or 'recon'."
                    )
                    
                input_seq.append(item)
                
        return torch.stack(input_seq, dim=1)

    def set_downstream_classifier(self, classifier):
        self.downstream_classifier = classifier.to(self.device)
        return self.downstream_classifier

    def _require_downstream_classifier(self):
        if self.downstream_classifier is None:
            raise RuntimeError(
                "The reconstructed-x downstream classifier has not been "
                "attached to this Trainer."
            )
        return self.downstream_classifier

    def train_phase4(
        self, hp=None, model_name="model", mode='snn', target='recon',
        save_path=None,
    ):
        print(f"\n>>> CLASSIFICATION TRAINING ({mode.upper()} | Input: {target.upper()})")
        if hp is None: hp = getattr(Config, 'PHASE4_CONFIG', {'epochs': 100, 'lr': 3e-4, 'lr_min': 1e-7, 'weight_decay': 0})
        
        th_val = self.model.get_threshold_value() if hasattr(self.model, 'get_threshold_value') else None
        
        classifier_module = self._require_downstream_classifier()

        self.sensor.eval()
        self.model.eval()
        for param in self.sensor.parameters(): param.requires_grad = False
        for param in self.model.parameters(): param.requires_grad = False
        
        classifier_module.train()
        for param in classifier_module.parameters(): param.requires_grad = True
        
        self.sensor.mode = mode
        if hasattr(self.model, 'mode'): 
            self.model.mode = 'rate' if mode == 'ann' else 'spiking'
        
        optimizer = optim.Adam(classifier_module.parameters(), lr=hp['lr'], weight_decay=hp['weight_decay'])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hp['epochs'], eta_min=hp['lr_min'])
        
        self.best_acc = 0.0
        
        if save_path is None:
            save_name = Config.get_cls_ckpt_name(
                mode, model_name, th_val, target=target
            )
            save_path = os.path.join(Config.RESULTS_DIR, save_name)

        best_epoch = 0
        
        for epoch in range(hp['epochs']):
            classifier_module.train()
            ep_loss_sum = 0.0
            ep_correct = 0
            ep_samples = 0
            
            for x_batch, labels in self.train_loader:
                x_batch = self._preprocess_batch(x_batch)
                
                optimizer.zero_grad(set_to_none=True)
                x_stack = self._make_classifier_input(x_batch, mode=mode, target=target)
                
                logits = classifier_module(x_stack)
                loss = self.criterion_cls(logits, labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(classifier_module.parameters(), max_norm=1.0)
                optimizer.step()
                
                batch_size = int(labels.size(0))
                ep_loss_sum += loss.item() * batch_size
                ep_correct += utils.count_correct(logits, labels)
                ep_samples += batch_size
                
            scheduler.step()
            # validate_cls deliberately fixes the AWGN draw for comparable
            # validation.  Restore the training RNG afterwards so every epoch
            # does not see the same shuffle and channel-error realization.
            training_rng_state = utils.capture_rng_state()
            try:
                val_acc = self.validate_cls(mode=mode, target=target)
            finally:
                utils.restore_rng_state(training_rng_state)
            
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                best_epoch = epoch + 1
                torch.save({
                    'classifier': classifier_module.state_dict(),
                    'epoch': best_epoch,
                    'val_accuracy': self.best_acc,
                    'target': target,
                    'phase4_config': dict(hp),
                    'downstream_dropout': float(
                        Config.DOWNSTREAM_DROPOUT
                    ),
                    'downstream_label_smoothing': float(
                        Config.DOWNSTREAM_LABEL_SMOOTHING
                    ),
                }, save_path)
            
            if (epoch+1) % 5 == 0 or epoch == 0:
                train_loss = ep_loss_sum / max(1, ep_samples)
                train_acc = 100.0 * ep_correct / max(1, ep_samples)
                print(
                    f"[{mode.upper()} CLS Ep {epoch+1}] "
                    f"Loss: {train_loss:.4f} | "
                    f"Train Acc: {train_acc:.2f}% | "
                    f"Val Acc: {val_acc:.2f}% "
                    f"(Best: {self.best_acc:.2f}% @ Ep {best_epoch})"
                )
        
        return self.best_acc

    def validate_recon_diagnostics(self, mode='spiking'):
        """Return the NMSE fields from the aligned joint evaluation pass."""
        joint = self.validate_joint_metrics(mode=mode)
        return {
            key: joint[key]
            for key in (
                'Temporal-Mean NMSE (dB)',
                'Full-Sequence NMSE (dB)',
                'Framewise Mean NMSE (dB)',
            )
        }

    def validate_recon(self, mode='spiking'):
        return self.validate_recon_diagnostics(mode=mode)[
            'Temporal-Mean NMSE (dB)'
        ]

    def validate_cls(self, mode='snn', epoch=None, target='recon'):
        utils.deterministic_eval()
        self.model.eval()
        self.sensor.eval()
        
        classifier_module = self._require_downstream_classifier()
        classifier_module.eval()
        
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for x_batch, labels in self.val_loader:
                x_batch = self._preprocess_batch(x_batch)
                x_stack = self._make_classifier_input(x_batch, mode=mode, target=target)
                
                logits = classifier_module(x_stack)
                total_correct += utils.count_correct(logits, labels)
                total_samples += int(labels.size(0))
        
        return 100.0 * total_correct / max(1, total_samples)
        
    def get_predictions(self, mode='snn', target='recon'):
        self.model.eval()
        self.sensor.eval()
        
        classifier_module = self._require_downstream_classifier()
        classifier_module.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for x_batch, labels in self.val_loader:
                x_batch = self._preprocess_batch(x_batch)
                x_stack = self._make_classifier_input(x_batch, mode=mode, target=target)
                logits = classifier_module(x_stack)
                
                _, preds = logits.topk(1, 1, True, True)
                
                all_preds.extend(preds.view(-1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        return all_labels, all_preds

    def collect_z_sequence(self, mode='spiking', max_samples=None):
        """Collect complete latent z sequences for frozen diagnostics."""
        utils.deterministic_eval()
        self.model.eval()
        self.sensor.eval()
        self.channel.eval()

        feature_parts = []
        label_parts = []
        collected = 0
        limit = (
            int(Config.TSNE_MAX_SAMPLES)
            if max_samples is None else int(max_samples)
        )
        with torch.no_grad():
            for x_batch, labels in self.val_loader:
                x_batch = self._preprocess_batch(x_batch)
                z_sequence = self._make_classifier_input(
                    x_batch, mode=mode, target='z'
                )
                remaining = max(0, limit - collected)
                if remaining == 0:
                    break
                take = min(int(z_sequence.shape[0]), remaining)
                feature_parts.append(z_sequence[:take].cpu().numpy())
                label_parts.append(labels[:take].cpu().numpy())
                collected += take
                if collected >= limit:
                    break

        if not feature_parts:
            raise RuntimeError("No latent z samples were collected for t-SNE.")
        return (
            np.concatenate(feature_parts, axis=0),
            np.concatenate(label_parts, axis=0),
        )

    def collect_time_mean_z(self, mode='spiking', max_samples=None):
        """Collect time-averaged latent z for qualitative t-SNE only."""
        z_sequence, labels = self.collect_z_sequence(
            mode=mode, max_samples=max_samples
        )
        return z_sequence.mean(axis=1), labels
        
    def evaluate_recon_statistics(self, mode='spiking', max_batches=3):
        utils.deterministic_eval()
        self.model.eval()
        self.sensor.eval()
        
        stats = {
            'target_mean_count': [],
            'recon_mean': [],
            'recon_min': [],
            'recon_max': [],
            'negative_ratio': [],
            'recon_hard_rate': [],
            'tx_hard_rate': [],
            'tx_prob_mean': [],
            'nmse': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'z_active_rate': [],
        }
        
        batches_processed = 0
        
        thresh_norm = 0.5
        
        with torch.no_grad():
            for x_batch, _ in self.val_loader:
                if batches_processed >= max_batches:
                    break
                    
                x_batch = self._preprocess_batch(x_batch)
                
                B, T, D = x_batch.shape
                
                model_states = None
                sensor_state = None
                channel_state = None
                
                x_rec_seq = []
                tx_hard_seq = []
                tx_prob_seq = []
                z_active_seq = []
                
                for t in range(T):
                    x_frame = x_batch[:, t]
                    
                    x_rec, layers, model_states, sensor_state, channel_state, _, _, aux_dict = \
                        self._process_reconstruction(x_frame, model_states, sensor_state, channel_state, mode=mode)
                    
                    if x_rec is not None:
                        x_rec_seq.append(x_rec)
                    if aux_dict and 'tx_hard' in aux_dict and aux_dict['tx_hard'] is not None:
                        tx_hard_seq.append(aux_dict['tx_hard'])
                    if aux_dict and 'tx_prob' in aux_dict and aux_dict['tx_prob'] is not None:
                        tx_prob_seq.append(aux_dict['tx_prob'])
                    if layers is not None and len(layers) > 0:
                        z_t = layers[-1]
                        z_active = (torch.abs(z_t) > 0.5).float().mean().item()
                        z_active_seq.append(z_active)
                
                if len(x_rec_seq) == T:
                    x_rec_batch = torch.stack(x_rec_seq, dim=1)
                    
                    stats['target_mean_count'].append(x_batch.float().mean().item())
                    stats['recon_mean'].append(x_rec_batch.mean().item())
                    stats['recon_min'].append(x_rec_batch.min().item())
                    stats['recon_max'].append(x_rec_batch.max().item())
                    stats['negative_ratio'].append((x_rec_batch < 0).float().mean().item())
                    stats['recon_hard_rate'].append((x_rec_batch > thresh_norm).float().mean().item())
                    
                    batch_nmse = utils.calculate_nmse_db(x_rec_batch, x_batch)
                    if not np.isnan(batch_nmse):
                        stats['nmse'].append(batch_nmse)
                        
                    p, r, f1 = utils.calculate_spike_prf(x_rec_batch, x_batch, threshold=thresh_norm)
                    stats['precision'].append(p)
                    stats['recall'].append(r)
                    stats['f1'].append(f1)
                    
                if len(tx_hard_seq) == T:
                    stats['tx_hard_rate'].append(
                        torch.stack(tx_hard_seq, dim=1)
                        .abs().mean().item()
                    )
                if len(tx_prob_seq) == T:
                    stats['tx_prob_mean'].append(torch.stack(tx_prob_seq, dim=1).mean().item())
                if len(z_active_seq) == T:
                    stats['z_active_rate'].append(sum(z_active_seq) / T)
                    
                batches_processed += 1
                
        def get_mean(lst):
            return sum(lst) / len(lst) if lst else float('nan')
            
        print("\n" + "="*50)
        print(f" >>> RECONSTRUCTION STATISTICS (Mode: {mode.upper()})")
        print("="*50)
        print(f" Target Mean (Norm): {get_mean(stats['target_mean_count']):.6f}")
        print(f" Recon Mean:         {get_mean(stats['recon_mean']):.6f}")
        print(f" Recon Min:          {get_mean(stats['recon_min']):.6f}")
        print(f" Recon Max:          {get_mean(stats['recon_max']):.6f}")
        print(f" Negative Ratio:     {get_mean(stats['negative_ratio']):.6f}")
        print(f" Recon Hard Rate:    {get_mean(stats['recon_hard_rate']):.6f}")
        print(f" Tx Hard Rate:       {get_mean(stats['tx_hard_rate']):.6f}")
        if stats['tx_prob_mean']:
            print(f" Tx Prob Mean:       {get_mean(stats['tx_prob_mean']):.6f}")
        if stats['z_active_rate']:
            print(
                f" Z Large Rate (|z|>0.5): "
                f"{get_mean(stats['z_active_rate']):.6f}"
            )
        print(f" NMSE (dB):          {get_mean(stats['nmse']):.4f}")
        print(
            f" Event Activity Precision "
            f"(thr_norm={thresh_norm:.4f}): "
            f"{get_mean(stats['precision']):.4f}"
        )
        print(
            f" Event Activity Recall "
            f"(thr_norm={thresh_norm:.4f}):    "
            f"{get_mean(stats['recall']):.4f}"
        )
        print(
            f" Event Activity F1 "
            f"(thr_norm={thresh_norm:.4f}):        "
            f"{get_mean(stats['f1']):.4f}"
        )
        print("="*50 + "\n")

        return {key: get_mean(values) for key, values in stats.items()}
