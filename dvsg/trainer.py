import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
import random
import torchvision.transforms.functional as TF
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
        self.criterion_cls = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        self.best_nmse = 1000.0
        self.best_acc = 0.0

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

    def _unfreeze_all_params(self):
        for param in self.model.parameters():
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
                    bits_per_sample = Config.LATENT_DIM * T
                    total_bits += bits_per_sample * B
                    total_samples += B
                else:
                    batch_events = 0
                    prev_tx_hard = None
                    
                    for t in range(T):
                        x_frame = x_batch[:, t]
                        _, sensor_state, aux_dict = self.sensor(x_frame, state=sensor_state, mode=mode)
                        tx_hard = aux_dict['tx_hard']
                        batch_events += tx_hard.sum().item()
                        
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
                    total_bits += batch_events * Config.RATE_BITS_PER_EVENT
                    total_samples += B
        
        avg_firing_rate = 0.0            
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
                
        return total_bits / max(1, total_samples), avg_firing_rate

    def train_dynamic_recon(self, hp=None, model_name="model", mode='spiking', freeze_sensor=False):
        print(f"\n>>> DYNAMIC RECONSTRUCTION TRAINING ({model_name} | Mode: {mode.upper()})")
        self._unfreeze_all_params()
        
        if freeze_sensor:
            print(" -> [Info] EdgeSensor is FROZEN for this phase to isolate LISTA pre-training.")
            for param in self.sensor.parameters():
                param.requires_grad = False
                
        if hp is None: hp = Config.PHASE3_CONFIG
        
        self.sensor.mode = mode
        if hasattr(self.model, 'mode'):
            self.model.mode = 'rate' if mode == 'ann' else 'spiking'

        threshold_suffix = ""
        if hasattr(self.model, 'get_threshold_value'):
            th_val = self.model.get_threshold_value()
            threshold_suffix = f"_th{th_val:.2f}"
            
        rho_suffix = f"_rho{Config.TX_RHO}"
        save_path = os.path.join(Config.RESULTS_DIR, f"dynamic_recon_{mode}_{model_name.lower()}{rho_suffix}{threshold_suffix}.pth")
        
        param_groups = self.model.get_optimizer_groups(hp)
        
        if not freeze_sensor:
            lr_sensor = hp.get('lr_sensor', hp.get('lr_lista_snn', 7e-4))
            sensor_params = [p for p in self.sensor.parameters() if p.requires_grad]
            if len(sensor_params) > 0:
                param_groups.append({'params': sensor_params, 'lr': lr_sensor})
        
        param_groups = [g for g in param_groups if len(list(g['params'])) > 0]
        
        optimizer = optim.Adam(param_groups, weight_decay=hp['weight_decay'])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hp['epochs'], eta_min=hp['lr_min'])
        
        self.best_nmse = 1000.0 
        
        for epoch in range(hp['epochs']):
            current_tau = Config.get_tau(epoch, hp['epochs'])
            
            self.model.train()
            if freeze_sensor:
                self.sensor.eval()
            else:
                self.sensor.train()
            
            ep_loss = 0
            epoch_activity_accum = 0.0 
            total_steps = 0
            
            epoch_tx_stats = {
                'total': 0, 'assisted': 0, 'cf': 0, 'state_contrib_sum': 0.0, 'elements': 0, 'threshold': 0.2
            }
            
            for x_batch, _ in self.train_loader:
                x_batch = x_batch.to(self.device)
                B, T, C, H, W = x_batch.shape
                
                model_states = None
                sensor_state = None
                channel_state = None
                loss_total_step = 0
                
                for t in range(T):
                    x_frame = x_batch[:, t]
                    
                    x_rec, layers, model_states, sensor_state, channel_state, y_spikes, stats, aux_dict = \
                        self._process_reconstruction(x_frame, model_states, sensor_state, channel_state, mode=mode, tau=current_tau)
                    
                    if mode == 'spiking' and Config.LOG_TX_STATE_STATS and aux_dict:
                        epoch_tx_stats['total'] += aux_dict.get('tx_total_spikes', 0)
                        epoch_tx_stats['assisted'] += aux_dict.get('tx_assisted_spikes', 0)
                        epoch_tx_stats['cf'] += aux_dict.get('tx_counterfactual_reset_spikes', 0)
                        epoch_tx_stats['state_contrib_sum'] += aux_dict.get('tx_state_contribution_sum', 0.0)
                        epoch_tx_stats['elements'] += aux_dict.get('tx_elements', 0)
                        epoch_tx_stats['threshold'] = aux_dict.get('tx_threshold', 0.2)
                    
                    step_mse_loss = 0
                    if x_rec is not None:
                        step_mse_loss = Config.ALPHA_MSE * self.criterion_mse(x_rec, x_frame)
                    
                    step_sparsity_loss = 0
                    if len(layers) > 0:
                        current_activity = 0
                        for l in layers: current_activity += l.abs().mean()
                        current_activity /= len(layers)
                        epoch_activity_accum += current_activity.item()
                        total_steps += 1
                        
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
                    
                    step_rate_loss = 0
                    if mode == 'spiking':
                        tx_prob = aux_dict.get('tx_prob')
                        tx_hard = aux_dict.get('tx_hard')
                        
                        if self.model.__class__.__name__ == 'SpikingCNNRecon':
                            alpha_rate = Config.ALPHA_RATE_SCNN
                        else:
                            alpha_rate = Config.ALPHA_RATE_LISTA
                            
                        if tx_prob is not None:
                            step_rate_loss = alpha_rate * Config.RATE_BITS_PER_EVENT * torch.mean(tx_prob)
                        elif tx_hard is not None:
                            step_rate_loss = alpha_rate * Config.RATE_BITS_PER_EVENT * torch.mean(tx_hard)

                    loss_total_step += (step_mse_loss + step_sparsity_loss + step_rate_loss)

                loss_final = loss_total_step / T
                
                optimizer.zero_grad()
                loss_final.backward()
                
                if epoch == 0 and total_steps == 1 and mode == 'spiking':
                    grad_norm = 0.0
                    for param in self.sensor.parameters():
                        if param.requires_grad and param.grad is not None:
                            grad_norm += param.grad.norm().item()
                    
                    ablation = Config.TX_ABLATION_MODE
                    relaxed = getattr(Config, 'USE_RELAXED_AER_BSC', False)
                    print(f"\n[DEBUG] Backward Pass - Mode: {mode}, Ablation: {ablation}, Relaxed AER: {relaxed}")
                    print(f"[DEBUG] Sensor Grad Norm: {grad_norm:.6f} -> {'VALID' if grad_norm > 0 else 'WARNING: ZERO GRADIENT'}\n")
                
                all_params = []
                for g in param_groups:
                    all_params.extend(g['params'])
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                
                optimizer.step()
                ep_loss += loss_final.item()
                    
            scheduler.step()
            
            val_nmse = self.validate_recon(mode=mode)
            
            if val_nmse < self.best_nmse:
                self.best_nmse = val_nmse
                torch.save({
                    'model': self.model.state_dict(), 
                    'sensor': self.sensor.state_dict()
                }, save_path)
            
            if (epoch+1) % 5 == 0 or epoch == 0:
                avg_activity = epoch_activity_accum / total_steps if total_steps > 0 else 0.0
                print(f"[Ep {epoch+1}] Loss: {ep_loss/len(self.train_loader):.4f} | Activity: {avg_activity:.4f} | Val NMSE: {val_nmse:.2f} dB (Best: {self.best_nmse:.2f} dB) | Tau: {current_tau:.4f}")
                
                if mode == 'spiking' and Config.LOG_TX_STATE_STATS and epoch_tx_stats['total'] > 0:
                    assist_ratio = epoch_tx_stats['assisted'] / epoch_tx_stats['total']
                    state_contribution = epoch_tx_stats['state_contrib_sum'] / (epoch_tx_stats['threshold'] * epoch_tx_stats['total'])
                    cf_rate = epoch_tx_stats['cf'] / max(1, epoch_tx_stats['elements'])
                    actual_tx_rate = epoch_tx_stats['total'] / max(1, epoch_tx_stats['elements'])
                    print(f" -> TX Diagnostics: Assist Ratio {assist_ratio*100:.1f}% | State Contrib {state_contribution*100:.1f}% | CF Rate {cf_rate*100:.2f}% | Actual Rate {actual_tx_rate*100:.2f}%")
        
        return self.best_nmse

    def train_phase4(self, hp=None, model_name="model", mode='snn', cls_type='latent'):
        print(f"\n>>> PHASE 4: Task-Oriented Classification ({mode.upper()} | Input: {cls_type.upper()})")
        if hp is None: hp = Config.PHASE4_CONFIG
        
        threshold_suffix = ""
        if hasattr(self.model, 'get_threshold_value'):
            th_val = self.model.get_threshold_value()
            threshold_suffix = f"_th{th_val:.2f}"

        rho_suffix = f"_rho{Config.TX_RHO}"
        cls_input_suffix = f"_{cls_type}"
        save_path = os.path.join(Config.RESULTS_DIR, f"phase4_cls_{mode}_{model_name.lower()}{rho_suffix}{threshold_suffix}{cls_input_suffix}.pth")
        
        if cls_type == 'latent':
            classifier_module = getattr(self.model, 'classifier_latent', None)
        else:
            classifier_module = getattr(self.model, 'classifier_image', None)

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
        
        for epoch in range(hp['epochs']):
            classifier_module.train()
            ep_loss = 0
            ep_acc = 0
            
            for x_batch, labels in self.train_loader:
                x_batch = x_batch.to(self.device)
                labels = labels.to(self.device)
                B, T, C, H, W = x_batch.shape
                
                optimizer.zero_grad()
                input_seq = []
                
                model_states = None
                sensor_state = None
                channel_state = None
                
                with torch.no_grad():
                    for t in range(T):
                        x_frame = x_batch[:, t]
                        x_rec, layers, model_states, sensor_state, channel_state, _, _, _ = \
                            self._process_reconstruction(x_frame, model_states, sensor_state, channel_state, mode=mode)
                        
                        if cls_type == 'latent' and len(layers) > 0:
                            item = layers[-1]
                        else:
                            item = x_rec if x_rec is not None else x_frame
                        input_seq.append(item)
                
                x_stack = torch.stack(input_seq, dim=1)
                
                B_s, T_s, C_s, H_s, W_s = x_stack.shape
                x_stack_aug = torch.zeros_like(x_stack)
                max_dx = int(0.1 * W_s)
                max_dy = int(0.1 * H_s)
                
                for b in range(B_s):
                    dx = int(random.uniform(-max_dx, max_dx))
                    dy = int(random.uniform(-max_dy, max_dy))
                    
                    if dx == 0 and dy == 0:
                        x_stack_aug[b] = x_stack[b]
                        continue
                        
                    src_y1 = 0 if dy >= 0 else -dy
                    src_y2 = H_s - dy if dy >= 0 else H_s
                    dst_y1 = dy if dy >= 0 else 0
                    dst_y2 = H_s if dy >= 0 else H_s + dy
                    
                    src_x1 = 0 if dx >= 0 else -dx
                    src_x2 = W_s - dx if dx >= 0 else W_s
                    dst_x1 = dx if dx >= 0 else 0
                    dst_x2 = W_s if dx >= 0 else W_s + dx
                    
                    x_stack_aug[b, :, :, dst_y1:dst_y2, dst_x1:dst_x2] = x_stack[b, :, :, src_y1:src_y2, src_x1:src_x2]
                    
                x_stack = x_stack_aug
                
                logits = classifier_module(x_stack)
                loss = self.criterion_cls(logits, labels)
                
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(classifier_module.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                ep_loss += loss.item()
                ep_acc += utils.calculate_accuracy(logits, labels)
                
            scheduler.step()
            
            val_acc = self.validate_cls(epoch_idx=epoch, mode=mode, cls_type=cls_type)
            
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                torch.save({
                    'model': self.model.state_dict(), 
                    'sensor': self.sensor.state_dict()
                }, save_path)
            
            if (epoch+1) % 5 == 0 or epoch == 0:
                print(f"[{mode.upper()} CLS {cls_type.upper()} Ep {epoch+1}] Loss: {ep_loss/len(self.train_loader):.4f} | Train Acc: {ep_acc/len(self.train_loader):.2f}% | Val Acc: {val_acc:.2f}% (Best: {self.best_acc:.2f}%)")
        
        return self.best_acc

    def validate_recon(self, mode='spiking', nmse_mode='temporal_mean'):
        """
        Formal validation NMSE.

        Default:
            global temporal-mean NMSE:
                10 log10( sum_i ||mean_t x_i,t - mean_t xhat_i,t||^2
                          / sum_i ||mean_t x_i,t||^2 + eps )

        Optional:
            nmse_mode='full_sequence':
                10 log10( sum_{i,t} ||x_i,t - xhat_i,t||^2
                          / sum_{i,t} ||x_i,t||^2 + eps )

        This is not batch-level dB averaging.
        """
        torch.manual_seed(42)
        np.random.seed(42)
        self.model.eval()
        self.sensor.eval()

        total_error_energy = 0.0
        total_target_energy = 0.0
        valid_batches = 0

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
                    return float('nan')

                x_rec_batch = torch.stack(x_rec_seq, dim=1)

                if nmse_mode == 'temporal_mean':
                    x_target_eval = x_batch.mean(dim=1)
                    x_rec_eval = x_rec_batch.mean(dim=1)
                elif nmse_mode == 'full_sequence':
                    x_target_eval = x_batch
                    x_rec_eval = x_rec_batch
                else:
                    raise ValueError(
                        f"Unknown nmse_mode={nmse_mode}. "
                        "Use 'temporal_mean' or 'full_sequence'."
                    )

                total_error_energy += torch.sum((x_target_eval - x_rec_eval) ** 2).item()
                total_target_energy += torch.sum(x_target_eval ** 2).item()
                valid_batches += 1

        if valid_batches == 0 or total_target_energy <= 0:
            return float('nan')

        nmse_linear = total_error_energy / max(total_target_energy, eps_den)
        nmse_db = 10.0 * np.log10(nmse_linear + eps_log)

        return float(nmse_db)

    def validate_cls(self, epoch_idx=0, mode='snn', cls_type='latent'):
        torch.manual_seed(42)
        np.random.seed(42)
        self.model.eval()
        self.sensor.eval()
        
        if cls_type == 'latent':
            classifier_module = getattr(self.model, 'classifier_latent', None)
        else:
            classifier_module = getattr(self.model, 'classifier_image', None)
            
        classifier_module.eval()
        
        total_acc = 0
        
        with torch.no_grad():
            for x_batch, labels in self.val_loader:
                x_batch = x_batch.to(self.device)
                labels = labels.to(self.device)
                B, T, C, H, W = x_batch.shape
                
                input_seq = []
                model_states = None
                sensor_state = None
                channel_state = None
                
                for t in range(T):
                    x_frame = x_batch[:, t]
                    x_rec, layers, model_states, sensor_state, channel_state, _, _, _ = \
                        self._process_reconstruction(x_frame, model_states, sensor_state, channel_state, mode=mode)
                    
                    if cls_type == 'latent' and len(layers) > 0:
                        item = layers[-1]
                    else:
                        item = x_rec if x_rec is not None else x_frame
                    input_seq.append(item)
                
                x_stack = torch.stack(input_seq, dim=1)
                logits = classifier_module(x_stack)
                
                total_acc += utils.calculate_accuracy(logits, labels)
        
        return total_acc / len(self.val_loader)

    def get_predictions(self, mode='snn', cls_type='latent'):
        self.model.eval()
        self.sensor.eval()
        
        if cls_type == 'latent':
            classifier_module = getattr(self.model, 'classifier_latent', None)
        else:
            classifier_module = getattr(self.model, 'classifier_image', None)
            
        classifier_module.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for x_batch, labels in self.val_loader:
                x_batch = x_batch.to(self.device)
                labels = labels.to(self.device)
                B, T, C, H, W = x_batch.shape
                
                input_seq = []
                model_states = None
                sensor_state = None
                channel_state = None
                
                for t in range(T):
                    x_frame = x_batch[:, t]
                    x_rec, layers, model_states, sensor_state, channel_state, _, _, _ = \
                        self._process_reconstruction(x_frame, model_states, sensor_state, channel_state, mode=mode)
                    
                    if cls_type == 'latent' and len(layers) > 0:
                        item = layers[-1]
                    else:
                        item = x_rec if x_rec is not None else x_frame
                    input_seq.append(item)
                
                x_stack = torch.stack(input_seq, dim=1)
                logits = classifier_module(x_stack)
                
                _, preds = logits.topk(1, 1, True, True)
                
                all_preds.extend(preds.view(-1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        return all_labels, all_preds