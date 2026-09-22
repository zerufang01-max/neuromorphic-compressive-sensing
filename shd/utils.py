import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from config import Config
from sklearn.metrics import confusion_matrix
import seaborn as sns
import torch.utils.data as data
import random
import torch.nn as nn


def count_reconstruction_parameters(model, sensor, mode=None):
    """
    Count the paper-facing scope consistently: measurement network plus the
    active reconstruction branch, excluding the downstream classifier and
    inactive ANN/SNN branches stored in a shared HybridLISTA checkpoint.
    """
    active_mode = mode or getattr(model, 'mode', 'spiking')
    is_ann = active_mode in ('ann', 'rate')
    model_class = model.__class__.__name__
    inactive_prefixes = ()
    if model_class == 'HybridLISTA':
        if is_ann:
            inactive_prefixes = ('P_snn.', 'PD_snn_k.', 'S_k.0.')
        else:
            inactive_prefixes = ('W_e.', 'S_k.', 'theta_ann.', 'PD_snn_k.0.')

    measurement_params = sum(p.numel() for p in sensor.parameters())
    reconstruction_params = 0
    classifier_params = 0
    semantic_head_params = 0
    inactive_params = 0
    for name, param in model.named_parameters():
        if name.startswith('classifier.'):
            classifier_params += param.numel()
        elif name.startswith('semantic_head.'):
            semantic_head_params += param.numel()
        elif name.startswith(inactive_prefixes):
            inactive_params += param.numel()
        else:
            reconstruction_params += param.numel()

    full_model_params = sum(p.numel() for p in model.parameters())
    if (
        reconstruction_params + classifier_params
        + semantic_head_params + inactive_params
        != full_model_params
    ):
        raise RuntimeError('Parameter audit partitions do not sum to the full model.')

    return {
        'Measurement Params': int(measurement_params),
        'Reconstruction Params': int(reconstruction_params),
        'Reported Total Params': int(measurement_params + reconstruction_params),
        'Excluded Classifier Params': int(classifier_params),
        'Semantic Head Params': int(semantic_head_params),
        'Task System Params': int(
            measurement_params + reconstruction_params
            + semantic_head_params
        ),
        'Inactive Branch Params': int(inactive_params),
        'Full Model Params': int(full_model_params),
    }

class SHDDataset(data.Dataset):
    def __init__(self, split='train'):
        self.split = split
        
        if split == 'train':
            file_name = getattr(Config, 'TRAIN_FILE', f"shd_train_T{Config.TIME_STEPS}_D{Config.INPUT_DIM}_rawcount.pt")
        else:
            file_name = getattr(Config, 'TEST_FILE', f"shd_test_T{Config.TIME_STEPS}_D{Config.INPUT_DIM}_rawcount.pt")
            
        path = os.path.join(Config.PROCESSED_DIR, file_name)
        
        print(f" -> Loading SHD {split} data from: {path}")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset file not found at {path}.")
            
        dataset_dict = torch.load(path, weights_only=False)
        
        self.x = dataset_dict["x"].float()  
        self.y = dataset_dict["y"].long()   
        
        print(f" -> Moving {split} raw-count dataset to {Config.DEVICE}...")

        self.x = self.x.to(Config.DEVICE)
        self.y = self.y.to(Config.DEVICE)
        
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, index):
        return self.x[index], self.y[index]

def get_dataloaders(batch_size):
    train_set = SHDDataset(split='train')
    val_set = SHDDataset(split='test')
    
    train_loader = data.DataLoader(
        train_set, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,     
        pin_memory=False   
    )
    
    val_loader = data.DataLoader(
        val_set, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,     
        pin_memory=False   
    )
    
    return train_loader, val_loader
    
def deterministic_eval(seed=None):
    if seed is None:
        seed = getattr(Config, 'EVAL_CHANNEL_SEED', getattr(Config, 'SEED', 42))
    seed = int(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state():
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch_cpu': torch.get_rng_state(),
        'torch_cuda': (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available() else None
        ),
    }


def restore_rng_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'])
    if torch.cuda.is_available() and state['torch_cuda'] is not None:
        torch.cuda.set_rng_state_all(state['torch_cuda'])


def calculate_accuracy(output, target):
    with torch.no_grad():
        batch_size = target.size(0)
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_k = correct[:1].reshape(-1).float().sum(0, keepdim=True)
        return correct_k.mul_(100.0 / batch_size).item()


def count_correct(output, target):
    """Return integer correct predictions for sample-weighted aggregation."""
    with torch.no_grad():
        return int(output.argmax(dim=1).eq(target).sum().item())


def reinitialize_classifier(module, scheme="orthogonal"):
    """Reset a standalone downstream classifier."""
    scheme = str(scheme).lower()
    if scheme not in {"orthogonal", "default"}:
        raise ValueError(
            f"Unknown classifier initialization {scheme!r}; "
            "expected 'orthogonal' or 'default'."
        )

    def reset(child):
        if hasattr(child, "reset_parameters"):
            child.reset_parameters()

    module.apply(reset)
    if scheme == "orthogonal":
        for child in module.modules():
            if isinstance(child, nn.Linear):
                nn.init.orthogonal_(child.weight, gain=0.5)
                if child.bias is not None:
                    nn.init.zeros_(child.bias)

def calculate_nmse_db(x_rec, x_target):
    error_energy = torch.sum((x_target - x_rec)**2).item()
    target_energy = torch.sum(x_target**2).item()
    if target_energy > 0:
        return 10 * np.log10((error_energy / target_energy) + 1e-10)
    return float('nan')
    
def calculate_spike_prf(x_rec, x_target, threshold=0.5):
    pred = (x_rec > threshold)
    target = (x_target > threshold)
    
    tp = (pred & target).sum().float()
    fp = (pred & ~target).sum().float()
    fn = (~pred & target).sum().float()
    
    eps = 1e-10
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * (precision * recall) / (precision + recall + eps)
    
    return precision.item(), recall.item(), f1.item()
    
def save_confusion_matrix(y_true, y_pred, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(Config.RESULTS_DIR, filename))
    plt.close()

def save_reconstruction_plot(model, sensor, channel, val_loader, filename, num_samples=3, mode='spiking'):
    model.eval()
    sensor.eval()
    
    x_target_seq = []
    x_recon_seq = []
    
    with torch.no_grad():
        for x_batch, _ in val_loader:
            x_batch = x_batch.to(Config.DEVICE)
            
            B, T, D = x_batch.shape
            
            model_states = None
            sensor_state = None
            channel_state = None
            
            batch_rec_seq = []
            for t in range(T):
                x_frame = x_batch[:, t]
                
                y_encoded, sensor_state, _ = sensor(x_frame, state=sensor_state, mode=mode)
                y_noisy, channel_state = channel(y_encoded, state=channel_state, mode=mode)
                
                prev_mode = getattr(model, 'mode', None)
                if hasattr(model, 'mode'): 
                    model.mode = 'rate' if mode == 'ann' else 'spiking'
                    
                x_rec_t, _, model_states, _ = model(y_noisy, states=model_states)
                
                if hasattr(model, 'mode') and prev_mode is not None: 
                    model.mode = prev_mode
                
                batch_rec_seq.append(x_rec_t if x_rec_t is not None else torch.zeros_like(x_frame))
            
            x_target_seq.append(x_batch.cpu())
            x_recon_seq.append(torch.stack(batch_rec_seq, dim=1).cpu())
            
            if sum(b.shape[0] for b in x_target_seq) >= num_samples:
                break
                
    x_target_all = torch.cat(x_target_seq, dim=0)[:num_samples]
    x_recon_all = torch.cat(x_recon_seq, dim=0)[:num_samples]
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 4 * num_samples))
    if num_samples == 1: 
        axes = np.expand_dims(axes, 0)
    
    fig.suptitle(f"T={Config.TIME_STEPS} | Mode={mode} | Raw Count", fontsize=14)
    
    for i in range(num_samples):
        ax1 = axes[i, 0]
        im1 = ax1.imshow(x_target_all[i].numpy().T, aspect='auto', cmap='inferno', origin='lower')
        ax1.set_title(f'Sample {i+1} - Original Target')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel(f'Features ({x_target_all.shape[-1]})')
        fig.colorbar(im1, ax=ax1)
        
        ax2 = axes[i, 1]
        im2 = ax2.imshow(x_recon_all[i].numpy().T, aspect='auto', cmap='inferno', origin='lower')
        ax2.set_title(f'Sample {i+1} - Reconstruction')
        ax2.set_xlabel('Time Step')
        fig.colorbar(im2, ax=ax2)
        
    plt.tight_layout()
    plt.savefig(os.path.join(Config.RESULTS_DIR, filename))
    plt.close()

def analyze_and_save_statistics(model, sensor, channel, val_loader, filename):
    model.eval()
    sensor.eval()
    
    stats = {
        'input_mean': [], 'input_std': [],
        'meas_mean': [], 'meas_std': [], 'meas_min': [], 'meas_max': [],
        'channel_mean': [], 'channel_std': [],
        'recon_mean': [], 'recon_std': [],
        'layer_sparsity': {} 
    }
    
    total_spikes = 0
    total_neurons = 0
    mode = getattr(sensor, 'mode', 'spiking')
    
    with torch.no_grad():
        for i, (x_batch, _) in enumerate(val_loader):
            if i >= 10: break 
            
            x_batch = x_batch.to(Config.DEVICE)
            
            B, T, D = x_batch.shape
            recon_states = None
            sensor_state = None
            channel_state = None
            
            for t in range(T):
                x_frame = x_batch[:, t]
                stats['input_mean'].append(x_frame.mean().item())
                stats['input_std'].append(x_frame.std().item())
                
                y_out, sensor_state, _ = sensor(x_frame, state=sensor_state, mode=mode)
                
                stats['meas_mean'].append(y_out.mean().item())
                stats['meas_std'].append(y_out.std().item())
                stats['meas_min'].append(y_out.min().item())
                stats['meas_max'].append(y_out.max().item())
                
                y_noisy, channel_state = channel(y_out, state=channel_state, mode=mode)
                stats['channel_mean'].append(y_noisy.mean().item())
                stats['channel_std'].append(y_noisy.std().item())
                
                temp_prev_mode = getattr(model, 'mode', None)
                if hasattr(model, 'mode'): 
                    model.mode = 'rate' if mode == 'ann' else 'spiking'

                x_rec, layers, recon_states, _ = model(y_noisy, states=recon_states)
                
                if hasattr(model, 'mode') and temp_prev_mode is not None:
                     model.mode = temp_prev_mode

                if x_rec is not None:
                    stats['recon_mean'].append(x_rec.mean().item())
                    stats['recon_std'].append(x_rec.std().item())
                
                if layers is not None:
                    for k, layer_z in enumerate(layers):
                        active_elements = (torch.abs(layer_z) > 1e-5).float()
                        sparsity = active_elements.mean().item()
                        if k not in stats['layer_sparsity']: stats['layer_sparsity'][k] = []
                        stats['layer_sparsity'][k].append(sparsity)
                        
                        if mode == 'spiking':
                            total_spikes += active_elements.sum().item()
                            total_neurons += layer_z.numel()

    report_path = os.path.join(Config.RESULTS_DIR, filename)
    with open(report_path, 'w') as f:
        f.write(f" SYSTEM STATISTICS REPORT ({mode.upper()})\n")
        f.write(f"--- 1. Input Data (Raw Count) ---\n")
        f.write(f"Mean: {np.mean(stats['input_mean']):.4f}\n")
        f.write(f"Std : {np.mean(stats['input_std']):.4f}\n\n")
        f.write(f"--- 2. Edge Compression ---\n")
        f.write(f"Mean: {np.mean(stats['meas_mean']):.4f}\n")
        f.write(f"Std : {np.mean(stats['meas_std']):.4f}\n\n")
        f.write(f"--- 3. Model Sparsity ---\n")
        for k in sorted(stats['layer_sparsity'].keys()):
            f.write(f"Layer {k+1}: {np.mean(stats['layer_sparsity'][k]) * 100:.2f}%\n")
        if mode == 'spiking':
            avg_fr = total_spikes / max(1, total_neurons)
            f.write(f"\nAvg LISTA Firing Rate: {avg_fr * 100:.2f}%\n")
        f.write(f"\n--- 4. Reconstruction ---\n")
        if stats['recon_mean']:
            f.write(f"Mean: {np.mean(stats['recon_mean']):.4f}\n")
            f.write(f"Std : {np.mean(stats['recon_std']):.4f}\n")

def save_feature_z_plot(model, sensor, channel, val_loader, filename, num_samples=3, mode='spiking'):
    model.eval()
    sensor.eval()
    
    x_target_seq = []
    z_seq = []
    
    with torch.no_grad():
        for x_batch, _ in val_loader:
            x_batch = x_batch.to(Config.DEVICE)
            B, T, D = x_batch.shape
            
            model_states = None
            sensor_state = None
            channel_state = None
            
            batch_z_seq = []
            for t in range(T):
                x_frame = x_batch[:, t]
                
                y_encoded, sensor_state, _ = sensor(x_frame, state=sensor_state, mode=mode)
                y_noisy, channel_state = channel(y_encoded, state=channel_state, mode=mode)
                
                prev_mode = getattr(model, 'mode', None)
                if hasattr(model, 'mode'): 
                    model.mode = 'rate' if mode == 'ann' else 'spiking'
                    
                _, layers, model_states, _ = model(y_noisy, states=model_states)
                
                if hasattr(model, 'mode') and prev_mode is not None: 
                    model.mode = prev_mode
                
                z_t = layers[-1] if (layers is not None and len(layers) > 0) else torch.zeros_like(x_frame)
                batch_z_seq.append(z_t)
            
            x_target_seq.append(x_batch.cpu())
            z_seq.append(torch.stack(batch_z_seq, dim=1).cpu())
            
            if sum(b.shape[0] for b in x_target_seq) >= num_samples:
                break
                
    x_target_all = torch.cat(x_target_seq, dim=0)[:num_samples]
    z_all = torch.cat(z_seq, dim=0)[:num_samples]
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 4 * num_samples))
    if num_samples == 1: 
        axes = np.expand_dims(axes, 0)
    
    fig.suptitle(f"Feature Z Visualization | Mode={mode} | T={Config.TIME_STEPS}", fontsize=14)
    
    for i in range(num_samples):
        ax1 = axes[i, 0]
        im1 = ax1.imshow(x_target_all[i].numpy().T, aspect='auto', cmap='inferno', origin='lower')
        ax1.set_title(f'Sample {i+1} - Original Target')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel(f'Features ({x_target_all.shape[-1]})')
        fig.colorbar(im1, ax=ax1)
        
        ax2 = axes[i, 1]
        im2 = ax2.imshow(z_all[i].numpy().T, aspect='auto', cmap='viridis', origin='lower')
        ax2.set_title(f'Sample {i+1} - Feature Z (Latent)')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel(f'Z Dim ({z_all.shape[-1]})')
        fig.colorbar(im2, ax=ax2)
        
    plt.tight_layout()
    plt.savefig(os.path.join(Config.RESULTS_DIR, filename))
    plt.close()
    
