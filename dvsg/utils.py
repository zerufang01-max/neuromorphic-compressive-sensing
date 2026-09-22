import torch
import torch.utils.data as data
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import torch.nn.functional as F
from torchvision import transforms 
import torchvision.transforms.functional as TF 
from config import Config
from sklearn.metrics import confusion_matrix
import seaborn as sns
import random
from PIL import Image
import io
import math

def calculate_accuracy(output, target):
    with torch.no_grad():
        batch_size = target.size(0)
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_k = correct[:1].reshape(-1).float().sum(0, keepdim=True)
        return correct_k.mul_(100.0 / batch_size).item()

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

def analyze_and_save_statistics(model, sensor, channel, val_loader, filename):
    model.eval()
    stats = {
        'input_mean': [], 'input_std': [],
        'meas_mean': [], 'meas_std': [], 'meas_min': [], 'meas_max': [],
        'channel_mean': [], 'channel_std': [],
        'recon_mean': [], 'recon_std': [],
        'layer_sparsity': {} 
    }
    total_spikes = 0
    total_neurons = 0
    is_snn = 'snn' in filename.lower() or (hasattr(model, 'mode') and model.mode == 'spiking')
    mode = 'spiking' if is_snn else 'ann'
    
    with torch.no_grad():
        for i, (x_batch, _) in enumerate(val_loader):
            if i >= 10: break 
            x_batch = x_batch.to(Config.DEVICE)
            B, T, C, H, W = x_batch.shape
            recon_states = None
            sensor_state = None
            channel_state = None
            
            if not is_snn:
                x_gt = x_batch.mean(dim=1)
                y_quant, _, _ = sensor(x_gt, mode='ann')
                stats['meas_mean'].append(y_quant.mean().item())
                
            for t in range(T):
                x_frame = x_batch[:, t]
                stats['input_mean'].append(x_frame.mean().item())
                stats['input_std'].append(x_frame.std().item())
                
                y_out, sensor_state, aux_dict = sensor(x_frame, state=sensor_state, mode=mode)
                
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
        f.write(f"--- 1. Input Data ---\n")
        f.write(f"Mean: {np.mean(stats['input_mean']):.4f}\n")
        f.write(f"Std : {np.mean(stats['input_std']):.4f}\n\n")
        f.write(f"--- 2. Edge Compression ---\n")
        f.write(f"Mean: {np.mean(stats['meas_mean']):.4f}\n")
        f.write(f"Std : {np.mean(stats['meas_std']):.4f}\n\n")
        f.write(f"--- 4. Model Sparsity ---\n")
        for k in sorted(stats['layer_sparsity'].keys()):
            f.write(f"Layer {k+1}: {np.mean(stats['layer_sparsity'][k]) * 100:.2f}%\n")
        if mode == 'spiking':
            avg_fr = total_spikes / max(1, total_neurons)
            f.write(f"\nAvg Firing Rate: {avg_fr:.4f}\n")
        f.write(f"\n--- 5. Reconstruction ---\n")
        if stats['recon_mean']:
            f.write(f"Mean: {np.mean(stats['recon_mean']):.4f}\n")
            f.write(f"Std : {np.mean(stats['recon_std']):.4f}\n")

def get_class_samples(loader, device):
    samples = {}
    with torch.no_grad():
        for x_batch, labels in loader:
            for i in range(len(labels)):
                lbl = labels[i].item()
                if lbl not in samples:
                    samples[lbl] = x_batch[i].to(device)
                if len(samples) >= Config.NUM_CLASSES:
                    break
            if len(samples) >= Config.NUM_CLASSES:
                break
    sorted_keys = sorted(samples.keys())
    if not sorted_keys: return None
    return torch.stack([samples[k] for k in sorted_keys])

def save_ann_visualization(model, sensor, channel, loader, filename):
    samples = get_class_samples(loader, Config.DEVICE)
    if samples is None: return

    num_classes = samples.shape[0]
    model.eval()
    sensor.eval()
    
    x_gt_mean = samples.mean(dim=1) 
    
    with torch.no_grad():
        y_quant, _, _ = sensor(x_gt_mean, mode='ann')
        y_noisy, _ = channel(y_quant, state=None, mode='ann')
        
        prev_mode = getattr(model, 'mode', None)
        if hasattr(model, 'mode'): model.mode = 'rate'
        x_rec, _, _, _ = model(y_noisy)
        if hasattr(model, 'mode') and prev_mode is not None: model.mode = prev_mode
    
    if x_rec is None: return 

    fig, axes = plt.subplots(2, num_classes, figsize=(num_classes * 2.5, 5))
    if num_classes == 1: axes = np.expand_dims(axes, axis=1)

    for i in range(num_classes):
        gt_img = x_gt_mean[i].sum(dim=0).cpu().numpy()
        axes[0, i].imshow(gt_img, cmap='gray', vmin=0, vmax=1)
        axes[0, i].axis('off')
        
        rec_img = x_rec[i].sum(dim=0).cpu().numpy()
        axes[1, i].imshow(rec_img, cmap='gray', vmin=0, vmax=1)
        axes[1, i].axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(Config.RESULTS_DIR, filename))
    plt.close()

def save_snn_visualization_gif(model, sensor, channel, loader, filename):
    samples = get_class_samples(loader, Config.DEVICE)
    if samples is None: return

    num_classes = samples.shape[0]
    T = samples.shape[1]
    
    model.eval()
    sensor.eval()
    
    recon_frames = [] 
    gt_frames = []    
    
    recon_state = None
    sensor_state = None
    channel_state = None
    
    mode = 'ann' if sensor.mode == 'ann' else 'spiking'

    with torch.no_grad():
        for t in range(T):
            x_t = samples[:, t] 
            
            y_quant, sensor_state, _ = sensor(x_t, state=sensor_state, mode=mode)
            y_noisy, channel_state = channel(y_quant, state=channel_state, mode=mode)
            
            prev_mode = getattr(model, 'mode', None)
            if hasattr(model, 'mode'): model.mode = 'rate' if mode == 'ann' else 'spiking'

            x_rec_t, _, recon_state, _ = model(y_noisy, states=recon_state)
            
            if hasattr(model, 'mode') and prev_mode is not None: model.mode = prev_mode
            
            gt_frames.append(x_t.sum(dim=1).cpu())       
            if x_rec_t is not None:
                recon_frames.append(x_rec_t.sum(dim=1).cpu())
            else:
                recon_frames.append(torch.zeros_like(x_t.sum(dim=1).cpu()))

    gif_images = []
    for t in range(T):
        fig, axes = plt.subplots(2, num_classes, figsize=(num_classes * 2.5, 5))
        if num_classes == 1: axes = np.expand_dims(axes, axis=1)
        
        for i in range(num_classes):
            gt_img = gt_frames[t][i].numpy()
            axes[0, i].imshow(gt_img, cmap='gray', vmin=0, vmax=1)
            axes[0, i].axis('off')
            
            rec_img = recon_frames[t][i].numpy()
            axes[1, i].imshow(rec_img, cmap='gray', vmin=0, vmax=1)
            axes[1, i].axis('off')
            
        fig.suptitle(f"Reconstruction - Time Step: {t+1}/{T}", fontsize=14)
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        gif_images.append(Image.open(buf).copy())
        plt.close(fig)
        buf.close()
        
    save_path = os.path.join(Config.RESULTS_DIR, filename)
    if gif_images:
        gif_images[0].save(save_path, save_all=True, append_images=gif_images[1:], optimize=False, duration=150, loop=0)

class VideoRandomAffine:
    def __init__(self, degrees=0, translate=None, scale=None):
        self.translate = translate

    def __call__(self, video_tensor):
        dx, dy = 0.0, 0.0
        
        if self.translate:
            max_dx = self.translate[0] * video_tensor.shape[3]
            max_dy = self.translate[1] * video_tensor.shape[2]
            dx = random.uniform(-max_dx, max_dx)
            dy = random.uniform(-max_dy, max_dy)
            
        transformed_frames = []
        for t in range(video_tensor.shape[0]):
            frame = video_tensor[t] 
            frame_aug = TF.affine(
                frame, 
                angle=0, 
                translate=(dx, dy), 
                scale=1.0, 
                shear=0
            )
            transformed_frames.append(frame_aug)
            
        return torch.stack(transformed_frames, dim=0)

class DVSGestureDataset(data.Dataset):
    def __init__(self, split='train', transform=None):
        self.H, self.W = Config.IMG_H, Config.IMG_W
        self.transform = transform 
        root_dir = "/workspace0/zf925/spiking_lista_dvsg/data/DVSGesture/frames_number_16_split_by_number"
        target_dir = os.path.join(root_dir, split)
        self.file_list = []
        self.labels = []
        
        if not os.path.exists(target_dir):
            pass
        else:
            classes = sorted(os.listdir(target_dir))
            valid_classes = []
            for c in classes:
                if os.path.isdir(os.path.join(target_dir, c)):
                    valid_classes.append(c)
            
            for class_name in valid_classes:
                try:
                    label_idx = int(class_name)
                except ValueError:
                    continue 
                
                class_dir = os.path.join(target_dir, class_name)
                pattern = os.path.join(class_dir, "*.pt")
                files = glob.glob(pattern)
                
                for f in files:
                    self.file_list.append(f)
                    self.labels.append(label_idx)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        path = self.file_list[index]
        label = self.labels[index]

        try:
            data_tensor = torch.load(path, weights_only=True)
        except Exception:
            data_tensor = torch.load(path)

        if isinstance(data_tensor, tuple): 
            data_tensor = data_tensor[0]

        frames = data_tensor.float()
        if frames.dim() == 3: 
            frames = frames.unsqueeze(1)

        T_in, C, h, w = frames.shape
        if h != self.H or w != self.W:
            frames = F.interpolate(frames, size=(self.H, self.W), mode='bilinear', align_corners=False)

        frames = torch.log1p(frames)
        max_val = frames.max()
        if max_val > 0:
            frames = frames / max_val

        if self.transform:
            frames = self.transform(frames)

        return frames, label

def get_dataloaders(batch_size):
    train_transform = VideoRandomAffine(
        degrees=0,            
        translate=(0.0, 0.0), 
        scale=None
    )
    
    train_set = DVSGestureDataset(split='train', transform=train_transform)
    val_set = DVSGestureDataset(split='test', transform=None) 
    
    train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, 
                                   num_workers=Config.NUM_WORKERS, pin_memory=True)
    val_loader = data.DataLoader(val_set, batch_size=batch_size, shuffle=False, 
                                 num_workers=Config.NUM_WORKERS, pin_memory=True)
                                 
    return train_loader, val_loader
    
    
def save_feature_visualization_gif(model, sensor, channel, loader, filename):
    samples = get_class_samples(loader, Config.DEVICE)
    if samples is None: return

    num_classes = samples.shape[0]
    T = samples.shape[1]
    
    model.eval()
    sensor.eval()
    
    feature_frames = [] 
    
    recon_state = None
    sensor_state = None
    channel_state = None
    
    mode = 'ann' if sensor.mode == 'ann' else 'spiking'

    with torch.no_grad():
        for t in range(T):
            x_t = samples[:, t] 
            
            y_quant, sensor_state, _ = sensor(x_t, state=sensor_state, mode=mode)
            y_noisy, channel_state = channel(y_quant, state=channel_state, mode=mode)
            
            prev_mode = getattr(model, 'mode', None)
            if hasattr(model, 'mode'): model.mode = 'rate' if mode == 'ann' else 'spiking'

            _, layers, recon_state, _ = model(y_noisy, states=recon_state)
            
            if hasattr(model, 'mode') and prev_mode is not None: model.mode = prev_mode
            
            if len(layers) > 0 and layers[-1] is not None:
                feat = layers[-1].mean(dim=1).cpu()
                feature_frames.append(feat)
            else:
                dummy_h = Config.IMG_H // Config.MEAS_STRIDE
                dummy_w = Config.IMG_W // Config.MEAS_STRIDE
                feature_frames.append(torch.zeros(num_classes, dummy_h, dummy_w))

    gif_images = []
    for t in range(T):
        fig, axes = plt.subplots(1, num_classes, figsize=(num_classes * 2.5, 2.5))
        if num_classes == 1: axes = [axes]
        
        for i in range(num_classes):
            feat_img = feature_frames[t][i].numpy()
            axes[i].imshow(feat_img, cmap='viridis')
            axes[i].axis('off')
            
        fig.suptitle(f"Latent Features - Time Step: {t+1}/{T}", fontsize=14)
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        gif_images.append(Image.open(buf).copy())
        plt.close(fig)
        buf.close()
        
    save_path = os.path.join(Config.RESULTS_DIR, filename)
    if gif_images:
        gif_images[0].save(save_path, save_all=True, append_images=gif_images[1:], optimize=False, duration=150, loop=0)
        
def save_temporal_intersection_gif(sensor, loader, filename):
    # Extract one sample per class for visualization
    samples = get_class_samples(loader, Config.DEVICE)
    if samples is None: return

    num_classes = samples.shape[0]
    T = samples.shape[1]
    
    sensor.eval()
    mode = 'spiking' 
    
    # Calculate spatial dimensions based on stride
    spatial_h = Config.IMG_H // Config.MEAS_STRIDE
    spatial_w = Config.IMG_W // Config.MEAS_STRIDE
    meas_c = Config.MEAS_CHANNELS

    intersection_frames = []
    current_spikes_frames = []
    
    sensor_state = None
    prev_tx_hard = None

    with torch.no_grad():
        for t in range(T):
            x_t = samples[:, t]
            _, sensor_state, aux_dict = sensor(x_t, state=sensor_state, mode=mode)
            
            # Reshape flat spike array back to spatial dimensions
            tx_hard_flat = aux_dict['tx_hard'] 
            tx_hard_spatial = tx_hard_flat.view(-1, meas_c, spatial_h, spatial_w)
            
            # Sum over the channel dimension to create a 2D activation heatmap
            current_map = tx_hard_spatial.sum(dim=1).cpu() 
            current_spikes_frames.append(current_map)
            
            if prev_tx_hard is not None:
                # Element-wise multiplication to isolate spikes active in both current and previous frame
                intersection_spatial = (tx_hard_spatial * prev_tx_hard).sum(dim=1).cpu()
                intersection_frames.append(intersection_spatial)
            else:
                # First frame has no temporal history
                intersection_frames.append(torch.zeros_like(current_map))
                
            prev_tx_hard = tx_hard_spatial

    # Generate and save the GIF
    gif_images = []
    for t in range(T):
        fig, axes = plt.subplots(2, num_classes, figsize=(num_classes * 2.5, 5))
        if num_classes == 1: axes = np.expand_dims(axes, axis=1)
        
        for i in range(num_classes):
            # Row 0: Current Spikes
            curr_img = current_spikes_frames[t][i].numpy()
            axes[0, i].imshow(curr_img, cmap='magma', vmin=0, vmax=meas_c)
            axes[0, i].axis('off')
            if i == 0: axes[0, i].set_title(f"Spikes (t={t+1})")
            
            # Row 1: Intersection with Previous Frame
            inter_img = intersection_frames[t][i].numpy()
            axes[1, i].imshow(inter_img, cmap='magma', vmin=0, vmax=meas_c)
            axes[1, i].axis('off')
            if i == 0: axes[1, i].set_title("Intersection w/ Prev")
            
        fig.suptitle(f"Temporal Spike Intersection (t={t+1}/{T})", fontsize=14)
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        gif_images.append(Image.open(buf).copy())
        plt.close(fig)
        buf.close()
        
    save_path = os.path.join(Config.RESULTS_DIR, filename)
    if gif_images:
        gif_images[0].save(save_path, save_all=True, append_images=gif_images[1:], optimize=False, duration=150, loop=0)