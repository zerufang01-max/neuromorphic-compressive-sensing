import torch
import numpy as np
import random
import os
import argparse 
import csv
from config import Config
from edge_compression import EdgeSensor
from wireless_channel import WirelessChannel
from snn_models import ConvolutionalLISTA, SpikingCNNRecon, ConvLSTMRecon, ConvolutionalLISTA_ImageSpace
from trainer import Trainer
import utils

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def safe_load_state_dict(model, state_dict):
    model_dict = model.state_dict()
    pretrained_dict = {
        k: v for k, v in state_dict.items() 
        if k in model_dict and v.shape == model_dict[k].shape
    }
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

def evaluate_inference(name, model, sensor, channel, loader, mode='rate', seed=42):
    print(f"Evaluating {name}...")
    original_model_mode = getattr(model, 'mode', None)
    original_sensor_mode = sensor.mode
    
    if hasattr(model, 'mode'):
        model.mode = 'rate' if mode == 'ann' else 'spiking'
    sensor.mode = 'ann' if mode == 'ann' else 'snn'

    trainer = Trainer(model, sensor, channel, loader, loader)
    
    if mode == 'spiking':
        nmse = trainer.validate_recon(mode='spiking')
        gif_filename = f"{name.lower()}_seed{seed}_reconstruction.gif"
        utils.save_snn_visualization_gif(model, sensor, channel, loader, gif_filename)
        
        feat_filename = f"{name.lower()}_seed{seed}_features.gif"
        utils.save_feature_visualization_gif(model, sensor, channel, loader, feat_filename)
        
        intersection_filename = f"{name.lower()}_seed{seed}_intersection.gif"
        utils.save_temporal_intersection_gif(sensor, loader, intersection_filename)
    else:
        nmse = trainer.validate_recon(mode='ann')
        gif_filename = f"{name.lower()}_seed{seed}_reconstruction.gif"
        utils.save_snn_visualization_gif(model, sensor, channel, loader, gif_filename)
        
        feat_filename = f"{name.lower()}_seed{seed}_features.gif"
        utils.save_feature_visualization_gif(model, sensor, channel, loader, feat_filename)
        
        intersection_filename = f"{name.lower()}_seed{seed}_intersection.gif"
        utils.save_temporal_intersection_gif(sensor, loader, intersection_filename)
        
    print(f"[{name}] NMSE: {nmse:.4f} dB")
    
    stats_filename = f"{name.lower()}_seed{seed}_stats.txt"
    utils.analyze_and_save_statistics(model, sensor, channel, loader, stats_filename)
    
    if original_model_mode is not None:
        model.mode = original_model_mode
    sensor.mode = original_sensor_mode
    
    return nmse

def run_ablation_suite(train_loader, val_loader_recon, train_loader_cls, val_loader_cls, channel):
    print(f"\n{'='*60}")
    print(" >>> RUNNING ABLATION SUITE (ConvLISTA_Img | RECON | SPIKING)")
    print(f"{'='*60}")
    
    variants = ['hard_spike']
    results_log = []
    
    for variant in variants:
        Config.TX_ABLATION_MODE = variant
        
        print(f"\n[{variant.upper()}] Starting Evaluation...")
        
       
        set_seed(Config.SEED)
        
        sensor = EdgeSensor().to(Config.DEVICE)
        model = ConvolutionalLISTA_ImageSpace().to(Config.DEVICE)
        
        trainer = Trainer(model, sensor, channel, train_loader_recon, val_loader_recon)
        
        th_val = model.get_threshold_value()
        model_name_ablation = f"convlista_img_ablate_{variant}_seed{Config.SEED}"
        
        rho_suffix = f"_rho{Config.TX_RHO}"
        recon_ckpt_name = f"dynamic_recon_spiking_{model_name_ablation.lower()}{rho_suffix}_th{th_val:.2f}.pth"
        recon_ckpt_path = os.path.join(Config.RESULTS_DIR, recon_ckpt_name)
        
        recon_loaded = False
        
        if os.path.exists(recon_ckpt_path):
            print(f"[{variant}] Loading Reconstruction Checkpoint: {recon_ckpt_path}")
            ckpt = torch.load(recon_ckpt_path, weights_only=False)
            safe_load_state_dict(model, ckpt['model'])
            if 'sensor' in ckpt: sensor.load_state_dict(ckpt['sensor'])
            recon_loaded = True
        else:
            print(f"[{variant}] Training Reconstruction from scratch...")
           
            set_seed(Config.SEED)
            trainer.train_dynamic_recon(model_name=model_name_ablation, mode='spiking', freeze_sensor=False)
            ckpt = torch.load(recon_ckpt_path, weights_only=False)
            safe_load_state_dict(model, ckpt['model'])
            if 'sensor' in ckpt: sensor.load_state_dict(ckpt['sensor'])
            
        recon_nmse = trainer.validate_recon(mode='spiking')
        print(f"[{variant}] Eval NMSE: {recon_nmse:.4f} dB")
        
        avg_bits, tx_rate = trainer.evaluate_bits(mode='spiking')
        print(f"[{variant}] Post-Recon Avg Bits/Sample: {avg_bits:.2f}")
        
        acc_results = {'latent': 0.0, 'image': 0.0}
        
        for cls_type in ['image','latent' ]:
           
            set_seed(Config.SEED)
            print(f"[{variant}] Training Phase 4 Classification ({cls_type.upper()})...")
            trainer_cls = Trainer(model, sensor, channel, train_loader_cls, val_loader_cls)
            
            cls_ckpt_name = f"phase4_cls_spiking_{model_name_ablation.lower()}{rho_suffix}_th{th_val:.2f}_{cls_type}.pth"
            cls_ckpt_path = os.path.join(Config.RESULTS_DIR, cls_ckpt_name)
            
            if os.path.exists(cls_ckpt_path):
                print(f"[{variant}] Loading Classification Checkpoint: {cls_ckpt_path}")
                ckpt = torch.load(cls_ckpt_path, weights_only=False)
                safe_load_state_dict(model, ckpt['model'])
                if 'sensor' in ckpt: sensor.load_state_dict(ckpt['sensor'])
            else:
                trainer_cls.train_phase4(model_name=model_name_ablation, mode='spiking', cls_type=cls_type)
                ckpt = torch.load(cls_ckpt_path, weights_only=False)
                safe_load_state_dict(model, ckpt['model'])
                if 'sensor' in ckpt: sensor.load_state_dict(ckpt['sensor'])
                
            acc = trainer_cls.validate_cls(mode='spiking', cls_type=cls_type)
            print(f"[{variant}] Eval Accuracy ({cls_type.upper()}): {acc:.2f}%")
            acc_results[cls_type] = acc
        
        tau_mode = 'annealed' if variant == 'annealed_soft' else ('1.0' if variant == 'fixed_tau_1.0' else ('0.1' if variant == 'fixed_tau_0.1' else 'hard'))
        row = {
            'Variant': variant,
            'Recon Checkpoint': recon_ckpt_name,
            'Recon Loaded Existing': recon_loaded,
            'Final Eval NMSE (dB)': recon_nmse,
            'Avg Bits/Sample': avg_bits,
            'Acc (Latent) (%)': acc_results['latent'],
            'Acc (Image) (%)': acc_results['image'],
            'Tau Mode': tau_mode,
            'Notes': 'Ablation variant'
        }
        results_log.append(row)
        
    csv_file = os.path.join(Config.RESULTS_DIR, "ablation_convlista_img.csv")
    txt_file = os.path.join(Config.RESULTS_DIR, "ablation_convlista_img.txt")
    md_file = os.path.join(Config.RESULTS_DIR, "ablation_convlista_img.md")
    
    headers = list(results_log[0].keys())
    
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(results_log)
        
    table_str = f"{'Variant':<15} | {'NMSE (dB)':<10} | {'Acc(Lat)':<8} | {'Acc(Img)':<8} | {'Bits/Samp':<10}\n"
    table_str += "-" * 60 + "\n"
    for r in results_log:
        table_str += f"{r['Variant']:<15} | {r['Final Eval NMSE (dB)']:<10.4f} | {r['Acc (Latent) (%)']:<8.2f} | {r['Acc (Image) (%)']:<8.2f} | {r['Avg Bits/Sample']:<10.2f}\n"
        
    with open(txt_file, 'w') as f:
        f.write("ABLATION SUITE RESULTS\n")
        f.write("="*60 + "\n")
        f.write(table_str)
        
    print("\n" + "="*60)
    print(" ABLATION SUITE COMPLETE - SUMMARY")
    print("="*60)
    print(table_str)
    
def main():
    parser = argparse.ArgumentParser(description='Run SNN Experiments')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--threshold', type=float, default=None, help='Initial threshold for LISTA')
    parser.add_argument('--tx_rho', type=float, default=None, help='Transmitter rho for state retention')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42], help='List of random seeds for grid search')
    parser.add_argument('--skip_recon', action='store_true', help='Skip Phase 3 reconstruction to dry-run Phase 4')
    
    parser.add_argument('--alpha_rates_lista', nargs='+', type=float, default=None)
    parser.add_argument('--alpha_rates_scnn', nargs='+', type=float, default=None)
                        
    parser.add_argument('--lr_lista_snn', nargs='+', type=float, default=None)
    parser.add_argument('--lr_lista_ann', nargs='+', type=float, default=None)
    parser.add_argument('--lr_lstm', nargs='+', type=float, default=None)
    parser.add_argument('--lr_spiking_cnn', nargs='+', type=float, default=None)

    parser.add_argument('--l1_snn_lista', nargs='+', type=float, default=None)
    parser.add_argument('--l1_ann_lista', nargs='+', type=float, default=None)
    parser.add_argument('--l1_lstm', nargs='+', type=float, default=None)
    parser.add_argument('--l1_scnn', nargs='+', type=float, default=None)

    parser.add_argument('--ablation', action='store_true', help='Run ablation suite for ConvLISTA_Img')
    args = parser.parse_args()

    Config.setup(args)

    if args.ablation:
        Config.SEED = args.seeds[0]
        Config.print_config()
        train_loader_recon, val_loader_recon = utils.get_dataloaders(Config.BATCH_SIZE_RECON)
        train_loader_cls, val_loader_cls = utils.get_dataloaders(Config.BATCH_SIZE_CLS)
        channel = WirelessChannel().to(Config.DEVICE)
        run_ablation_suite(train_loader_recon, val_loader_recon, train_loader_cls, val_loader_cls, channel)
        return

    grid_results_log = []

    for seed in args.seeds:
        Config.SEED = seed
        print(f"\n{'#'*60}")
        print(f" ### INITIALIZING GRID SEARCH FOR SEED: {seed} ### ")
        print(f"{'#'*60}")
        Config.print_config()
        
        train_loader_recon, val_loader_recon = utils.get_dataloaders(Config.BATCH_SIZE_RECON)
        train_loader_cls, val_loader_cls = utils.get_dataloaders(Config.BATCH_SIZE_CLS)
        channel = WirelessChannel().to(Config.DEVICE)
        
        experiments = []
        suffix = "_RelaxedAER" if getattr(Config, 'USE_RELAXED_AER_BSC', False) else ""
        
        is_lr_sweep = any(arg is not None for arg in [args.lr_lista_snn, args.lr_lista_ann, args.lr_lstm, args.lr_spiking_cnn])
        is_l1_sweep = any(arg is not None for arg in [args.l1_snn_lista, args.l1_ann_lista, args.l1_lstm, args.l1_scnn])

        if is_l1_sweep:
            if args.l1_snn_lista is not None:
                for l1 in args.l1_snn_lista:
                    experiments.append((f'ConvLISTA_Img{suffix}_l1{l1}', ConvolutionalLISTA_ImageSpace, 'spiking', f'ContinuousSensor_SNN_LISTA (L1={l1})'))
            if args.l1_ann_lista is not None:
                for l1 in args.l1_ann_lista:
                    experiments.append((f'ANN_ConvLISTA_l1{l1}', ConvolutionalLISTA, 'ann', f'ContinuousSensor_ANN_LISTA (L1={l1})'))
            if args.l1_lstm is not None:
                for l1 in args.l1_lstm:
                    experiments.append((f'ConvLSTM_l1{l1}', ConvLSTMRecon, 'ann', f'ContinuousSensor_ANN_LSTM (L1={l1})'))
            if args.l1_scnn is not None:
                for l1 in args.l1_scnn:
                    experiments.append((f'SpikingCNN{suffix}_l1{l1}', SpikingCNNRecon, 'spiking', f'ContinuousSensor_SpikingCNN (L1={l1})'))
            print(f"\n[Info] Running L1 Penalty Sweep...")

        elif is_lr_sweep:
            if args.lr_lista_snn is not None:
                for lr in args.lr_lista_snn:
                    experiments.append((f'ConvLISTA_Img{suffix}_lr{lr}', ConvolutionalLISTA_ImageSpace, 'spiking', f'ContinuousSensor_SNN_LISTA_Img{suffix} (LR={lr})'))
            if args.lr_lista_ann is not None:
                for lr in args.lr_lista_ann:
                    experiments.append((f'ANN_ConvLISTA_lr{lr}', ConvolutionalLISTA, 'ann', f'ContinuousSensor_ANN_LISTA (LR={lr})'))
            if args.lr_lstm is not None:
                for lr in args.lr_lstm:
                    experiments.append((f'ConvLSTM_lr{lr}', ConvLSTMRecon, 'ann', f'ContinuousSensor_ANN_LSTM (LR={lr})'))
            if args.lr_spiking_cnn is not None:
                for lr in args.lr_spiking_cnn:
                    experiments.append((f'SpikingCNN{suffix}_lr{lr}', SpikingCNNRecon, 'spiking', f'ContinuousSensor_SpikingCNN{suffix} (LR={lr})'))
            print(f"\n[Info] Running Learning Rate Sweep...")
            
        elif args.alpha_rates_lista is not None or args.alpha_rates_scnn is not None:
            if args.alpha_rates_lista is not None:
                for alpha in args.alpha_rates_lista:
                    exp_name = f"ConvLISTA_Img{suffix}_alpha{alpha}"
                    experiments.append((exp_name, ConvolutionalLISTA_ImageSpace, 'spiking', f'ContinuousSensor_SNN_LISTA_Img{suffix} (Alpha={alpha})'))
                print(f"\n[Info] Running Alpha Rate Sweep for S-LISTA with values: {args.alpha_rates_lista}")
                
            if args.alpha_rates_scnn is not None:
                for alpha in args.alpha_rates_scnn:
                    exp_name = f"SpikingCNN{suffix}_alpha{alpha}"
                    experiments.append((exp_name, SpikingCNNRecon, 'spiking', f'ContinuousSensor_SpikingCNN{suffix} (Alpha={alpha})'))
                print(f"\n[Info] Running Alpha Rate Sweep for SpikingCNN with values: {args.alpha_rates_scnn}")
        else:
            experiments = [
                (f'ConvLISTA_Img{suffix}', ConvolutionalLISTA_ImageSpace, 'spiking', f'ContinuousSensor_SNN_LISTA_Img{suffix}'),
                ('ConvLSTM', ConvLSTMRecon, 'ann', 'ContinuousSensor_ANN_LSTM'),
                (f'SpikingCNN{suffix}', SpikingCNNRecon, 'spiking', f'ContinuousSensor_SpikingCNN{suffix}'),
                ('ANN_ConvLISTA', ConvolutionalLISTA, 'ann', 'ContinuousSensor_ANN_LISTA'),
            ]
        
        for name, ModelClass, mode, desc in experiments:
            base_name = name
            alpha_val_record = "N/A"
            
            if 'alpha' in base_name:
                alpha_val = float(base_name.split('alpha')[1])
                alpha_val_record = alpha_val
                if 'ConvLISTA' in base_name: Config.ALPHA_RATE_LISTA = alpha_val
                elif 'SpikingCNN' in base_name: Config.ALPHA_RATE_SCNN = alpha_val

            if '_lr' in base_name:
                lr_val = float(base_name.split('_lr')[1])
                if 'ConvLISTA_Img' in base_name: Config.PHASE3_CONFIG['lr_lista_snn'] = lr_val
                elif 'ANN_ConvLISTA' in base_name: Config.PHASE3_CONFIG['lr_lista_ann'] = lr_val
                elif 'ConvLSTM' in base_name: Config.PHASE3_CONFIG['lr_lstm'] = lr_val
                elif 'SpikingCNN' in base_name: Config.PHASE3_CONFIG['lr_spiking_cnn'] = lr_val

            if '_l1' in base_name:
                l1_val = float(base_name.split('_l1')[1])
                if 'ConvLISTA_Img' in base_name: Config.ALPHA_L1_LISTA_SNN = l1_val
                elif 'ANN_ConvLISTA' in base_name: Config.ALPHA_L1_LISTA_ANN = l1_val
                elif 'ConvLSTM' in base_name: Config.ALPHA_L1_LSTM = l1_val
                elif 'SpikingCNN' in base_name: Config.ALPHA_L1_SCNN = l1_val

            print(f"\n{'='*50}")
            print(f" >>> Experiment: {desc}")
            print(f" >>> Mode: {mode.upper()}")
            print(f"{'='*50}")
            
         
            set_seed(seed)
            
            sensor = EdgeSensor().to(Config.DEVICE)
            model = ModelClass().to(Config.DEVICE)
            
            trainer = Trainer(model, sensor, channel, train_loader_recon, val_loader_recon)
            
            recon_nmse = float('nan')
            
            threshold_suffix = ""
            if hasattr(model, 'get_threshold_value'):
                th_val = model.get_threshold_value()
                threshold_suffix = f"_th{th_val:.2f}"
            
            rho_suffix = f"_rho{Config.TX_RHO}"
            
            model_name_with_seed = f"{base_name.lower()}_seed{seed}"
            recon_ckpt_name = f"dynamic_recon_{mode}_{model_name_with_seed}{rho_suffix}{threshold_suffix}.pth"
            recon_ckpt_path = os.path.join(Config.RESULTS_DIR, recon_ckpt_name)
            
            if args.skip_recon:
                print(f"\n[WARNING] --skip_recon is ACTIVE. Skipping Phase 3.")
            elif os.path.exists(recon_ckpt_path):
                print(f"Loading Reconstruction Checkpoint: {recon_ckpt_path}")
                try: ckpt = torch.load(recon_ckpt_path, weights_only=False)
                except: ckpt = torch.load(recon_ckpt_path, weights_only=False)
                
                safe_load_state_dict(model, ckpt['model'])
                if 'sensor' in ckpt: sensor.load_state_dict(ckpt['sensor'])
            else:
                print(f"No checkpoint found at {recon_ckpt_path}, starting training on DVSG...")
              
                set_seed(seed)
                _ = trainer.train_dynamic_recon(model_name=model_name_with_seed, mode=mode, freeze_sensor=False)
                
                print(f"Reloading best reconstruction checkpoint for evaluation...")
                ckpt = torch.load(recon_ckpt_path, weights_only=False)
                safe_load_state_dict(model, ckpt['model'])
                if 'sensor' in ckpt: sensor.load_state_dict(ckpt['sensor'])
            
            recon_nmse = evaluate_inference(f"{name}", model, sensor, channel, val_loader_recon, mode=mode, seed=seed)
            
            avg_bits, tx_rate = trainer.evaluate_bits(mode=mode)
            print(f"[{name}] Post-Recon Avg Bits/Sample: {avg_bits:.2f}")
            
            acc_results = {'latent': 0.0, 'image': 0.0}
            
            for cls_type in ['image', 'latent']:
              
                set_seed(seed) 
                
                print(f"\n >>> Classification Evaluation for {name} ({cls_type.upper()})")
                trainer_cls = Trainer(model, sensor, channel, train_loader_cls, val_loader_cls)
                
                cls_ckpt_name = f"phase4_cls_{mode}_{model_name_with_seed}{rho_suffix}{threshold_suffix}_{cls_type}.pth"
                cls_ckpt_path = os.path.join(Config.RESULTS_DIR, cls_ckpt_name)
                
                acc = 0.0
                
                if os.path.exists(cls_ckpt_path):
                    print(f"Loading {name} Classification Checkpoint: {cls_ckpt_path}")
                    try: ckpt = torch.load(cls_ckpt_path, weights_only=False)
                    except: ckpt = torch.load(cls_ckpt_path, weights_only=False)
                    
                    safe_load_state_dict(model, ckpt['model'])
                    if 'sensor' in ckpt: sensor.load_state_dict(ckpt['sensor'])
                    
                    acc = trainer_cls.validate_cls(mode=mode, cls_type=cls_type)
                    print(f"Loaded Model Accuracy ({cls_type.upper()}): {acc:.2f}%")
                else:
                    acc = trainer_cls.train_phase4(model_name=model_name_with_seed, mode=mode, cls_type=cls_type)
                    
                    print(f"Reloading best classification checkpoint for evaluation...")
                    ckpt = torch.load(cls_ckpt_path, weights_only=False)
                    safe_load_state_dict(model, ckpt['model'])
                    if 'sensor' in ckpt: sensor.load_state_dict(ckpt['sensor'])
                    acc = trainer_cls.validate_cls(mode=mode, cls_type=cls_type)
                
                acc_results[cls_type] = acc
                y_true, y_pred = trainer_cls.get_predictions(mode=mode, cls_type=cls_type)
                utils.save_confusion_matrix(y_true, y_pred, f"confusion_matrix_{name.lower()}_seed{seed}_{cls_type}.png")

            grid_results_log.append({
                'Seed': seed,
                'Model': name,
                'Mode': mode.upper(),
                'Alpha Rate': alpha_val_record,
                'Tx Rho': Config.TX_RHO,
                'Tx Firing Rate': tx_rate,
                'Bandwidth (Bits/Sample)': avg_bits,
                'NMSE (dB)': recon_nmse,
                'Acc (Latent) (%)': acc_results['latent'],
                'Acc (Image) (%)': acc_results['image']
            })

    csv_path = os.path.join(Config.RESULTS_DIR, "grid_search_rate_results.csv")
    headers = ['Seed', 'Model', 'Mode', 'Alpha Rate', 'Tx Rho', 'Tx Firing Rate', 'Bandwidth (Bits/Sample)', 'NMSE (dB)', 'Acc (Latent) (%)', 'Acc (Image) (%)']
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(grid_results_log)
        
    table_path = os.path.join(Config.RESULTS_DIR, "final_performance_table.txt")
    with open(table_path, "w") as f:
        f.write(f"{'Seed':<6} | {'Model':<35} | {'Alpha Rate':<12} | {'Tx Rho':<8} | {'NMSE (dB)':<12} | {'Acc(Lat)':<10} | {'Acc(Img)':<10} | {'Bits/Sample':<15} | {'Tx Firing Rate':<15}\n")
        f.write("-" * 155 + "\n")
        for res in grid_results_log:
            nmse_str = f"{res['NMSE (dB)']:.4f}" if not np.isnan(res['NMSE (dB)']) else "N/A"
            acc_lat_str = f"{res['Acc (Latent) (%)']:.2f}" if res['Acc (Latent) (%)'] != 0 else "N/A"
            acc_img_str = f"{res['Acc (Image) (%)']:.2f}" if res['Acc (Image) (%)'] != 0 else "N/A"
            bits_str = f"{res['Bandwidth (Bits/Sample)']:.2f}"
            tx_str = f"{res['Tx Firing Rate']:.4f}"
            f.write(f"{res['Seed']:<6} | {res['Model']:<35} | {str(res['Alpha Rate']):<12} | {res['Tx Rho']:<8} | {nmse_str:<12} | {acc_lat_str:<10} | {acc_img_str:<10} | {bits_str:<15} | {tx_str:<15}\n")

    print(f"\n[Info] Grid Search Complete. Logged to {csv_path} and {table_path}")

if __name__ == "__main__":
    main()