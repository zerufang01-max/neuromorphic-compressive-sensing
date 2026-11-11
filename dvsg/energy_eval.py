"""Table 3 energy and INT8 post-training quantisation for DVS-G.

    python energy_eval.py --gpu 0                 # self-checks, energy, quantisation
    python energy_eval.py --gpu 0 --skip-quant    # energy only

Seed 42, AWGN 10 dB, test split (288 samples).  Table 3 uses the AER operating points
(slista_aer, vanilla_aer = scnn_rate_0p002 of baselines/train.py) and the two ANN baselines; the dense
S-LISTA / Vanilla models are also evaluated for reference.  The counting rules live in
energy_model.py; this script only collects event statistics with the trained models.

Outputs (outputs/energy/):
  selfcheck_seed42.txt      checks 1-3 (check 1 against analysis/energy_seed42.csv)
  table3_seed42.csv         transmitter / receiver / total energy (uJ per sample) and ratios
  components_seed42.csv     every counted item (amount per sample, energy)
  slista_layers_seed42.csv  S-LISTA ||xi||_0, ||z||_0, ||z||_1 per layer and frame (per sample)
  stats_seed42.json         raw event statistics
  quant_seed42.csv          float vs INT8 NMSE and downstream accuracy (AWGN 10 dB)
"""

import argparse
import csv
import json
import math
from collections import defaultdict

import config as C
import energy_model as EV
import experiments as E

TABLE_KEYS = list(E.ENERGY_KEYS)                      # slista_aer, vanilla_aer, ann_lista, ann_lstm
EXTRA_KEYS = ["slista_dense", "vanilla_dense"]


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


# ----------------------------------------------------------------- collection
def collect(experiment, model, sensor, loader, device):
    """Per-sample event statistics in one deterministic AWGN 10 dB pass (same RNG use as
    trainer.evaluate, so the counts match analysis/energy_seed42.csv)."""
    import torch
    import torch.nn as nn

    import trainer as T
    from channels import make_channel
    from slista import SpikingLISTA

    channel = make_channel("awgn", C.TRAIN_SNR_DB, experiment.transport).to(device)
    T.deterministic_eval(C.EVAL_CHANNEL_SEED)
    model.eval(); sensor.eval(); channel.eval()
    mode = experiment.mode
    acc = defaultdict(float)
    handles = []

    def watch(key, which="input"):
        def hook(module, inputs, output):
            t = inputs[0] if which == "input" else (output[0] if isinstance(output, tuple) else output)
            acc[key + "_nnz"] += float((t != 0).sum())
            acc[key + "_l1"] += float(t.abs().sum())
        return hook

    def readout_hook(module, inputs, output):
        x = inputs[0]
        k = 1
        for v in module.kernel_size:
            k *= int(v)
        if isinstance(module, nn.ConvTranspose2d):
            acc["readout_macs"] += float(x.numel() * module.out_channels * k)
        else:
            acc["readout_macs"] += float(output.numel() * (module.in_channels // module.groups) * k)
        acc["readout_in_bytes"] += float(x.numel())

    convs = [m for m in sensor.encoder if isinstance(m, nn.Conv2d)]
    if mode == "spiking":
        handles.append(convs[1].register_forward_hook(watch("spikes1")))
        handles.append(convs[2].register_forward_hook(watch("spikes2")))
    if experiment.method == "vanilla":
        handles.append(model.head_1.register_forward_hook(watch("y")))
        handles.append(model.res1.conv1.register_forward_hook(watch("spikes_head")))
        handles.append(model.res1.conv2.register_forward_hook(watch("spikes_res1")))
        handles.append(model.res1.lif2.register_forward_hook(watch("spikes_res2", "output")))
        readouts = [model.conv_up1, model.conv_up2, model.tail]
    else:
        readouts = [model.tail]
    for block in readouts:
        for m in ([block] if isinstance(block, (nn.Conv2d, nn.ConvTranspose2d)) else block.modules()):
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                handles.append(m.register_forward_hook(readout_hook))

    is_slista = isinstance(model, SpikingLISTA)
    layers = getattr(model, "spiking_layers", 0)
    frames = C.Config.TIME_STEPS
    per_step = {k: [[0.0] * layers for _ in range(frames)] for k in ("xi_nnz", "z_nnz", "z_l1")}
    samples = 0
    try:
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device, non_blocking=True)
                states = sensor_state = channel_state = None
                for t in range(x.shape[1]):
                    y, sensor_state, aux = sensor(x[:, t], state=sensor_state, mode=mode)
                    if mode == "spiking":
                        acc["tx_spikes"] += float(aux["tx_hard"].sum())
                    y, channel_state = channel(y, state=channel_state, mode=mode,
                                               tx_prob=aux["tx_prob"], tx_hard=aux["tx_hard"])
                    _, _, states, trace = T.step(model, y, states)
                    if is_slista:
                        acc["y_nnz"] += float((trace["input"] != 0).sum())
                        acc["y_l1"] += float(trace["input"].abs().sum())
                        acc["mask"] += float(trace["mask"].sum())
                        for l in range(layers):
                            per_step["xi_nnz"][t][l] += float((trace["spikes"][l] != 0).sum())
                            per_step["z_nnz"][t][l] += float((trace["codes"][l] != 0).sum())
                            per_step["z_l1"][t][l] += float(trace["codes"][l].abs().sum())
                samples += x.shape[0]
    finally:
        for h in handles:
            h.remove()

    avg = {k: v / samples for k, v in acc.items()}
    stats = {"samples": samples, "readout_macs": avg.get("readout_macs", 0.0),
             "readout_in_bytes": avg.get("readout_in_bytes", 0.0)}
    if mode == "spiking":
        stats["sender"] = {"spikes1": avg["spikes1_nnz"], "spikes2": avg["spikes2_nnz"],
                           "spikes1_l1": avg["spikes1_l1"], "spikes2_l1": avg["spikes2_l1"],
                           "tx_spikes": avg["tx_spikes"]}
    if is_slista:
        per_step = {k: [[v / samples for v in row] for row in rows] for k, rows in per_step.items()}
        stats["recon"] = {"y_events": avg["y_nnz"], "y_magnitude": avg["y_l1"], "mask": avg["mask"],
                          **{k: [sum(per_step[k][t][l] for t in range(frames)) for l in range(layers)]
                             for k in per_step}}
        stats["per_step"] = per_step
    elif experiment.method == "vanilla":
        stats["recon"] = {"y_events": avg["y_nnz"], "y_magnitude": avg["y_l1"],
                          "spikes_head": avg["spikes_head_nnz"], "spikes_res1": avg["spikes_res1_nnz"],
                          "spikes_res2": avg["spikes_res2_nnz"],
                          "spikes_head_l1": avg["spikes_head_l1"], "spikes_res1_l1": avg["spikes_res1_l1"]}
    return stats


def ledger_for(experiment, stats, rho, opcount=False, membrane=True):
    return EV.build(experiment.method, stats, experiment.channels, rho=rho, opcount=opcount, membrane=membrane)


# ----------------------------------------------------------------- self-checks
def self_checks(ledgers, stats_by_key, ref_path, lines):
    ok = True
    # 1. operation-count prices and rules reproduce analysis/energy_seed42.csv
    lines.append("Check 1: E_B = 0, E_AC = 0.9, E_MAC = 4.6, E_ADD = E_MUL = 0, eta as MAC, "
                 "LSTM first step with W_hh h0 (op_counter.py)")
    if not ref_path.is_file():
        lines.append(f"  {ref_path} missing -> run `python main.py analysis` first")
        ok = False
    else:
        ref = {r["experiment"]: r for r in csv.DictReader(open(ref_path))}
        worst = 0.0
        for key, (experiment, opcount_ledger) in ledgers.items():
            if key not in ref:
                lines.append(f"  {key}: not in {ref_path.name}")
                continue
            new = EV.opcount_counts(opcount_ledger)
            for col in ("sender_mac_mop", "sender_ac_mop", "recon_mac_mop", "recon_ac_mop",
                        "sender_uj", "recon_uj", "total_uj"):
                a, b = new[col], float(ref[key][col])
                rel = abs(a - b) / max(abs(b), 1e-12) if (a or b) else 0.0
                worst = max(worst, rel)
                flag = "" if rel < 1e-5 else "   <-- differs"
                lines.append(f"  {key:14s} {col:15s} model {a:14.6f}  analysis {b:14.6f}  rel {rel:.2e}{flag}")
        passed = worst < 1e-5
        ok &= passed
        lines.append(f"  max relative difference {worst:.2e} -> {'PASS' if passed else 'FAIL'}")
    # 2. all-ones code: event-driven (U2) == dense (U1) with E_MAC -> E_AC
    lines.append("Check 2: code set to all ones -> U2 equals U1 with E_MAC replaced by E_AC")
    n_events = EV.CODE * EV.GRID * EV.GRID * EV.FRAMES
    dense, event = EV.Ledger(), EV.Ledger()
    dense.dense_conv("recon", "G", EV.CODE, EV.CODE, EV.GRID * EV.GRID, EV.GRID * EV.GRID, EV.FRAMES)
    event.event_conv("recon", "G", EV.CODE, n_events, n_events)
    prices = dict(EV.PRICES, mac=EV.PRICES["ac"])
    a, b = dense.energy(prices)["total"], event.energy()["total"]
    passed = abs(a - b) <= 1e-9 * max(a, b)
    ok &= passed
    lines.append(f"  one 128->128 3x3 layer, 16 frames: U1 {a:.6f} uJ, U2 {b:.6f} uJ -> {'PASS' if passed else 'FAIL'}")
    ones = {"y_events": 0.0, "y_magnitude": 0.0, "mask": 0.0,
            "xi_nnz": [0.0] * 7, "z_nnz": [n_events] * 7, "z_l1": [n_events] * 7}
    led = EV.Ledger()
    EV.slista(led, ones, 4, rho=0.0)
    g_ac = led.amount("ac")
    want = 7 * EV.CODE * EV.K2 * EV.CODE * EV.GRID * EV.GRID * EV.FRAMES
    passed = abs(g_ac - want) < 1e-6 * want
    ok &= passed
    lines.append(f"  S-LISTA with all-ones codes: G accumulations {g_ac:.4e} = 7 dense layers {want:.4e} "
                 f"-> {'PASS' if passed else 'FAIL'}")
    # 3. ANN LISTA weight reads ~ 8x compute
    lines.append("Check 3: ANN LISTA weight-read energy / compute energy (about 2.5 / 0.3 = 8.3)")
    led = EV.Ledger()
    EV.ann_lista(led, 4)
    e = led.energy()
    ratio = e["weights"] / e["compute"]
    passed = 7.5 < ratio < 8.5
    ok &= passed
    lines.append(f"  receiver: weights {e['weights']:.1f} uJ / compute {e['compute']:.1f} uJ = {ratio:.2f} "
                 f"-> {'PASS' if passed else 'FAIL'}")
    # consistency of the collected statistics
    for key, stats in stats_by_key.items():
        sender = stats.get("sender")
        if sender:
            for name in ("spikes1", "spikes2"):
                if abs(sender[name] - sender[name + "_l1"]) > 1e-6 * max(1.0, sender[name]):
                    lines.append(f"  note: {key} {name} not binary (nnz {sender[name]}, l1 {sender[name + '_l1']})")
    return ok


# ----------------------------------------------------------------- quantisation
def int8_per_channel_(weight):
    """Symmetric per-output-channel INT8 (round to the nearest of 255 levels, dequantise)."""
    import torch
    w = weight.data
    amax = w.abs().flatten(1).amax(1).clamp_min(1e-12).view(-1, *([1] * (w.dim() - 1)))
    scale = amax / 127.0
    w.copy_(torch.round(w / scale).clamp(-127, 127) * scale)


def quantise_(experiment, model, sensor):
    """INT8 on every weight matrix that is priced in the energy estimate :
    transmitter convolutions and the receiver convolutions; dictionary readout, task head,
    downstream classifier, thresholds, eta and biases stay in floating point.  BatchNorm
    scales act per output channel, so quantising the convolution per output channel is the
    same as quantising the folded weight."""
    import torch.nn as nn
    from torch.nn import utils
    names = []
    for i, m in enumerate([m for m in sensor.encoder if isinstance(m, nn.Conv2d)]):
        if hasattr(m, "weight_g"):
            utils.remove_weight_norm(m)                       # effective weight g * v / ||v||
        int8_per_channel_(m.weight)
        names.append(f"sensor.conv{i + 1}")
    if experiment.method == "slista":
        targets = {"P": model.P, **{f"G{l + 1}": g for l, g in enumerate(model.G)}}
    elif experiment.method == "vanilla":
        targets = {"head_1": model.head_1, "head_2": model.head_2,
                   "res1.conv1": model.res1.conv1, "res1.conv2": model.res1.conv2}
    elif experiment.method == "ann_lista":
        targets = {"W_e": model.W_e, **{f"S_k.{k}": model.S_k[k] for k in range(1, len(model.S_k))}}
    else:
        targets = {"lstm_cell.conv": model.lstm_cell.conv}
    for name, m in targets.items():
        int8_per_channel_(m.weight)
        names.append(name)
    return names


# ----------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpu", type=int, default=C.DEFAULT_GPUS[0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-quant", action="store_true")
    args = parser.parse_args()

    device = C.setup_torch(args.gpu)
    import data
    import trainer as T
    from channels import make_channel
    from main import load_run_models

    out = C.OUTPUTS / "energy"
    out.mkdir(parents=True, exist_ok=True)
    seed = args.seed
    _, loader = data.make_loaders(seed)

    stats_by_key, opcount, ledgers, rho_by_key = {}, {}, {}, {}
    for key in TABLE_KEYS + EXTRA_KEYS:
        experiment = E.REGISTRY[key]
        spec = E.make_spec(key, seed, E.formal_run_dir(key, seed))
        model, sensor, _, _ = load_run_models(experiment, spec, device)
        rho = float(getattr(model, "rho", 0.0))
        stats = collect(experiment, model, sensor, loader, device)
        stats_by_key[key], rho_by_key[key] = stats, rho
        opcount[key] = (experiment, ledger_for(experiment, stats, rho, opcount=True))
        ledgers[key] = (experiment, ledger_for(experiment, stats, rho))
        print(f"[energy] collected {key}", flush=True)
        del model, sensor

    (out / f"stats_seed{seed}.json").write_text(json.dumps(stats_by_key, indent=1))
    lines = []
    ok = self_checks(opcount, stats_by_key, C.OUTPUTS / "analysis" / f"energy_seed{seed}.csv", lines)
    lines.append("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    (out / f"selfcheck_seed{seed}.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)

    # Table 3 and components
    table, components = [], []
    ref = ledgers["ann_lista"][1]
    ref_s, ref_r = ref.energy(scope="sender")["total"], ref.energy(scope="recon")["total"]
    ref_t = ref.energy()["total"]
    for key, (experiment, led) in ledgers.items():
        s, r, tot = led.energy(scope="sender"), led.energy(scope="recon"), led.energy()
        stats = stats_by_key[key]
        row = {"experiment": key, "method": experiment.display, "transport": experiment.transport,
               "in_table3": key in TABLE_KEYS,
               "sender_compute_uj": s["compute"], "sender_weights_uj": s["weights"],
               "sender_state_uj": s["state"], "sender_uj": s["total"],
               "recon_compute_uj": r["compute"], "recon_weights_uj": r["weights"],
               "recon_state_uj": r["state"], "recon_uj": r["total"],
               "total_uj": tot["total"],
               "sender_ratio_vs_ann_lista": s["total"] / ref_s,
               "recon_ratio_vs_ann_lista": r["total"] / ref_r,
               "total_ratio_vs_ann_lista": tot["total"] / ref_t,
               "readout_uj_not_counted": EV.readout_estimate(stats["readout_macs"], stats["readout_in_bytes"]),
               "readout_mac_mop_not_counted": stats["readout_macs"] / 1e6}
        if experiment.method == "slista" and rho_by_key[key]:
            plain = ledger_for(experiment, stats, rho_by_key[key], membrane=False).energy(scope="recon")["total"]
            row["recon_uj_without_membrane_memory"] = plain
        table.append(row)
        for scope, comp, kind, amount in led.rows:
            components.append({"experiment": key, "scope": scope, "component": comp, "kind": kind,
                               "amount_per_sample": amount, "energy_uj": amount * EV.PRICES[kind] / 1e6})
    write_csv(out / f"table3_seed{seed}.csv", table)
    write_csv(out / f"components_seed{seed}.csv", components)

    layer_rows = []
    for key in ("slista_aer", "slista_dense"):
        per_step = stats_by_key[key]["per_step"]
        for t in range(len(per_step["xi_nnz"])):
            for l in range(len(per_step["xi_nnz"][t])):
                layer_rows.append({"experiment": key, "frame": t + 1, "layer": l + 1,
                                   **{k: per_step[k][t][l] for k in per_step}})
    write_csv(out / f"slista_layers_seed{seed}.csv", layer_rows)

    print("\n[energy] Table 3 (uJ per sample; seed 42, AWGN 10 dB)")
    for row in table:
        print(f"  {row['experiment']:14s} sender {row['sender_uj']:10.2f}  recon {row['recon_uj']:10.2f}  "
              f"total {row['total_uj']:10.2f}  recon/ANN LISTA {row['recon_ratio_vs_ann_lista']:.4f}  "
              f"(readout not counted: {row['readout_uj_not_counted']:.0f})", flush=True)

    if args.skip_quant:
        return
    quant_rows = []
    for key in TABLE_KEYS + EXTRA_KEYS:
        experiment = E.REGISTRY[key]
        spec = E.make_spec(key, seed, E.formal_run_dir(key, seed))
        row = {"experiment": key, "method": experiment.display, "transport": experiment.transport}
        for tag in ("float", "int8"):
            model, sensor, classifier, _ = load_run_models(experiment, spec, device)
            if tag == "int8":
                row["quantised"] = " ".join(quantise_(experiment, model, sensor))
            result = T.evaluate(model, sensor, make_channel("awgn", C.TRAIN_SNR_DB, experiment.transport),
                                loader, experiment, device, classifier=classifier)
            row[f"{tag}_nmse"] = result["nmse"]
            row[f"{tag}_acc"] = result["acc"]
            del model, sensor, classifier
        row["delta_nmse_db"] = row["int8_nmse"] - row["float_nmse"]
        row["delta_acc"] = row["int8_acc"] - row["float_acc"]
        quant_rows.append(row)
        print(f"[quant] {key:14s} NMSE {row['float_nmse']:.2f} -> {row['int8_nmse']:.2f} dB, "
              f"acc {row['float_acc']:.2f} -> {row['int8_acc']:.2f}", flush=True)
    write_csv(out / f"quant_seed{seed}.csv", quant_rows)
    assert all(math.isfinite(r["int8_nmse"]) for r in quant_rows)


if __name__ == "__main__":
    main()
