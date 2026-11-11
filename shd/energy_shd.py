"""SHD energy per sample (energy model of Supplementary Note 2, Energy
estimation) and INT8 post-training quantization.

Table 3 uses the AER models of S-LISTA and the Vanilla SNN (slista_aer,
scnn_aer; the transmission scheme of Table 2 and of DVS-G) and the ANN
baselines (8-bit dense transmission, their only scheme): seed 42, AWGN
10 dB, all 2,264 test samples, 25 time steps per sample counted step by
step and summed (no weight reuse across steps or samples).  The dense
(full-bitmap) S-LISTA / Vanilla models are counted as well: the
operation-count estimate of main.py (analysis_energy_seed42.csv) is computed
on them, so self-check 1 is run on that set.  The dictionary readout (and the equivalent final linear readout
of LSTM / Vanilla SNN) is excluded, as in the paper.

    python -u energy_shd.py --gpu 0               # energy + self-checks + INT8
    python energy_shd.py --gpu 0 --skip-quant     # energy only
    python energy_shd.py --unit-test              # pure counting formulas (no GPU)

Outputs in outputs/energy/:
    energy_counts_seed42.json        per-sample counts per model and side
    energy_table_aer_seed42.csv      Table 3 (AER S-LISTA / Vanilla)
    energy_table_dense_seed42.csv    same with the dense S-LISTA / Vanilla
    selfcheck_seed42.json            self-checks
    slista_layers_{aer,dense}_seed42.csv  ||xi||_0, ||z||_0, ||z||_1 per layer
    quant_int8_seed42.csv            float vs INT8 NMSE / accuracy, AWGN 10 dB
"""

import argparse
import copy
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TABLES = {
    "aer": ["slista_aer", "scnn_aer", "ann_lista", "ann_lstm"],      # Table 3
    "dense": ["slista_dense", "scnn_dense", "ann_lista", "ann_lstm"],  # self-check 1
}
KEYS = ["slista_aer", "scnn_aer", "slista_dense", "scnn_dense", "ann_lista", "ann_lstm"]
DISPLAY = {"slista_aer": "S-LISTA", "scnn_aer": "Vanilla SNN",
           "slista_dense": "S-LISTA", "scnn_dense": "Vanilla SNN",
           "ann_lista": "ANN LISTA", "ann_lstm": "ANN LSTM"}
TRANSMISSION = {"slista_aer": "block AER", "scnn_aer": "block AER",
                "slista_dense": "dense bitmap", "scnn_dense": "dense bitmap",
                "ann_lista": "dense 8-bit", "ann_lstm": "dense 8-bit"}

# Unit prices (pJ).
PRICES = {"MAC": 0.3, "AC": 0.1, "ADD": 0.1, "MUL": 3.1, "B": 2.5, "eta_as_mac": False}
# Self-check 1: the operation-count estimate (4.6 / 0.9 pJ, operations only;
# the readout eta multiplication counted as one MAC).
OPCOUNT_PRICES = {"MAC": 4.6, "AC": 0.9, "ADD": 0.0, "MUL": 0.0, "B": 0.0, "eta_as_mac": True}
FIELDS = ("mac", "ac", "add", "mul", "mul_eta", "w_bytes", "s_bytes")


# ----------------------------------------------------------------- counting
def ledger(**counts):
    out = {field: 0.0 for field in FIELDS}
    for field, value in counts.items():
        out[field] += float(value)
    return out


def add_to(total, part):
    for field in FIELDS:
        total[field] += part[field]
    return total


def u1(a, b, samples=1):
    """Dense matrix-vector product a -> b with a real input (unit U1)."""
    return ledger(mac=a * b * samples, w_bytes=a * b * samples, s_bytes=a * samples)


def u2(nnz, l1, b):
    """Event-driven accumulation (unit U2): one column read per non-zero
    input, |z_j| accumulations; nnz / l1 are summed over the batch."""
    return ledger(ac=b * l1, w_bytes=b * nnz, s_bytes=nnz)


def energy(counts, prices):
    """Energy split into compute, weight reads and state reads/writes (pJ)."""
    eta_price = prices["MAC"] if prices["eta_as_mac"] else prices["MUL"]
    compute = (counts["mac"] * prices["MAC"] + counts["ac"] * prices["AC"]
               + counts["add"] * prices["ADD"] + counts["mul"] * prices["MUL"]
               + counts["mul_eta"] * eta_price)
    weight = counts["w_bytes"] * prices["B"]
    state = counts["s_bytes"] * prices["B"]
    return {"compute": compute, "weight": weight, "state": state,
            "total": compute + weight + state}


def sender_spiking(samples, x_nnz, x_l1, d_in, m, leak=True):
    """Spiking sender: U2 on the integer event counts of the frame
    (N = ||x||_0, M = ||x||_1), then m LIF neurons (one E_MUL for
    the leak, two E_ADD, membrane read and write of 2 bytes each)."""
    del d_in
    out = u2(x_nnz, x_l1, m)
    neurons = m * samples
    return add_to(out, ledger(mul=neurons if leak else 0.0, add=2 * neurons,
                              s_bytes=4 * neurons))


def sender_ann(samples, d_in, m):
    """ANN sender: dense U1 (d_in -> m), sigmoid by look-up (1 E_ADD per
    element; the 8-bit level is read from the same table)."""
    return add_to(u1(d_in, m, samples), ledger(add=m * samples))


def slista_frame(samples, n, y_nnz, layers, mask_nnz):
    """One frame of S-LISTA .

    y_nnz: received spikes (binary, so N = M); layers: per spiking layer
    (xi_nnz, z_nnz, z_l1) of xi^(l) and z^(l); mask_nnz = ||m||_0.  All
    counts are summed over the batch; samples is the batch size."""
    out = u2(y_nnz, y_nnz, n)                          # I^(1) = P y
    add_to(out, ledger(s_bytes=4 * n * samples))       # write I^(1) once
    for xi_nnz, z_nnz, z_l1 in layers:
        add_to(out, ledger(add=2 * n * samples))       # threshold on the accumulator
        add_to(out, ledger(add=xi_nnz, s_bytes=2 * xi_nnz))   # z += xi (read + write)
        add_to(out, ledger(s_bytes=4 * n * samples))   # next layer: init with I^(1)
        add_to(out, u2(z_nnz, z_l1, n))                # ... minus G^(l) z^(l), all n rows
    # Readout z = z^(L-1) + eta * m * I^(L): eta multiply, add, write.
    add_to(out, ledger(mul_eta=mask_nnz, add=mask_nnz, s_bytes=mask_nnz))
    return out


def ann_lista_frame(samples, m, n, layers):
    """One frame of ANN LISTA : W y once (bias written as 4n
    bytes, layer 1 soft-thresholded on the accumulator), then layers-1
    S matrices (the S acting on the all-zero initial code is not counted)."""
    out = u1(m, n, samples)
    add_to(out, ledger(s_bytes=4 * n * samples, add=2 * n * samples))
    add_to(out, ledger(s_bytes=n * samples))           # layer-1 output
    for _ in range(layers - 1):
        add_to(out, ledger(s_bytes=4 * n * samples))   # init with W y
        add_to(out, u1(n, n, samples))
        add_to(out, ledger(add=2 * n * samples, s_bytes=n * samples))
    return out


def lstm_frame(samples, m, h, first=False):
    """One LSTM step : U1((m + h) -> 4h) (its input read covers
    y and h); 5h activations by look-up (4 gates + tanh c); 3h gate
    products; c = f*c + i*g adds h; c read + c write + h write.
    First step: h_0 = 0 is known, so W_hh h_0 and the read of
    h_0 are not counted, i.e. U1(m -> 4h)."""
    out = u1(m if first else m + h, 4 * h, samples)
    return add_to(out, ledger(add=(5 * h + h) * samples, mul=3 * h * samples,
                              s_bytes=3 * h * samples))


def vanilla_frame(samples, synapses, lif_neurons, residual_nnz, spike_writes):
    """One frame of the Vanilla SNN : every synaptic layer is U2
    on its spike input; each LIF neuron 1 E_MUL + 2 E_ADD + 4 bytes;
    residual identity added per non-zero spike; output spikes written."""
    out = ledger()
    for out_features, nnz, l1 in synapses:
        add_to(out, u2(nnz, l1, out_features))
    neurons = lif_neurons * samples
    add_to(out, ledger(mul=neurons, add=2 * neurons, s_bytes=4 * neurons))
    add_to(out, ledger(add=residual_nnz, s_bytes=spike_writes))
    return out


# ------------------------------------------------------------- unit tests
def unit_test():
    """Self-check 2 and basic identities (no torch)."""
    ok = True

    def check(condition, text):
        nonlocal ok
        ok &= bool(condition)
        print(("PASS " if condition else "FAIL ") + text)

    ac_prices = dict(PRICES, MAC=PRICES["AC"])
    # Single product: all-ones input of length a -> U2 == U1 with MAC -> AC.
    for a, b in [(256, 700), (700, 700), (700, 256)]:
        e2 = energy(u2(a, a, b), PRICES)["total"]
        e1 = energy(u1(a, b), ac_prices)["total"]
        check(abs(e1 - e2) < 1e-9, f"U2(all ones {a}) == U1({a}->{b}) with E_MAC -> E_AC")
    # Whole S-LISTA frame with y and every code all ones (xi = 1 only in layer 1).
    n, m, layers = 700, 256, 7
    frame = slista_frame(1, n, m, [(n if k == 0 else 0, n, n) for k in range(layers)], n)
    dense = u1(m, n)
    for _ in range(layers):
        add_to(dense, u1(n, n))
    matrix_part = ledger(ac=frame["ac"], w_bytes=frame["w_bytes"],
                         s_bytes=m + layers * n)
    e2 = energy(matrix_part, PRICES)["total"]
    e1 = energy(dense, ac_prices)["total"]
    check(abs(e1 - e2) < 1e-6, "S-LISTA with all-ones codes: matrix part == dense U1 (MAC -> AC)")
    # Operation-count identity: S-LISTA = 0.9 n (||y||_1 + sum ||z||_1) + 4.6 ||m||_0.
    frame = slista_frame(2, n, 30, [(10, 10, 12), (5, 14, 20)], 9)
    opcount = energy(frame, OPCOUNT_PRICES)["total"]
    check(abs(opcount - (0.9 * n * (30 + 12 + 20) + 4.6 * 9)) < 1e-6,
          "operation-count prices reproduce 0.9 n (||y||_1 + sum ||z||_1) + 4.6 ||m||_0")
    # ANN LISTA ratio of weight-read to compute energy (self-check 3, ~8.3).
    lista = ann_lista_frame(1, m, n, 8)
    split = energy(lista, PRICES)
    check(7.5 < split["weight"] / split["compute"] < 8.4,
          f"ANN LISTA weight / compute energy = {split['weight'] / split['compute']:.2f} (~8.3)")
    check(lista["mac"] == m * n + 7 * n * n, "ANN LISTA MACs = m n + 7 n^2")
    check(lstm_frame(1, 256, 512)["mac"] == 4 * 512 * (256 + 512), "LSTM MACs = 4h(m + h)")
    first, later = lstm_frame(1, 256, 512, first=True), lstm_frame(1, 256, 512)
    check(later["mac"] - first["mac"] == 4 * 512 * 512
          and later["w_bytes"] - first["w_bytes"] == 4 * 512 * 512
          and later["s_bytes"] - first["s_bytes"] == 512,
          "LSTM first step drops W_hh h_0 (MACs, weight reads) and the read of h_0")
    print("All unit tests passed." if ok else "UNIT TESTS FAILED")
    return ok


# ------------------------------------------------------------ measurement
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-batches", type=int, default=0, help="smoke test")
    parser.add_argument("--skip-quant", action="store_true")
    parser.add_argument("--unit-test", action="store_true")
    parser.add_argument("--results-dir", default=str(ROOT / "outputs"))
    return parser.parse_args()


def setup(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    sys.path.insert(0, str(ROOT))
    import torch
    from config import Config
    Config.RESULTS_DIR = str(Path(args.results_dir).resolve())
    os.environ["SHD_OUTPUT_DIR"] = Config.RESULTS_DIR
    Config.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Config._CLI_CLASSIFIER_LR = 1e-3
    out = Path(Config.RESULTS_DIR) / "energy"
    out.mkdir(parents=True, exist_ok=True)
    return out


def count_method(key, seed, loader, max_batches):
    """Run the formal model of `key` over the test set exactly as
    collect_analysis does (same channel realisations) and accumulate the
    ledgers per side, plus S-LISTA layer statistics."""
    import torch
    import main as shd
    from config import Config
    from snn_models import SpikingLISTA

    experiment = shd.experiment_registry()[key]
    trainer = shd.load_for_analysis(experiment, seed, loader)
    model, sensor = trainer.model, trainer.sensor
    m, d_in = int(Config.MEAS_DIM), int(Config.INPUT_DIM)
    sides = {"sender": ledger(), "reconstruction": ledger()}
    stats = None
    hooks, vanilla = [], {}
    if isinstance(model, SpikingLISTA):
        if float(model.rho) != 0.0:
            raise RuntimeError("S-LISTA with membrane memory (rho != 0) is not covered by this accounting.")
        layers = model.spiking_layers
        stats = {"y_nnz": 0.0, "mask_nnz": 0.0,
                 "xi_nnz": [0.0] * layers, "z_nnz": [0.0] * layers, "z_l1": [0.0] * layers}
    vanilla_model = key.startswith("scnn")
    if vanilla_model:
        named = [("head_1", model.head_1), ("head_2", model.head_2)]
        for index, block in enumerate(model.res_blocks):
            named += [(f"b{index}.fc1", block.fc1), (f"b{index}.fc2", block.fc2)]
        named.append(("tail", model.tail))

        def make_hook(name, module):
            def hook(_module, inputs, _output):
                x = inputs[0].detach()
                vanilla[name] = (module.out_features, float((x != 0).sum()), float(x.abs().sum()))
            return hook
        for name, module in named:
            hooks.append(module.register_forward_hook(make_hook(name, module)))
    total_samples = 0
    total_bits = 0.0
    shd.set_seed(Config.EVAL_CHANNEL_SEED)
    with torch.no_grad():
        for batch_index, (x_batch, _labels) in enumerate(loader):
            if max_batches > 0 and batch_index >= max_batches:
                break
            x_batch = trainer._preprocess_batch(x_batch)
            batch, steps, _ = x_batch.shape
            total_samples += batch
            sensor_state = channel_state = model_states = None
            for t in range(steps):
                frame = x_batch[:, t]
                if experiment.mode == "spiking":
                    add_to(sides["sender"], sender_spiking(
                        batch, float((frame != 0).sum()), float(frame.abs().sum()), d_in, m,
                        leak=float(getattr(sensor, "rho", 0.0)) != 0.0))
                else:
                    add_to(sides["sender"], sender_ann(batch, d_in, m))
                transmitted, sensor_state, auxiliary = sensor(frame, state=sensor_state,
                                                              mode=experiment.mode)
                if experiment.mode != "spiking":
                    total_bits += batch * m * int(Config.ANN_QUANT_BITS)
                elif experiment.transport == "block_aer_awgn_hard":
                    total_bits += float(trainer.channel.block_aer_bits(auxiliary.get("tx_hard")).sum())
                else:
                    total_bits += batch * m
                received, channel_state = trainer.channel(
                    transmitted, state=channel_state, mode=experiment.mode,
                    tx_prob=auxiliary.get("tx_prob"), tx_hard=auxiliary.get("tx_hard"),
                    tx_aux=auxiliary)
                _, _, model_states, _ = model(received, states=model_states)
                if stats is not None:
                    trace = model.last_trace
                    y_nnz = float((received != 0).sum())
                    if float(received.abs().sum()) != y_nnz:
                        raise RuntimeError("S-LISTA received frame is not binary.")
                    per_layer = []
                    for k, (xi, z) in enumerate(zip(trace["spikes"], trace["codes"])):
                        xi_nnz, z_nnz, z_l1 = (float((xi != 0).sum()), float((z != 0).sum()),
                                               float(z.abs().sum()))
                        per_layer.append((xi_nnz, z_nnz, z_l1))
                        stats["xi_nnz"][k] += xi_nnz
                        stats["z_nnz"][k] += z_nnz
                        stats["z_l1"][k] += z_l1
                    mask_nnz = float(trace["mask"].sum())
                    stats["y_nnz"] += y_nnz
                    stats["mask_nnz"] += mask_nnz
                    add_to(sides["reconstruction"], slista_frame(
                        batch, int(model.n), y_nnz, per_layer, mask_nnz))
                elif key == "ann_lista":
                    add_to(sides["reconstruction"], ann_lista_frame(
                        batch, m, int(model.sparse_dim), int(model.K)))
                elif key == "ann_lstm":
                    hidden = int(model.hidden_dim)
                    add_to(sides["reconstruction"], lstm_frame(batch, m, hidden, first=(t == 0)))
                    if t == 0:
                        # Counted by the operation-count estimate; kept apart for self-check 1 only.
                        add_to(sides.setdefault("excluded_h0", ledger()),
                               ledger(mac=4 * hidden * hidden * batch))
                elif vanilla_model:
                    order = ["head_1", "head_2"] + [f"b{i}.fc{j}" for i in range(len(model.res_blocks))
                                                    for j in (1, 2)]
                    synapses = [vanilla[name] for name in order]
                    # LIF outputs are the inputs of the next synaptic layer.
                    writes = sum(vanilla[name][1] for name in order[1:] + ["tail"])
                    residual = sum(vanilla[f"b{i}.fc1"][1] for i in range(len(model.res_blocks)))
                    neurons = model.feat_dim + model.sparse_dim * (1 + 2 * len(model.res_blocks))
                    add_to(sides["reconstruction"], vanilla_frame(
                        batch, synapses, neurons, residual, writes))
    for handle in hooks:
        handle.remove()
    per_sample = {side: {f: v / total_samples for f, v in counts.items()}
                  for side, counts in sides.items()}
    if stats is not None:
        stats = {k: ([v / total_samples for v in val] if isinstance(val, list)
                     else val / total_samples) for k, val in stats.items()}
    per_sample["bits_per_sample"] = total_bits / total_samples
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return per_sample, stats, total_samples


# ------------------------------------------------------------ quantization
def quantize_rows_(weight):
    """Symmetric per-output-channel INT8 (fake quantization, in place)."""
    import torch
    with torch.no_grad():
        scale = weight.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 127.0
        weight.copy_(torch.clamp(torch.round(weight / scale), -127, 127) * scale)


def quant_matrices(key, model, sensor):
    """(network, readout) weight matrices of sender + reconstruction.
    'readout' is the dictionary (or the final linear readout of LSTM /
    Vanilla SNN), excluded from the energy count."""
    network = [sensor.fc.weight]
    if key.startswith("slista"):
        network += [model.P.weight] + [g.weight for g in model.G]
        readout = [model.synthesis_dict.weight]
    elif key == "ann_lista":
        network += [model.W_e.weight] + [s.weight for s in model.S_k]
        readout = [model.synthesis_dict.weight]
    elif key == "ann_lstm":
        network += [model.lstm.weight_ih, model.lstm.weight_hh]
        readout = [model.fc.weight]
    else:
        network += [model.head_1.weight, model.head_2.weight]
        for block in model.res_blocks:
            network += [block.fc1.weight, block.fc2.weight]
        readout = [model.tail.weight]
    return network, readout


def evaluate_10db(trainer, experiment):
    trainer.channel.set_snr_db(10.0)
    joint = trainer.validate_joint_metrics(mode=experiment.mode)
    accuracy = trainer.validate_cls(mode=experiment.mode, target="recon")
    return {"NMSE (dB)": float(joint["Temporal-Mean NMSE (dB)"]),
            "Full-Sequence NMSE (dB)": float(joint["Full-Sequence NMSE (dB)"]),
            "Accuracy (%)": float(accuracy)}


def run_quant(seed, out):
    import pandas as pd
    import torch
    from types import SimpleNamespace
    import main as shd
    import utils
    from config import Config

    rows = []
    train_loader, val_loader = utils.get_dataloaders(Config.BATCH_SIZE_RECON)
    flags = SimpleNamespace(force_retrain=False, force_retrain_classifier=False,
                            eval_only=True, epochs=None)
    for key in KEYS:
        experiment = shd.experiment_registry()[key]
        trainer, _, _, _ = shd.train_or_load(experiment, seed, flags, train_loader, val_loader)
        formal = pd.read_csv(shd.point_path(experiment, seed))
        formal = formal[formal["Eval SNR (dB)"].astype(float).sub(10.0).abs() < 1e-9].iloc[0]
        original = {"model": copy.deepcopy(trainer.model.state_dict()),
                    "sensor": copy.deepcopy(trainer.sensor.state_dict())}
        results = {"float": evaluate_10db(trainer, experiment)}
        for tier in ("network", "network+readout"):
            trainer.model.load_state_dict(original["model"])
            trainer.sensor.load_state_dict(original["sensor"])
            network, readout = quant_matrices(key, trainer.model, trainer.sensor)
            for weight in network + (readout if tier == "network+readout" else []):
                quantize_rows_(weight)
            results[tier] = evaluate_10db(trainer, experiment)
        trainer.model.load_state_dict(original["model"])
        trainer.sensor.load_state_dict(original["sensor"])
        row = {"Method": DISPLAY[key], "Experiment Key": key, "Transmission": TRANSMISSION[key],
               "Seed": seed,
               "Formal NMSE (dB)": float(formal["Mean NMSE (dB)"]),
               "Formal Accuracy (%)": float(formal["Accuracy (%)"])}
        for tier, metrics in results.items():
            for name, value in metrics.items():
                row[f"{tier} {name}"] = value
        rows.append(row)
        print(f"[INT8] {DISPLAY[key]} ({key}): float {results['float']['NMSE (dB)']:.3f} dB / "
              f"{results['float']['Accuracy (%)']:.2f}% (formal {row['Formal NMSE (dB)']:.3f} / "
              f"{row['Formal Accuracy (%)']:.2f}) | INT8 network {results['network']['NMSE (dB)']:.3f} / "
              f"{results['network']['Accuracy (%)']:.2f}% | +readout "
              f"{results['network+readout']['NMSE (dB)']:.3f} / "
              f"{results['network+readout']['Accuracy (%)']:.2f}%", flush=True)
        del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    frame = pd.DataFrame(rows)
    frame.to_csv(out / f"quant_int8_seed{seed}.csv", index=False)
    return frame


def main():
    args = parse_args()
    if args.unit_test:
        raise SystemExit(0 if unit_test() else 1)
    if not unit_test():
        raise SystemExit("Counting formulas failed their unit tests.")
    out = setup(args)
    import pandas as pd
    import utils
    from config import Config

    _, loader = utils.get_dataloaders(Config.BATCH_SIZE_CLS)
    counts, layer_stats, samples = {}, {}, 0
    for key in KEYS:
        counts[key], stats, samples = count_method(key, args.seed, loader, args.max_batches)
        if stats is not None:
            layer_stats[key] = stats
        print(f"[Counted] {DISPLAY[key]} ({TRANSMISSION[key]}, {key}) over {samples} samples",
              flush=True)
    sides = ("sender", "reconstruction")
    (out / f"energy_counts_seed{args.seed}.json").write_text(json.dumps(
        {"samples": samples, "prices_pJ": PRICES, "tables": TABLES,
         "counts_per_sample": counts, "slista_layers_per_sample": layer_stats}, indent=2))

    # ---------------- self-check 1 (dense set): operation-count prices reproduce analysis_energy
    checks = {}
    ref_path = Path(Config.RESULTS_DIR) / f"analysis_energy_seed{args.seed}.csv"
    if ref_path.is_file() and args.max_batches == 0:
        ref_table = pd.read_csv(ref_path).set_index("Model")
        for key in TABLES["dense"]:
            ledgers = [c for name, c in counts[key].items() if isinstance(c, dict)]
            mine = sum(energy(c, OPCOUNT_PRICES)["total"] for c in ledgers) / 1e6
            ref = float(ref_table.loc[DISPLAY[key], "Total Energy (uJ/Sample)"])
            mac = sum(c["mac"] for c in ledgers) + counts[key]["reconstruction"]["mul_eta"]
            ac = counts[key]["sender"]["ac"] + counts[key]["reconstruction"]["ac"]
            ref_mac = float(ref_table.loc[DISPLAY[key], ["Sender MAC/Sample", "Reconstruction MAC/Sample"]].sum())
            ref_ac = float(ref_table.loc[DISPLAY[key], ["Sender AC/Sample", "Reconstruction AC/Sample"]].sum())
            checks[key] = {
                "main_total_uJ": ref, "recomputed_uJ": mine,
                "rel_diff": abs(mine - ref) / ref, "mac": [mac, ref_mac], "ac": [ac, ref_ac],
                "pass": abs(mine - ref) / ref < 1e-5}
            print(f"[Self-check 1] {DISPLAY[key]} ({key}): main.py {ref:.6f} uJ, recomputed {mine:.6f} uJ "
                  f"-> {'PASS' if checks[key]['pass'] else 'FAIL'}", flush=True)
    else:
        print("[Self-check 1] skipped (needs the full test set and analysis_energy_seed*.csv)")
    split = energy(counts["ann_lista"]["reconstruction"], PRICES)
    checks["ANN LISTA weight/compute (self-check 3, ~8.3)"] = split["weight"] / split["compute"]
    checks["self-check 2 (unit test)"] = True
    (out / f"selfcheck_seed{args.seed}.json").write_text(json.dumps(checks, indent=2))

    # ---------------- Table 3 (AER) and the dense counterpart
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    for name, keys in TABLES.items():
        rows, base = [], None
        for key in keys:
            row = {"Method": DISPLAY[key], "Experiment Key": key,
                   "Transmission": TRANSMISSION[key],
                   "Bits/Sample": counts[key]["bits_per_sample"]}
            for side in sides:
                split = energy(counts[key][side], PRICES)
                for part in ("compute", "weight", "state", "total"):
                    row[f"{side} {part} (uJ)"] = split[part] / 1e6
                row[f"{side} memory (uJ)"] = (split["weight"] + split["state"]) / 1e6
                c = counts[key][side]
                row[f"{side} MAC (M)"] = c["mac"] / 1e6
                row[f"{side} AC (M)"] = c["ac"] / 1e6
                row[f"{side} weight bytes (MB)"] = c["w_bytes"] / 1e6
                row[f"{side} state bytes (MB)"] = c["s_bytes"] / 1e6
            row["total (uJ)"] = row["sender total (uJ)"] + row["reconstruction total (uJ)"]
            rows.append(row)
            if key == "ann_lista":
                base = row
        for row in rows:
            for column in ("sender total (uJ)", "reconstruction total (uJ)", "total (uJ)"):
                row[f"{column} / ANN LISTA"] = row[column] / base[column]
        table = pd.DataFrame(rows)
        table.to_csv(out / f"energy_table_{name}_seed{args.seed}.csv", index=False)
        show = ["Method", "Bits/Sample", "sender compute (uJ)", "sender weight (uJ)", "sender state (uJ)",
                "sender total (uJ)", "reconstruction compute (uJ)", "reconstruction weight (uJ)",
                "reconstruction state (uJ)", "reconstruction total (uJ)", "total (uJ)",
                "reconstruction total (uJ) / ANN LISTA", "total (uJ) / ANN LISTA"]
        label = "Table 3: AER S-LISTA / Vanilla" if name == "aer" else "dense S-LISTA / Vanilla (reference)"
        print(f"\n=== Energy per sample seed {args.seed}, AWGN 10 dB, {label} ===")
        print(table[show].to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    for key, stats in layer_stats.items():
        layers = pd.DataFrame({
            "Layer": range(1, len(stats["xi_nnz"]) + 1),
            "||xi||_0": stats["xi_nnz"], "||z||_0": stats["z_nnz"], "||z||_1": stats["z_l1"]})
        layers["||z||_0/||z||_1"] = layers["||z||_0"] / layers["||z||_1"]
        tag = "aer" if key.endswith("aer") else "dense"
        layers.to_csv(out / f"slista_layers_{tag}_seed{args.seed}.csv", index=False)
        print(f"\nS-LISTA ({key}) per sample (25 frames): ||y||_0 = "
              f"{stats['y_nnz']:.1f}, ||m||_0 = {stats['mask_nnz']:.1f}")
        print(layers.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    if not args.skip_quant:
        print("\n=== INT8 post-training quantization, AWGN 10 dB ===", flush=True)
        run_quant(args.seed, out)
    print(f"\n[Done] {out}")


if __name__ == "__main__":
    main()
