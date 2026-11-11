#!/usr/bin/env python3
"""ECG energy accounting (energy model of Supplementary Note 2, Energy estimation).

Analytical estimate: the trained models of outputs/seed*/learned are run on
the test set (DS2, AWGN at the training SNR, test channel seed, as in the
reported results), operations and memory traffic are counted per sample, and
multiplied by unit costs.  The dictionary readout z D is excluded for every
method.

Unit costs (pJ): E_AC 0.1, E_MAC 0.3, E_ADD 0.1, E_MUL 3.1,
E_B 2.5 per byte.  Byte widths: INT8 weights, 1-byte activations / codes /
events / inputs, 4-byte biases kept across layers (I^(1), W y).
Accumulator convention: the bias I^(1) = P y or b = W y is
thresholded in the accumulator at layer 1 (no read), written back once
(4n bytes) and read once per layer from layer 2 on.  The S-LISTA readout
current is computed for all n coordinates.

The script also
  * runs three self-checks:
      1. operation-count unit costs (E_B = 0, E_AC = 0.9, E_MAC = 4.6, E_ADD = E_MUL = 0,
         readout multiply eta*v counted as one MAC, as in the training pipeline)
         against the Energy_uJ stored in results.csv;
      2. U2 with an all-ones code equals U1 with E_MAC replaced by E_AC;
      3. weight-read / compute energy of ANN LISTA (about 2.5 / 0.3 = 8.3);
  * evaluates INT8 post-training quantization: per-output-
    channel symmetric INT8 weights, dequantized, test set re-evaluated, no
    retraining.  Variant "net" quantizes the reconstruction weights that the
    energy table counts (S-LISTA P, G; ANN LISTA W, S; LAMP B_k and the
    operator A = Phi D^T; ALISTA W and A).  Variant "all" also quantizes the
    sensing matrix Phi and the dictionary D;
  * records, for S-LISTA, per-layer ||xi||_0, ||z||_0, ||z||_1 and ||m||_0.

Usage:
  python energy_ecg.py --output outputs --gpu 0 --save energy > energy.log 2>&1
"""
import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

TARGETS = ["alista", "lamp", "lista_ann", "lista_snn_1"]
LABELS = {"alista": "ALISTA", "lamp": "LAMP", "lista_ann": "ANN LISTA", "lista_snn_1": "S-LISTA"}


# --------------------------------------------------------------------- costs
@dataclass(frozen=True)
class Costs:
    name: str
    ac: float
    mac: float
    add: float
    mul: float
    byte: float


FULL = Costs("full model", ac=0.1, mac=0.3, add=0.1, mul=3.1, byte=2.5)
OPCOUNT = Costs("operation count (4.6/0.9, no memory)", ac=0.9, mac=4.6, add=0.0, mul=0.0, byte=0.0)
# The training pipeline counts the S-LISTA readout gain eta*v as a MAC.
OPCOUNT_READOUT_AS_MAC = Costs("operation count, readout multiply as MAC", ac=0.9, mac=4.6, add=0.0, mul=4.6, byte=0.0)


class Count:
    """Operation and byte counts for one sample ."""

    def __init__(self):
        self.mac = self.ac = self.add = self.mul = 0.0
        self.weight_bytes = self.state_bytes = 0.0

    # U1: dense matrix-vector product, real input, a -> b
    def u1(self, a, b):
        self.mac += a * b
        self.weight_bytes += a * b
        self.state_bytes += a

    # U2: event-driven accumulation, input with N nonzeros and |.|-sum M, a -> b
    def u2(self, b, nonzeros, magnitude):
        self.ac += b * magnitude
        self.weight_bytes += b * nonzeros
        self.state_bytes += nonzeros

    # U4: state / vector reads and writes
    def rw(self, nbytes):
        self.state_bytes += nbytes

    def energy_pj(self, c):
        compute = self.mac * c.mac + self.ac * c.ac + self.add * c.add + self.mul * c.mul
        weights = self.weight_bytes * c.byte
        state = self.state_bytes * c.byte
        return {"compute": compute, "weights": weights, "state": state,
                "memory": weights + state, "total": compute + weights + state}

    def as_dict(self):
        return {"mac": self.mac, "ac": self.ac, "add": self.add, "mul": self.mul,
                "weight_bytes": self.weight_bytes, "state_bytes": self.state_bytes}


# ------------------------------------------------------- per-method counting
def count_slista(m, n, z0, z1, xi0, gate):
    """z0[l], z1[l], xi0[l]: mean ||z^(l)||_0, ||z^(l)||_1,
    ||xi^(l)||_0 of the L-1 spiking layers; gate: mean ||m||_0."""
    c = Count()
    spiking = len(z0)
    c.u1(m, n)                      # I^(1) = P y, y real valued
    c.rw(4 * n)                     # write I^(1) back once
    c.add += 2 * n                  # layer 1: threshold in the accumulator (no read)
    for layer in range(2, spiking + 2):          # layers 2 .. L (L = readout)
        c.rw(4 * n)                 # accumulator initialised with I^(1)
        c.u2(n, z0[layer - 2], z1[layer - 2])    # G^(l-1) z^(l-1)
        if layer <= spiking:
            c.add += 2 * n          # threshold on the accumulator
    for l in range(spiking):        # code update z^(l) = z^(l-1) + xi^(l)
        c.rw(2 * xi0[l]); c.add += xi0[l]
    c.mul += gate; c.add += gate; c.rw(gate)     # readout z^(L-1) + eta m I^(L)
    return c


def count_lista(m, n, layers):
    """ANN LISTA: b = W y once; z_k = soft(b + S z_{k-1}).
    Layer 1 soft-thresholds b in the accumulator (no read); b is written back
    once and read from layer 2 on (same as I^(1) in S-LISTA)."""
    c = Count()
    c.u1(m, n); c.rw(4 * n)         # b = W y, write 4n
    for k in range(1, layers + 1):
        if k >= 2:
            c.rw(4 * n)             # read b into the accumulator
            c.u1(n, n)              # S z_{k-1}
        c.add += 2 * n              # soft threshold
        c.rw(n)                     # write z_k
    return c


def count_lamp(m, n, layers):
    """LAMP (snn_models.LAMP.forward):
    pseudo = z + B_k r;  thr = theta_k ||r|| / sqrt(M);  z = soft(pseudo, thr)
    r <- y - A z + (||z||_0 / M) r   (k < K)."""
    c = Count()
    for k in range(1, layers + 1):
        c.u1(m, n)                  # B_k r (reads r; r = y at k = 1)
        if k >= 2:
            c.rw(n)                 # accumulator initialised with z_{k-1}
        c.mul += m + 1; c.add += m  # ||r||^2 and theta_k * scale
        c.add += 2 * n; c.rw(n)     # soft threshold, write z_k
        if k < layers:
            c.add += n; c.mul += 1  # divergence: count of active entries / M
            c.u1(n, m)              # A z_k (reads z_k)
            c.rw(m)                 # accumulator initialised with y
            c.rw(m); c.mul += m; c.add += m   # Onsager term: read r, scale, add
            c.rw(m)                 # write the new residual
    return c


def count_alista(m, n, layers):
    """ALISTA (snn_models.ALISTA.forward):
    r = A z - y;  z = soft(z - gamma_k W r, theta_k).  At k = 1, z = 0 and r = -y."""
    c = Count()
    for k in range(1, layers + 1):
        if k >= 2:
            c.u1(n, m)              # A z_{k-1} (reads z)
            c.rw(m)                 # accumulator initialised with -y
            c.rw(m)                 # write r (input of the next product)
        c.u1(m, n)                  # W r (reads r; r = -y at k = 1)
        c.mul += n                  # gamma_k * step
        if k >= 2:
            c.rw(n); c.add += n     # read z_{k-1}, subtract
        c.add += 2 * n; c.rw(n)     # soft threshold, write z_k
    return c


def count_sensing(m, n):
    """Transmitter y = Phi x (same for every method; reported separately)."""
    c = Count()
    c.u1(n, m)
    c.rw(m)
    return c


def self_check_u1_u2(n, spiking_layers, m):
    """Self-check 2.  (a) One unit: U2(n -> n) with an all-ones input
    equals U1(n -> n) with E_MAC replaced by E_AC.  (b) Whole S-LISTA: setting
    every code z^(l) to all ones adds exactly (L-1) such U1 products relative
    to all-zero codes (the G^(l) z^(l) terms)."""
    as_ac = Costs("U1 with E_MAC -> E_AC", ac=FULL.ac, mac=FULL.ac, add=FULL.add, mul=FULL.mul, byte=FULL.byte)
    u2 = Count(); u2.u2(n, n, n)
    u1 = Count(); u1.u1(n, n)
    unit_ok = all(math.isclose(u2.energy_pj(FULL)[k], u1.energy_pj(as_ac)[k]) for k in ("compute", "weights", "state"))
    ones = count_slista(m, n, [float(n)] * spiking_layers, [float(n)] * spiking_layers, [0.0] * spiking_layers, 0.0)
    zeros = count_slista(m, n, [0.0] * spiking_layers, [0.0] * spiking_layers, [0.0] * spiking_layers, 0.0)
    e1, e0, eu = ones.energy_pj(FULL), zeros.energy_pj(FULL), u1.energy_pj(as_ac)
    model_ok = all(math.isclose(e1[k] - e0[k], spiking_layers * eu[k]) for k in ("compute", "weights", "state"))
    return unit_ok and model_ok, {"unit": unit_ok, "model": model_ok,
                                  "U2 all-ones total pJ": u2.energy_pj(FULL)["total"],
                                  "U1 (E_AC) total pJ": u1.energy_pj(as_ac)["total"]}


# ---------------------------------------------------------------- evaluation
def quantize_per_channel(weight, out_axis):
    """Symmetric per-output-channel INT8, dequantized back to float."""
    import torch
    w = weight.detach()
    moved = w.transpose(0, out_axis) if out_axis != 0 else w
    flat = moved.reshape(moved.shape[0], -1)
    scale = flat.abs().amax(dim=1).clamp_min(1e-12) / 127.0
    q = torch.clamp(torch.round(flat / scale[:, None]), -127, 127) * scale[:, None]
    q = q.reshape(moved.shape)
    return q.transpose(0, out_axis) if out_axis != 0 else q


def apply_ptq(obj, variant):
    """Quantize the weights of a loaded model in place (variant 'net' or 'all')."""
    import torch
    model, sensor = obj.model, obj.sensor
    with torch.no_grad():
        if variant == "all":
            sensor.Phi.copy_(quantize_per_channel(sensor.Phi, 0))    # y = Phi x: rows
            model.D.copy_(quantize_per_channel(model.D, 1))           # x = z D: columns
        target = obj.target
        if target == "lista_snn_1":
            model.P.weight.copy_(quantize_per_channel(model.P.weight, 0))
            for g in model.G:
                g.weight.copy_(quantize_per_channel(g.weight, 0))
        elif target == "lista_ann":
            model.W_e.weight.copy_(quantize_per_channel(model.W_e.weight, 0))
            for s in model.S_k:
                s.weight.copy_(quantize_per_channel(s.weight, 0))
        elif target in ("lamp", "alista"):
            if target == "lamp":
                for b in model.B_k:
                    b.weight.copy_(quantize_per_channel(b.weight, 0))
            else:
                model.W_t.copy_(quantize_per_channel(model.W_t, 0))   # step = r W_t^T: rows
            operator = (sensor.effective_phi() @ model.D.t()).detach()
            operator_q = quantize_per_channel(operator, 0)            # z A^T: rows of A
            model.current_operator = lambda phi_override=None, _a=operator_q: _a


def load(folder, target, plot_results, main_mod, torch, Config):
    job = json.loads((folder / "job.json").read_text())
    Config.QUANT_BITS = int(job.get("bits", 8))
    Config.BITS_PER_SAMPLE = Config.M * Config.QUANT_BITS
    Config.CS_MODE = job.get("cs_mode", "learnable_dictionary")
    main_mod.apply_method_overrides(job.get("overrides", {}))
    row = main_mod.read_result(folder)
    model_class, mode = plot_results.MODEL_SPECS[target]
    sensor = plot_results.EdgeSensor().to(Config.DEVICE)
    model = model_class(phi=sensor.effective_phi().detach()).to(Config.DEVICE)
    if hasattr(model, "set_mode"):
        model.set_mode(mode)
    checkpoint = torch.load(folder / row["checkpoint"], map_location=Config.DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    sensor.load_state_dict(checkpoint["sensor"], strict=True)
    channel = plot_results.WirelessChannel(snr_db=float(job.get("train_snr_db", 5.0))).to(Config.DEVICE)
    model.eval(); sensor.eval()
    return SimpleNamespace(model=model, sensor=sensor, channel=channel, row=row, target=target)


def run_test(obj, loader, torch, Config, utils, collect_slista=False):
    """NMSE on the test set with the test channel; S-LISTA code statistics."""
    err = energy = 0.0
    samples = 0
    stats = None
    with utils.deterministic_eval(Config.TEST_CHANNEL_SEED), torch.no_grad():
        for batch in loader:
            x = (batch[0] if isinstance(batch, (tuple, list)) else batch).to(Config.DEVICE)
            y, _, _ = obj.sensor(x, mode="shared")
            y_recv, _ = obj.channel(y, mode="shared", channel_mode=Config.EVAL_CHANNEL_MODE)
            x_hat, _, _, out = obj.model(y_recv, states=None, return_debug=False,
                                         phi_override=obj.sensor.effective_phi())
            err += float(((x_hat - x) ** 2).sum())
            energy += float((x ** 2).sum())
            samples += x.shape[0]
            if collect_slista:
                trace = out["trace"]
                z0 = torch.stack([(c != 0).sum(1) for c in trace["codes"]]).double().sum(1)
                z1 = torch.stack([c.abs().sum(1) for c in trace["codes"]]).double().sum(1)
                xi0 = torch.stack([(s != 0).sum(1) for s in trace["spikes"]]).double().sum(1)
                gate = trace["mask"].double().sum()
                part = {"z0": z0.cpu(), "z1": z1.cpu(), "xi0": xi0.cpu(), "gate": float(gate)}
                if stats is None:
                    stats = part
                else:
                    for key in ("z0", "z1", "xi0"):
                        stats[key] += part[key]
                    stats["gate"] += part["gate"]
    nmse = 10 * math.log10(max(err, 1e-12) / max(energy, 1e-12))
    if stats is not None:
        stats = {"z0": (stats["z0"] / samples).tolist(), "z1": (stats["z1"] / samples).tolist(),
                 "xi0": (stats["xi0"] / samples).tolist(), "gate": stats["gate"] / samples}
    return nmse, stats, samples


def mean_std(values):
    values = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not values:
        return float("nan"), float("nan")
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def fmt(values, digits=3):
    mu, sd = mean_std(values)
    return f"{mu:.{digits}f} ± {sd:.{digits}f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/mitdb_pt"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--save", type=Path, default=Path("energy"))
    parser.add_argument("--skip-ptq", action="store_true")
    args = parser.parse_args()

    import torch
    from config import Config
    import main as main_mod
    import plot_results
    import utils

    plot_results.configure(SimpleNamespace(train_snr_db=5.0, data_dir=args.data_dir.resolve(), gpu=args.gpu))
    Config.NUM_WORKERS = 0
    m, n, layers = Config.M, Config.N, Config.NUM_LAYERS
    spiking = layers - 1
    args.save.mkdir(parents=True, exist_ok=True)

    print(f"ECG energy accounting.  m = {m}, n = {n}, L = {layers}")
    print(f"unit costs (pJ): {FULL}")

    # Self-check 2 (no data needed).
    ok, detail = self_check_u1_u2(n, spiking, m)
    print(f"\n[self-check 2] all-ones code: U2 == U1 with E_MAC -> E_AC: {'PASS' if ok else 'FAIL'}  {detail}")

    _, test_loader = utils.get_dataloaders(eval_split="test")
    fixed_counts = {"lista_ann": count_lista(m, n, layers), "lamp": count_lamp(m, n, layers),
                    "alista": count_alista(m, n, layers)}

    records = {t: [] for t in TARGETS}
    for seed in args.seeds:
        for target in TARGETS:
            folder = (args.output / f"seed{seed}" / "learned" / target).resolve()
            obj = load(folder, target, plot_results, main_mod, torch, Config)
            nmse, stats, samples = run_test(obj, test_loader, torch, Config, utils,
                                            collect_slista=(target == "lista_snn_1"))
            if target == "lista_snn_1":
                count = count_slista(m, n, stats["z0"], stats["z1"], stats["xi0"], stats["gate"])
            else:
                count = fixed_counts[target]
            rec = {"seed": seed, "target": target, "samples": samples,
                   "reported_test_nmse": float(obj.row["Test_NMSE_dB"]), "test_nmse": nmse,
                   "reported_energy_uj": float(obj.row.get("Energy_uJ", "nan") or "nan"),
                   "reported_macs": float(obj.row.get("MACs", "nan") or "nan"),
                   "reported_acs": float(obj.row.get("ACs", "nan") or "nan"),
                   "counts": count.as_dict(), "slista_stats": stats,
                   "energy_pj": count.energy_pj(FULL), "energy_opcount_pj": count.energy_pj(OPCOUNT),
                   "energy_opcount_readout_mac_pj": count.energy_pj(OPCOUNT_READOUT_AS_MAC)}
            if not args.skip_ptq:
                for variant in ("net", "all"):
                    qobj = load(folder, target, plot_results, main_mod, torch, Config)
                    apply_ptq(qobj, variant)
                    rec[f"int8_{variant}_nmse"], _, _ = run_test(qobj, test_loader, torch, Config, utils)
            records[target].append(rec)
            print(f"seed {seed} {LABELS[target]:<10} NMSE {nmse:7.3f} (reported {rec['reported_test_nmse']:7.3f})"
                  f"  energy {rec['energy_pj']['total'] * 1e-6:.4f} uJ"
                  + ("" if args.skip_ptq else f"  INT8 net {rec['int8_net_nmse']:7.3f}  all {rec['int8_all_nmse']:7.3f}"),
                  flush=True)

    (args.save / "records.json").write_text(json.dumps(records, indent=1))

    # ------------------------------------------------------------ self-check 1
    print("\n[self-check 1] operation-count unit costs (E_B = 0, E_AC = 0.9, E_MAC = 4.6, E_ADD = E_MUL = 0,"
          " eta*v as one MAC) vs Energy_uJ in results.csv (uJ, five-seed mean)")
    print(f"  {'method':<10} {'check':>10} {'reported':>10} {'max rel diff':>13}  result  MAC/AC recount vs reported")
    for target in TARGETS:
        rs = records[target]
        recount = [r["energy_opcount_pj"]["total"] * 1e-6 for r in rs]
        # Only the S-LISTA readout multiply is a MAC in the training pipeline; the ANN
        # element-wise multiplies (gamma, Onsager, ||r||) are not counted there.
        key = "energy_opcount_readout_mac_pj" if target == "lista_snn_1" else "energy_opcount_pj"
        recount2 = [r[key]["total"] * 1e-6 for r in rs]
        reported = [r["reported_energy_uj"] for r in rs]
        diff = max(abs(a - b) / max(abs(b), 1e-12) for a, b in zip(recount2, reported))
        mac_new = statistics.mean(r["counts"]["mac"] + (r["counts"]["mul"] if target == "lista_snn_1" else 0) for r in rs)
        print(f"  {LABELS[target]:<10} {statistics.mean(recount2):10.5f} "
              f"{statistics.mean(reported):10.5f} {diff:13.2e}  {'PASS' if diff < 1e-6 else 'FAIL'}"
              f"  MAC {mac_new:.1f}/{statistics.mean(r['reported_macs'] for r in rs):.1f}"
              f"  AC {statistics.mean(r['counts']['ac'] for r in rs):.1f}/{statistics.mean(r['reported_acs'] for r in rs):.1f}")
    print("  (the official results count eta*v as E_MUL + E_ADD; with the operation-count costs that version gives "
          + ", ".join(f"{LABELS[t]} {statistics.mean(r['energy_opcount_pj']['total'] * 1e-6 for r in records[t]):.5f}" for t in TARGETS) + ")")
    print("  NMSE recomputed vs reported (max |diff| over seeds): "
          + ", ".join(f"{LABELS[t]} {max(abs(r['test_nmse'] - r['reported_test_nmse']) for r in records[t]):.1e}"
                      for t in TARGETS))

    # ------------------------------------------------------------ self-check 3
    lc = fixed_counts["lista_ann"]
    e = lc.energy_pj(FULL)
    print(f"\n[self-check 3] ANN LISTA weight-read / compute energy = {e['weights'] / e['compute']:.2f}"
          f"  (MAC-only ratio 2.5/0.3 = {2.5 / 0.3:.2f}; compute includes {lc.add:.0f} ADD)")

    # --------------------------------------------------------------- tables
    print("\n=== Energy per sample (uJ), five seeds (mean ± std), test set")
    print(f"  {'method':<10} {'E_compute':>17} {'E_memory':>17} {'(weights':>9} {'state)':>8} {'E_total':>17} {'/ANN LISTA':>11}")
    ann_total = statistics.mean(r["energy_pj"]["total"] for r in records["lista_ann"])
    table1 = []
    for target in TARGETS:
        rs = records[target]
        comp = [r["energy_pj"]["compute"] * 1e-6 for r in rs]
        mem = [r["energy_pj"]["memory"] * 1e-6 for r in rs]
        wts = [r["energy_pj"]["weights"] * 1e-6 for r in rs]
        st = [r["energy_pj"]["state"] * 1e-6 for r in rs]
        tot = [r["energy_pj"]["total"] * 1e-6 for r in rs]
        ratio = statistics.mean(r["energy_pj"]["total"] for r in rs) / ann_total
        table1.append({"method": LABELS[target], "compute": mean_std(comp), "memory": mean_std(mem),
                       "weights": mean_std(wts), "state": mean_std(st), "total": mean_std(tot), "ratio": ratio})
        print(f"  {LABELS[target]:<10} {fmt(comp, 4):>17} {fmt(mem, 4):>17} {statistics.mean(wts):9.4f} "
              f"{statistics.mean(st):8.4f} {fmt(tot, 4):>17} {ratio:11.3f}")
    sens = count_sensing(m, n).energy_pj(FULL)
    print(f"  (transmitter y = Phi x, same for all methods, not included: {sens['total'] * 1e-6:.4f} uJ)")
    opcount_rows = {t: statistics.mean(r["reported_energy_uj"] for r in records[t]) for t in TARGETS}
    print("  operation-count estimate (results.csv): " + ", ".join(f"{LABELS[t]} {opcount_rows[t]:.3f}" for t in TARGETS))
    s_new = statistics.mean(r["energy_pj"]["total"] for r in records["lista_snn_1"])
    for ref in ("lista_ann", "lamp", "alista"):
        ref_new = statistics.mean(r["energy_pj"]["total"] for r in records[ref])
        print(f"  S-LISTA vs {LABELS[ref]}: {100 * (1 - s_new / ref_new):.1f}% lower"
              f" (operation count {100 * (1 - opcount_rows['lista_snn_1'] / opcount_rows[ref]):.1f}% lower)")

    print("\n  counts per sample (five-seed mean): MAC, AC, ADD, MUL, weight bytes, state bytes")
    for target in TARGETS:
        rs = records[target]
        keys = ("mac", "ac", "add", "mul", "weight_bytes", "state_bytes")
        print(f"  {LABELS[target]:<10} " + "  ".join(f"{k} {statistics.mean(r['counts'][k] for r in rs):,.0f}" for k in keys))

    if not args.skip_ptq:
        print("\n=== INT8 post-training quantization (test NMSE, dB, five seeds)")
        print(f"  {'method':<10} {'float':>17} {'INT8 (network)':>17} {'change':>8} {'INT8 (+Phi, D)':>17} {'change':>8}")
        for target in TARGETS:
            rs = records[target]
            fl = [r["test_nmse"] for r in rs]
            qn = [r["int8_net_nmse"] for r in rs]
            qa = [r["int8_all_nmse"] for r in rs]
            print(f"  {LABELS[target]:<10} {fmt(fl, 2):>17} {fmt(qn, 2):>17} {statistics.mean(qn) - statistics.mean(fl):8.3f}"
                  f" {fmt(qa, 2):>17} {statistics.mean(qa) - statistics.mean(fl):8.3f}")

    print("\n=== Internal: S-LISTA per spiking layer (five-seed mean per sample)")
    rs = records["lista_snn_1"]
    print(f"  {'layer':>5} {'||xi||_0':>10} {'||z||_0':>10} {'||z||_1':>10} {'z0/z1':>7}")
    for l in range(spiking):
        x0 = statistics.mean(r["slista_stats"]["xi0"][l] for r in rs)
        a0 = statistics.mean(r["slista_stats"]["z0"][l] for r in rs)
        a1 = statistics.mean(r["slista_stats"]["z1"][l] for r in rs)
        print(f"  {l + 1:>5} {x0:10.2f} {a0:10.2f} {a1:10.2f} {a0 / max(a1, 1e-12):7.3f}")
    tz0 = statistics.mean(sum(r["slista_stats"]["z0"]) for r in rs)
    tz1 = statistics.mean(sum(r["slista_stats"]["z1"]) for r in rs)
    print(f"  total ||z||_0 {tz0:.1f}, ||z||_1 {tz1:.1f}, ratio {tz0 / tz1:.3f}; "
          f"||m||_0 {statistics.mean(r['slista_stats']['gate'] for r in rs):.1f}; "
          f"code density sum||z||_0/(n(L-1)) {tz0 / (n * spiking):.3f}")
    (args.save / "table1.json").write_text(json.dumps(table1, indent=1))
    print(f"\nSaved {args.save / 'records.json'} and {args.save / 'table1.json'}")


if __name__ == "__main__":
    main()
