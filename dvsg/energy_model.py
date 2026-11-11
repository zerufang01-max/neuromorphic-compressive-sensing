"""DVS-G energy accounting (energy model of Supplementary Note 2, Energy estimation).

Pure Python: turns per-sample event statistics (collected by energy_eval.py) into
operation and byte counts, and prices them.  No PyTorch here, so the rules can be
checked without a GPU.

Units: E_AC 0.1, E_MAC 0.3, E_ADD 0.1, E_MUL 3.1 pJ, SRAM 2.5 pJ/byte.
Every quantity is per sample (16 frames summed); one sample at a time, no weight
reuse across samples, time steps or spatial positions.

Kinds of amounts:  mac, ac, add, mul (operations), wbytes (weight reads),
sbytes (state / activation / event reads and writes).

Architecture:
  transmitter   conv1 2->16 s2 (128->64), BN, LIF | conv2 16->32 (64), BN, LIF |
                conv3 32->C s2 (64->32), sigmoid, output LIF (rho 0.1)
                ANN methods: ReLU instead of the two LIFs, sigmoid + 8-bit quantiser
  S-LISTA       P C->128, G^(1..7) 128->128 (3x3, 32x32 grid), readout layer, tail (not counted)
  Vanilla SNN   head_1 C->64 (spikes in), BN, ReLU | head_2 64->128, BN, LIF |
                res1: conv1 128->128, BN, LIF, conv2 128->128, BN, + identity, LIF | up-sampling tail
  ANN LISTA     W_e C->128, S_k[1..7] 128->128, soft threshold (S_k[0] acts on z = 0: not counted)
  ANN LSTM      ConvLSTM cell conv (C + 128) -> 4 x 128, tail
BatchNorm at inference is folded into the preceding convolution (per-channel scale into
the weights, shift = bias, not counted).  The dictionary readout of every
method (S-LISTA / ANN LISTA / LSTM tail, Vanilla conv_up1, conv_up2, tail) is not counted
; its cost is reported separately for information.

Conventions used here: event convolutions with stride 2, spike-output writes, S-LISTA
membrane memory (rho = 0.3), input reads of dense convolutions, memoryless internal
transmitter LIFs.
"""

GRID = 32                    # code / measurement grid
CODE = 128                   # S-LISTA / ANN LISTA code channels, Vanilla and LSTM hidden channels
K2 = 9                       # 3 x 3 kernels
FRAMES = 16

PRICES = {"mac": 0.3, "ac": 0.1, "add": 0.1, "mul": 3.1, "wbytes": 2.5, "sbytes": 2.5}      # pJ
OPCOUNT_PRICES = {"mac": 4.6, "ac": 0.9, "add": 0.0, "mul": 0.0, "wbytes": 0.0, "sbytes": 0.0}
COMPUTE_KINDS = ("mac", "ac", "add", "mul")


class Ledger:
    """Per-sample amounts, one entry per (scope, component, kind)."""

    def __init__(self):
        self.rows = []

    def add(self, scope, component, kind, amount):
        if amount:
            self.rows.append((scope, component, kind, float(amount)))

    # ---- units ------------------------------------------------------------
    def dense_conv(self, scope, name, cin, cout, out_hw, in_hw, frames):
        """U1 / dense convolution: MAC = weight reads = Cin K^2 Cout Hout Wout; input read once."""
        ops = cin * K2 * cout * out_hw * frames
        self.add(scope, name, "mac", ops)
        self.add(scope, name, "wbytes", ops)
        self.add(scope, name, "sbytes", cin * in_hw * frames)

    def event_conv(self, scope, name, cout, events, magnitude, taps=1.0):
        """U2 / event-driven convolution: per input event Cout * K^2 * taps weights read,
        |z| accumulations per weight; the event itself read once (1 byte).
        taps = Hout Wout / (Hin Win): 1 for the stride-1 'same' convolutions; 1/4 for the
        stride-2 transmitter convolution (an event reaches 9/4 kernel taps on average)."""
        self.add(scope, name, "ac", cout * K2 * taps * magnitude)
        self.add(scope, name, "wbytes", cout * K2 * taps * events)
        self.add(scope, name, "sbytes", events)

    def lif(self, scope, name, neurons, spikes):
        """Section 3.4: 1 E_MUL + 2 E_ADD per neuron and step, membrane read + write 2 bytes
        each; the output spikes are written as events (1 byte each)."""
        self.add(scope, name, "mul", neurons)
        self.add(scope, name, "add", 2 * neurons)
        self.add(scope, name, "sbytes", 4 * neurons)
        self.add(scope, name + " spikes out", "sbytes", spikes)

    # ---- totals -----------------------------------------------------------
    def energy(self, prices=PRICES, scope=None):
        """{'compute', 'weights', 'state', 'total'} in microjoules per sample."""
        out = {"compute": 0.0, "weights": 0.0, "state": 0.0}
        for s, _, kind, amount in self.rows:
            if scope is not None and s != scope:
                continue
            pj = amount * prices[kind]
            out["compute" if kind in COMPUTE_KINDS else "weights" if kind == "wbytes" else "state"] += pj
        out = {k: v / 1e6 for k, v in out.items()}
        out["total"] = out["compute"] + out["weights"] + out["state"]
        return out

    def amount(self, kind, scope=None):
        return sum(a for s, _, k, a in self.rows if k == kind and (scope is None or s == scope))


# ============================================================== transmitter
def transmitter(ledger, stats, meas_channels, spiking, frames=FRAMES):
    """stats (per sample, summed over frames): spikes1, spikes2 (internal LIF outputs),
    tx_spikes (output LIF).  Internal transmitter LIFs have decay 0 in the trained
    models (no membrane memory); they are still charged as LIFs."""
    s = "sender"
    m = meas_channels * GRID * GRID
    ledger.dense_conv(s, "conv1 (real frames, dense)", 2, 16, 64 * 64, 128 * 128, frames)
    if spiking:
        ledger.lif(s, "LIF1", 16 * 64 * 64 * frames, stats["spikes1"])
        ledger.event_conv(s, "conv2 (events)", 32, stats["spikes1"], stats["spikes1"])
        ledger.lif(s, "LIF2", 32 * 64 * 64 * frames, stats["spikes2"])
        ledger.event_conv(s, "conv3 (events, stride 2)", meas_channels, stats["spikes2"], stats["spikes2"],
                          taps=(GRID * GRID) / (64 * 64))
        ledger.add(s, "sigmoid (LUT)", "add", m * frames)
        ledger.lif(s, "output LIF", m * frames, 0)            # spikes to the channel: not counted
    else:
        a1, a2 = 16 * 64 * 64 * frames, 32 * 64 * 64 * frames
        ledger.add(s, "ReLU1", "add", a1)
        ledger.add(s, "act1 write", "sbytes", a1)
        ledger.dense_conv(s, "conv2 (dense)", 16, 32, 64 * 64, 64 * 64, frames)
        ledger.add(s, "ReLU2", "add", a2)
        ledger.add(s, "act2 write", "sbytes", a2)
        ledger.dense_conv(s, "conv3 (dense)", 32, meas_channels, GRID * GRID, 64 * 64, frames)
        ledger.add(s, "sigmoid + 8-bit quantiser (LUT)", "add", 2 * m * frames)


# ============================================================== receivers
def slista(ledger, stats, meas_channels, rho, opcount=False, membrane=True, frames=FRAMES):
    """S-LISTA per frame, summed over frames.
    stats: y_events, y_magnitude, xi_nnz[7], z_nnz[7], z_l1[7], mask (per sample, frame sums).
    opcount: readout gain eta counted as one MAC per gated coordinate (self-check 1).
    membrane (rho != 0): the spiking layers keep u^(l)_{t-1} across frames, charged
    like an LIF membrane: per neuron and frame rho*u (1 E_MUL) + add to the
    current (1 E_ADD) and read 2 bytes when a previous state exists, write 2 bytes every
    frame, and the soft reset u - theta*xi (1 E_ADD per spike)."""
    r = "recon"
    n = CODE * GRID * GRID
    L1 = len(stats["z_nnz"])                                  # spiking layers (7)
    ledger.event_conv(r, "P y (received spikes)", CODE, stats["y_events"], stats["y_magnitude"])
    ledger.add(r, "I(1) write (4 B)", "sbytes", 4 * n * frames)
    ledger.add(r, "layer 1 threshold", "add", 2 * n * frames)
    for l in range(1, L1 + 1):                                # layers 2..L-1 and the readout layer L
        name = f"layer {l + 1}" if l < L1 else "readout layer"
        ledger.add(r, f"{name}: I(1) read (4 B)", "sbytes", 4 * n * frames)
        ledger.event_conv(r, f"{name}: G({l}) z({l})", CODE, stats["z_nnz"][l - 1], stats["z_l1"][l - 1])
        if l < L1:
            ledger.add(r, f"{name} threshold", "add", 2 * n * frames)
    for l in range(L1):
        xi = stats["xi_nnz"][l]
        ledger.add(r, f"code update {l + 1}", "add", xi)
        ledger.add(r, f"code update {l + 1}", "sbytes", 2 * xi)
    if opcount:
        ledger.add(r, "readout eta * I(L) (as MAC)", "mac", stats["mask"])
    else:
        ledger.add(r, "readout eta * I(L)", "mul", stats["mask"])
        ledger.add(r, "readout eta * I(L)", "add", stats["mask"])
        ledger.add(r, "readout write", "sbytes", stats["mask"])
    if rho and membrane:
        neurons = L1 * n
        ledger.add(r, "membrane rho*u + I", "mul", neurons * (frames - 1))
        ledger.add(r, "membrane rho*u + I", "add", neurons * (frames - 1))
        ledger.add(r, "membrane read (2 B)", "sbytes", 2 * neurons * (frames - 1))
        ledger.add(r, "membrane write (2 B)", "sbytes", 2 * neurons * frames)
        ledger.add(r, "membrane soft reset", "add", sum(stats["xi_nnz"]))


def vanilla(ledger, stats, meas_channels, frames=FRAMES):
    """stats: y_events, y_magnitude, spikes_head, spikes_res1, spikes_res2 (per sample)."""
    r = "recon"
    n = CODE * GRID * GRID
    feat = 64 * GRID * GRID
    ledger.event_conv(r, "head_1 (received spikes)", 64, stats["y_events"], stats["y_magnitude"])
    ledger.add(r, "head_1 ReLU", "add", feat * frames)
    ledger.add(r, "head_1 act write", "sbytes", feat * frames)
    ledger.dense_conv(r, "head_2 (dense)", 64, CODE, GRID * GRID, GRID * GRID, frames)
    ledger.lif(r, "LIF head", n * frames, stats["spikes_head"])
    ledger.event_conv(r, "res1.conv1 (events)", CODE, stats["spikes_head"], stats["spikes_head"])
    ledger.lif(r, "LIF res1", n * frames, stats["spikes_res1"])
    ledger.event_conv(r, "res1.conv2 (events)", CODE, stats["spikes_res1"], stats["spikes_res1"])
    ledger.add(r, "residual add (identity events)", "add", stats["spikes_head"])
    ledger.add(r, "residual add (identity events)", "sbytes", stats["spikes_head"])
    ledger.lif(r, "LIF res2", n * frames, stats["spikes_res2"])


def ann_lista(ledger, meas_channels, layers=8, frames=FRAMES):
    """Section 3.2: W_e y once, S_k[1..L-1] (S_k[0] acts on z = 0)."""
    r = "recon"
    n = CODE * GRID * GRID
    ledger.dense_conv(r, "W_e y", meas_channels, CODE, GRID * GRID, GRID * GRID, frames)
    ledger.add(r, "b = W_e y write (4 B)", "sbytes", 4 * n * frames)
    ledger.add(r, "layer 1 soft threshold", "add", 2 * n * frames)
    ledger.add(r, "layer 1 output write", "sbytes", n * frames)
    for k in range(1, layers):
        ledger.add(r, f"layer {k + 1}: b read (4 B)", "sbytes", 4 * n * frames)
        ledger.dense_conv(r, f"layer {k + 1}: S_{k} z", CODE, CODE, GRID * GRID, GRID * GRID, frames)
        ledger.add(r, f"layer {k + 1} soft threshold", "add", 2 * n * frames)
        ledger.add(r, f"layer {k + 1} output write", "sbytes", n * frames)


def ann_lstm(ledger, meas_channels, opcount=False, frames=FRAMES):
    """ConvLSTM per time step; W_hh h_0 with h_0 = 0 and f * c_0 with c_0 = 0 are not
    counted; opcount keeps the full first-step convolution (as op_counter.py)."""
    r = "recon"
    n = CODE * GRID * GRID
    full_cin = meas_channels + CODE
    if opcount:
        ledger.dense_conv(r, "cell conv [y, h]", full_cin, 4 * CODE, GRID * GRID, GRID * GRID, frames)
        return
    ledger.dense_conv(r, "cell conv [y, h] (t >= 2)", full_cin, 4 * CODE, GRID * GRID, GRID * GRID, frames - 1)
    ledger.dense_conv(r, "cell conv [y] (t = 1, h0 = 0)", meas_channels, 4 * CODE, GRID * GRID, GRID * GRID, 1)
    ledger.add(r, "gate LUTs (3 sigmoid + tanh)", "add", 4 * n * frames)
    ledger.add(r, "tanh(c) LUT", "add", n * frames)
    ledger.add(r, "c = f*c + i*g", "mul", 2 * n * (frames - 1) + n)
    ledger.add(r, "c = f*c + i*g", "add", n * (frames - 1))
    ledger.add(r, "h = o*tanh(c)", "mul", n * frames)
    ledger.add(r, "c read", "sbytes", n * (frames - 1))
    ledger.add(r, "c write", "sbytes", n * frames)
    ledger.add(r, "h write", "sbytes", n * frames)


def build(method, stats, meas_channels, rho=0.0, opcount=False, membrane=True):
    """Ledger of one model (transmitter + receiver)."""
    ledger = Ledger()
    transmitter(ledger, stats.get("sender", {}), meas_channels, spiking=method in ("slista", "vanilla"))
    if method == "slista":
        slista(ledger, stats["recon"], meas_channels, rho, opcount=opcount, membrane=membrane)
    elif method == "vanilla":
        vanilla(ledger, stats["recon"], meas_channels)
    elif method == "ann_lista":
        ann_lista(ledger, meas_channels)
    elif method == "ann_lstm":
        ann_lstm(ledger, meas_channels, opcount=opcount)
    else:
        raise ValueError(method)
    return ledger


def opcount_counts(ledger):
    """Operation counts (MOP) and energies (uJ) at the operation-count prices, for self-check 1."""
    out = {}
    for scope in ("sender", "recon"):
        mac, ac = ledger.amount("mac", scope), ledger.amount("ac", scope)
        out[f"{scope}_mac_mop"] = mac / 1e6
        out[f"{scope}_ac_mop"] = ac / 1e6
        out[f"{scope}_uj"] = ledger.energy(OPCOUNT_PRICES, scope)["total"]
    out["total_uj"] = out["sender_uj"] + out["recon_uj"]
    return out


def readout_estimate(macs, input_bytes):
    """Cost of the dictionary readout if it were counted as dense convolutions (U1):
    MAC + one weight read per MAC + input reads.  Information only."""
    return (macs * (PRICES["mac"] + PRICES["wbytes"]) + input_bytes * PRICES["sbytes"]) / 1e6
