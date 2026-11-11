"""Energy model of the paper (Supplementary Note 2, Energy estimation) for the synthetic task.

Pure Python (no torch): every function takes per-sample operation statistics and
returns an itemised Ledger.  energy_eval.py collects the statistics from the
trained models; consistency checks are in selfcheck().

Units: pJ per sample in the Ledger, converted to uJ for reporting.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Prices:
    """Unit prices, 45 nm (Horowitz 2014)."""
    ac: float = 0.1      # INT8 weight into INT32 accumulator
    mac: float = 0.3     # INT8 x INT8, INT32 accumulate
    add: float = 0.1     # element-wise add / sub / compare
    mul: float = 3.1     # element-wise multiply (32-bit)
    byte: float = 2.5    # SRAM read or write, per byte
    eta_as_mac: bool = False   # check 1 only: readout eta multiply = one MAC


SPEC = Prices()
# Operation-count prices of evaluate.py (4.6 / 0.9 pJ, no memory); used by check 1 only.
OPCOUNT = Prices(ac=0.9, mac=4.6, add=0.0, mul=0.0, byte=0.0, eta_as_mac=True)

BIAS_BYTES = 4        # bias kept across layers (I1 = P y, b = W y), per element
ELEM_BYTES = 1        # activations, codes, spike addresses, inputs, per element


@dataclass
class Ledger:
    """Energy split of the reported tables: compute vs memory (weights, state)."""
    compute: float = 0.0          # pJ
    weight_bytes: float = 0.0
    state_bytes: float = 0.0
    notes: dict = field(default_factory=dict)

    def total(self, p):
        return self.compute + (self.weight_bytes + self.state_bytes) * p.byte

    def summary(self, p):
        return {"compute_uj": self.compute * 1e-6,
                "weight_uj": self.weight_bytes * p.byte * 1e-6,
                "state_uj": self.state_bytes * p.byte * 1e-6,
                "memory_uj": (self.weight_bytes + self.state_bytes) * p.byte * 1e-6,
                "total_uj": self.total(p) * 1e-6}

    # ---------------------------------------------------------------- units
    def u1(self, a, b, p):
        """Dense matrix-vector product, real input a -> b."""
        self.compute += a * b * p.mac
        self.weight_bytes += a * b
        self.state_bytes += a * ELEM_BYTES            # read the input

    def u2(self, b, nnz, mag, p):
        """Event-driven accumulation of an integer code (nnz = ||z||_0, mag = ||z||_1)
        into b outputs: one weight column read per nonzero, |z_j| additions."""
        self.compute += b * mag * p.ac
        self.weight_bytes += b * nnz
        self.state_bytes += nnz * ELEM_BYTES          # read the events

    def add(self, count, p):
        self.compute += count * p.add

    def mul(self, count, p):
        self.compute += count * p.mul

    def rw(self, nbytes):
        self.state_bytes += nbytes


# ----------------------------------------------------------------- S-LISTA
def slista(m, n, layers, mask_nnz, p):
    """layers: one dict per spiking layer l = 1..L-1 with per-sample means
    xi0 = ||xi^(l)||_0, z0 = ||z^(l)||_0, z1 = ||z^(l)||_1.
    mask_nnz: ||m||_0 of the readout.  Input y is real (toy): P y by U1."""
    led = Ledger()
    led.u1(m, n, p)                                   # I1 = P y
    led.rw(BIAS_BYTES * n)                            # write I1 back once
    for index, layer in enumerate(layers):            # spiking layer l = index + 1
        if index > 0:                                 # l >= 2: I_l = I1 - G z^(l-1)
            prev = layers[index - 1]
            led.rw(BIAS_BYTES * n)                    # accumulator init: read I1
            led.u2(n, prev["z0"], prev["z1"], p)
        led.add(2 * n, p)                             # threshold (two-sided) in accumulator
        led.rw(2 * layer["xi0"] * ELEM_BYTES)         # code update: read + write
        led.add(layer["xi0"], p)
    last = layers[-1]                                 # readout layer L
    led.rw(BIAS_BYTES * n)
    led.u2(n, last["z0"], last["z1"], p)              # I_L over all n outputs
    if p.eta_as_mac:
        led.compute += mask_nnz * p.mac
    else:
        led.mul(mask_nnz, p)
        led.add(mask_nnz, p)
    led.rw(mask_nnz * ELEM_BYTES)                     # write the gated outputs
    return led


# ------------------------------------------------------------- ANN LISTA
def lista(m, n, depth, p):
    led = Ledger()
    led.u1(m, n, p)                                   # b = W y
    led.rw(BIAS_BYTES * n)                            # write b back once
    for layer in range(depth):
        if layer > 0:
            led.rw(BIAS_BYTES * n)                    # accumulator init: read b
            led.u1(n, n, p)                           # S z
        led.add(2 * n, p)                             # soft threshold
        led.rw(n * ELEM_BYTES)                        # write the layer output
    return led


# ---------------------------------------------------------------- ALISTA
def alista(m, n, depth, p):
    """r = A z - y (layer 1: z = 0, r = -y); z = soft(z - gamma * W r).  Accounting:
    y read once per layer (1 byte/element); from layer 2 r is written once and
    read once as the input of W, and z_(k-1) is read twice (input of A z and in
    z - gamma * step); gamma: n E_MUL per layer, plus n E_ADD from layer 2."""
    led = Ledger()
    for layer in range(depth):
        if layer == 0:
            led.u1(m, n, p)                           # W y
        else:
            led.rw(m * ELEM_BYTES)                    # accumulator init: read y
            led.u1(n, m, p)                           # A z
            led.rw(m * ELEM_BYTES)                    # write residual
            led.u1(m, n, p)                           # W residual (reads residual)
            led.rw(n * ELEM_BYTES)                    # read z to add it
            led.add(n, p)
        led.mul(n, p)                                 # gamma scaling
        led.add(2 * n, p)                             # soft threshold
        led.rw(n * ELEM_BYTES)                        # write z
    return led


# ------------------------------------------------------------------ LAMP
def lamp(m, n, depth, p):
    """pre = z + B_k r; thr = alpha_k ||r|| / sqrt(m); z = soft(pre);
    r = y - A z + (||pre > thr||_0 / m) r  (layers 1..L-1).  Accounting:
    - y read once per layer (layer 1 as the input of B_1 y, later at the
      accumulator initialisation of r);
    - r written once per layer (1..L-1), read once by the Onsager term and once
      as the input of the next B_k r;
    - ||r||: m E_MUL + m E_ADD, plus one E_MUL for alpha_k * scale; the read of
      r is shared with the B_k r input;
    - Onsager: divergence n E_ADD + 1 E_MUL; divergence * r: m E_MUL + m E_ADD;
    - no separate read of z / addition is counted for pre = z + B_k r (layers
      >= 2)."""
    led = Ledger()
    for layer in range(depth):
        led.u1(m, n, p)                               # B_k r (r = y in layer 1)
        led.mul(m + 1, p)                             # ||r||^2 and alpha_k * scale
        led.add(m, p)
        led.add(2 * n, p)                             # soft threshold (incl. |pre| > thr)
        led.rw(n * ELEM_BYTES)                        # write z
        if layer + 1 < depth:
            led.add(n, p)                             # divergence: count nonzeros
            led.mul(1, p)                             # ... / M
            led.rw(m * ELEM_BYTES)                    # accumulator init: read y
            led.u1(n, m, p)                           # A z
            led.rw(m * ELEM_BYTES)                    # read previous r (Onsager)
            led.mul(m, p)                             # divergence * r
            led.add(m, p)
            led.rw(m * ELEM_BYTES)                    # write new r
    return led


# ------------------------------------------------------------------ S-LCA
def slca(m, n, steps, spikes, p):
    """Signed LIF S-LCA with 2n neurons, K = steps simulation steps, `spikes`
    spikes in total (per sample).  Accounting:
    - drive b = A^T y once: U1(m -> n) (the negated half is free);
    - every neuron, every step: LIF (1 E_MUL + 2 E_ADD, membrane read
      and write 2 bytes each); reading b each step is not counted;
    - every spike: one inhibition column of 2n - 1 INT8 weights, 2n - 1 E_AC,
      1-byte event read; the membrane update is already counted above;
    - final code from spike counts = dictionary readout, not counted."""
    led = Ledger()
    led.u1(m, n, p)
    neurons = 2 * n
    led.mul(steps * neurons, p)
    led.add(2 * steps * neurons, p)
    led.rw(4 * steps * neurons)
    led.u2(neurons - 1, spikes, spikes, p)
    return led


ANN = {"lista": lista, "alista": alista, "lamp": lamp}


# ------------------------------------------------------------- self-checks
def opcount_ann_energy_uj(method, m, n, depth):
    """evaluate.py: macs_per_sample * 4.6 pJ."""
    macs = m * n + (depth - 1) * n * n if method == "lista" else (2 * depth - 1) * m * n
    return macs * 4.6e-6


def selfcheck(m=141, n=256):
    out = {}
    # 1. operation-count prices reproduce evaluate.py (ANN here; S-LISTA in energy_eval.py)
    worst = 0.0
    for method, fn in ANN.items():
        for depth in range(2, 11):
            new = fn(m, n, depth, OPCOUNT).total(OPCOUNT) * 1e-6
            ref = opcount_ann_energy_uj(method, m, n, depth)
            worst = max(worst, abs(new - ref) / ref)
    out["check1_ann_max_rel_err"] = worst
    # 2. an all-ones code: U2 must equal U1 with E_MAC replaced by E_AC
    a, b = Ledger(), Ledger()
    a.u2(n, n, n, SPEC)
    b.u1(n, n, Prices(mac=SPEC.ac))
    out["check2_u2_equals_u1"] = (a.compute == b.compute and a.weight_bytes == b.weight_bytes
                                  and a.state_bytes == b.state_bytes)
    # 3. ANN LISTA: weight-read energy / compute energy ~ 2.5 / 0.3
    led = lista(m, n, 10, SPEC)
    out["check3_lista_weight_over_compute"] = led.weight_bytes * SPEC.byte / led.compute
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(selfcheck(), indent=2))
    for method, fn in ANN.items():
        for depth in (2, 4, 6, 8, 10):
            print(method, depth, {k: round(v, 4) for k, v in fn(141, 256, depth, SPEC).summary(SPEC).items()})
