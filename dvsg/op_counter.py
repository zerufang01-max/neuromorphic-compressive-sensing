"""Operation counts (4.6 pJ per MAC, 0.9 pJ per AC), one convention for all methods.

Used by ``main.py analysis``; the energy model of the paper is in energy_model.py
(energy_eval.py), which checks its operation counts against these.

- the first transmitter convolution receives real-valued frames, so it is counted
  as dense MAC for every method;
- ANN LISTA counts the 7 effective recurrent convolutions; S_k[0] acts on the
  all-zero initial code and is excluded;
- dense ANN layers are MAC; layers driven by binary spikes or integer
  codes are AC, with an input of magnitude n costing n additions per connection;
- the dictionary readout (tail / transposed convolutions), communication, memory
  access and the downstream classifier are excluded.

S-LISTA receiver counts come from SpikingLISTA.frame_counts (P and G as AC on
the received bits and integer codes, eta * I^(L) on the support as MAC).
"""

import torch
import torch.nn as nn

import config as C


class OperationCounter:
    def __init__(self):
        self.counts = {("sender", "mac"): 0.0, ("sender", "ac"): 0.0,
                       ("recon", "mac"): 0.0, ("recon", "ac"): 0.0}
        self._pending = []
        self.handles = []

    def _hook(self, scope, rule):
        def hook(module, inputs, output):
            x = inputs[0]
            y = output[0] if isinstance(output, tuple) else output
            kernel = 1
            for k in module.kernel_size:
                kernel *= int(k)
            dense = float(y.numel() * (module.in_channels // module.groups) * kernel)
            if rule == "mac":
                self.counts[(scope, "mac")] += dense
            else:   # event-driven: dense x mean |input| = fan_out * sum |input|
                self._pending.append((scope, dense * x.detach().abs().sum() / max(1, x.numel())))
        return hook

    def attach(self, experiment, model, sensor):
        convs = [m for m in sensor.encoder if isinstance(m, nn.Conv2d)]
        for index, conv in enumerate(convs):
            rule = "mac" if (index == 0 or experiment.mode == "ann") else "ac"
            self.handles.append(conv.register_forward_hook(self._hook("sender", rule)))
        if experiment.method == "vanilla":
            plan = {"head_1": "ac", "head_2": "mac", "res1.conv1": "ac", "res1.conv2": "ac"}
        elif experiment.method == "ann_lista":
            plan = {"W_e": "mac", **{f"S_k.{k}": "mac" for k in range(1, C.Config.NUM_LAYERS)}}
        elif experiment.method == "ann_lstm":
            plan = {"lstm_cell.conv": "mac"}
        else:
            plan = {}     # S-LISTA: SpikingLISTA.frame_counts
        modules = dict(model.named_modules())
        for name, rule in plan.items():
            self.handles.append(modules[name].register_forward_hook(self._hook("recon", rule)))
        return self

    def close(self):
        for handle in self.handles:
            handle.remove()
        for scope, value in self._pending:
            self.counts[(scope, "ac")] += float(value)
        self._pending = []


def energy_row(experiment, counter, result):
    """Per-sample counts (MOP) and energies (uJ) from one evaluation pass."""
    samples = float(result["samples"])
    s_mac = counter.counts[("sender", "mac")] / samples
    s_ac = counter.counts[("sender", "ac")] / samples
    r_mac = counter.counts[("recon", "mac")] / samples
    r_ac = counter.counts[("recon", "ac")] / samples
    if experiment.method == "slista":
        r_mac += result["rx_mac_per_sample"]
        r_ac += result["rx_ac_per_sample"]
    pj = lambda mac, ac: (C.ENERGY_MAC_PJ * mac + C.ENERGY_AC_PJ * ac) / 1e6   # noqa: E731
    return {"method": experiment.display, "experiment": experiment.key,
            "sender_mac_mop": s_mac / 1e6, "sender_ac_mop": s_ac / 1e6, "sender_uj": pj(s_mac, s_ac),
            "recon_mac_mop": r_mac / 1e6, "recon_ac_mop": r_ac / 1e6, "recon_uj": pj(r_mac, r_ac),
            "total_uj": pj(s_mac + r_mac, s_ac + r_ac),
            "rx_rate": result.get("rx_rate", float("nan")), "tx_rate": result.get("tx_rate", float("nan")),
            "bits": result["bits"]}


def count_energy(experiment, model, sensor, loader, device):
    from channels import make_channel
    from trainer import evaluate
    counter = OperationCounter().attach(experiment, model, sensor)
    try:
        with torch.no_grad():
            result = evaluate(model, sensor, make_channel("awgn", C.TRAIN_SNR_DB, experiment.transport),
                              loader, experiment, device)
    finally:
        counter.close()
    return energy_row(experiment, counter, result)
