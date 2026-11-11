"""Supplementary Fig. 3 of SN3 (SHD) from the csv files of sn3_fig3.py.

Style follows main-text Fig. 4: DejaVu Serif text with STIX mathtext, markers on
lines, framed in-panel legends, light dashed grid, bold lower-case panel letters.

Panels
  a  fraction of samples (solid) and time steps (dashed) whose receiver spikes are
     all unchanged, AWGN and Rayleigh; error bars are CP 95% intervals
  b  distribution of r_{k,i} over all neuron-time-step pairs (r_star = 1 for every
     sample, noted in the panel)
  c  R_t at class-(iii) time steps (median, IQR), AWGN
  d  share of sum ||z' - z||^2 by source, AWGN, log scale (wide)
  e  time-step classes (i)-(iv), AWGN, log scale (wide)
Writes <input>/figures/supp_fig3_shd.{pdf,png} and <input>/fig3_numbers.json.

Usage : python sn3_fig3_plot.py [--input outputs/sn3_fig3]
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
MM = 1 / 25.4
RED = "#D94B3D"            # S-LISTA colour of plot_results.COLORS
CHANNEL = {"awgn": ("#222222", "o", "AWGN"), "rayleigh": (RED, "s", "Rayleigh")}
CLASS_KEYS = ["i_unchanged", "ii_spikes_changed_d0", "iii_same_support", "iv_support_changed"]
CLASS_STYLE = [("#3E86BB", "o", "(i) spikes unchanged"),
               ("#3BA275", "v", r"(ii) spikes changed, $\mathbf{d}_t=\mathbf{0}$"),
               ("#E69F00", "^", r"(iii) $\mathbf{d}_t\neq\mathbf{0}$, $\Lambda_t^{′}=\Lambda_t$"),
               ("#7C5AAC", "D", r"(iv) $\Lambda_t^{′}\neq\Lambda_t$")]
SOURCE_STYLE = [("Out energy (i)+(ii)", "#3E86BB", "o", "-", "(i)+(ii)"),
                ("Out energy (iii)", "#E69F00", "^", "-", "(iii)"),
                ("Out energy (iv) common", "#7C5AAC", "D", "-", r"(iv), on $\Lambda_t\cap\Lambda_t^{′}$"),
                ("Out energy (iv) symdiff", "#555555", "s", "--", r"(iv), on $\Lambda_t\,\triangle\,\Lambda_t^{′}$")]
XLABEL = r"Average $E_b/N_0$ (dB)"
XTICKS = [-5, 0, 5, 10, 15, 20]
MIN_FRAMES_R = 20


FONTS = {
    # Main-text Fig. 4: DejaVu Serif + STIX mathtext.
    "serif": {"font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "stix"},
    # Main-text Fig. 2: DejaVu Sans + DejaVu mathtext.
    "sans": {"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "mathtext.fontset": "dejavusans"},
}


def style(font="sans"):
    plt.rcParams.update(FONTS[font])
    plt.rcParams.update({
        "axes.unicode_minus": True,
        "font.size": 7, "axes.labelsize": 7.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "legend.fontsize": 5.8, "legend.frameon": True, "legend.framealpha": 0.92,
        "legend.edgecolor": "0.75", "legend.fancybox": False, "legend.borderpad": 0.35,
        "legend.handlelength": 1.8, "legend.labelspacing": 0.3,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "lines.linewidth": 1.0, "lines.markersize": 3.0, "lines.markeredgewidth": 0.6,
        "savefig.dpi": 300, "pdf.fonttype": 42,
    })


def panel_letter(axis, letter):
    # Fixed offset in points, so the letter sits at the same place on narrow
    # and wide panels and clear of the y label.
    axis.annotate(letter, xy=(0, 1), xycoords="axes fraction", xytext=(-36, 5),
                  textcoords="offset points", fontsize=9, fontweight="bold", va="bottom", ha="left")


def snr_axis(axis):
    axis.set_xlim(-5.8, 20.8)
    axis.set_xticks(XTICKS)
    axis.set_xlabel(XLABEL)
    axis.grid(True, linestyle="--", linewidth=0.4, alpha=0.35)
    axis.tick_params(direction="out")


def panel_a(axis, summary):
    for channel, (color, marker, name) in CHANNEL.items():
        s = summary[summary["Channel"] == channel].sort_values("Eb/N0 (dB)")
        x = s["Eb/N0 (dB)"].to_numpy()
        y = s["Frac Samples Unchanged"].to_numpy()
        err = np.vstack([y - s["CP lo Samples"].to_numpy(), s["CP hi Samples"].to_numpy() - y])
        axis.errorbar(x, y, yerr=err, color=color, marker=marker, linestyle="-", capsize=1.2,
                      elinewidth=0.5, zorder=3)
        axis.plot(x, s["Frac Frames Unchanged"], color=color, marker=marker, linestyle="--",
                  markerfacecolor="white", zorder=3)
    axis.set_ylabel("Fraction with all spikes unchanged")
    axis.set_ylim(-0.03, 1.03)
    snr_axis(axis)
    # Two small keys placed where there are no data: colour = channel,
    # line style = samples (all 25 frames) or individual frames.
    colours = [Line2D([], [], color=c, marker=m, linestyle="-", label=n) for c, m, n in CHANNEL.values()]
    first = axis.legend(handles=colours, loc="upper left", bbox_to_anchor=(0.0, 0.97))
    axis.add_artist(first)
    styles = [Line2D([], [], color="0.35", marker="o", linestyle="-", label="samples"),
              Line2D([], [], color="0.35", marker="o", linestyle="--", markerfacecolor="white", label="time steps")]
    axis.legend(handles=styles, loc="lower right")


def panel_b(axis, rstar):
    columns = [f"r_ki={q}" for q in range(1, 11)]
    total = rstar[columns + ["r_ki>10", "r_ki=inf"]].to_numpy().sum()
    counts = [rstar[f"r_ki={q}"].sum() for q in range(1, 6)]
    counts.append(sum(rstar[f"r_ki={q}"].sum() for q in range(6, 11)))
    counts += [rstar["r_ki>10"].sum(), rstar["r_ki=inf"].sum()]
    pct = 100.0 * np.asarray(counts, dtype=float) / total
    labels = ["1", "2", "3", "4", "5", "6\u201310", "\u226511", "\u221e"]
    bars = axis.bar(range(len(pct)), pct, color=RED, edgecolor="black", linewidth=0.5, width=0.7, zorder=3)
    for bar, value in zip(bars, pct):
        axis.text(bar.get_x() + bar.get_width() / 2, value * 1.15, f"{value:.1f}" if value >= 1 else f"{value:.2f}",
                  ha="center", va="bottom", fontsize=5.6)
    axis.set_yscale("log")
    axis.set_ylim(0.1, 400)
    axis.set_xticks(range(len(pct)), labels)
    axis.tick_params(axis="x", labelsize=5.6)        # keeps "6-10" and ">=11" apart
    axis.set_xlabel(r"$r_{k,i}$ (fewest bit errors that change the spike)")
    axis.set_ylabel(r"Neuron$-$time-step pairs (%)")
    axis.grid(True, axis="y", linestyle="--", linewidth=0.4, alpha=0.35)
    r = rstar["r_star"].to_numpy(dtype=float)
    n1 = int((r == 1).sum())
    axis.text(0.03, 0.96, rf"$r_\star=1$ for {n1:,}/{len(r):,} samples", transform=axis.transAxes,
              ha="left", va="top", fontsize=6,
              bbox=dict(boxstyle="square,pad=0.3", facecolor="white", edgecolor="0.75", linewidth=0.5))
    return pct


def log_lines(axis, x, values, color, marker, linestyle, label, floor):
    values = np.asarray(values, dtype=float)
    keep = values > floor
    axis.plot(x[keep], values[keep], color=color, marker=marker, linestyle=linestyle, label=label, zorder=3)


def panel_c(axis, awgn):
    x = awgn["Eb/N0 (dB)"].to_numpy()
    for key, (color, marker, label) in zip(CLASS_KEYS, CLASS_STYLE):
        log_lines(axis, x, awgn[f"Frac {key}"], color, marker, "-", label, 0)
    axis.set_yscale("log")
    axis.set_ylim(1e-6, 3)
    axis.set_ylabel("Fraction of time steps")
    snr_axis(axis)
    axis.legend(loc="lower center", ncol=2, title="Time-step class", alignment="left",
                columnspacing=1.2)


def panel_d(axis, awgn):
    x = awgn["Eb/N0 (dB)"].to_numpy()
    total = awgn["Out energy total"].to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        for key, color, marker, linestyle, label in SOURCE_STYLE:
            share = np.where(total > 0, awgn[key].to_numpy() / total, np.nan)
            share = np.nan_to_num(share, nan=0.0)
            log_lines(axis, x, share, color, marker, linestyle, label, 0)
    axis.set_yscale("log")
    axis.set_ylim(3e-9, 5)
    axis.set_ylabel("Share of output change")
    snr_axis(axis)
    axis.legend(loc="lower right", title="Time-step class", alignment="left")


def panel_e(axis, awgn):
    s = awgn[awgn["R_t n"] >= MIN_FRAMES_R]
    x = s["Eb/N0 (dB)"].to_numpy()
    color = CLASS_STYLE[2][0]
    axis.fill_between(x, s["R_t q25"], s["R_t q75"], color=color, alpha=0.25, linewidth=0, zorder=2)
    axis.plot(x, s["R_t median"], color=color, marker="^", zorder=3)
    axis.axhline(1.0, color="black", linestyle="--", linewidth=0.7, zorder=1)
    axis.set_ylim(0.9, 1.1)
    axis.set_ylabel(r"$R_t$ at class-(iii) time steps")
    snr_axis(axis)
    axis.legend(handles=[Line2D([], [], color=color, marker="^", label="median"),
                         plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.25, label="interquartile range"),
                         Line2D([], [], color="black", linestyle="--", linewidth=0.7, label=r"$R_t=1$")],
                loc="lower right")


def cp_interval(k, n):
    from scipy.stats import beta
    lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return lo, hi


def numbers(summary, rstar, pct):
    r = rstar["r_star"].to_numpy(dtype=float)
    ge2 = int((r >= 2).sum())
    out = {"samples": int(len(r)),
           "r_star_counts": {str(int(v)) if np.isfinite(v) else "inf": int((r == v).sum()) for v in sorted(set(r))},
           "r_star>=2": {"k": ge2, "frac": ge2 / len(r), "cp95": cp_interval(ge2, len(r))},
           "r_ki_percent (1,2,3,4,5,6-10,>10,inf)": [float(v) for v in pct],
           "unchanged_samples": {}, "representative": {}}
    for _, row in summary.iterrows():
        out["unchanged_samples"][f"{row['Channel']} {row['Eb/N0 (dB)']:g}"] = {
            "frac": row["Frac Samples Unchanged"], "cp95": [row["CP lo Samples"], row["CP hi Samples"]],
            "frac_frames": row["Frac Frames Unchanged"]}
    for channel, snr in (("awgn", 0.0), ("rayleigh", 5.0)):
        row = summary[(summary["Channel"] == channel) & (summary["Eb/N0 (dB)"] == snr)]
        if row.empty:
            continue
        row = row.iloc[0]
        total = row["Out energy total"]
        out["representative"][f"{channel} {snr:g}"] = {
            "class_fractions": {k: row[f"Frac {k}"] for k in CLASS_KEYS},
            "output_change_shares": {k[0]: (row[k[0]] / total if total > 0 else None) for k in SOURCE_STYLE},
            "R_t median": row["R_t median"], "R_t IQR": [row["R_t q25"], row["R_t q75"]], "R_t n": int(row["R_t n"]),
        }
    return out


def main():
    global XLABEL
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(ROOT / "outputs" / "sn3_fig3"))
    parser.add_argument("--font", choices=sorted(FONTS), default="sans",
                        help="sans = main-text Fig. 2 and Supplementary Figs. 1-2 (default); "
                             "serif = main-text Fig. 4")
    parser.add_argument("--xlabel", default=XLABEL,
                        help="x label of the SNR panels; must equal that of main-text Fig. 4")
    args = parser.parse_args()
    XLABEL = args.xlabel
    folder = Path(args.input)
    summary = pd.read_csv(folder / "mc_summary.csv")
    rstar = pd.read_csv(folder / "rstar_per_sample.csv")
    awgn = summary[summary["Channel"] == "awgn"].sort_values("Eb/N0 (dB)")
    style(args.font)
    figure = plt.figure(figsize=(180 * MM, 110 * MM))
    top = figure.add_gridspec(1, 3, left=0.075, right=0.985, top=0.96, bottom=0.6, wspace=0.42)
    bottom = figure.add_gridspec(1, 2, left=0.075, right=0.985, top=0.44, bottom=0.08, wspace=0.2)
    axes = [figure.add_subplot(top[0, j]) for j in range(3)] + \
           [figure.add_subplot(bottom[0, j]) for j in range(2)]
    panel_a(axes[0], summary)
    pct = panel_b(axes[1], rstar)
    panel_e(axes[2], awgn)          # c: R_t at class-(iii) time steps
    panel_d(axes[3], awgn)          # d: source of the output change
    panel_c(axes[4], awgn)          # e: frame classes
    for axis, letter in zip(axes, "abcde"):
        panel_letter(axis, letter)
    out = folder / "figures"
    out.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        figure.savefig(out / f"supp_fig3_shd.{ext}", bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)
    facts = numbers(summary, rstar, pct)
    (folder / "fig3_numbers.json").write_text(json.dumps(facts, indent=2, default=float))
    print(f"Saved {out / 'supp_fig3_shd.pdf'}")


if __name__ == "__main__":
    main()
