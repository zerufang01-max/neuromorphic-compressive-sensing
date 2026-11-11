"""Draw Supplementary Figs. 1 and 2 of SN3 from the outputs of sn3_figs.py.

  python sn3_plot.py --dir runs/final/sn3_figs

Needs only summary.json and per_sample_s*.csv (no torch, no checkpoints).
Style follows plot_fig2.py (font, S-LISTA red, grid, spines); 180 mm wide,
7.5 pt text, bold lowercase panel letters, vector PDF + 300 dpi PNG.
Categorical colours: Okabe-Ito (colour-blind safe).
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

MM = 1 / 25.4
WIDTH = 180 * MM
INK = "#253746"
GRID = "#E8EAED"
SLISTA = "#D84A3A"            # plot_fig2.COLORS["slista"]
BEFORE = "#8A949E"
OKABE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]
SPLIT_COLORS = {"amp": "#0072B2", "miss": "#E69F00", "false": "#009E73"}
TERM_COLORS = {"T1": "#0072B2", "T2": "#E69F00", "T3": "#009E73"}
T12_COLOR = "#CC79A7"
SCATTER_POINTS = 1000
SCATTER_SEED = 0


FONT = 7.0          # every text element (ticks, labels, legends, annotations)
LETTER = 8.0        # bold panel letters


def resolve_font_family(requested="Aptos"):
    """Same chain as plot_fig2.resolve_font_family (main Fig. 2)."""
    from matplotlib import font_manager
    for family in (requested, "Arial", "Helvetica", "DejaVu Sans"):
        try:
            font_manager.findfont(family, fallback_to_default=False)
            return family
        except ValueError:
            continue
    return "DejaVu Sans"


def configure_style(font=None):
    """Font of main Fig. 2 (plot_fig2.py), also used for every formula (mathtext
    'custom' set to the same family), one font size throughout."""
    family = font or resolve_font_family()
    mpl.rcParams.update({
        "font.family": family, "font.sans-serif": [family, "DejaVu Sans"],
        "mathtext.fontset": "custom", "mathtext.rm": family,
        "mathtext.it": f"{family}:italic", "mathtext.bf": f"{family}:bold",
        "mathtext.sf": family, "mathtext.fallback": "stixsans",
        "font.size": FONT, "axes.titlesize": FONT, "axes.labelsize": FONT,
        "xtick.labelsize": FONT, "ytick.labelsize": FONT, "legend.fontsize": FONT,
        "axes.linewidth": 0.6, "lines.linewidth": 1.0, "lines.markersize": 3.5,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 2.2, "ytick.major.size": 2.2,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "legend.frameon": False, "legend.handlelength": 1.5, "legend.handletextpad": 0.4,
        "legend.labelspacing": 0.3, "legend.columnspacing": 1.0, "legend.borderaxespad": 0.3,
    })
    return family


def clean_axis(ax, grid_axis="both"):
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.4, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(pad=1.5)


def panel_letters(fig, axes, letters, dx=0.045, dy=0.035):
    """Bold letter just above the top-left corner of each axes, all at one height."""
    top = max(ax.get_position().y1 for ax in axes)
    for ax, letter in zip(axes, letters):
        fig.text(ax.get_position().x0 - dx, top + dy, letter, fontsize=LETTER,
                 fontweight="bold", ha="left", va="bottom")


def read_columns(path):
    with open(path) as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


# --------------------------------------------------------------- Supp. Fig. 1
def fig1(summary):
    per = summary["per_s"]
    s_values = sorted(int(k) for k in per)
    rows = [per[str(s)]["fig1"] for s in s_values]
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 58 * MM))
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.17, top=0.90, wspace=0.38)

    # a: error split (S15), stacked percentages; NMSE above each bar
    ax = axes[0]
    x = np.arange(len(s_values))
    bottom = np.zeros(len(s_values))
    for key, label in (("amp", "Amplitude"), ("miss", "Missed"), ("false", "False")):
        values = np.array([r[f"pct_{key}"] for r in rows])
        ax.bar(x, values, 0.6, bottom=bottom, color=SPLIT_COLORS[key], edgecolor="white",
               linewidth=0.4, label=label, zorder=2)
        bottom += values
    for xi, r in zip(x, rows):
        ax.text(xi, 102, f"{r['nmse_db']:.1f}".replace("-", "\u2212"),
                ha="center", va="bottom", fontsize=FONT)
    ax.set_xticks(x, [str(s) for s in s_values])
    ax.set_xlim(-0.55, len(s_values) - 0.45)
    ax.set_ylim(0, 132)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.spines["left"].set_bounds(0, 100)
    ax.set_xlabel("Sparsity $s$")
    ax.set_ylabel("Share of squared error (%)")
    clean_axis(ax, "y")
    ax.legend(loc="upper center", ncol=3, handlelength=0.9, columnspacing=0.8,
              borderaxespad=0.1)

    # b: error on the correctly detected coordinates before / after the readout
    ax = axes[1]
    before = [r["before_db"] for r in rows]
    after = [r["after_db"] for r in rows]
    for s, b, a in zip(s_values, before, after):
        ax.annotate("", xy=(s, a), xytext=(s, b), zorder=1,
                    arrowprops=dict(arrowstyle="-|>", color="#A9B0B7", lw=0.6,
                                    mutation_scale=6, shrinkA=3, shrinkB=3))
        ax.text(s + 1.3, 0.5 * (a + b), f"{a - b:.1f}".replace("-", "\u2212"),
                ha="left", va="center", fontsize=FONT)
    ax.plot(s_values, before, color=BEFORE, marker="o", markerfacecolor="white",
            label="Before readout", zorder=3)
    ax.plot(s_values, after, color=SLISTA, marker="s", label="After readout", zorder=3)
    ax.set_xticks(s_values)
    ax.set_xlim(s_values[0] - 5, s_values[-1] + 9)
    ax.set_xlabel("Sparsity $s$")
    ax.set_ylabel("Error on $\\Lambda\\cap S$ (dB)")
    clean_axis(ax)
    ax.legend(loc="lower right")

    # c: size of the terms of (S14) on Lambda
    ax = axes[2]
    styles = {"T1": dict(marker="o", markersize=4.2, linewidth=1.5, zorder=3),
              # T1 and T2 nearly cancel and coincide; T2 is drawn on top, hollow
              "T2": dict(marker="s", markersize=2.4, linewidth=0.8, linestyle=(0, (3, 1.5)),
                         markerfacecolor="white", zorder=4),
              "T3": dict(marker="o", zorder=3)}
    for key, name in (("T1", "$T_1$"), ("T2", "$T_2$"), ("T3", "$T_3$")):
        values = [float(r[f"{key}_db"]) for r in rows]
        values = [v if np.isfinite(v) else np.nan for v in values]
        ax.plot(s_values, values, color=TERM_COLORS[key], label=name, **styles[key])
    if all("T12_db" in r for r in rows):
        ax.plot(s_values, [float(r["T12_db"]) for r in rows], color=T12_COLOR, marker="^",
                markersize=3.6, label="$T_1 + T_2$", zorder=3)
    ax.plot(s_values, [float(r["err_lambda_db"]) for r in rows], color=INK, linestyle="--",
            linewidth=0.9, label="Total", zorder=5)
    ax.set_xticks(s_values)
    ax.set_xlim(s_values[0] - 5, s_values[-1] + 5)
    lo = min(float(r["T3_db"]) for r in rows)
    ax.set_ylim(10 * np.floor((lo - 3) / 10), 0)
    ax.set_xlabel("Sparsity $s$")
    ax.set_ylabel("Squared norm on $\\Lambda$ (dB)")
    clean_axis(ax)
    ax.legend(loc="upper left", ncol=2)
    panel_letters(fig, axes, "abc")
    return fig


# --------------------------------------------------------------- Supp. Fig. 2
def fig2(summary, columns):
    per = summary["per_s"]
    const = summary["constants"]
    s_values = sorted(int(k) for k in per)
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 62 * MM))
    fig.subplots_adjust(left=0.08, right=0.92, bottom=0.16, top=0.91, wspace=0.32)

    # a: code-dependent energy vs the code-dependent part of the bound (S18)
    ax = axes[0]
    rng = np.random.default_rng(SCATTER_SEED)
    top = 0.0
    for color, s in zip(OKABE, s_values):
        col = columns[s]
        idx = rng.choice(col["B_pJ"].size, size=min(SCATTER_POINTS, col["B_pJ"].size),
                         replace=False)
        xb, ye = col["B_pJ"][idx] * 1e-3, col["E_minus_E0_pJ"][idx] * 1e-3
        top = max(top, xb.max(), ye.max())
        ax.scatter(xb, ye, s=2.0, color=color, alpha=0.5, linewidths=0, zorder=3,
                   rasterized=True)
    top = 100 * np.ceil(top / 100)
    ax.plot([0, top], [0, top], color=INK, linewidth=0.7, linestyle="--", zorder=2)
    ax.text(0.80 * top, 0.80 * top, "$y = x$", rotation=0, ha="right", va="bottom",
            fontsize=FONT)
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_xlabel("Code-dependent bound (nJ)")
    ax.set_ylabel("Code-dependent energy (nJ)")
    clean_axis(ax)
    handles = [Line2D([0], [0], marker="o", linestyle="none", markersize=3.2, color=c)
               for c in OKABE[:len(s_values)]]
    ax.legend(handles, [f"$s$ = {s}" for s in s_values], loc="upper left",
              handletextpad=0.2)

    # b: F / s per sparsity, secondary axis C'_L F / (C_L s)
    ax = axes[1]
    data = [columns[s]["F"] / s for s in s_values]
    box = ax.boxplot(data, positions=np.arange(len(s_values)), widths=0.5, whis=(5, 95),
                     showfliers=False, showmeans=True, patch_artist=True,
                     meanprops=dict(marker="D", markersize=2.8, markerfacecolor="white",
                                    markeredgecolor=INK, markeredgewidth=0.6),
                     medianprops=dict(color=INK, linewidth=0.9),
                     whiskerprops=dict(color=INK, linewidth=0.6),
                     capprops=dict(color=INK, linewidth=0.6),
                     boxprops=dict(linewidth=0.6, edgecolor=INK))
    for patch, color in zip(box["boxes"], OKABE):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_xticks(np.arange(len(s_values)), [str(s) for s in s_values])
    ax.set_xlabel("Sparsity $s$")
    ax.set_ylabel("False detections per true coefficient, $F/s$")
    ax.set_ylim(bottom=0)
    clean_axis(ax, "y")
    k = const["C_L_prime"] / const["C_L"]
    sec = ax.secondary_yaxis("right", functions=(lambda v: v * k, lambda v: v / k))
    sec.set_ylabel("False-detection term / sparsity term")
    sec.tick_params(pad=1.5)
    ax.legend([Line2D([0], [0], color=INK, linewidth=0.9),
               Line2D([0], [0], marker="D", linestyle="none", markersize=2.8,
                      markerfacecolor="white", markeredgecolor=INK)],
              ["Median", "Mean"], loc="upper left")
    panel_letters(fig, axes, "ab")
    return fig


def make_figures(directory, formats=("pdf", "png"), dpi=300, font=None):
    directory = Path(directory)
    print(f"font: {configure_style(font)}", flush=True)
    summary = json.loads((directory / "summary.json").read_text())
    columns = {int(s): read_columns(directory / f"per_sample_s{s}.csv") for s in summary["per_s"]}
    for name, fig in (("supp_fig1", fig1(summary)), ("supp_fig2", fig2(summary, columns))):
        for ext in formats:
            fig.savefig(directory / f"{name}.{ext}", dpi=dpi)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", default="runs/final/sn3_figs")
    parser.add_argument("--font", default=None,
                        help="override the font chain of plot_fig2.py (Aptos, Arial, Helvetica, DejaVu Sans)")
    args = parser.parse_args()
    make_figures(args.dir, font=args.font)
