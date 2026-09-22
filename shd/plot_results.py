"""Publication-style plots for the formal five-seed AWGN experiments."""

import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import Config


MODEL_ORDER = ["S-LISTA", "Vanilla SNN", "ANN LISTA", "ANN LSTM"]
SNN_ORDER = ["S-LISTA", "Vanilla SNN"]
COLORS = {
    "S-LISTA": "#D94B3D",
    "Vanilla SNN": "#3BA275",
    "ANN LISTA": "#3E86BB",
    "ANN LSTM": "#7C5AAC",
}
MARKERS = {
    "S-LISTA": "o",
    "Vanilla SNN": "s",
    "ANN LISTA": "^",
    "ANN LSTM": "D",
}
PANEL_SIZE = (7.4, 5.5)


def publication_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 18,
        "axes.labelsize": 20,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 12,
        "axes.linewidth": 1.4,
        "savefig.dpi": 300,
    })


def style_axis(axis):
    axis.grid(True, linestyle="--", linewidth=0.6, alpha=0.3)
    axis.tick_params(direction="out")


def save_figure(figure, results_dir, stem):
    output = os.path.join(results_dir, "figures")
    os.makedirs(output, exist_ok=True)
    figure.tight_layout()
    for extension in ("png", "pdf"):
        figure.savefig(
            os.path.join(output, f"{stem}.{extension}"),
            bbox_inches="tight",
        )
    plt.close(figure)
    print(f"[Saved figure] {stem}")


def read_csv(results_dir, name):
    path = os.path.join(results_dir, name)
    if not os.path.exists(path):
        return pd.DataFrame()
    frame = pd.read_csv(path)
    source = "Display Model" if "Display Model" in frame.columns else "Model"
    normalized = frame[source].replace({
        "SPIKING LISTA": "S-LISTA",
        "Spiking LISTA": "S-LISTA",
        "SCNN": "Vanilla SNN",
        "SPIKING CNN": "Vanilla SNN",
        "Spiking CNN": "Vanilla SNN",
        "Generic SNN": "Vanilla SNN",
        "SPIKING DENSE": "Vanilla SNN",
        "SpikingDenseRecon": "Vanilla SNN",
    })
    frame["Model"] = normalized
    frame["Display Model"] = normalized
    return frame


def plot_curve(frame, metric, ylabel, results_dir, stem, yticks=None):
    publication_style()
    figure, axis = plt.subplots(figsize=PANEL_SIZE)
    for model in MODEL_ORDER:
        part = frame[frame["Display Model"] == model]
        if part.empty:
            continue
        grouped = part.groupby("Eval SNR (dB)")[metric]
        mean = grouped.mean().sort_index()
        std = grouped.std(ddof=1).reindex(mean.index).fillna(0.0)
        x = mean.index.to_numpy(dtype=float)
        y = mean.to_numpy(dtype=float)
        s = std.to_numpy(dtype=float)
        axis.plot(
            x, y, color=COLORS[model], marker=MARKERS[model],
            linewidth=2.8, markersize=8.5, label=model,
        )
        axis.fill_between(x, y - s, y + s, color=COLORS[model], alpha=0.12)
    axis.set_xlabel("Eb/N0 (dB)")
    axis.set_ylabel(ylabel)
    if yticks is not None:
        axis.set_yticks(yticks)
        axis.set_ylim(yticks[0], yticks[-1])
    axis.legend(frameon=True)
    style_axis(axis)
    save_figure(figure, results_dir, stem)


def plot_snr_robustness(results_dir):
    awgn = read_csv(results_dir, "snr_robustness_per_seed.csv")
    fading = read_csv(results_dir, "snr_robustness_rayleigh_per_seed.csv")
    combined = pd.concat([awgn.assign(_channel="awgn"), fading.assign(_channel="rayleigh")], ignore_index=True)
    if combined.empty:
        return
    for metric, ylabel, stem in [
        ("Mean NMSE (dB)", "Mean NMSE (dB)", "snr_robustness_nmse"),
        ("Accuracy (%)", "Reconstructed-x accuracy (%)", "snr_robustness_accuracy"),
    ]:
        grouped = combined.groupby(["_channel", "Display Model", "Eval SNR (dB)"])[metric].agg(["mean", "std"])
        lower = (grouped["mean"] - grouped["std"].fillna(0)).min()
        upper = (grouped["mean"] + grouped["std"].fillna(0)).max()
        pad = max((upper-lower)*0.05, 0.1)
        ticks = (np.arange(0, 101, 20) if metric == "Accuracy (%)" else
                 matplotlib.ticker.MaxNLocator(nbins=6).tick_values(lower-pad, upper+pad))
        for frame, suffix in ((awgn, ""), (fading, "_rayleigh")):
            if not frame.empty:
                plot_curve(frame, metric, ylabel, results_dir, stem+suffix, ticks)
    if fading.empty:
        print("[Pending] Run main.py --rayleigh-all for Rayleigh figures")


def mean_std_text(mean, std, digits=2):
    return f"{float(mean):.{digits}f} ± {float(std):.{digits}f}"


def save_main_table(results_dir, rayleigh=False):
    frame = read_csv(results_dir, "main_rayleigh20_per_seed.csv" if rayleigh else "main_results_per_seed.csv")
    if frame.empty:
        return
    rows = []
    for model in MODEL_ORDER:
        part = frame[frame["Display Model"] == model]
        if part.empty:
            continue
        row = {"Algorithm": model, "N": int(part["Seed"].nunique())}
        for metric in ["Mean NMSE (dB)", "Accuracy (%)", "Bits/Sample"]:
            values = pd.to_numeric(part[metric], errors="coerce").dropna()
            row[metric] = mean_std_text(
                values.mean(), values.std(ddof=1), 0 if metric == "Bits/Sample" else 2
            )
        rows.append(row)
    table = pd.DataFrame(rows)
    stem = "table_main_results_rayleigh20" if rayleigh else "table_main_results"
    table.to_csv(os.path.join(results_dir, stem + ".csv"), index=False)
    publication_style()
    figure, axis = plt.subplots(figsize=(11.2, 3.4))
    axis.axis("off")
    rendered = axis.table(
        cellText=table.values,
        colLabels=table.columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    rendered.auto_set_font_size(False)
    rendered.set_fontsize(12.5)
    rendered.scale(1.0, 1.55)
    for (row, _column), cell in rendered.get_celld().items():
        cell.set_edgecolor("#444444")
        if row == 0:
            cell.set_facecolor("#DCEAF4")
            cell.set_text_props(weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F5F5F5")
    save_figure(figure, results_dir, stem)


def plot_performance_bit(results_dir, seed=None):
    awgn = read_csv(results_dir, "performance_bit_per_seed.csv")
    rayleigh = read_csv(results_dir, "performance_bit_rayleigh20_per_seed.csv")
    common = pd.concat([awgn, rayleigh], ignore_index=True)
    if seed is not None and not common.empty:
        common = common[common["Seed"].eq(int(seed))]
    limits = {}
    for metric in ("Mean NMSE (dB)", "Accuracy (%)"):
        if common.empty:
            continue
        values = pd.to_numeric(common.loc[common["Display Model"].isin(SNN_ORDER), metric], errors="coerce").dropna()
        if values.empty:
            continue
        pad = max(float(values.max() - values.min()) * 0.05, 0.1)
        limits[metric] = (np.arange(max(0.0, min(80.0, 5.0*np.floor((values.min()-pad)/5.0))), 101.0, 5.0) if metric == "Accuracy (%)"
                          else matplotlib.ticker.MaxNLocator(nbins=6).tick_values(values.min()-pad, values.max()+pad))
    for frame, suffix in ((awgn, ""), (rayleigh, "_rayleigh20")):
        _plot_performance_bit_frame(frame, results_dir, seed, suffix, limits)
    if rayleigh.empty:
        print("[Pending] Rayleigh scatter: run main.py --rayleigh-all first; AWGN plots retained.")


def _plot_performance_bit_frame(frame, results_dir, seed, suffix, limits):
    if frame.empty:
        return
    if seed is not None:
        frame = frame[frame["Seed"] == int(seed)].copy()
        if frame.empty:
            return
    publication_style()
    for metric, ylabel, stem in [
        ("Mean NMSE (dB)", "Mean NMSE (dB)", "performance_bit_nmse"),
        ("Accuracy (%)", "Reconstructed-x accuracy (%)", "performance_bit_accuracy"),
    ]:
        figure, axis = plt.subplots(figsize=PANEL_SIZE)
        for model in SNN_ORDER:
            part = frame[frame["Display Model"] == model]
            if part.empty:
                continue
            x = pd.to_numeric(part["Bits/Sample"], errors="coerce") / 1000.0
            y = pd.to_numeric(part[metric], errors="coerce")
            axis.scatter(
                x, y, s=42, marker=MARKERS[model],
                color=COLORS[model], linewidth=0,
                alpha=0.28,
            )
            operating_points = part.assign(
                _bits=pd.to_numeric(part["Bits/Sample"], errors="coerce"),
                _metric=pd.to_numeric(part[metric], errors="coerce"),
            ).groupby("Alpha Rate", as_index=False).agg(
                {"_bits": "mean", "_metric": "mean"}
            )
            axis.scatter(
                operating_points["_bits"] / 1000.0,
                operating_points["_metric"],
                s=145, marker=MARKERS[model],
                color=COLORS[model], edgecolor="black", linewidth=0.8,
                label=model, zorder=3,
            )
        axis.set_xlabel(r"Bits/sample ($\times 10^3$)")
        axis.set_ylabel(ylabel)
        if metric in limits:
            ticks = limits[metric]
            axis.set_yticks(ticks)
            axis.set_ylim(ticks[0], ticks[-1])
        axis.legend(frameon=True)
        style_axis(axis)
        output_stem = stem + suffix
        if seed is not None:
            output_stem += f"_seed{int(seed)}"
        save_figure(figure, results_dir, output_stem)


def plot_core_ablation(results_dir, eval_snr=None):
    frame = read_csv(results_dir, "core_ablation_per_seed.csv")
    if frame.empty:
        return
    if eval_snr is None:
        eval_snr = 20.0 if os.path.basename(os.path.normpath(results_dir)) == "rayleigh_paper" else 10.0
    available = pd.to_numeric(frame["Eval SNR (dB)"], errors="coerce")
    selected = frame[np.isclose(available, eval_snr)].copy()
    if selected.empty:
        raise ValueError(
            f"No ablation rows at {eval_snr:g} dB in {results_dir}; "
            f"available SNRs: {sorted(available.dropna().unique().tolist())}"
        )
    frame = selected
    order = [
        "S-LISTA with AER",
        "Hard surrogate",
        "Without temperature relaxation",
        "No rate regularization",
        "No semantic supervision",
    ]
    order += [v for v in frame["Variant"].dropna().unique() if v not in order]
    rows = []
    for variant in order:
        part = frame[frame["Variant"] == variant]
        if part.empty:
            continue
        row = {"Variant": variant, "N": int(part["Seed"].nunique())}
        for metric in ["Mean NMSE (dB)", "Accuracy (%)", "Bits/Sample"]:
            values = pd.to_numeric(part[metric], errors="coerce").dropna()
            row[metric] = mean_std_text(
                values.mean(), values.std(ddof=1), 0 if metric == "Bits/Sample" else 2
            )
        rows.append(row)
    if not rows:
        raise ValueError(f"No valid ablation variants in {results_dir}")
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(results_dir, "table_core_ablation.csv"), index=False)
    publication_style()
    figure, axis = plt.subplots(figsize=(11.2, 3.5))
    axis.axis("off")
    rendered = axis.table(
        cellText=table.values, colLabels=table.columns,
        cellLoc="center", colLoc="center", loc="center",
    )
    rendered.auto_set_font_size(False)
    rendered.set_fontsize(12.5)
    rendered.scale(1.0, 1.55)
    for (row, _column), cell in rendered.get_celld().items():
        cell.set_edgecolor("#444444")
        if row == 0:
            cell.set_facecolor("#DCEAF4")
            cell.set_text_props(weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F5F5F5")
    save_figure(figure, results_dir, "table_core_ablation")


def plot_semantic_robustness(results_dir):
    frame = read_csv(results_dir, "semantic_snr_robustness_per_seed.csv")
    if frame.empty:
        return
    palette = {
        "With semantic supervision": "#D94B3D",
        "Without semantic supervision": "#3E86BB",
    }
    publication_style()
    figure, axis = plt.subplots(figsize=PANEL_SIZE)
    for variant in palette:
        part = frame[frame["Variant"] == variant]
        grouped = part.groupby("Eval SNR (dB)")["Accuracy (%)"]
        mean = grouped.mean().sort_index()
        std = grouped.std(ddof=1).reindex(mean.index).fillna(0.0)
        x = mean.index.to_numpy(dtype=float)
        y = mean.to_numpy(dtype=float)
        s = std.to_numpy(dtype=float)
        axis.plot(x, y, color=palette[variant], marker="o",
                  linewidth=2.8, markersize=8.0, label=variant)
        axis.fill_between(x, y - s, y + s,
                          color=palette[variant], alpha=0.12)
    axis.set_xlabel("Eb/N0 (dB)")
    axis.set_ylabel("Reconstructed-x accuracy (%)")
    axis.legend(frameon=True)
    style_axis(axis)
    save_figure(figure, results_dir, "semantic_snr_accuracy")


def between_within_ratio(features, labels):
    global_center = features.mean(axis=0)
    between = 0.0
    within = 0.0
    for class_index in np.unique(labels):
        points = features[labels == class_index]
        center = points.mean(axis=0)
        between += len(points) * float(np.sum((center - global_center) ** 2))
        within += float(np.sum((points - center) ** 2))
    return between / max(within, 1e-12)


def plot_tsne(results_dir, seed):
    paths = {
        "Without semantic supervision": os.path.join(
            results_dir, f"z_features_slista_dense_no_semantic_seed{seed}.npz"
        ),
        "With semantic supervision": os.path.join(
            results_dir, f"z_features_slista_dense_seed{seed}.npz"
        ),
    }
    if not all(os.path.exists(path) for path in paths.values()):
        return
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE, trustworthiness
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    regimes = {}
    for name, path in paths.items():
        archive = np.load(path)
        features = np.asarray(archive["features"], dtype=np.float32)
        target = np.asarray(archive["labels"], dtype=int)
        scaled = StandardScaler().fit_transform(features)
        pca_dim = min(50, scaled.shape[0] - 1, scaled.shape[1])
        reduced = PCA(
            n_components=pca_dim, random_state=seed
        ).fit_transform(scaled)
        regimes[name] = (reduced, target)

    minimum_samples = min(value[0].shape[0] for value in regimes.values())
    perplexities = [
        value for value in (10, 20, 30, 40)
        if value < minimum_samples
    ]
    if not perplexities:
        perplexities = [max(2, minimum_samples - 1)]
    candidate_rows = []
    candidate_coordinates = {}
    for perplexity in perplexities:
        for tsne_seed in (seed, seed + 1, seed + 2):
            metrics = {}
            for name, (reduced, target) in regimes.items():
                coordinates = TSNE(
                    n_components=2, perplexity=perplexity,
                    init="pca", learning_rate="auto",
                    random_state=tsne_seed,
                ).fit_transform(reduced)
                neighbors = min(10, max(1, (len(target) - 1) // 2))
                metrics[name] = {
                    "silhouette": float(silhouette_score(coordinates, target)),
                    "trustworthiness": float(trustworthiness(
                        reduced, coordinates, n_neighbors=neighbors
                    )),
                }
                candidate_coordinates[(perplexity, tsne_seed, name)] = coordinates
            without = metrics["Without semantic supervision"]
            with_semantics = metrics["With semantic supervision"]
            candidate_rows.append({
                "Perplexity": int(perplexity),
                "t-SNE Seed": int(tsne_seed),
                "Without 2D Silhouette": without["silhouette"],
                "With 2D Silhouette": with_semantics["silhouette"],
                "Silhouette Gain": (
                    with_semantics["silhouette"] - without["silhouette"]
                ),
                "Mean Trustworthiness": 0.5 * (
                    without["trustworthiness"]
                    + with_semantics["trustworthiness"]
                ),
            })
    candidates = pd.DataFrame(candidate_rows)
    trust_limit = candidates["Mean Trustworthiness"].max() - 0.02
    eligible = candidates[candidates["Mean Trustworthiness"] >= trust_limit]
    selected = eligible.sort_values(
        ["Silhouette Gain", "Mean Trustworthiness"], ascending=False
    ).iloc[0]
    selected_perplexity = int(selected["Perplexity"])
    selected_seed = int(selected["t-SNE Seed"])
    candidates["Selected"] = (
        (candidates["Perplexity"] == selected_perplexity)
        & (candidates["t-SNE Seed"] == selected_seed)
    )
    candidates.to_csv(
        os.path.join(results_dir, f"tsne_selection_seed{seed}.csv"),
        index=False,
    )
    palette = plt.get_cmap("tab20")
    publication_style()
    coordinate_rows = []
    stems = {
        "Without semantic supervision": "tsne_without_semantic_supervision",
        "With semantic supervision": "tsne_with_semantic_supervision",
    }
    for name, (reduced, target) in regimes.items():
        xy = candidate_coordinates[(selected_perplexity, selected_seed, name)]
        xy = xy - xy.mean(axis=0, keepdims=True)
        figure, axis = plt.subplots(figsize=(7.2, 6.2))
        latent_silhouette = silhouette_score(reduced, target)
        bw_ratio = between_within_ratio(reduced, target)
        for class_index in range(Config.NUM_CLASSES):
            mask = target == class_index
            axis.scatter(
                xy[mask, 0], xy[mask, 1], s=18,
                color=palette(class_index), alpha=0.72,
                linewidths=0,
            )
        condition = (
            "Without task supervision" if name.startswith("Without")
            else "With task supervision"
        )
        axis.set_title(
            f"{condition}\nB/W = {bw_ratio:.2f}"
        )
        axis.set_xlabel("t-SNE 1")
        axis.set_ylabel("t-SNE 2")
        axis.autoscale(enable=True, tight=True)
        axis.margins(x=0.04, y=0.06)
        style_axis(axis)
        for point, class_index in zip(xy, target):
            coordinate_rows.append({
                "Regime": name,
                "Class": int(class_index),
                "t-SNE 1": float(point[0]),
                "t-SNE 2": float(point[1]),
                "Perplexity": selected_perplexity,
                "t-SNE Seed": selected_seed,
                "Latent Silhouette": float(latent_silhouette),
                "Between/Within": float(bw_ratio),
            })
        save_figure(figure, results_dir, stems[name])
    pd.DataFrame(coordinate_rows).to_csv(
        os.path.join(results_dir, f"tsne_alpha0_alpha3_seed{seed}.csv"),
        index=False,
    )
    print(
        f"[t-SNE] selected perplexity={selected_perplexity}, "
        f"seed={selected_seed}, gain={selected['Silhouette Gain']:.4f}, "
        f"trustworthiness={selected['Mean Trustworthiness']:.4f}"
    )


def ecdf(values):
    values = np.sort(np.asarray(values, dtype=float))
    return values, np.arange(1, len(values) + 1) / len(values)


def plot_sample_analysis(results_dir, seed):
    samples = read_csv(results_dir, f"analysis_samples_seed{seed}.csv")
    aer_bits = read_csv(results_dir, f"analysis_aer_bits_seed{seed}.csv")
    classes = read_csv(results_dir, f"analysis_classes_seed{seed}.csv")
    energy = read_csv(results_dir, f"analysis_energy_seed{seed}.csv")
    if samples.empty:
        return
    publication_style()
    figure, axis = plt.subplots(figsize=PANEL_SIZE)
    for model in MODEL_ORDER:
        part = samples[samples["Model"] == model]
        x, y = ecdf(part["NMSE (dB)"])
        axis.plot(x, y, color=COLORS[model], linewidth=2.4, label=model)
    axis.set_xlabel("Per-sample NMSE (dB)")
    axis.set_ylabel("Empirical CDF")
    axis.legend(frameon=True)
    style_axis(axis)
    save_figure(figure, results_dir, "sample_nmse_ecdf")

    figure, axis = plt.subplots(figsize=PANEL_SIZE)
    bit_source = aer_bits if not aer_bits.empty else samples
    for model in SNN_ORDER:
        part = bit_source[bit_source["Model"] == model]
        x, y = ecdf(part["Bits/Sample"])
        axis.plot(x / 1000.0, y, color=COLORS[model], linewidth=2.4, label=model)
    axis.set_xlabel(r"Bits/sample ($\times 10^3$)")
    axis.set_ylabel("Empirical CDF")
    axis.legend(frameon=True)
    style_axis(axis)
    save_figure(figure, results_dir, "sample_bits_ecdf")

    matrix = []
    for model in MODEL_ORDER:
        indexed = classes[classes["Model"] == model].set_index("Class")
        matrix.append([
            float(indexed.loc[index, "NMSE (dB)"])
            if index in indexed.index else np.nan
            for index in range(Config.NUM_CLASSES)
        ])
    figure, axis = plt.subplots(figsize=(8.4, 5.5))
    image = axis.imshow(np.asarray(matrix), aspect="auto", cmap="viridis")
    axis.set_xticks(np.arange(Config.NUM_CLASSES))
    axis.set_yticks(np.arange(len(MODEL_ORDER)))
    axis.set_xticklabels([str(index) for index in range(Config.NUM_CLASSES)])
    axis.set_yticklabels(MODEL_ORDER)
    axis.set_xlabel("SHD class")
    figure.colorbar(image, ax=axis, label="Mean NMSE (dB)")
    save_figure(figure, results_dir, "classwise_nmse_heatmap")

    if os.path.basename(os.path.normpath(results_dir)) == 'rayleigh_paper':
        return

    for model in ["S-LISTA"]:
        part = samples[samples["Model"] == model]
        confusion = np.zeros((Config.NUM_CLASSES, Config.NUM_CLASSES), dtype=int)
        for truth, prediction in zip(part["Class"], part["Prediction"]):
            confusion[int(truth), int(prediction)] += 1
        figure, axis = plt.subplots(figsize=(10.5, 9.2))
        image = axis.imshow(confusion, cmap="Blues", aspect="equal")
        axis.set_xticks(np.arange(Config.NUM_CLASSES))
        axis.set_yticks(np.arange(Config.NUM_CLASSES))
        axis.set_xticklabels(range(Config.NUM_CLASSES), rotation=45, ha="right")
        axis.set_yticklabels(range(Config.NUM_CLASSES))
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("True class")
        threshold = 0.52 * confusion.max()
        for row in range(Config.NUM_CLASSES):
            for column in range(Config.NUM_CLASSES):
                value = confusion[row, column]
                axis.text(
                    column, row, str(value), ha="center", va="center",
                    fontsize=8, color="white" if value > threshold else "black",
                )
        figure.colorbar(image, ax=axis, label="Count")
        safe = re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")
        save_figure(figure, results_dir, f"confusion_raw_{safe}")

    if energy.empty:
        return
    indexed = energy.set_index("Model").loc[MODEL_ORDER]
    x = np.arange(len(MODEL_ORDER))
    width = 0.34
    figure, axes = plt.subplots(1, 4, figsize=(16.0, 4.5))
    for column, scope in enumerate(("Sender", "Reconstruction")):
        mac = indexed[f"{scope} MAC/Sample"].to_numpy() / 1e9
        ac = indexed[f"{scope} AC/Sample"].to_numpy() / 1e9
        axis = axes[column]
        axis.bar(
            x - width / 2, mac, width,
            color="#4c78a8", edgecolor="black", label="MAC",
        )
        axis.bar(
            x + width / 2, ac, width,
            color="#f2a541", edgecolor="black", hatch="///", label="AC",
        )
        axis.set_title("Sensing" if scope == "Sender" else "Reconstruction")
        axis.set_ylabel(r"Operations/sample ($\times 10^9$)")
        axis.legend(frameon=True)
        style_axis(axis)
        energy = (4.6 * mac * 1e9 + 0.9 * ac * 1e9) / 1e6
        energy_axis = axes[column + 2]
        energy_axis.bar(
            x, energy, color=[COLORS[model] for model in MODEL_ORDER],
            edgecolor="black",
        )
        energy_axis.set_title(
            "Sensing" if scope == "Sender" else "Reconstruction"
        )
        energy_axis.set_ylabel(r"Analytical energy/sample ($\mu$J)")
        style_axis(energy_axis)
    for axis in axes:
        axis.set_xticks(x)
        axis.set_xticklabels(MODEL_ORDER, rotation=18, ha="right")
    save_figure(figure, results_dir, "computational_cost_breakdown")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-rayleigh", action="store_true")
    parser.add_argument("--results-dir", default=Config.RESULTS_DIR)
    parser.add_argument("--analysis-seed", type=int, default=Config.DEFAULT_SEEDS[0])
    parser.add_argument("--tsne-seed", type=int, default=Config.DEFAULT_SEEDS[0])
    parser.add_argument(
        "--figures", nargs="+",
        choices=["all", "main", "snr", "performance_bit", "ablation", "semantic", "tsne", "analysis"],
        default=["all"],
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.paper_rayleigh:
        args.results_dir = os.path.join(args.results_dir, 'rayleigh_paper')
        if args.figures == ['all']:
            args.figures = ['ablation', 'semantic', 'tsne', 'analysis']
    selected = set(args.figures)
    all_figures = "all" in selected
    if all_figures or "main" in selected:
        for extension in ("png", "pdf"):
            obsolete = os.path.join(
                args.results_dir, "figures",
                f"main_performance_metrics.{extension}",
            )
            if os.path.exists(obsolete):
                os.remove(obsolete)
        save_main_table(args.results_dir)
        save_main_table(args.results_dir, rayleigh=True)
    if all_figures or "snr" in selected:
        plot_snr_robustness(args.results_dir)
    if all_figures or "performance_bit" in selected:
        plot_performance_bit(args.results_dir)
    if all_figures or "ablation" in selected:
        plot_core_ablation(args.results_dir, eval_snr=20.0 if args.paper_rayleigh else None)
    if all_figures or "semantic" in selected:
        plot_semantic_robustness(args.results_dir)
    if all_figures or "tsne" in selected:
        plot_tsne(args.results_dir, args.tsne_seed)
    if all_figures or "analysis" in selected:
        plot_sample_analysis(args.results_dir, args.analysis_seed)


if __name__ == "__main__":
    main()
