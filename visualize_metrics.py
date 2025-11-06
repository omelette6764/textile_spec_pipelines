# visualize_metrics.py
# Usage:
#   python visualize_metrics.py --metrics out_clean/metrics_pct.csv --out figs
# Generates:
#   - Heatmaps (amplitude%, AUC%)
#   - Grouped bars
#   - Scatter amplitude% vs AUC%
#   - Per-channel spectral bar charts, radar plots, and heatmaps

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def radar_plot(ax, values, labels, title):
    # Simple radar for <=12 axes
    N = len(values)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    values = values.tolist() + [values[0]]
    angles += [angles[0]]
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.plot(angles, values)
    ax.fill(angles, values, alpha=0.15)
    ax.set_title(title, pad=12)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True, help="Path to out_clean/metrics_pct_of_effortful.csv")
    ap.add_argument("--out", default="figs", help="Output folder for figures")
    args = ap.parse_args()

    outdir = Path(args.out); ensure_dir(outdir)
    m = pd.read_csv(args.metrics)

    # Clean/expected categorical values
    m["location"] = m["location"].str.strip().str.lower()
    m["gulp"] = m["gulp"].str.strip().str.lower()

    # ------------- A) Visualize amplitude% & AUC% (MEAN trace) -------------
    mean_rows = m[m["channel"]=="mean"].copy()
    locations = ["top middle", "bottom middle","first right","first left", "second right", "second left"]
    #gulps = ["gulp 1","gulp 2","gulp 3"]
    gulps = ["effortful swallow 1","effortful swallow 2","effortful swallow 3", "effortful swallow 4", "effortful swallow 5", "masako maneuver 1", "masako maneuver 2", "masako maneuver 3", "masako maneuver 4", "masako maneuver 5", "masako maneuver 6", "masako maneuver 7", "masako maneuver 8", "masako maneuver 9", "masako maneuver 10"]

    # Heatmaps
    amp_tbl = mean_rows.pivot_table(index="location", columns="gulp", values="amplitude_pct", aggfunc="mean").reindex(index=locations, columns=gulps)
    auc_tbl = mean_rows.pivot_table(index="location", columns="gulp", values="auc_pct", aggfunc="mean").reindex(index=locations, columns=gulps)

    def heatmap(df, title, fname):
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(df.values, aspect="auto")
        ax.set_xticks(range(df.shape[1])); ax.set_xticklabels(df.columns, rotation=0)
        ax.set_yticks(range(df.shape[0])); ax.set_yticklabels(df.index)
        ax.set_title(title)
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                val = df.iat[i,j]
                ax.text(j, i, f"{val:.1f}" if pd.notna(val) else "", ha="center", va="center", color="w")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout(); fig.savefig(outdir / fname, dpi=150); plt.close(fig)

    heatmap(amp_tbl, "Amplitude Δ% (mean) — location × gulp", "heatmap_amplitude_pct_mean_of_effortful.png")
    heatmap(auc_tbl, "AUC Δ% (mean) — location × gulp", "heatmap_auc_pct_mean_of_effortful.png")

    # Grouped bars (amplitude)
    def grouped_bar(df, title, fname, ylabel):
        fig, ax = plt.subplots(figsize=(7,4))
        x = np.arange(df.shape[0])
        w = 0.25
        for k, col in enumerate(df.columns):
            ax.bar(x + (k-1)*w, df[col].values, width=w, label=col)
        ax.set_xticks(x); ax.set_xticklabels(df.index)
        ax.set_ylabel(ylabel); ax.set_title(title); ax.legend()
        fig.tight_layout(); fig.savefig(outdir / fname, dpi=150); plt.close(fig)

    grouped_bar(amp_tbl, "Amplitude Δ% (mean) — grouped bars", "bars_amplitude_pct_mean_of_effortful.png", "Amplitude Δ%")
    grouped_bar(auc_tbl, "AUC Δ% (mean) — grouped bars", "bars_auc_pct_mean_of_effortful.png", "AUC Δ%·s")

    # Scatter amplitude% vs AUC%
    fig, ax = plt.subplots(figsize=(5,4))
    for loc in locations:
        for g in gulps:
            row = mean_rows[(mean_rows["location"]==loc) & (mean_rows["gulp"]==g)]
            if len(row):
                ax.scatter(row["amplitude_pct"], row["auc_pct"])
                ax.annotate(f"{loc} / {g}", (row["amplitude_pct"].values[0], row["auc_pct"].values[0]), fontsize=8)
    ax.set_xlabel("Amplitude Δ%"); ax.set_ylabel("AUC Δ%·s"); ax.set_title("Amplitude vs AUC (mean)")
    fig.tight_layout(); fig.savefig(outdir / "scatter_amp_vs_auc_mean_of_effortful.png", dpi=150); plt.close(fig)

    # ------------- B) Per-wavelength (data_0..data_5) before averaging -------------
    ch_rows = m[m["channel"].str.startswith("data_")].copy()
    channels = sorted(ch_rows["channel"].unique(), key=lambda s: int(s.split("_")[1]))

    # 1) Channel bar charts per (location, gulp) for amplitude%
    for loc in locations:
        for g in gulps:
            sub = ch_rows[(ch_rows["location"]==loc) & (ch_rows["gulp"]==g)]
            if sub.empty:
                continue
            sub = sub.set_index("channel").reindex(channels)
            fig, ax = plt.subplots(figsize=(7,4))
            ax.bar(range(len(channels)), sub["amplitude_pct"].values)
            ax.set_xticks(range(len(channels))); ax.set_xticklabels(channels)
            ax.set_ylabel("Amplitude Δ%"); ax.set_title(f"{loc} — {g} (per-channel amplitude Δ%)")
            fig.tight_layout(); fig.savefig(outdir / f"{loc.replace(' ','_')}_{g.replace(' ','_')}_channel_bars_amplitude_of_effortful.png", dpi=150); plt.close(fig)

    # 2) Radar plots (“spectral fingerprints”) per (location, gulp) for amplitude%
    for loc in locations:
        for g in gulps:
            sub = ch_rows[(ch_rows["location"]==loc) & (ch_rows["gulp"]==g)]
            if sub.empty:
                continue
            sub = sub.set_index("channel").reindex(channels)
            fig = plt.figure(figsize=(4.5,4.5))
            ax = fig.add_subplot(111, polar=True)
            radar_plot(ax, sub["amplitude_pct"].values, channels, f"{loc}\n{g} — amplitude Δ%")
            fig.tight_layout(); fig.savefig(outdir / f"{loc.replace(' ','_')}_{g.replace(' ','_')}_radar_amplitude_of_effortful.png", dpi=150); plt.close(fig)

    # 3) Heatmap: channels × (location/gulp) for amplitude%
    cols = []
    for loc in locations:
        for g in gulps:
            lbl = f"{loc}\n{g}"
            sub = ch_rows[(ch_rows["location"]==loc) & (ch_rows["gulp"]==g)]
            if sub.empty: continue
            sub = sub.set_index("channel").reindex(channels)
            cols.append((lbl, sub["amplitude_pct"].values))
    if cols:
        labels = [c[0] for c in cols]
        mat = np.column_stack([c[1] for c in cols])  # rows=channels
        fig, ax = plt.subplots(figsize=(1.2*len(labels), 4))
        im = ax.imshow(mat, aspect="auto")
        ax.set_yticks(range(len(channels))); ax.set_yticklabels(channels)
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title("Per-channel amplitude Δ% — channels × (location/gulp)")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout(); fig.savefig(outdir / "heatmap_channels_by_condition_amplitude_of_effortful.png", dpi=150); plt.close(fig)

    # 4) Same as (1–3) but for AUC%
    for loc in locations:
        for g in gulps:
            sub = ch_rows[(ch_rows["location"]==loc) & (ch_rows["gulp"]==g)]
            if sub.empty: continue
            sub = sub.set_index("channel").reindex(channels)
            fig, ax = plt.subplots(figsize=(7,4))
            ax.bar(range(len(channels)), sub["auc_pct"].values)
            ax.set_xticks(range(len(channels))); ax.set_xticklabels(channels)
            ax.set_ylabel("AUC Δ%·s"); ax.set_title(f"{loc} — {g} (per-channel AUC Δ%)")
            fig.tight_layout(); fig.savefig(outdir / f"{loc.replace(' ','_')}_{g.replace(' ','_')}_channel_bars_auc_of_effortful.png", dpi=150); plt.close(fig)

    cols_auc = []
    for loc in locations:
        for g in gulps:
            sub = ch_rows[(ch_rows["location"]==loc) & (ch_rows["gulp"]==g)]
            if sub.empty: continue
            sub = sub.set_index("channel").reindex(channels)
            cols_auc.append((f"{loc}\n{g}", sub["auc_pct"].values))
    if cols_auc:
        labels = [c[0] for c in cols_auc]
        mat = np.column_stack([c[1] for c in cols_auc])
        fig, ax = plt.subplots(figsize=(1.2*len(labels), 4))
        im = ax.imshow(mat, aspect="auto")
        ax.set_yticks(range(len(channels))); ax.set_yticklabels(channels)
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title("Per-channel AUC Δ%·s — channels × (location/gulp)")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout(); fig.savefig(outdir / "heatmap_channels_by_condition_auc_of_effortful.png", dpi=150); plt.close(fig)

    # 5) Simple derived spectral features (for quick stats later)
    #    slope across channels (index 0..5) and red/blue ratio
    spec = []
    for loc in locations:
        for g in gulps:
            sub = ch_rows[(ch_rows["location"]==loc) & (ch_rows["gulp"]==g)].set_index("channel").reindex(channels)
            if sub.isna().any().any(): continue
            y = sub["amplitude_pct"].values
            x = np.arange(len(y))
            slope = np.polyfit(x, y, 1)[0]
            red_blue_ratio = (y[-1] + y[-2]) / max(1e-9, (y[0] + y[1]))  # last two vs first two
            spec.append(dict(location=loc, gulp=g, slope_amppct=slope, red_blue_amppct=red_blue_ratio))
    if spec:
        df_spec = pd.DataFrame(spec)
        df_spec.to_csv(outdir / "spectral_features_amplitude_pct_of_effortful.csv", index=False)

    # ------------- C) PCA / Cosine similarity / Parallel coordinates (Amplitude Δ%) -------------
    # Build matrix X (conditions × channels) using amplitude_pct
    ch_rows_amp = ch_rows.copy()  # already filtered to data_*
    channels = sorted(ch_rows_amp["channel"].unique(), key=lambda s: int(s.split("_")[1]))

    records, labels = [], []
    for loc in locations:
        for g in gulps:
            sub = ch_rows_amp[(ch_rows_amp["location"]==loc) & (ch_rows_amp["gulp"]==g)]
            if sub.empty:
                continue
            v = sub.set_index("channel").reindex(channels)["amplitude_pct"].to_numpy()
            if np.any(~np.isfinite(v)):
                continue
            records.append(v)
            labels.append(f"{loc} | {g}")

    X = np.vstack(records) if records else np.empty((0, len(channels)))
    if X.size:

        def pca_svd(A, standardize=True):
            A = A.copy()
            if standardize:
                A = (A - A.mean(0)) / (A.std(0) + 1e-12)
            else:
                A = A - A.mean(0)
            U, S, Vt = np.linalg.svd(A, full_matrices=False)
            scores = U * S             # projection of samples onto PCs
            loadings = Vt.T            # channel weights per PC
            explained = (S**2) / (S**2).sum()
            return scores, loadings, explained

        scores, loadings, explained = pca_svd(X, standardize=True)

        # PCA scatter (PC1 vs PC2)
        fig, ax = plt.subplots(figsize=(6,5))
        color_map = {"bottom middle":"C0", "first right":"C1", "first left":"C2"}
        marker_map = {"gulp 1":"o", "gulp 2":"s", "gulp 3":"^"}
        for i, lbl in enumerate(labels):
            loc, gulp = lbl.split(" | ")
            ax.scatter(scores[i,0], scores[i,1], c=color_map.get(loc,"C7"), marker=marker_map.get(gulp,"o"))
            ax.annotate(lbl, (scores[i,0], scores[i,1]), fontsize=8, xytext=(3,3), textcoords="offset points")
        ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}%)")
        ax.set_title("PCA (Amplitude Δ%)")
        fig.tight_layout(); fig.savefig(outdir / "pca_scatter_amplitude_of_effortful.png", dpi=150); plt.close(fig)

        # PCA loadings (which wavelengths drive PC1/PC2)
        fig, ax = plt.subplots(figsize=(6,5))
        ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
        scale = 1.0
        for j, ch in enumerate(channels):
            ax.arrow(0, 0, loadings[j,0]*scale, loadings[j,1]*scale,
                    head_width=0.03*scale, head_length=0.05*scale, length_includes_head=True)
            ax.text(loadings[j,0]*scale*1.1, loadings[j,1]*scale*1.1, ch, fontsize=9)
        ax.set_xlabel("PC1 loading"); ax.set_ylabel("PC2 loading")
        ax.set_title("PCA Loadings (Amplitude Δ%)")
        fig.tight_layout(); fig.savefig(outdir / "pca_loadings_amplitude_of_effortful.png", dpi=150); plt.close(fig)

        # Cosine similarity (shape-only similarity across conditions)
        Xu = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        cosM = Xu @ Xu.T
        fig, ax = plt.subplots(figsize=(6,5))
        im = ax.imshow(cosM, vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8)
        ax.set_title("Cosine similarity — spectral shape (Amplitude Δ%)")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout(); fig.savefig(outdir / "cosine_similarity_amplitude_of_effortful.png", dpi=150); plt.close(fig)

        # Parallel coordinates (z-score each condition so shape stands out)
        df_pc = []
        for i, lbl in enumerate(labels):
            x = X[i]
            z = (x - x.mean()) / (x.std() + 1e-12)
            row = {"label": lbl}
            row.update({ch: z[j] for j, ch in enumerate(channels)})
            df_pc.append(row)
        df_pc = pd.DataFrame(df_pc)

        fig = plt.figure(figsize=(8,5))
        ax = fig.gca()
        parallel_coordinates(df_pc, "label", color=None, ax=ax, linewidth=1.0, alpha=0.6)
        ax.set_title("Parallel coordinates (z-scored per condition) — Amplitude Δ%")
        ax.set_ylabel("z-score per condition")
        ax.legend([],[], frameon=False)  # hide cluttered legend
        fig.tight_layout(); fig.savefig(outdir / "parallel_coords_amplitude_of_effortful.png", dpi=150); plt.close(fig)

    # (Optional) Repeat for AUC%: copy the block above and replace 'amplitude_pct' with 'auc_pct',
    # and change filenames to '*_auc.png' to produce the AUC versions as well.

    print(f"Figures saved to: {outdir.resolve()}")

if __name__ == "__main__":
    main()
