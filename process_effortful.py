# process_spectraderma.py
# Usage:
#   pip install pandas numpy scipy pywt matplotlib
#   python process_spectraderma.py --csv "BTVIZ_2025-11-03_effortful_swallow_and_masako_maneuver_and_water.csv" --denoise
# Options:
#   --wavelet db2 --level 3 --zero-levels 1,2,3 --sample-rate 30 --out out_clean

import argparse, re
from pathlib import Path
import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt

def monotonic_local_time(series: pd.Series):
    """Make a monotonic local time axis from a numeric or datetime timestamp; fallback to sample index."""
    if series is None:
        return None
    s = series
    if pd.api.types.is_numeric_dtype(s):
        base = s.to_numpy(dtype=float)
        base = base - base[0]
        diffs = np.diff(base)
        pos = diffs[diffs > 0]
        step = np.median(pos) if pos.size > 0 else 1.0
        t = base.copy()
        for i in range(1, len(t)):
            if t[i] <= t[i-1]:
                t[i] = t[i-1] + step
        return t
    else:
        dt = pd.to_datetime(s, errors="coerce")
        if dt.notna().mean() > 0.7:
            return (dt - dt.iloc[0]).dt.total_seconds().to_numpy()
    return None

def swt_denoise_array(X, wavelet="db2", level=3, zero_levels=(1,2,3)):
    """Apply SWT denoising to a (T, C) array and return (T, C)."""
    X = np.asarray(X, float)
    T, C = X.shape
    block = 2 ** level
    T_pad = int(np.ceil(T / block) * block)
    pad = T_pad - T
    Y = np.empty((T, C), dtype=float)
    zl = {l for l in zero_levels if 1 <= l <= level}

    for c in range(C):
        xc = X[:, c]
        if pad > 0:
            xc = np.pad(xc, (0, pad), mode="reflect")
        coeffs = pywt.swt(xc, wavelet=wavelet, level=level, norm=True)
        new_coeffs = []
        for j, (cA, cD) in enumerate(coeffs, start=1):
            if j in zl:
                cD = np.zeros_like(cD)
            new_coeffs.append((cA, cD))
        xc_hat = pywt.iswt(new_coeffs, wavelet=wavelet, norm=True)
        if pad > 0:
            xc_hat = xc_hat[:T]
        Y[:, c] = xc_hat
    return Y

def normalize_percent(y):
    """Δ% = 100*(y - baseline)/baseline, baseline = median of first 10% or 20 samples."""
    n = len(y)
    if n == 0:
        return np.full(0, np.nan), np.nan
    b_len = max(20, int(0.1*n))
    b = float(np.nanmedian(y[:b_len]))
    if not np.isfinite(b) or b == 0:
        return np.full_like(y, np.nan, dtype=float), b
    return (y - b) / b * 100.0, b

def seg_metrics(y_pct, dt=1.0):
    """On normalized (Δ%) signal: baseline%, amplitude%, t_peak_idx, AUC% (≥0), FWHM (samples & seconds)."""
    n = len(y_pct)
    if n == 0:
        return dict(baseline_pct=np.nan, amplitude_pct=np.nan, t_peak_idx=np.nan,
                    auc_pct=np.nan, fwhm_samples=np.nan, fwhm_seconds=np.nan)
    b_len = max(20, int(0.1*n))
    baseline_pct = float(np.nanmedian(y_pct[:b_len]))
    yb = y_pct - baseline_pct
    amplitude_pct = float(np.nanmax(yb))
    t_peak_idx = int(np.nanargmax(yb))
    auc_pct = float(np.nansum(np.maximum(yb, 0.0))) * dt  # %·s if dt in seconds
    if np.isfinite(amplitude_pct) and amplitude_pct > 0:
        half = amplitude_pct/2.0
        idx = np.where(yb >= half)[0]
        fwhm_samples = float(idx[-1] - idx[0] + 1) if len(idx) >= 2 else np.nan
    else:
        fwhm_samples = np.nan
    return dict(baseline_pct=baseline_pct,
                amplitude_pct=amplitude_pct,
                t_peak_idx=t_peak_idx,
                auc_pct=auc_pct,
                fwhm_samples=fwhm_samples,
                fwhm_seconds=(fwhm_samples*dt if np.isfinite(fwhm_samples) else np.nan))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="out_clean")
    ap.add_argument("--denoise", action="store_true", help="Apply SWT denoising")
    ap.add_argument("--wavelet", default="db2")
    ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--zero-levels", default="1,2,3", help="Comma list, e.g. 1,2,3 or 2,3")
    ap.add_argument("--sample-rate", type=float, default=30.0, help="Hz (for AUC/FWHM in seconds)")
    ap.add_argument("--plot-channels", action="store_true",
                help="Generate per-(location,gulp) plots with all 6 channels (uses denoised signals if --denoise is set)")
    ap.add_argument("--plot-channels-mode", default="raw", choices=["raw","norm"],
                help="Plot 'raw' filtered signals or 'norm' (Δ%% from baseline) per channel")

    args = ap.parse_args()

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv, low_memory=False).ffill()

    # Detect labels
    def norm_loc(x):
        t = str(x).strip().lower()
        if t.startswith("First right"): return "first right"
        if t.startswith("first left"):  return "first left"
        if "top" in t and "middle" in t and "(actual)" in t: return "top middle"
        if "bottom" in t and "middle" in t: return "bottom middle"
        if t.startswith("second left"): return "second left"
        if t.startswith("second right"): return "second right"
        return t
    def norm_gulp(x):
        t = str(x).strip().lower()
        if "Effortful swallow" not in t: return ""
        m = re.search(r"Effortful swallow\s*([12345])", t)
        return f"Effortful swallow {m.group(1)}" if m else "Effortful swallow"

    df["_location"] = df["Environment"].map(norm_loc) if "Environment" in df.columns else ""
    df["_gulp"] = df["Activity"].map(norm_gulp) if "Activity" in df.columns else ""

    # Choose time and channel columns
    time_col = "notificationTimestamp" if "notificationTimestamp" in df.columns else ("timestamp" if "timestamp" in df.columns else None)
    channel_cols = [c for c in df.columns if c.startswith("data_")]
    X = df[channel_cols].to_numpy(dtype=float)

    # Optional SWT denoise
    if args.denoise:
        zl = tuple(int(x) for x in args.zero_levels.split(",") if x.strip())
        X = swt_denoise_array(X, wavelet=args.wavelet, level=args.level, zero_levels=zl)
        df[channel_cols] = X
        df.to_csv(outdir / "denoised_signals_of_effortful.csv", index=False)

    # Build monotonic local time per segment when plotting; metrics use dt=1/fs
    fs = float(args.sample_rate); dt = 1.0/fs

    # === Compute per-segment metrics (mean across channels + channel-wise) ===
    rows = []
    locations = [v for v in ["top middle", "bottom middle","first right","first left", "second right", "second left"] if (df["_location"]==v).any()]
    gulps = [v for v in ["Effortful swallow 1","Effortful swallow 2","Effortful swallow 3", "Effortful swallow 4", "Effortful swallow 5"] if (df["_gulp"]==v).any()]

    for loc in locations:
        for g in gulps:
            seg = df[(df["_location"]==loc) & (df["_gulp"]==g)]
            if len(seg) < 5:
                continue
            Y = seg[channel_cols].to_numpy(float)
            ymean = np.nanmean(Y, axis=1)
            ymean_pct, _ = normalize_percent(ymean)
            m = seg_metrics(ymean_pct, dt=dt)
            rows.append(dict(location=loc, gulp=g, channel="mean", n=len(seg), **m))
            # channel-wise too
            for i, cname in enumerate(channel_cols):
                ypc, _ = normalize_percent(Y[:, i])
                m = seg_metrics(ypc, dt=dt)
                rows.append(dict(location=loc, gulp=g, channel=cname, n=len(seg), **m))

    metrics = pd.DataFrame(rows)
    metrics.to_csv(outdir / "metrics_pct_of_effortful.csv", index=False)

    print("metrics columns:", metrics.columns.tolist())
    print("metrics shape:", metrics.shape)
    print("unique channels:", metrics["channel"].unique() if "channel" in metrics.columns else "no channel column")
    print("locations:", locations)
    print("gulps:", gulps)


    # === Pivot tables (what you asked to compare) ===
    mean_only = metrics[metrics["channel"]=="mean"].copy()
    amp_tbl = mean_only.pivot_table(index="location", columns="gulp", values="amplitude_pct", aggfunc="mean")
    auc_tbl = mean_only.pivot_table(index="location", columns="gulp", values="auc_pct", aggfunc="mean")
    amp_tbl.to_csv(outdir / "pivot_mean_amplitude_pct_by_loc_gulp_of_effortful.csv")
    auc_tbl.to_csv(outdir / "pivot_mean_auc_pct_by_loc_gulp_of_effortful.csv")

    # === Quick overlays (normalized mean) ===
    def time_axis_for(seg):
        if time_col is None: return np.arange(len(seg))
        t = monotonic_local_time(seg[time_col])
        return t if t is not None else np.arange(len(seg))

    # per location: overlay gulps
    for loc in locations:
        plt.figure(figsize=(10,4)); any_plot=False
        for g in gulps:
            seg = df[(df["_location"]==loc) & (df["_gulp"]==g)]
            if len(seg) < 5: continue
            t = time_axis_for(seg)
            Y = seg[channel_cols].to_numpy(float)
            ymean = np.nanmean(Y, axis=1)
            ypc, _ = normalize_percent(ymean)
            plt.plot(t, ypc, label=g); any_plot=True
        if any_plot:
            plt.title(f"{loc} — gulps overlay (Δ% from baseline)"); plt.xlabel("time (a.u.)"); plt.ylabel("Δ%")
            plt.legend(); plt.tight_layout()
            plt.savefig(outdir / f"{loc.replace(' ','_')}_gulps_overlay_norm_of_effortful.png", dpi=150); plt.close()

        # per gulp: overlay locations
        for g in gulps:
            plt.figure(figsize=(10,4)); any_plot=False
            for loc in locations:
                seg = df[(df["_location"]==loc) & (df["_gulp"]==g)]
                if len(seg) < 5: continue
                t = time_axis_for(seg)
                Y = seg[channel_cols].to_numpy(float)
                ymean = np.nanmean(Y, axis=1)
                ypc, _ = normalize_percent(ymean)
                plt.plot(t, ypc, label=loc); any_plot=True
            if any_plot:
                plt.title(f"{g} — locations overlay (Δ% from baseline)"); plt.xlabel("time (a.u.)"); plt.ylabel("Δ%")
                plt.legend(); plt.tight_layout()
                plt.savefig(outdir / f"{g.replace(' ','_')}_locations_overlay_norm_of_effortful.png", dpi=150); plt.close()
        # === NEW: Per-(location, gulp) plots showing all 6 channels ===
    # Uses SWT-filtered signals if you ran with --denoise (because df[channel_cols] was overwritten)

    def slug(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")

    # Reuse the same location/gulp sets and channel list already computed above
    channel_cols = [c for c in df.columns if c.startswith("data_")]

    # Helper to get a time axis (reuses your earlier helper if present)
    def _time_axis_for(seg):
        try:
            return time_axis_for(seg)  # you defined this earlier in the script
        except NameError:
            # Fallback: monotonic index if helper not available
            return np.arange(len(seg), dtype=float)

    if args.plot_channels:
        chan_dir = outdir / "channel_plots_of_effortful"
        chan_dir.mkdir(parents=True, exist_ok=True)

        for loc in locations:
            for g in gulps:
                seg = df[(df["_location"] == loc) & (df["_gulp"] == g)]
                if len(seg) < 5:
                    continue

                t = _time_axis_for(seg)
                Y = seg[channel_cols].to_numpy(dtype=float)

                fig, ax = plt.subplots(figsize=(10, 4))

                if args.plot_channels_mode == "norm":
                    # Δ% normalization per channel (same baseline rule you use elsewhere)
                    for i, cname in enumerate(channel_cols):
                        y = Y[:, i]
                        n = len(y)
                        b_len = max(20, int(0.1 * n))
                        b = float(np.nanmedian(y[:b_len])) if n > 0 else np.nan
                        ypc = np.full_like(y, np.nan, dtype=float) if (not np.isfinite(b) or b == 0) else (y - b) / b * 100.0
                        ax.plot(t, ypc, label=cname)
                    ax.set_ylabel("Δ% from baseline")
                    suffix = "norm_of_effortful"
                else:
                    # Raw filtered units (after SWT if --denoise)
                    for i, cname in enumerate(channel_cols):
                        ax.plot(t, Y[:, i], label=cname)
                    ax.set_ylabel("Signal (a.u.)")
                    suffix = "raw_of_effortful"

                ax.set_title(f"{loc} — {g} (6 channels, {'SWT' if args.denoise else 'raw'})")
                ax.set_xlabel("Time (a.u.)")
                ax.legend(ncol=3, fontsize=8)
                fig.tight_layout()
                fig.savefig(chan_dir / f"{slug(loc)}_{slug(g)}_channels_{suffix}_of_effortful.png", dpi=150)
                plt.close(fig)

    print("Done. Outputs (of effortful) in:", outdir.as_posix())

if __name__ == "__main__":
    main()
