#!/usr/bin/env python3
"""
make_clinical_ui_mockup.py

Generates a clinician/patient-facing UI mockup figure from your BTVIZ CSV and metrics CSV.

Default behavior reproduces the "CARDUI9" style mockup:
- labeled-window-only x-axis (dotted bounds on both sides)
- trough detection on detrended MEAN trace (rolling median)
- primary trough = most prominent detrended trough
- local troughs = closest 6 in the last 25s before primary
- per-channel Δ% traces + rolling baseline line
- right-side cards with zebra rows and baseline "pill" at top-right of Channels card

Example:
python scripts/mockup_UI.py \
  --csv data/BTVIZ_2025-11-03_effortful_swallow_and_masako_maneuver_and_water.csv \
  --metrics data/metrics_pct_of_effortful.csv \
  --location "first right" \
  --trial "3oz water (3)" \
  --out out/clinical_ui_first_right_3oz_water_3_CLOSEST_CARDUI9.png
"""

import argparse
import os
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from scipy.signal import find_peaks


# ----------------------------
# Label parsing (matches your pipeline behavior)
# ----------------------------
def normalize_label_text(x: str) -> str:
    if x is None:
        return ""
    t = str(x).strip().lower()
    return re.sub(r"[_\s]+", " ", t)


def norm_loc(x):
    t = normalize_label_text(x)
    if "first right" in t: return "first right"
    if "first left" in t: return "first left"
    if "second right" in t: return "second right"
    if "second left" in t: return "second left"
    if "top" in t and "middle" in t: return "top middle"
    if "bottom" in t and "middle" in t: return "bottom middle"
    return t


def classify_task(x: str) -> str:
    t = normalize_label_text(x)
    if "effortful" in t: return "effortful swallow"
    if "masako" in t: return "masako maneuver"
    if "oz" in t or "water" in t or "gulp" in t: return "3oz water"
    return ""


def norm_gulp_text(x: str) -> str:
    raw = normalize_label_text(x)
    task = classify_task(raw)
    if task:
        m = re.findall(r"(\d+)", raw)
        if m:
            n = m[-1]
            if task == "3oz water":
                return f"3oz water ({n})"
            return f"{task} {n}"
    return task

def has_true_trial_number(label: str) -> bool:
    t = normalize_label_text(label)
    return re.match(
        r"^(effortful swallow|masako maneuver|3oz water)(?:\s+trial)?\s*\(?\s*(\d+)\s*\)?$",
        t,
    ) is not None


# ----------------------------
# Time axis
# ----------------------------
def monotonic_local_time(series: pd.Series):
    """Make a monotonic local time axis from numeric or datetime; fallback to None."""
    if series is None:
        return None

    if pd.api.types.is_numeric_dtype(series):
        base = series.to_numpy(dtype=float)
        base = base - base[0]
        diffs = np.diff(base)
        pos = diffs[diffs > 0]
        step = np.median(pos) if pos.size > 0 else 1.0
        t = base.copy()
        for i in range(1, len(t)):
            if t[i] <= t[i - 1]:
                t[i] = t[i - 1] + step
        return t

    dt = pd.to_datetime(series, errors="coerce")
    if dt.notna().mean() > 0.7:
        return (dt - dt.iloc[0]).dt.total_seconds().to_numpy()

    return None


# ----------------------------
# Small smoothing for plotting (UI-friendly)
# ----------------------------
def smooth_moving_average(x: np.ndarray, k: int = 5) -> np.ndarray:
    if k <= 1:
        return x
    k = int(k)
    if k % 2 == 0:
        k += 1
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    ker = np.ones(k, dtype=float) / k
    return np.convolve(xp, ker, mode="valid")


# ----------------------------
# CARD UI drawing helpers
# ----------------------------
def add_card(ax, x, y, w, h, title, face="#FFFFFF", edge="#D9E7EF", title_pad_frac=0.22):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.0,
        edgecolor=edge,
        facecolor=face
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.03 * w,
        y + h - (title_pad_frac * h),
        title,
        fontsize=12,
        weight="bold",
        color="#1A365D",
        va="center"
    )
    return patch

def add_zebra_kv(ax, x, y, w, h, rows, font=10.6, key_w=0.56, pad=0.02):
    n = max(len(rows), 1)
    rh = h / n
    for i, (k, v) in enumerate(rows):
        yy = y + h - (i + 1) * rh
        bg = "#F6FAFC" if i % 2 == 0 else "#FFFFFF"
        ax.add_patch(Rectangle((x, yy), w, rh, facecolor=bg, edgecolor="none"))
        ax.text(x + pad * w, yy + rh / 2, str(k), va="center", ha="left", fontsize=font, color="#4A5568")
        ax.text(x + key_w * w, yy + rh / 2, str(v), va="center", ha="left", fontsize=font, color="#1A202C", weight="bold")


# ----------------------------
# Main generator
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to raw BTVIZ CSV")
    ap.add_argument("--metrics", required=False, default=None, help="Path to metrics_pct_of_effortful.csv (optional)")
    ap.add_argument("--location", default="first right")
    ap.add_argument("--trial", default="3oz water (3)")
    ap.add_argument("--out", default="clinical_ui_mockup.png")
    ap.add_argument("--fs", type=float, default=30.0)

    # Trough detection settings (match current figure defaults)
    ap.add_argument("--rolling_median_s", type=float, default=7.0)
    ap.add_argument("--local_window_s", type=float, default=25.0)
    ap.add_argument("--local_n", type=int, default=6)
    ap.add_argument("--min_sep_s", type=float, default=0.7)
    ap.add_argument("--prom_mad0", type=float, default=1.00)
    ap.add_argument("--height_mad0", type=float, default=0.50)

    # Plot aesthetics
    ap.add_argument("--smooth_k", type=int, default=5)
    args = ap.parse_args()

    csv_path = args.csv
    metrics_csv = args.metrics
    fs = float(args.fs)
    dt = 1.0 / fs

    df = pd.read_csv(csv_path, low_memory=False)

    # Clean labels, ffill only data columns
    for col in ["Activity", "Environment"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace("", np.nan)

    channel_cols = [c for c in df.columns if c.startswith("data_")]
    if not channel_cols:
        raise RuntimeError("No channel columns found (expected columns starting with 'data_').")

    df[channel_cols] = df[channel_cols].ffill()

    # Build _location and _gulp exactly like the earlier logic
    df["_location"] = df["Environment"].map(norm_loc) if "Environment" in df.columns else ""
    df["task"] = df["Activity"].map(classify_task)
    df["_gulp_raw"] = df["Activity"].map(norm_gulp_text)

    has_true_numbers = df["_gulp_raw"].apply(has_true_trial_number)
    if has_true_numbers.any():
        df["_gulp"] = df["_gulp_raw"]
    else:
        counters = defaultdict(int)
        gulp_labels = []
        prev_gap = True
        for task, loc, act, env in zip(df["task"], df["_location"], df["Activity"], df["Environment"]):
            if (str(act).strip() == "" or pd.isna(act)) and (str(env).strip() == "" or pd.isna(env)):
                gulp_labels.append("")
                prev_gap = True
                continue

            if task and loc:
                if prev_gap:
                    counters[task] += 1
                gulp_labels.append(f"{task} {counters[task]}")
                prev_gap = False
            else:
                gulp_labels.append("")
                prev_gap = True

        df["_gulp"] = gulp_labels

    loc = normalize_label_text(args.location)
    trial = normalize_label_text(args.trial)

    if "3oz water" in trial:
        m = re.findall(r"(\d+)", trial)
        if m:
            trial = f"3oz water ({m[-1]})"

    seg = df[(df["_location"] == loc) & (df["_gulp"].astype(str).str.strip().str.lower() == trial)].copy()
    if seg.empty:
        raise RuntimeError(f"No rows found for location='{loc}' and trial='{trial}'. Check labels in CSV.")

    # time axis
    time_col = "notificationTimestamp" if "notificationTimestamp" in seg.columns else ("timestamp" if "timestamp" in seg.columns else None)
    t = monotonic_local_time(seg[time_col]) if time_col else None
    if t is None:
        t = np.arange(len(seg), dtype=float)

    # signals
    Y = seg[channel_cols].to_numpy(float)
    ymean = np.nanmean(Y, axis=1)

    n = len(ymean)
    b_len = max(20, int(0.1 * n))
    baseline = float(np.nanmedian(ymean[:b_len]))
    if not np.isfinite(baseline) or baseline == 0:
        raise RuntimeError("Baseline invalid (nan or 0).")

    ymean_pct = (ymean - baseline) / baseline * 100.0

    # rolling median detrend
    win_s = float(args.rolling_median_s)
    win = max(3, int(round(win_s * fs)))
    if win % 2 == 0:
        win += 1
    roll_med = pd.Series(ymean_pct).rolling(window=win, center=True, min_periods=max(3, win // 3)).median().to_numpy()
    detr = ymean_pct - roll_med

    finite = np.isfinite(detr)
    if not finite.any():
        raise RuntimeError("Detrended signal has no finite values.")

    mad = np.nanmedian(np.abs(detr[finite] - np.nanmedian(detr[finite])))
    mad0 = mad if np.isfinite(mad) and mad > 0 else float(np.nanstd(detr[finite]))
    if not np.isfinite(mad0) or mad0 <= 0:
        mad0 = 1.0

    prom = float(args.prom_mad0) * mad0
    height = float(args.height_mad0) * mad0
    min_sep_s = float(args.min_sep_s)
    distance = max(1, int(round(min_sep_s * fs)))

    # troughs on detrended mean
    trough_idx, props = find_peaks(-detr, prominence=prom, height=height, distance=distance)

    if len(trough_idx) == 0:
        raise RuntimeError("No troughs detected with current thresholds. Try lowering prom/height or min_sep.")

    prominences = props.get("prominences", np.zeros(len(trough_idx)))
    primary_idx = int(trough_idx[int(np.argmax(prominences))])

    # local troughs = closest 6 in last 25s pre-primary
    local_window_s = float(args.local_window_s)
    local_n = int(args.local_n)
    pre = trough_idx[trough_idx < primary_idx]
    pre = pre[pre >= primary_idx - int(round(local_window_s * fs))]
    locals_idx = list(pre[-local_n:]) if len(pre) else []
    locals_idx = [int(i) for i in locals_idx]

    # helper to compute boundaries for shading around a trough
    def trough_metrics(idx: int):
        all_tr = np.array(sorted(set(list(trough_idx) + [primary_idx])), dtype=int)
        all_tr = all_tr[(all_tr >= 0) & (all_tr < len(detr))]

        left = 0
        right = len(detr) - 1
        pos = np.where(all_tr == idx)[0]
        if len(pos):
            k = int(pos[0])
            if k > 0:
                left = int((all_tr[k - 1] + idx) / 2)
            if k < len(all_tr) - 1:
                right = int((idx + all_tr[k + 1]) / 2)

        depth = float(-detr[idx])
        dur = (right - left) * dt
        t_to = (idx - left) * dt
        t_rec = (right - idx) * dt
        half = -depth / 2.0
        segd = detr[left:right + 1]
        below = np.where(segd <= half)[0]
        t_below = float(len(below) * dt)

        return dict(left=left, right=right, depth=depth, duration=dur, time_to_trough=t_to, recovery=t_rec, time_below_half=t_below)

    primary_metrics = trough_metrics(primary_idx)

    # trough summary
    trough_times = trough_idx * dt
    inter = np.diff(trough_times) if len(trough_times) >= 2 else np.array([])
    inter_mean = float(np.nanmean(inter)) if inter.size else np.nan
    inter_cov = float(np.nanstd(inter) / np.nanmean(inter)) if inter.size and np.nanmean(inter) != 0 else np.nan

    # ----------------------------
    # Build figure (CARDUI9 style)
    # ----------------------------
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(14.5, 6.8), facecolor="#EAF6F8")
    gs = fig.add_gridspec(1, 2, width_ratios=[2.55, 1.05], wspace=0.08)

    # Left plot
    ax = fig.add_subplot(gs[0, 0])
    ax.set_facecolor("#EAF6F8")

    # Plot channels Δ%
    lines = []
    for i, cname in enumerate(channel_cols):
        y = Y[:, i]
        b = float(np.nanmedian(y[:b_len]))
        ypct = (y - b) / b * 100.0 if np.isfinite(b) and b != 0 else np.full_like(y, np.nan)
        ln, = ax.plot(t, smooth_moving_average(ypct, args.smooth_k), lw=1.5, alpha=0.95)
        lines.append(ln)

    # Rolling baseline line (mean trace rolling median)
    ax.plot(t, smooth_moving_average(roll_med, args.smooth_k), lw=1.5, alpha=0.85, color="#C77DFF")

    # dotted bounds of labeled window
    ax.axvline(t[0], color="#2B6CB0", lw=2.0, ls=(0, (4, 4)), alpha=0.85)
    ax.axvline(t[-1], color="#2B6CB0", lw=2.0, ls=(0, (4, 4)), alpha=0.85)

    # local trough bands (light) + numbering
    shade_color = "#BEE3F8"
    y_top = float(np.nanmax(ymean_pct)) if np.isfinite(np.nanmax(ymean_pct)) else 25.0

    for k, idx in enumerate(locals_idx, start=1):
        m = trough_metrics(idx)
        # slightly tighter band around trough
        left = max(m["left"], idx - int(round(0.6 * fs)))
        right = min(m["right"], idx + int(round(0.6 * fs)))
        ax.axvspan(t[left], t[right], color=shade_color, alpha=0.45, zorder=0)
        ax.text(t[idx], y_top * 0.95, f"{k}", ha="center", va="top", fontsize=12, color="#1A365D")

    # primary gulp band (darker)
    pm = primary_metrics
    ax.axvspan(t[pm["left"]], t[pm["right"]], color="#90CDF4", alpha=0.55, zorder=0)
    ax.text(
        t[primary_idx] + (t[-1] - t[0]) * 0.01,
        float(np.nanmin(ymean_pct)) * 0.2,
        "Primary gulp",
        fontsize=10,
        color="#1A365D",
        weight="bold"
    )

    ax.set_title(
        "Clinician/patient-facing summary screen (concept figure from real trial data)\n"
        f"{loc} — {trial} — multi-wavelength signals (Δ%)\n"
        "local troughs = closest 6 within last 25s pre-primary (labeled window only)",
        fontsize=12,
        color="#1A202C",
    )
    ax.set_xlabel("time (a.u.)")
    ax.set_ylabel("Δ% from baseline")
    ax.grid(True, alpha=0.15)

    # Right panel
    axr = fig.add_subplot(gs[0, 1])
    axr.axis("off")
    axr.set_facecolor("#EAF6F8")

    # Channels card
    cx, cy, cw, ch = 0.02, 0.80, 0.96, 0.17
    add_card(axr, cx, cy, cw, ch, "Channels", face="#D7ECFB", edge="#B6D9F7", title_pad_frac=0.22)

    # Two columns, two entries per row
    sw_x1 = cx + 0.06 * cw
    sw_x2 = cx + 0.55 * cw
    start_y = cy + 0.62 * ch
    dy = 0.20 * ch

    for i, cname in enumerate(channel_cols):
        col = 0 if i % 2 == 0 else 1
        row = i // 2
        xx = sw_x1 if col == 0 else sw_x2
        yy = start_y - row * dy
        color = lines[i].get_color()
        axr.add_patch(Rectangle((xx, yy - 0.02), 0.05, 0.018, facecolor=color, edgecolor=color))
        axr.text(xx + 0.07, yy - 0.012, cname, fontsize=10.2, color="#1A202C", va="center")

    # Rolling baseline pill at far top-right of Channels card (same header row)
    pill_text = f"rolling baseline\n(median, {win_s:.1f}s)"
    pill_w = 0.36 * cw
    pill_h = 0.34 * ch
    pill_x = cx + cw - pill_w - 0.04 * cw
    pill_y = cy + ch - pill_h - 0.08 * ch
    pill = FancyBboxPatch(
        (pill_x, pill_y), pill_w, pill_h,
        boxstyle="round,pad=0.01,rounding_size=0.03",
        linewidth=1.0,
        edgecolor="#B6D9F7",
        facecolor="#FFFFFF"
    )
    axr.add_patch(pill)
    axr.text(pill_x + 0.06 * pill_w, pill_y + 0.55 * pill_h, pill_text, fontsize=9.6, color="#1A202C", va="center")

    # Summary metrics card
    sx, sy, sw, sh = 0.02, 0.48, 0.96, 0.30
    add_card(axr, sx, sy, sw, sh, "Summary metrics", title_pad_frac=0.20)

    rows1 = [
        ("Task", args.trial),
        ("Location", args.location),
        ("Segmentation", "A (labeled rows only)"),
        ("Plot window", "labeled window only"),
        ("Local select", f"{args.local_n} closest within last {args.local_window_s:.0f}s pre-primary"),
        ("Trough detection", "detrended mean"),
        ("Detrend", f"rolling median {win_s:.1f} s"),
    ]
    add_zebra_kv(axr, sx + 0.02 * sw, sy + 0.10 * sh, sw * 0.96, sh * 0.72, rows1, font=10.6, key_w=0.52)
    axr.text(
        sx + 0.04 * sw,
        sy + 0.03 * sh,
        f"find_peaks: prom {args.prom_mad0:.2f}×MAD₀, height {args.height_mad0:.2f}×MAD₀, min_sep {args.min_sep_s:.1f}s",
        fontsize=9.6,
        color="#4A5568"
    )

    # Trough summary card (padding so title doesn't overlap first row)
    tx, ty, tw, th = 0.02, 0.30, 0.96, 0.15
    add_card(axr, tx, ty, tw, th, "Trough summary", title_pad_frac=0.22)

    rows2 = [
        ("Troughs (labeled window)", len(trough_idx)),
        ("Local troughs shown", len(locals_idx)),
        ("Inter-trough mean (s)", f"{inter_mean:.2f}" if np.isfinite(inter_mean) else "—"),
        ("Inter-trough CoV", f"{inter_cov:.2f}" if np.isfinite(inter_cov) else "—"),
    ]
    add_zebra_kv(axr, tx + 0.02 * tw, ty + 0.12 * th, tw * 0.96, th * 0.70, rows2, font=10.6, key_w=0.60)

    # Primary gulp metrics card
    px, py, pw, ph = 0.02, 0.11, 0.96, 0.18
    add_card(axr, px, py, pw, ph, "Primary gulp metrics (detrended)", title_pad_frac=0.22)

    rows3 = [
        ("Depth", f"{primary_metrics['depth']:.2f}"),
        ("Duration (window) (s)", f"{primary_metrics['duration']:.2f}"),
        ("Time to trough (s)", f"{primary_metrics['time_to_trough']:.2f}"),
        ("Recovery time (s)", f"{primary_metrics['recovery']:.2f}"),
        ("Time below half-depth (s)", f"{primary_metrics['time_below_half']:.2f}"),
    ]
    add_zebra_kv(axr, px + 0.02 * pw, py + 0.12 * ph, pw * 0.96, ph * 0.70, rows3, font=10.6, key_w=0.62)

    # From pipeline (mean trace) card
    mx, my, mw, mh = 0.02, 0.01, 0.96, 0.09
    add_card(axr, mx, my, mw, mh, "From pipeline (mean trace)", title_pad_frac=0.24)

    pipe = {"Amplitude (Δ%)": "—", "AUC (Δ%·s)": "—", "FWHM (s)": "—", "n_humps (pipeline)": "—"}
    if metrics_csv and os.path.exists(metrics_csv):
        mm = pd.read_csv(metrics_csv)
        mrow = mm[
            (mm["location"].astype(str).str.lower() == loc) &
            (mm["gulp"].astype(str).str.lower() == trial) &
            (mm["channel"].astype(str) == "mean")
        ]
        if len(mrow):
            r = mrow.iloc[0]
            pipe = {
                "Amplitude (Δ%)": f"{float(r.get('amplitude_pct', np.nan)):.2f}" if np.isfinite(r.get("amplitude_pct", np.nan)) else "—",
                "AUC (Δ%·s)": f"{float(r.get('auc_pct', np.nan)):.2f}" if np.isfinite(r.get("auc_pct", np.nan)) else "—",
                "FWHM (s)": f"{float(r.get('fwhm_seconds', np.nan)):.2f}" if np.isfinite(r.get("fwhm_seconds", np.nan)) else "—",
                "n_humps (pipeline)": f"{int(r.get('n_humps', 0))}" if np.isfinite(r.get("n_humps", np.nan)) else "—",
            }

    rows4 = list(pipe.items())
    add_zebra_kv(axr, mx + 0.02 * mw, my + 0.15 * mh, mw * 0.96, mh * 0.68, rows4, font=10.2, key_w=0.62)

    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"Saved: {out_path.resolve()}")


if __name__ == "__main__":
    main()
