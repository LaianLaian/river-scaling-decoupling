"""
Step 3. Climate modulation of runoff-efficiency scaling.

This script quantifies how the runoff-efficiency scaling exponent (beta) varies along a continuous precipitation gradient.

For each sliding window of rivers sorted by long-term mean precipitation (P), the following model is fitted:
    log10(R/P) = beta * log10(A) + c
where:
    R = runoff (mm)
    P = precipitation (mm)
    A = drainage area (km²)
    beta = runoff-efficiency scaling exponent

The workflow:

1. Load the cleaned river dataset.
2. Filter records with valid precipitation, runoff, and drainage area.
3. Calculate runoff efficiency (R/P).
4. Sort rivers by precipitation.
5. Apply sliding-window ordinary least squares (OLS) regression.
6. Estimate beta and its uncertainty for each window.
7. Quantify the relationship between beta and precipitation.
8. Fit a LOWESS curve to characterize nonlinear trends.
9. Generate publication-quality figures and summary tables.

"""

from __future__ import annotations

import json
import logging
import re
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

MERGED_DATA_PATH = Path(r"F:\Lake\Info.xlsx")
MERGED_SHEET_NAME = "Merge"
CLEANED_CSV_PATH: Path | None = None
OUTPUT_DIR = Path(r"F:\Lake\Result3")
MAIN_RESULTS_MD = Path(r"F:\Lake\Result\RESULTS.md")

WINDOW_SIZE = 2000
STEP_SIZE = 500
RP_MAX = 1.5
MIN_VALID_POINTS_IN_WINDOW = 100  # for stable local OLS

from run_hacks_scaling_analysis import (  # noqa: E402
    BOOTSTRAP_B,
    RNG,
    bootstrap_h,
    clean_for_hacks,
    fit_hacks_ols,
    load_merged,
    _set_pub_style,
)
COL_PRECIP_MM_CANDIDATES = (  "P_mm","P(mm)")

COL_RUNOFF_MM_CANDIDATES = (
    "R（mm）",
    "R(mm)",
    "R_mm",
    "ZB_R_mm",
    "ML_R_mm",
)


def _setup_logger() -> logging.Logger:
    log = logging.getLogger("rp_scaling_43")
    log.setLevel(logging.INFO)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.handlers.clear()
    log.addHandler(h)
    return log


LOG = _setup_logger()


def resolve_precipitation_column(df: pd.DataFrame) -> str | None:
    for c in COL_PRECIP_MM_CANDIDATES:
        if c in df.columns:
            return c
    return None


def resolve_runoff_column(df: pd.DataFrame) -> str | None:
    for c in COL_RUNOFF_MM_CANDIDATES:
        if c in df.columns:
            return c
    return None


def prepare_rp_scaling_data(cleaned: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Use main cleaned table (grades 0/99 already removed). Additional filters:
    P>0, R>0, A>0, RP=R/P, RP>0, RP<=RP_MAX.
    """
    meta: dict[str, Any] = {"module": "rp_scaling_sliding_beta", "RP_max_filter": RP_MAX}
    n0 = int(len(cleaned))
    meta["n_after_main_clean"] = n0

    pcol = resolve_precipitation_column(cleaned)
    rcol = resolve_runoff_column(cleaned)
    if pcol is None:
        raise ValueError( "Precipitation column not found. Tried: " + ", ".join(COL_PRECIP_MM_CANDIDATES))
    if rcol is None:
        raise ValueError( "Runoff column not found. Tried: " + ", ".join(COL_RUNOFF_MM_CANDIDATES))
    meta["precip_column"] = pcol
    meta["runoff_column"] = rcol

    d = cleaned.copy()
    d["P_mm"] = pd.to_numeric(d[pcol], errors="coerce")
    d["R_mm"] = pd.to_numeric(d[rcol], errors="coerce")
    d["A_km2"] = pd.to_numeric(d["A_km2"], errors="coerce")

    drops: list[dict[str, Any]] = []

    def _drop(mask: pd.Series, reason: str) -> None:
        nonlocal d
        before = len(d)
        d = d.loc[mask].copy()
        drops.append({"reason": reason, "removed": int(before - len(d)), "remaining": int(len(d))})

    _drop(d["P_mm"].notna() & d["R_mm"].notna() & d["A_km2"].notna(), "drop_na_P_R_A")
    _drop(d["P_mm"] > 0, "P_gt_0")
    _drop(d["R_mm"] > 0, "R_gt_0")
    _drop(d["A_km2"] > 0, "A_gt_0")

    d["RP"] = d["R_mm"] / d["P_mm"]
    _drop(d["RP"] > 0, "RP_gt_0")
    _drop(d["RP"] <= RP_MAX, f"RP_le_{RP_MAX}")

    meta["cleaning_drops"] = drops
    meta["n_raw_for_43"] = n0
    meta["n_final"] = int(len(d))
    meta["n_removed_total"] = int(n0 - len(d))

    rp = d["RP"].astype(float)
    meta["RP_stats"] = {
        "mean": float(rp.mean()),
        "median": float(rp.median()),
        "std": float(rp.std(ddof=1)) if len(rp) > 1 else float("nan"),
        "min": float(rp.min()),
        "max": float(rp.max()),
    }
    LOG.info(
        "4.3 prepare: n_main=%d -> n_final=%d (removed=%d); RP mean=%.4f median=%.4f",
        n0,
        len(d),
        n0 - len(d),
        meta["RP_stats"]["mean"],
        meta["RP_stats"]["median"],
    )
    return d, meta


def run_sliding_beta_analysis(
    df: pd.DataFrame,
    window_size: int = WINDOW_SIZE,
    step_size: int = STEP_SIZE,
) -> pd.DataFrame:
    """Sort by P ascending; sliding windows; each row = one window regression."""
    d = df.sort_values("P_mm", ascending=True).reset_index(drop=True)
    n = len(d)
    if n < window_size:
        raise ValueError(
            f"Sample size n={n} is smaller than " f"window_size={window_size}"
            "Reduce WINDOW_SIZE or provide more observations."
        )

    rows: list[dict[str, Any]] = []
    window_id = 0
    start = 0
    while start + window_size <= n:
        sub = d.iloc[start : start + window_size].copy()
        log_a = np.log10(sub["A_km2"].to_numpy(float))
        log_rp = np.log10(sub["RP"].to_numpy(float))
        m = np.isfinite(log_a) & np.isfinite(log_rp)
        n_valid = int(m.sum())
        if n_valid < MIN_VALID_POINTS_IN_WINDOW:
            LOG.warning(
                "window_id=%d start=%d: valid OLS points=%d < %d, skip",
                window_id,
                start,
                n_valid,
                MIN_VALID_POINTS_IN_WINDOW,
            )
            start += step_size
            window_id += 1
            continue

        log_a = log_a[m]
        log_rp = log_rp[m]
        o = fit_hacks_ols(log_a, log_rp)
        bmed, blo, bhi = bootstrap_h(log_a, log_rp, B=BOOTSTRAP_B, rng=RNG)

        pwin = sub.loc[m, "P_mm"]
        rows.append(
            {
                "window_id": window_id,
                "start_index": int(start),
                "end_index": int(start + window_size - 1),
                "n": n_valid,
                "P_mean": float(pwin.mean()),
                "P_median": float(pwin.median()),
                "P_min": float(pwin.min()),
                "P_max": float(pwin.max()),
                "beta": float(o["h"]),
                "intercept": float(o["intercept_c"]),
                "R2": float(o["r2"]),
                "p": float(o["p_slope"]),
                "beta_ci_low": float(o["h_ci95_low"]),
                "beta_ci_high": float(o["h_ci95_high"]),
                "beta_bootstrap_median": float(bmed) if np.isfinite(bmed) else float("nan"),
                "beta_bootstrap_ci_low": float(blo) if np.isfinite(blo) else float("nan"),
                "beta_bootstrap_ci_high": float(bhi) if np.isfinite(bhi) else float("nan"),
            }
        )
        start += step_size
        window_id += 1

    if not rows:
        raise RuntimeError(    "No valid sliding windows were generated. "
    "Please check WINDOW_SIZE and data quality.")
    LOG.info("sliding windows: %d windows fitted", len(rows))
    return pd.DataFrame(rows)


def trend_beta_vs_P_mean(win_tbl: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    from scipy.stats import pearsonr, spearmanr

    p = win_tbl["P_mean"].to_numpy(float)
    b = win_tbl["beta"].to_numpy(float)
    r_p, pv_p = pearsonr(p, b)
    r_s, pv_s = spearmanr(p, b)
    meta = {
        "pearson_r": float(r_p),
        "pearson_p": float(pv_p),
        "spearman_rho": float(r_s),
        "spearman_p": float(pv_s),
        "n_windows": float(len(win_tbl)),
    }
    tbl = pd.DataFrame(
        [
            {"test": "Pearson_P_mean_vs_beta", "statistic": float(r_p), "pvalue": float(pv_p), "n": len(win_tbl)},
            {"test": "Spearman_P_mean_vs_beta", "statistic": float(r_s), "pvalue": float(pv_s), "n": len(win_tbl)},
        ]
    )
    return tbl, meta


def lowess_beta_on_P(win_tbl: pd.DataFrame) -> pd.DataFrame:
    from statsmodels.nonparametric.smoothers_lowess import lowess

    w = win_tbl.sort_values("P_mean").reset_index(drop=True)
    x = w["P_mean"].to_numpy(float)
    y = w["beta"].to_numpy(float)
    n = len(x)
    frac = min(0.55, max(0.2, 12.0 / max(n, 5)))
    smoothed = lowess(y, x, frac=frac, it=3, return_sorted=True)
    out = pd.DataFrame({"P_mean": smoothed[:, 0], "beta_lowess": smoothed[:, 1]})
    out.attrs["lowess_frac"] = frac
    LOG.info("LOWESS on (P_mean, beta): n_windows=%d, frac=%.3f", n, frac)
    return out


def plot_beta_vs_precipitation(win_tbl: pd.DataFrame, smooth: pd.DataFrame, corr: dict[str, float], png: Path, pdf: Path) -> None:
    import matplotlib.pyplot as plt

    _set_pub_style()
    w = win_tbl.sort_values("P_mean").reset_index(drop=True)
    x = w["P_mean"].to_numpy(float)
    y = w["beta"].to_numpy(float)
    lo = w["beta_ci_low"].to_numpy(float)
    hi = w["beta_ci_high"].to_numpy(float)

    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    order = np.argsort(x)
    xs = x[order]
    ys = y[order]
    ylo = lo[order]
    yhi = hi[order]
    ax.fill_between(xs, ylo, yhi, color="0.75", alpha=0.55, linewidth=0, label="OLS 95% CI for beta")
    ax.plot(xs, ys, color="0.35", lw=1.0, alpha=0.85, label="Window estimates (polyline)")
    ax.scatter(x, y, s=28, c="darkgreen", edgecolors="white", linewidths=0.4, zorder=4, label="Sliding-window beta")

    sx = smooth["P_mean"].to_numpy(float)
    sy = smooth["beta_lowess"].to_numpy(float)
    ax.plot(sx, sy, color="crimson", lw=2.2, label="LOWESS smooth")

    ax.text(
        0.03,
        0.97,
        f"Pearson r = {corr['pearson_r']:.3f}, p = {corr['pearson_p']:.3g}\n"
        f"Spearman rho = {corr['spearman_rho']:.3f}, p = {corr['spearman_p']:.3g}\n"
        f"(across {int(corr['n_windows'])} windows)",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.6", alpha=0.92),
    )
    ax.set_xlabel("Mean precipitation in window, " + r"$\overline{P}$ (mm)")
    ax.set_ylabel(r"Scaling exponent $\beta$ in $\log_{10}(R/P) = \beta \log_{10}(A) + c$")
    ax.set_title("Continuous variation of the runoff-efficiency scaling exponent along the precipitation gradient")
    ax.legend(loc="best", fontsize=8)
    fig.savefig(png, bbox_inches="tight", dpi=180)
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def plot_regression_diagnostics(win_tbl: pd.DataFrame, png: Path, pdf: Path) -> None:
    import matplotlib.pyplot as plt

    _set_pub_style()
    w = win_tbl.sort_values("P_mean").reset_index(drop=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 4.2), constrained_layout=True)
    ax1.scatter(w["P_mean"], w["R2"], s=26, c="steelblue", edgecolors="white", linewidths=0.35, alpha=0.9)
    ax1.set_xlabel(r"$\overline{P}$ in window (mm)")
    ax1.set_ylabel(r"Local $R^2$")
    ax1.set_title("Local regression fit vs. precipitation")

    ax2.scatter(w["P_mean"], w["n"], s=26, c="darkorange", edgecolors="white", linewidths=0.35, alpha=0.9)
    ax2.set_xlabel(r"$\overline{P}$ in window (mm)")
    ax2.set_ylabel("Effective sample size n")
    ax2.set_title("Window size used in OLS (valid points)")
    fig.savefig(png, bbox_inches="tight", dpi=170)
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def build_results_section_4_3_md(meta: dict[str, Any], win_tbl: pd.DataFrame, corr: dict[str, Any]) -> str:
    b = win_tbl["beta"].astype(float)
    interp = []
    pr, ppr = corr["pearson_r"], corr["pearson_p"]
    sr, sps = corr["spearman_rho"], corr["spearman_p"]
    if ppr < 0.05:
        direction = "positive" if pr > 0 else "negative"
        interp.append(
            f"Across windows, Pearson correlation between **P_mean** and **beta** is statistically detectable (r={pr:.3f}, p={ppr:.3g}), suggesting a **{direction}** linear association in the window-level meta-relationship."
        )
    else:
        interp.append(
            f"Across windows, Pearson correlation is weak or not significant at conventional levels (r={pr:.3f}, p={ppr:.3g})."
        )
    if sps < 0.05 and (ppr >= 0.05 or abs(sr - pr) > 0.15):
        interp.append("Spearman results differ noticeably from Pearson, which can indicate **nonlinearity** or leverage from a few windows; compare the LOWESS curve to the pointwise pattern.")
    elif sps >= 0.05:
        interp.append("Spearman correlation is also not strong evidence of a monotone window-level trend; interpret the LOWESS curve as descriptive only.")

    lines = [
        "## Continuous climate dependence of runoff-efficiency scaling",
        "",
        "### 1. Data cleaning (this module, after main `clean_for_hacks`)",
        f"- Rows after main cleaning: **{meta.get('n_after_main_clean', '—')}**",
        f"- Final rows for 4.3: **{meta.get('n_final', '—')}**; removed in this module: **{meta.get('n_removed_total', '—')}**",
        f"- Precipitation column: `{meta.get('precip_column', '')}`; runoff column: `{meta.get('runoff_column', '')}`",
        f"- Filters: `P>0`, `R>0`, `A>0`, `RP=R/P`, `RP>0`, `RP<={meta.get('RP_max_filter', RP_MAX)}`",
        "",
        "### 2. RP summary (after filters)",
        f"- mean = **{meta['RP_stats']['mean']:.4f}**, median = **{meta['RP_stats']['median']:.4f}**, std = **{meta['RP_stats']['std']:.4f}**",
        f"- min = **{meta['RP_stats']['min']:.4f}**, max = **{meta['RP_stats']['max']:.4f}**",
        "",
        "### 3. Sliding-window regression",
        f"- Model (within each window): **log10(RP) = beta * log10(A) + c**",
        f"- `window_size` = **{meta.get('window_size', WINDOW_SIZE)}**, `step_size` = **{meta.get('step_size', STEP_SIZE)}**",
        f"- Number of windows fitted: **{len(win_tbl)}**",
        f"- Bootstrap replications per window: **{BOOTSTRAP_B}**",
        "",
        "### 4. Range of beta along the precipitation gradient",
        f"- min(beta) = **{float(b.min()):.4f}**, max(beta) = **{float(b.max()):.4f}**",
        "",
        "### 5. Window-level association: P_mean vs. beta",
        f"- Pearson **r** = **{pr:.4f}**, **p** = **{ppr:.4e}**",
        f"- Spearman **rho** = **{sr:.4f}**, **p** = **{sps:.4e}**",
        "",
        "### 6. Brief interpretation (descriptive; not causal)",
    ]
    lines.extend(["- " + s for s in interp])
    lines.append("")
    lines.append(
        "- Figures: **`Fig7_beta_vs_precipitation_continuous.png`** (core), **`Fig8_local_regression_diagnostics.png`** (R² and n vs. P_mean). "
        "Tables: **`rp_scaling_sliding_windows.csv`**, **`beta_vs_precipitation_smooth.csv`**, **`beta_P_window_correlation.csv`**."
    )
    lines.append("")
    return "\n".join(lines)


def embed_section_in_main_results(main_path: Path, section_md: str) -> None:
    if not main_path.exists():
        LOG.warning("Main RESULTS.md not found (%s), skip embed.", main_path)
        return
    start_m = "<!-- AUTO_SECTION_4_3_START -->\n"
    end_m = "<!-- AUTO_SECTION_4_3_END -->\n"
    block = start_m + section_md.strip() + "\n" + end_m
    text = main_path.read_text(encoding="utf-8")
    if start_m in text and end_m in text:
        text = re.sub(re.escape(start_m) + r".*?" + re.escape(end_m), block, text, count=1, flags=re.DOTALL)
    else:
        text = text.rstrip() + "\n\n" + block
    main_path.write_text(text, encoding="utf-8")
    LOG.info("Updated embedded Section 4.3 in %s", main_path)


def load_cleaned_data() -> pd.DataFrame:
    if CLEANED_CSV_PATH is not None and Path(CLEANED_CSV_PATH).exists():
        LOG.info("load cleaned CSV: %s", CLEANED_CSV_PATH)
        return pd.read_csv(CLEANED_CSV_PATH)
    raw = load_merged(MERGED_DATA_PATH, MERGED_SHEET_NAME)
    cleaned, _ = clean_for_hacks(raw)
    return cleaned


def run_all() -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cleaned = load_cleaned_data()
    df, prep_meta = prepare_rp_scaling_data(cleaned)
    prep_meta["window_size"] = WINDOW_SIZE
    prep_meta["step_size"] = STEP_SIZE

    win_tbl = run_sliding_beta_analysis(df, WINDOW_SIZE, STEP_SIZE)
    win_tbl.to_csv(OUTPUT_DIR / "rp_scaling_sliding_windows.csv", index=False, encoding="utf-8-sig")

    corr_tbl, corr_meta = trend_beta_vs_P_mean(win_tbl)
    corr_tbl.to_csv(OUTPUT_DIR / "beta_P_window_correlation.csv", index=False, encoding="utf-8-sig")

    smooth = lowess_beta_on_P(win_tbl)
    smooth.to_csv(OUTPUT_DIR / "beta_vs_precipitation_smooth.csv", index=False, encoding="utf-8-sig")

    plot_beta_vs_precipitation(
        win_tbl,
        smooth,
        corr_meta,
        OUTPUT_DIR / "Fig7_beta_vs_precipitation_continuous.png",
        OUTPUT_DIR / "Fig7_beta_vs_precipitation_continuous.pdf",
    )
    plot_regression_diagnostics(
        win_tbl,
        OUTPUT_DIR / "Fig8_local_regression_diagnostics.png",
        OUTPUT_DIR / "Fig8_local_regression_diagnostics.pdf",
    )

    prep_meta.update(corr_meta)
    prep_meta["beta_min"] = float(win_tbl["beta"].min())
    prep_meta["beta_max"] = float(win_tbl["beta"].max())
    prep_meta["n_windows"] = int(len(win_tbl))
    (OUTPUT_DIR / "rp_scaling_summary.json").write_text(
        json.dumps(prep_meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    section_md = build_results_section_4_3_md(prep_meta, win_tbl, corr_meta)
    (OUTPUT_DIR / "RESULTS_section_4_3.md").write_text(section_md, encoding="utf-8")
    embed_section_in_main_results(MAIN_RESULTS_MD, section_md)

    LOG.info("4.3 done -> %s", OUTPUT_DIR)
    return prep_meta


def main() -> int:
    try:
        run_all()
        return 0
    except Exception as e:
        LOG.exception("4.3 failed: %s", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
