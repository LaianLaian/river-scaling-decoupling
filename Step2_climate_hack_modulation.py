"""
Step 2. Climate modulation of river scaling.

This script quantifies the relationship between long-term mean precipitation (P) and river-scale Hack exponent (h).

For each river:
    h = log10(L_km) / log10(A_km2)
where:
    L_km   = river length (km)
    A_km2  = drainage area (km²)

The script:
1. Loads the merged river dataset.
2. Applies the same quality-control procedures used in the  main scaling analysis.
3. Calculates river-level h values.
4. Evaluates correlations between precipitation and h.
5. Produces scatterplots and precipitation-group boxplots.
6. Exports summary tables and figures.

"""

from __future__ import annotations

import json
import logging
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
OUTPUT_DIR = Path(r"F:\Lake\Result")

from run_hacks_scaling_analysis import (  # noqa: E402
    clean_for_hacks,
    load_merged,
    _set_pub_style,
)

COL_PRECIP_MM_CANDIDATES = (  "P_mm","P(mm)")
# Avoid division by zero when log10(A)=0
# (A = 1 km² exactly)
MIN_LOG10_A_ABS = 1e-6
# Number of precipitation quantile bins
# used for boxplot visualization
N_P_QUANTILE_BINS_FOR_BOXPLOT = 25


def _setup_logger() -> logging.Logger:
    log = logging.getLogger("p_h_42")
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


def build_p_h_per_river(cleaned: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """P_mm & h=log10(L)/log10(A)"""
    meta: dict[str, Any] = {"module": "P_vs_h_per_river", "h_definition": "log10(L_km) / log10(A_km2)"}
    col = resolve_precipitation_column(cleaned)
    if col is None:
        meta["error"] = "Not found: " + ", ".join(COL_PRECIP_MM_CANDIDATES)
        LOG.warning(meta["error"])
        return pd.DataFrame(), meta

    df = cleaned.copy()
    df["P_mm"] = pd.to_numeric(df[col], errors="coerce")
    df["L_km"] = pd.to_numeric(df["L_km"], errors="coerce")
    df["A_km2"] = pd.to_numeric(df["A_km2"], errors="coerce")
    ok = df["P_mm"].notna() & df["L_km"].notna() & df["A_km2"].notna()
    ok = ok & (df["L_km"] > 0) & (df["A_km2"] > 0)
    df = df.loc[ok].copy()
    meta["precip_column_used"] = col
    meta["n_after_basic_filters"] = int(len(df))
    if len(df) == 0:
        meta["error"] = "no rows with valid P, L, A"
        return df, meta

    log_a = np.log10(df["A_km2"].to_numpy(float))
    log_l = np.log10(df["L_km"].to_numpy(float))
    valid = np.isfinite(log_a) & np.isfinite(log_l) & (np.abs(log_a) > MIN_LOG10_A_ABS)
    df = df.loc[valid].copy()
    log_a = np.log10(df["A_km2"].to_numpy(float))
    log_l = np.log10(df["L_km"].to_numpy(float))
    df["h"] = log_l / log_a
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["h", "P_mm"])
    meta["n_final"] = int(len(df))
    if len(df) == 0:
        meta["error"] = "no rows after h=P_mm finite filter"
        return df, meta

    out_cols = ["P_mm", "h"]
    if "River ID" in df.columns:
        out_cols = ["River ID"] + out_cols
    for c in ("L_km", "A_km2"):
        if c in df.columns:
            out_cols.append(c)
    return df[out_cols].copy(), meta


def p_h_correlation_tests(p: np.ndarray, h: np.ndarray) -> pd.DataFrame:
    from scipy.stats import pearsonr, spearmanr

    if len(p) < 3:
        return pd.DataFrame([{"test": "skipped", "note": "n < 3"}])
    rho_p, pv_p = pearsonr(p, h)
    rho_s, pv_s = spearmanr(p, h)
    return pd.DataFrame(
        [
            {
                "test": "Pearson_P_mm_vs_h",
                "statistic": float(rho_p),
                "pvalue": float(pv_p),
                "n": int(len(p)),
            },
            {
                "test": "Spearman_P_mm_vs_h",
                "statistic": float(rho_s),
                "pvalue": float(pv_s),
                "n": int(len(p)),
            },
        ]
    )


def plot_p_vs_h(out: pd.DataFrame, out_png: Path, out_pdf: Path) -> None:
    import matplotlib.pyplot as plt

    _set_pub_style()
    p = out["P_mm"].to_numpy(float)
    h = out["h"].to_numpy(float)

    fig, ax = plt.subplots(figsize=(6.6, 4.4), constrained_layout=True)
    if len(out) > 8000:
        hb = ax.hexbin(p, h, gridsize=(55, 45), mincnt=1, cmap="viridis", linewidths=0)
        cb = fig.colorbar(hb, ax=ax, shrink=0.85, label="count")
        cb.ax.tick_params(labelsize=8)
    else:
        ax.scatter(p, h, s=8, alpha=0.35, c="0.25", edgecolors="none", rasterized=True)

    if len(p) >= 2:
        slope, intercept = np.polyfit(p, h, 1)
        xs = np.linspace(np.nanmin(p), np.nanmax(p), 120)
        ax.plot(xs, slope * xs + intercept, color="crimson", lw=2.0, label="OLS: h ~ P")
        ax.legend(loc="best", fontsize=9)

    ax.set_xlabel("Multi-year mean precipitation P (mm)")
    ax.set_ylabel(r"$h = \log_{10}(L) / \log_{10}(A)$")
    ax.set_title("Per-river P vs. h (no precipitation binning)")
    fig.savefig(out_png, bbox_inches="tight", dpi=160)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def assign_p_quantile_bins(df: pd.DataFrame, n_bins: int) -> tuple[pd.DataFrame, int]:
    """Assign precipitation quantile bins using P_mm.

Returns a copy of the dataframe containing a categorical
variable ('P_bin'). The effective number of bins may be
smaller than the requested value if duplicated quantile
boundaries are removed.
"""
    w = df.copy()
    try:
        w["P_bin"] = pd.qcut(w["P_mm"], q=n_bins, duplicates="raise")
    except ValueError:
        w["P_bin"] = pd.qcut(w["P_mm"], q=n_bins, duplicates="drop")
    if not isinstance(w["P_bin"].dtype, pd.CategoricalDtype):
        w["P_bin"] = w["P_bin"].astype("category")
    k = int(w["P_bin"].nunique(dropna=True))
    return w, k


def p_quantile_group_summary(w: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    order = w.groupby("P_bin", observed=True)["P_mm"].mean().sort_values().index.tolist()
    for i, b in enumerate(order):
        g = w.loc[w["P_bin"] == b]
        rows.append(
            {
                "bin_order": i + 1,
                "P_interval": str(b),
                "mean_P_mm": float(g["P_mm"].mean()),
                "min_P_mm": float(g["P_mm"].min()),
                "max_P_mm": float(g["P_mm"].max()),
                "n": int(len(g)),
                "median_h": float(g["h"].median()),
                "q1_h": float(g["h"].quantile(0.25)),
                "q3_h": float(g["h"].quantile(0.75)),
            }
        )
    return pd.DataFrame(rows)


def plot_h_boxplot_by_p_quantiles(w: pd.DataFrame, out_png: Path, out_pdf: Path, n_bins_requested: int) -> None:
    import matplotlib.pyplot as plt

    _set_pub_style()
    order = w.groupby("P_bin", observed=True)["P_mm"].mean().sort_values().index.tolist()
    data = [w.loc[w["P_bin"] == b, "h"].dropna().to_numpy(float) for b in order]
    mean_ps = [float(w.loc[w["P_bin"] == b, "P_mm"].mean()) for b in order]
    k = len(data)
    if k == 0:
        LOG.warning("boxplot: no P bins")
        return

    fig_w = max(9.0, 0.32 * k + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, 4.8), constrained_layout=True)
    pos = np.arange(1, k + 1)
    bp = ax.boxplot(
        data,
        positions=pos,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        whis=(5, 95),
        medianprops={"color": "crimson", "linewidth": 2.0},
        boxprops={"facecolor": "0.88", "edgecolor": "0.35"},
        whiskerprops={"color": "0.35"},
        capprops={"color": "0.35"},
    )

    medians = [float(np.median(d)) if len(d) else np.nan for d in data]
    ax.plot(pos, medians, color="crimson", marker="o", ms=4, lw=1.2, alpha=0.9, label="bin median")

    ax.set_xticks(pos)
    ax.set_xticklabels([f"{m:.0f}" for m in mean_ps], rotation=55, ha="right", fontsize=8)
    ax.set_xlabel(r"Precipitation group: mean $\overline{P}$ in bin (mm), low $\to$ high")
    ax.set_ylabel(r"$h = \log_{10}(L) / \log_{10}(A)$")
    ax.set_title(
        f"h distribution by precipitation quantile groups (n_bins requested={n_bins_requested}, effective={k})"
    )
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(out_png, bbox_inches="tight", dpi=160)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def write_results_md(out_dir: Path, summary: dict[str, Any], n_rows: int) -> None:
    lines = [
        "# Section 4.2 — P vs. h (per river, no binning)",
        "",
        "## Definition",
        "- **P_mm**: multi-year mean precipitation depth (mm) from the merged table.",
        r"- **h** (per river): $h = \log_{10}(L)/\log_{10}(A)$ with $L$ in km and $A$ in km$^2$. This is a **river-level** length–area index, not the single pooled OLS Hack slope.",
        "- Rivers with grade codes **0** and **99** are removed upstream (`clean_for_hacks`), same as the main scaling pipeline.",
        "",
        "## Outputs",
        "- `P_and_h_per_river.csv` — core columns **`P_mm`**, **`h`** (plus optional id / L / A).",
        "- `P_h_correlation.csv` — Pearson and Spearman between P and h.",
        "- `Fig_P_vs_h_per_river.png` / `.pdf` — scatter or hexbin + linear fit **h ~ P**.",
        "- `Fig_h_boxplot_by_P_quantile_groups.png` / `.pdf` — **boxplot of h** in many **equal-frequency P bins** (trend of medians overlaid).",
        "- `P_quantile_group_h_summary.csv` — per-bin mean/min/max P, n, median/q1/q3 of h.",
        "- `climate_modulation_summary.json` — run metadata and correlations.",
        "",
        "## Correlations (this run)",
        f"- Pearson r(P, h): **{summary.get('pearson_r_P_h', '—')}**, p = **{summary.get('pearson_p', '—')}**",
        f"- Spearman rho(P, h): **{summary.get('spearman_rho_P_h', '—')}**, p = **{summary.get('spearman_p', '—')}**",
        f"- n = **{n_rows}**",
        "",
        "",
    ]
    (out_dir / "RESULTS_climate_modulation.md").write_text("\n".join(lines), encoding="utf-8")


def run_all(cleaned: pd.DataFrame, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"module": "P_vs_h_per_river", "output_dir": str(out_dir)}

    out, meta = build_p_h_per_river(cleaned)
    summary.update(meta)
    if out.empty or meta.get("error"):
        LOG.warning("4.2 skipped: %s", meta.get("error", "empty"))
        (out_dir / "climate_modulation_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        write_results_md(out_dir, summary, 0)
        return summary

    out.to_csv(out_dir / "P_and_h_per_river.csv", index=False, encoding="utf-8-sig")
    p_arr = out["P_mm"].to_numpy(float)
    h_arr = out["h"].to_numpy(float)
    trend = p_h_correlation_tests(p_arr, h_arr)
    trend.to_csv(out_dir / "P_h_correlation.csv", index=False, encoding="utf-8-sig")

    pr = trend[trend["test"] == "Pearson_P_mm_vs_h"]
    sp = trend[trend["test"] == "Spearman_P_mm_vs_h"]
    if len(pr):
        summary["pearson_r_P_h"] = float(pr["statistic"].iloc[0])
        summary["pearson_p"] = float(pr["pvalue"].iloc[0])
    if len(sp):
        summary["spearman_rho_P_h"] = float(sp["statistic"].iloc[0])
        summary["spearman_p"] = float(sp["pvalue"].iloc[0])

    plot_p_vs_h(out, out_dir / "Fig_P_vs_h_per_river.png", out_dir / "Fig_P_vs_h_per_river.pdf")

    w_binned, k_eff = assign_p_quantile_bins(out, N_P_QUANTILE_BINS_FOR_BOXPLOT)
    summary["P_quantile_bins_requested"] = int(N_P_QUANTILE_BINS_FOR_BOXPLOT)
    summary["P_quantile_bins_effective"] = int(k_eff)
    gsum = p_quantile_group_summary(w_binned)
    gsum.to_csv(out_dir / "P_quantile_group_h_summary.csv", index=False, encoding="utf-8-sig")
    plot_h_boxplot_by_p_quantiles(
        w_binned,
        out_dir / "Fig_h_boxplot_by_P_quantile_groups.png",
        out_dir / "Fig_h_boxplot_by_P_quantile_groups.pdf",
        N_P_QUANTILE_BINS_FOR_BOXPLOT,
    )

    (out_dir / "climate_modulation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    write_results_md(out_dir, summary, len(out))
    LOG.info("4.2 P vs h (per river) done -> %s, n=%d", out_dir, len(out))
    return summary


def load_cleaned_data() -> pd.DataFrame:
    if CLEANED_CSV_PATH is not None and Path(CLEANED_CSV_PATH).exists():
        LOG.info("load cleaned CSV: %s", CLEANED_CSV_PATH)
        return pd.read_csv(CLEANED_CSV_PATH)
    LOG.info("load merged + clean: %s [%s]", MERGED_DATA_PATH, MERGED_SHEET_NAME)
    raw = load_merged(MERGED_DATA_PATH, MERGED_SHEET_NAME)
    cleaned, _ = clean_for_hacks(raw)
    return cleaned


def main() -> int:
    try:
        cleaned = load_cleaned_data()
        run_all(cleaned, OUTPUT_DIR)
        return 0
    except Exception as e:
        LOG.exception("4.2 failed: %s", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
