"""
Step 1. River-network scaling analysis based on Hack's law.

This script quantifies geometric scaling relationships in river networks using Hack's law:
    L = cA^h
or equivalently:
    log10(L) = h·log10(A) + c
where:
    L = river length (km)
    A = drainage area (km²)
    h = Hack exponent
    c = intercept

The script performs the following analyses:

1. Data loading and quality control.
2. Construction of river-length records using available sources.
3. Removal of missing values, non-positive values, duplicates,
   and placeholder river-grade codes.
4. Estimation of Hack's law parameters using ordinary least squares (OLS).
5. Bootstrap estimation of uncertainty in the Hack exponent.
6. Comparison between:
      - the full river dataset, and
      - representative rivers selected from each basin.
7. River-grade-specific scaling analyses.
8. Nested-model evaluation of river-grade effects on
   river-length scaling.
9. Generation of publication-quality figures and summary tables.
"""
from __future__ import annotations

import json
import logging
import re
import sys
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
MERGED_DATA_PATH = Path(r"F:\Lake\Info.xlsx")
MERGED_SHEET_NAME = "Merge"
OUTPUT_DIR = Path(r"F:\Lake\Result")
BOOTSTRAP_B = 1000
RNG_SEED = 42
RNG = np.random.default_rng(RNG_SEED)
# Minimum sample size required for grade-specific Hack-law fitting
MIN_STRATUM_N = 300

# Original field names retained from the source database
COL_L_ML = "L_ML_km"
COL_L_ZB = "L_ZB_km"
COL_A = "A_km2"
COL_BASIN = "BASIN "
COL_LEVEL = "LEVEL"
COL_CODE = "CODE"


def _setup_logger() -> logging.Logger:
    log = logging.getLogger("hacks_scaling")
    log.setLevel(logging.INFO)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.handlers.clear()
    log.addHandler(h)
    return log


LOG = _setup_logger()


def load_merged(path: Path, sheet: str) -> pd.DataFrame:
    LOG.info("Loading dataset: %s [%s]", path, sheet)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_excel(path, sheet_name=sheet)
    LOG.info(
        "Dataset loaded: %d rows, %d columns",
        len(df),
        df.shape[1]
    )
    return df


def attach_length_km(df: pd.DataFrame) -> pd.DataFrame:
    """Construct river length values. Prefer the indicator-derived length and fall back to the inventory-derived length when unavailable."""
    out = df.copy()
    l_zb = pd.to_numeric(out[COL_L_ZB], errors="coerce")
    l_ml = pd.to_numeric(out[COL_L_ML], errors="coerce")
    out["L_km"] = l_zb.combine_first(l_ml)
    return out


def clean_for_hacks(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean the dataset by removing missing values, non-positive values, and duplicates. Return the cleaned dataframe and summary statistics."""
    n0 = len(df)
    d = attach_length_km(df)
    a = pd.to_numeric(d[COL_A], errors="coerce")

    summary: dict[str, Any] = {
        "raw_rows": int(n0),
        "after_attach_L": int(len(d)),
    }

    d["A_km2"] = a

    # Descriptive statistics (before cleaning, for the original interpretable numerical values)
    summary["L_describe_raw"] = d["L_km"].describe().to_dict()
    summary["A_describe_raw"] = d["A_km2"].describe().to_dict()

    m_valid = d["L_km"].notna() & d["A_km2"].notna()
    d1 = d.loc[m_valid].copy()
    summary["after_drop_na_L_or_A"] = int(len(d1))

    d2 = d1[(d1["L_km"] > 0) & (d1["A_km2"] > 0)].copy()
    summary["after_drop_nonpositive"] = int(len(d2))

    # Remove exact duplicates based on river code, length, and drainage area
    before_dedup = len(d2)
    dup_subset = ["L_km", "A_km2"]
    if COL_CODE in d2.columns:
        dup_subset = [COL_CODE] + dup_subset
    d3 = d2.drop_duplicates(subset=dup_subset, keep="first").copy()
    summary["duplicate_rows_removed"] = int(before_dedup - len(d3))
    summary["after_dedup"] = int(len(d3))

    # Exclude placeholder river-grade codes (0 and 99)
    before_lvl = len(d3)
    if COL_LEVEL in d3.columns:
        lv = pd.to_numeric(d3[COL_LEVEL], errors="coerce")
        n99 = int(lv.eq(99).sum())
        n0 = int(lv.eq(0).sum())
        d3 = d3.loc[~lv.isin([0, 99])].copy()
        summary["dropped_level_99"] = n99
        summary["dropped_level_0"] = n0
        summary["dropped_levels_0_99_total"] = int(before_lvl - len(d3))
    else:
        summary["dropped_level_99"] = 0
        summary["dropped_level_0"] = 0
        summary["dropped_levels_0_99_total"] = 0
    summary["after_exclude_levels_0_99"] = int(len(d3))

    summary["L_describe_clean"] = d3["L_km"].describe().to_dict()
    summary["A_describe_clean"] = d3["A_km2"].describe().to_dict()

    LOG.info(
         "Cleaning: raw {} -> after drop NA {} -> after drop non-positive {} -> after dedup {} -> after exclude level 0/99 {} (dropped level 0={}, dropped level 99={})",
        n0,
        summary["after_drop_na_L_or_A"],
        summary["after_drop_nonpositive"],
        summary["after_dedup"],
        summary["after_exclude_levels_0_99"],
        summary.get("dropped_level_0", 0),
        summary.get("dropped_level_99", 0),
    )
    return d3, summary


def select_representative_per_basin(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Select one representative river per basin.

    Priority:
    1. Smallest river-grade code (higher grade).
    2. Longest river length.
    3. First record if ties remain.
    """
    msgs: list[str] = []
    need = [COL_BASIN, COL_LEVEL, "L_km"]
    for c in need:
        if c not in df.columns:
            raise KeyError(f"Missing required column: {c}")

    sub = df.copy()
    sub["_lvl"] = pd.to_numeric(sub[COL_LEVEL], errors="coerce")
    sub = sub[sub[COL_BASIN].notna() & (sub[COL_BASIN].astype(str).str.strip() != "")]
    sub = sub[sub["_lvl"].notna() & sub["L_km"].notna()]

    picks: list[pd.Series] = []
    for basin, g in sub.groupby(COL_BASIN, sort=False):
        # Select the smallest river-grade code
        min_grade = float(g["_lvl"].min())
        top = g[g["_lvl"] == min_grade]
        max_L = float(top["L_km"].max())
        top2 = top[top["L_km"] == max_L]
        if len(top2) > 1:
            msg = (
                f"Tie detected in basin '{basin}' "
                f"(grade={min_grade:g}, length={max_L:g}); "
                f"the first record was retained."
            )
            LOG.warning(msg)
            msgs.append(msg)
        pick = top2.iloc[0]
        picks.append(pick)

    rep = pd.DataFrame(picks).reset_index(drop=True)
    # Remove temporary columns
    rep = rep.drop(columns=[c for c in ["_lvl"] if c in rep.columns])

    LOG.info(
        "Representative-river dataset generated: n=%d",
        len(rep)
    )
    return rep, msgs


@dataclass
class HackResult:
    version: str
    n: int
    h: float
    intercept_c: float
    r2: float
    p_slope: float
    h_ci95_low: float
    h_ci95_high: float
    bootstrap_h_median: float | None
    bootstrap_h_ci95_low: float | None
    bootstrap_h_ci95_high: float | None


def fit_hacks_ols(log10_A: np.ndarray, log10_L: np.ndarray) -> dict[str, float]:
    import statsmodels.api as sm
    from scipy.stats import linregress

    X = sm.add_constant(log10_A)
    y = log10_L
    m = sm.OLS(y, X).fit()
    params = np.asarray(m.params, dtype=float).ravel()
    pvals = np.asarray(m.pvalues, dtype=float).ravel()
    ci = np.asarray(m.conf_int(alpha=0.05), dtype=float)
    # columns: [const, slope]
    c = float(params[0])
    h = float(params[1])
    lr = linregress(log10_A, log10_L)
    if abs(float(lr.slope) - h) > 1e-6:
        LOG.warning(
            "Slope estimates differ between scipy.linregress and OLS (difference=%.3e); OLS result retained.",
            abs(float(lr.slope) - h)        )
    return {
        "h": h,
        "intercept_c": c,
        "r2": float(m.rsquared),
        "p_slope": float(pvals[1]),
        "h_ci95_low": float(ci[1, 0]),
        "h_ci95_high": float(ci[1, 1]),
    }


def bootstrap_h(
    log10_A: np.ndarray, log10_L: np.ndarray, B: int = BOOTSTRAP_B, rng: np.random.Generator = RNG
) -> tuple[float, float, float]:
    import statsmodels.api as sm

    n = len(log10_A)
    hs: list[float] = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        Xb = sm.add_constant(log10_A[idx])
        yb = log10_L[idx]
        try:
            mb = sm.OLS(yb, Xb).fit()
            hs.append(float(np.asarray(mb.params, dtype=float).ravel()[1]))
        except Exception:
            continue
    if not hs:
        return float("nan"), float("nan"), float("nan")
    arr = np.asarray(hs, dtype=float)
    return float(np.median(arr)), float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))


def run_hacks(df: pd.DataFrame, version: str) -> HackResult:
    a = df["A_km2"].to_numpy(dtype=float)
    ell = df["L_km"].to_numpy(dtype=float)
    if np.any(a <= 0) or np.any(ell <= 0):
        raise ValueError(            "Hack's law requires positive drainage area (A) and river length (L)."        )

    log10_A = np.log10(a)
    log10_L = np.log10(ell)

    o = fit_hacks_ols(log10_A, log10_L)
    med, lo, hi = bootstrap_h(log10_A, log10_L)

    return HackResult(
        version=version,
        n=int(len(df)),
        h=o["h"],
        intercept_c=o["intercept_c"],
        r2=o["r2"],
        p_slope=o["p_slope"],
        h_ci95_low=o["h_ci95_low"],
        h_ci95_high=o["h_ci95_high"],
        bootstrap_h_median=med,
        bootstrap_h_ci95_low=lo,
        bootstrap_h_ci95_high=hi,
    )


def level_stats(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["_lvl"] = pd.to_numeric(d[COL_LEVEL], errors="coerce")
    d = d[d["_lvl"].notna() & d["L_km"].notna()]
    g = d.groupby("_lvl", dropna=False)["L_km"]
    out = pd.DataFrame(
        {
            "river_level": g.count().index.astype(int),
            "count": g.count().values,
            "mean_length_km": g.mean().values,
            "median_length_km": g.median().values,
            "std_length_km": g.std(ddof=1).values,
        }
    ).sort_values("river_level")
    return out.reset_index(drop=True)


def _prep_hierarchy_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare a valid subset for hierarchical analyses with valid L, A, and river-grade values."""
    d = df.copy()
    d["_lvl"] = pd.to_numeric(d[COL_LEVEL], errors="coerce")
    d = d[d["_lvl"].notna() & d["L_km"].notna() & d["A_km2"].notna()].copy()
    d = d[(d["L_km"] > 0) & (d["A_km2"] > 0)].copy()
    d["log10_L"] = np.log10(d["L_km"].astype(float))
    d["log10_A"] = np.log10(d["A_km2"].astype(float))
    return d


def hierarchy_hacks_by_level(work: pd.DataFrame) -> pd.DataFrame:
    """
    Fit Hack's law separately for each river-grade class.

    This analysis evaluates whether the Hack exponent (h)
    varies systematically across river grades.
    """
    rows: list[dict[str, Any]] = []
    for lv, g in work.groupby("_lvl", sort=True):
        n = len(g)
        if n < MIN_STRATUM_N:
            continue
        try:
            o = fit_hacks_ols(g["log10_A"].to_numpy(float), g["log10_L"].to_numpy(float))
        except Exception as e:
            LOG.warning(
                "Stratified Hack's-law fitting failed for level=%s: %s",
                lv,
                e,
            )
            continue
        rows.append(
            {
                "river_level": int(lv),
                "n": n,
                "h": o["h"],
                "intercept_c": o["intercept_c"],
                "r2": o["r2"],
                "p_slope": o["p_slope"],
                "h_ci95_low": o["h_ci95_low"],
                "h_ci95_high": o["h_ci95_high"],
            }
        )
    return pd.DataFrame(rows)


def hierarchy_nested_ols(
    work: pd.DataFrame,
) -> tuple[pd.DataFrame | None, dict[str, Any], pd.DataFrame | None, Any]:
    """
    Nested OLS analysis of river-grade effects.

    Models
    ------
    M0: log10_L ~ log10_A

    M1: log10_L ~ log10_A + C(_lvl)

    The increase in R² (ΔR²) quantifies the additional
    explanatory power provided by river-grade classes
    after accounting for drainage area.
    """
    import statsmodels.formula.api as smf

    info: dict[str, Any] = {}
    if work["_lvl"].nunique() < 2:
        info["error"] = "Fewer than two river-grade classes are available; nested-model analysis was skipped."
        LOG.warning(info["error"])
        return None, info, None, None

    m0 = smf.ols("log10_L ~ log10_A", data=work).fit()
    m1 = smf.ols("log10_L ~ log10_A + C(_lvl)", data=work).fit()

    try:
        cf = m1.compare_f_test(m0)
        info["nested_F_stat"] = float(cf[0])
        info["nested_F_pvalue"] = float(cf[1])
        info["nested_df_diff"] = float(cf[2]) if len(cf) > 2 else None
    except Exception as e:
        LOG.warning("compare_f_test Failed: %s", e)
        info["nested_F_stat"] = None
        info["nested_F_pvalue"] = None
        info["nested_df_diff"] = None

    info["n"] = int(m0.nobs)
    info["M0_r2"] = float(m0.rsquared)
    info["M1_r2"] = float(m1.rsquared)
    info["delta_r2_level_after_area"] = float(m1.rsquared - m0.rsquared)
    info["M0_r2_adj"] = float(m0.rsquared_adj)
    info["M1_r2_adj"] = float(m1.rsquared_adj)

    comp = pd.DataFrame(
        [
            {"model": "M0_baseline", "formula": "log10_L ~ log10_A", "n": info["n"], "r2": info["M0_r2"], "r2_adj": info["M0_r2_adj"], "k_params": int(len(m0.params))},
            {"model": "M1_add_grade", "formula": "log10_L ~ log10_A + C(river_grade)", "n": info["n"], "r2": info["M1_r2"], "r2_adj": info["M1_r2_adj"], "k_params": int(len(m1.params))},
        ]
    )
    # Use the actual model specification for reporting
    comp.loc[
        comp["model"] == "M1_add_grade",
        "formula"
    ] = "log10_L ~ log10_A + C(_lvl)"

    # Coefficients of river-grade dummy variables
    # relative to the reference category
    coef_tbl = pd.DataFrame(
        {
            "param": m1.params.index.astype(str),
            "coef": m1.params.values.astype(float),
            "ci_low": m1.conf_int(alpha=0.05)[0].values.astype(float),
            "ci_high": m1.conf_int(alpha=0.05)[1].values.astype(float),
            "pvalue": m1.pvalues.values.astype(float),
        }
    )
    coef_tbl = coef_tbl[coef_tbl["param"].str.contains("_lvl", na=False)].reset_index(drop=True)

    return comp, info, coef_tbl, m1


def run_hierarchy_analysis(cleaned: pd.DataFrame, out_dir: Path) -> dict[str, Any]:
    """
    Hierarchical-effect analysis.

    This module:
    1. Performs stratified Hack's-law analyses.
    2. Fits nested OLS models with drainage area and river grade.
    3. Exports summary tables and figures.

    Returns
    -------
    dict
        Summary statistics for report generation.
    """
    summary: dict[str, Any] = {"module": "hierarchy_effects", "min_stratum_n": MIN_STRATUM_N}
    work = _prep_hierarchy_frame(cleaned)
    summary["n_for_hierarchy"] = int(len(work))

    strat = hierarchy_hacks_by_level(work)
    p_strat = out_dir / "hierarchy_hacks_by_level.csv"
    strat.to_csv(p_strat, index=False, encoding="utf-8-sig")
    LOG.info("Stratified Hack's-law results saved: %s (%d grade classes)",p_strat, len(strat))
    summary["stratified_levels"] = int(len(strat))
    if len(strat) >= 2:
        summary["h_min_across_levels"] = float(strat["h"].min())
        summary["h_max_across_levels"] = float(strat["h"].max())
        summary["h_range"] = float(strat["h"].max() - strat["h"].min())

    comp, nest_info, coef_tbl, m1 = hierarchy_nested_ols(work)
    summary["nested"] = nest_info

    if comp is not None:
        comp.to_csv(out_dir / "hierarchy_model_comparison.csv", index=False, encoding="utf-8-sig")
    if coef_tbl is not None and len(coef_tbl):
        coef_tbl.to_csv(out_dir / "hierarchy_grade_dummy_coefficients.csv", index=False, encoding="utf-8-sig")

    (out_dir / "hierarchy_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    if len(strat):
        plot_hack_exponent_by_level(
            strat,
            out_dir / "Fig5_Hack_exponent_by_grade_stratum.png",
            out_dir / "Fig5_Hack_exponent_by_grade_stratum.pdf",
        )
    if coef_tbl is not None and len(coef_tbl):
        plot_grade_dummy_forest(
            coef_tbl,
            out_dir / "Fig6_Grade_effects_on_log10_L_given_area.png",
            out_dir / "Fig6_Grade_effects_on_log10_L_given_area.pdf",
        )

    LOG.info(
        "nested OLS: dR2(level|log10A)=%s, nested F p=%s",
        nest_info.get("delta_r2_level_after_area"),
        nest_info.get("nested_F_pvalue"),
    )
    return summary


def plot_hack_exponent_by_level(strat: pd.DataFrame, out_png: Path, out_pdf: Path) -> None:
    """
    Plot Hack exponent (h) across river-grade classes.
    Error bars indicate OLS 95% confidence intervals.
    """
    import matplotlib.pyplot as plt

    _set_pub_style()
    fig, ax = plt.subplots(figsize=(5.4, 4.0), constrained_layout=True)
    x = strat["river_level"].to_numpy(int)
    y = strat["h"].to_numpy(float)
    yerr = np.vstack([y - strat["h_ci95_low"].to_numpy(float), strat["h_ci95_high"].to_numpy(float) - y])
    ax.errorbar(x, y, yerr=yerr, fmt="o", color="darkred", ecolor="0.35", capsize=3, ms=5, lw=1.2)
    ax.axhline(np.nanmean(y), color="0.5", ls="--", lw=1, label="Mean h across strata")
    ax.set_xticks(x)
    ax.set_xlabel("River grade code")
    ax.set_ylabel("Hack exponent $h$ (per stratum)")
    ax.set_title(f"Stratified Hack's law: $h$ by river grade code\n(min n per stratum = {MIN_STRATUM_N})")
    ax.legend(loc="best", fontsize=8)
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)


def plot_grade_dummy_forest(coef_tbl: pd.DataFrame, out_png: Path, out_pdf: Path) -> None:
    """
    Plot coefficients of river-grade dummy variables from Model M1.

    Coefficients represent shifts in log10(L) relative to
    the reference category after controlling for log10(A).
    """
    import matplotlib.pyplot as plt

    _set_pub_style()
    fig, ax = plt.subplots(figsize=(5.8, max(3.5, 0.35 * len(coef_tbl))), constrained_layout=True)
    y = np.arange(len(coef_tbl))
    x = coef_tbl["coef"].to_numpy(float)
    xerr = np.vstack([x - coef_tbl["ci_low"].to_numpy(float), coef_tbl["ci_high"].to_numpy(float) - x])
    ax.errorbar(x, y, xerr=xerr, fmt="o", color="teal", ecolor="0.35", capsize=2, ms=4, lw=1)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(coef_tbl["param"].tolist(), fontsize=8)
    ax.set_xlabel("Coefficient (log10 km shift vs. reference river grade code)")
    ax.set_title("River-grade-code dummies in M1: log10(L) ~ log10(A) + C(code)")
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)


def _set_pub_style() -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
        }
    )
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")


def plot_hacks_scatter(
    df: pd.DataFrame,
    res: HackResult,
    out_png: Path,
    out_pdf: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    _set_pub_style()
    x = np.log10(df["A_km2"].to_numpy(float))
    y = np.log10(df["L_km"].to_numpy(float))

    fig, ax = plt.subplots(figsize=(5.2, 4.2), constrained_layout=True)
    ax.scatter(x, y, s=8, alpha=0.25, color="0.25", edgecolors="none", rasterized=True)

    xs = np.linspace(np.nanmin(x), np.nanmax(x), 200)
    ys = res.h * xs + res.intercept_c
    ax.plot(xs, ys, color="crimson", lw=2.0, label="OLS fit")

    txt = (
        f"$h={res.h:.3f}$\n"
        f"$R^2={res.r2:.3f}$\n"
        f"$p={res.p_slope:.2e}$\n"
        f"$n={res.n}$"
    )
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", ha="left", fontsize=10)

    ax.set_title(title)
    ax.set_xlabel(r"$\log_{10}$(Drainage area, km$^2$)")
    ax.set_ylabel(r"$\log_{10}$(River length, km)")
    ax.legend(loc="lower right")
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)


def plot_level_bar(stats: pd.DataFrame, ycol: str, ylabel: str, title: str, out_png: Path, out_pdf: Path) -> None:
    import matplotlib.pyplot as plt

    _set_pub_style()
    fig, ax = plt.subplots(figsize=(5.2, 4.0), constrained_layout=True)
    x = stats["river_level"].to_numpy(int)
    y = stats[ycol].to_numpy(float)
    ax.bar(x, y, color="steelblue", edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xlabel("River grade code ")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)

def main() -> int:
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        LOG.info("Output directory: %s", OUTPUT_DIR)

        raw = load_merged(MERGED_DATA_PATH, MERGED_SHEET_NAME)
        cleaned, clean_summary = clean_for_hacks(raw)

        cleaned_path = OUTPUT_DIR / "cleaned_merged_for_scaling.csv"
        cleaned.to_csv(cleaned_path, index=False, encoding="utf-8-sig")
        LOG.info("Saved: %s", cleaned_path)

        try:
            cleaned_xlsx = OUTPUT_DIR / "cleaned_merged_for_scaling.xlsx"
            cleaned.to_excel(cleaned_xlsx, index=False)
            LOG.info("Saved: %s", cleaned_xlsx)
        except Exception as e:
            LOG.warning(
                "Failed to export cleaned dataset as xlsx (csv retained): %s",
                e,
            )

        rep_df, tie_msgs = select_representative_per_basin(cleaned)

        rep_path = OUTPUT_DIR / "representative_rivers_by_basin.csv"
        rep_df.to_csv(rep_path, index=False, encoding="utf-8-sig")
        LOG.info("Saved: %s", rep_path)

        r_full = run_hacks(cleaned, "full_sample")
        r_rep = run_hacks(rep_df, "representative_per_basin")

        reg_tbl = pd.DataFrame([asdict(r_full), asdict(r_rep)])
        reg_path = OUTPUT_DIR / "hacks_law_regression_results.csv"
        reg_tbl.to_csv(reg_path, index=False, encoding="utf-8-sig")
        LOG.info("Saved: %s", reg_path)

        lvl = level_stats(cleaned)

        lvl_path = OUTPUT_DIR / "river_grade_statistics.csv"
        lvl.to_csv(lvl_path, index=False, encoding="utf-8-sig")
        LOG.info("Saved: %s", lvl_path)

        try:
            hsum = run_hierarchy_analysis(cleaned, OUTPUT_DIR)
        except Exception as e:
            LOG.warning(
                "Hierarchical-effect analysis failed: %s",
                e,
            )
            hsum = {"error": repr(e)}

        (OUTPUT_DIR / "cleaning_summary.json").write_text(
            json.dumps(clean_summary, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

        # Figures
        plot_hacks_scatter(
            cleaned,
            r_full,
            OUTPUT_DIR / "Fig1_Hacks_law_full_sample.png",
            OUTPUT_DIR / "Fig1_Hacks_law_full_sample.pdf",
            title="Hack's law (full sample)",
        )

        plot_hacks_scatter(
            rep_df,
            r_rep,
            OUTPUT_DIR / "Fig2_Hacks_law_representative_sample.png",
            OUTPUT_DIR / "Fig2_Hacks_law_representative_sample.pdf",
            title="Hack's law (representative main stem per basin)",
        )

        plot_level_bar(
            lvl,
            "count",
            "Number of rivers",
            "River count vs. river grade code",
            OUTPUT_DIR / "Fig3_River_count_vs_level.png",
            OUTPUT_DIR / "Fig3_River_count_vs_level.pdf",
        )

        plot_level_bar(
            lvl,
            "mean_length_km",
            "Mean river length (km)",
            "Mean river length vs. river grade code",
            OUTPUT_DIR / "Fig4_Mean_length_vs_level.png",
            OUTPUT_DIR / "Fig4_Mean_length_vs_level.pdf",
        )

        LOG.info("Figures saved (PNG and PDF)")

        n_basins = int(cleaned[COL_BASIN].nunique())


        LOG.info(
            "Analysis completed. h_full=%.4f, h_rep=%.4f",
            r_full.h,
            r_rep.h,
        )

        return 0

    except Exception as e:
        LOG.exception(
            "Analysis failed: %s",
            e,
        )
        return 1
if __name__ == "__main__":
    raise SystemExit(main())
