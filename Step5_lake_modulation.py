"""
Step 5. Lake modulation of runoff efficiency and scaling behavior.

This script evaluates whether lake abundance influences:

    1. Runoff efficiency (R/P)
    2. The runoff-efficiency scaling exponent (beta)

Lake information is aggregated at the provincial level and merged with river records using province identifiers inferred from geographic attributes.

Lake metrics:

    lake_density  = lake area / drainage area
    lake_density2 = lake count / drainage area

The workflow:

1. Load and clean river-network data.
2. Load and aggregate lake inventory data.
3. Merge lake and river datasets.
4. Evaluate lake effects on log10(R/P).
5. Quantify changes in local scaling exponent beta along  a lake-density gradient using sliding-window regression.
6. Generate publication-quality figures and summary tables.

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
LAKE_EXCEL_PATH = Path(r"F:\Lake\Lake_Province_1km2.xlsx")
CLEANED_CSV_PATH: Path | None = None
OUTPUT_DIR = Path(r"F:\Lake\Result")
MAIN_RESULTS_MD = Path(r"F:\Lake\Result\RESULTS.md")


RP_MAX = 1.5

WINDOW_SIZE = 2000
STEP_SIZE = 500
MIN_VALID_IN_WINDOW = 100

from run_hacks_scaling_analysis import (  # noqa: E402
    clean_for_hacks,
    fit_hacks_ols,
    load_merged,
    _set_pub_style,
)

COL_PRECIP_MM_CANDIDATES = (
    "P_mm",
    "P（mm）",
    "P(mm)",
)
COL_RUNOFF_MM_CANDIDATES = (
    "R（mm）",
    "R(mm)",
    "R_mm",
    "ZB_R_mm",
    "ML_R_mm",
)

RIVER_GEO_KEYWORDS = (
    "流经区县",
    "流经",
    "区县",
    "所在省",
    "省级",
    "省区",
    "行政区",
)


def _setup_logger() -> logging.Logger:
    log = logging.getLogger("lake_mod_45")
    log.setLevel(logging.INFO)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.handlers.clear()
    log.addHandler(h)
    return log


LOG = _setup_logger()


def find_col(cols: pd.Index, keywords: list[str]) -> str | None:
    for c in cols:
        cs = str(c)
        for k in keywords:
            if k in cs:
                return c
    return None


def find_lake_area_column(cols: pd.Index) -> str | None:
    for kw in ("湖泊面积", "水面面积", "水面", "面积"):
        for c in cols:
            if kw in str(c):
                return c
    return None


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


def find_river_geo_column(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        for k in RIVER_GEO_KEYWORDS:
            if k in str(c):
                return c
    return None


def extract_province_from_river_geo(x: Any) -> Any:
    """Infer province-level label for merge with provincial lake table."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return np.nan
    if "省" in s:
        return s.split("省", 1)[0] + "省"
    for m in ("北京市", "上海市", "天津市", "重庆市"):
        if s.startswith(m) or m in s:
            return m
    if "自治区" in s:
        for p in (
            "内蒙古自治区",
            "新疆维吾尔自治区",
            "广西壮族自治区",
            "宁夏回族自治区",
            "西藏自治区",
        ):
            if s.startswith(p[:2]) or p in s:
                return p
        if "自治区" in s:
            return s.split("自治区", 1)[0] + "自治区"
    return s


def normalize_lake_region_key(x: Any) -> Any:
    if pd.isna(x):
        return np.nan
    s = str(x).strip().replace("\u3000", " ")
    if not s:
        return np.nan
    if s.endswith("省") or s.endswith("自治区") or s in ("北京市", "上海市", "天津市", "重庆市"):
        return s
    if "省" not in s and s.endswith("市") and len(s) <= 3:
        return s
    if s.endswith("市") and s in ("北京市", "上海市", "天津市", "重庆市"):
        return s
    return s


def load_lake_table(path: Path) -> tuple[pd.DataFrame, str, str, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Lake Excel not found: {path}")
    lake_df = pd.read_excel(path)
    meta: dict[str, Any] = {"lake_excel": str(path), "lake_raw_rows": int(len(lake_df))}
    col_region = find_col(lake_df.columns, ["省", "行政", "地区", "区域", "省市"])
    col_area = find_lake_area_column(lake_df.columns)
    if col_area is None:
        raise ValueError("Lake table: could not find a lake-area column.")
    if col_region is None:
        raise ValueError("Lake table: could not find region column (tried 省/行政/地区/区域).")
    meta["lake_col_region"] = col_region
    meta["lake_col_area"] = col_area
    ld = lake_df.copy()
    ld[col_area] = pd.to_numeric(ld[col_area], errors="coerce")
    ld = ld.dropna(subset=[col_area])
    ld = ld[ld[col_area] > 0].copy()
    ld["_region_key"] = ld[col_region].map(normalize_lake_region_key)
    lg = (
        ld.groupby("_region_key", dropna=True)
        .agg(lake_area_sum=(col_area, "sum"), lake_count=(col_area, "count"))
        .reset_index()
    )
    lg.rename(columns={"_region_key": "province_merge_key"}, inplace=True)
    meta["lake_provinces"] = int(len(lg))
    LOG.info("lakes: aggregated %d provinces from %d lake rows", len(lg), len(ld))
    return lg, col_region, col_area, meta


def prepare_river_with_rp(cleaned: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    meta: dict[str, Any] = {"n_input": int(len(cleaned))}
    pcol = resolve_precipitation_column(cleaned)
    rcol = resolve_runoff_column(cleaned)
    if pcol is None or rcol is None:
        raise ValueError("Precip or runoff column missing for R/P analysis.")
    geo = find_river_geo_column(cleaned)
    if geo is None:
        raise ValueError(
            "No river geo column for province (tried keywords: " + ", ".join(RIVER_GEO_KEYWORDS) + ")."
        )
    meta["precip_col"] = pcol
    meta["runoff_col"] = rcol
    meta["river_geo_col"] = geo

    d = cleaned.copy()
    d["P_mm"] = pd.to_numeric(d[pcol], errors="coerce")
    d["R_mm"] = pd.to_numeric(d[rcol], errors="coerce")
    d["A_km2"] = pd.to_numeric(d["A_km2"], errors="coerce")
    d = d[d["P_mm"].notna() & d["R_mm"].notna() & d["A_km2"].notna() & d[geo].notna()].copy()
    d = d[(d["A_km2"] > 0) & (d["P_mm"] > 0) & (d["R_mm"] > 0)]
    d["RP"] = d["R_mm"] / d["P_mm"]
    d = d[(d["RP"] > 0) & (d["RP"] <= RP_MAX)].copy()
    d["log10_A"] = np.log10(d["A_km2"].astype(float))
    d["log10_P"] = np.log10(d["P_mm"].astype(float))
    d["log10_RP"] = np.log10(d["RP"].astype(float))
    d = d[np.isfinite(d["log10_A"]) & np.isfinite(d["log10_P"]) & np.isfinite(d["log10_RP"])].copy()
    d["province_merge_key"] = d[geo].map(extract_province_from_river_geo)
    meta["n_after_rp_filters"] = int(len(d))
    LOG.info("river RP prep: n=%d (geo col=%s)", len(d), geo)
    return d, meta


def merge_lakes_and_density(d: pd.DataFrame, lake_group: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    meta: dict[str, Any] = {}
    before = len(d)
    m = d.merge(lake_group, on="province_merge_key", how="left", validate="m:1")
    m["lake_density"] = m["lake_area_sum"] / m["A_km2"].astype(float)
    m["lake_density2"] = m["lake_count"].astype(float) / m["A_km2"].astype(float)
    m = m.replace([np.inf, -np.inf], np.nan)
    m = m[m["lake_density"].notna() & (m["lake_density"] > 0)].copy()
    meta["n_after_merge_positive_density"] = int(len(m))
    meta["n_lost_no_lake_or_nonpositive_density"] = int(before - len(m))
    LOG.info("after lake merge + lake_density>0: n=%d (dropped from merge step=%d)", len(m), before - len(m))
    return m, meta


def fit_ols_lake_models(df: pd.DataFrame) -> dict[str, Any]:
    import statsmodels.api as sm

    y = df["log10_RP"]
    X0 = sm.add_constant(df[["log10_A", "log10_P"]])
    X1 = sm.add_constant(df[["log10_A", "log10_P", "lake_density"]])
    m0 = sm.OLS(y, X0).fit()
    m1 = sm.OLS(y, X1).fit()
    rss_r, rss_f = float(m0.ssr), float(m1.ssr)
    df_f = int(m1.df_resid)
    df_diff = int(m1.df_model - m0.df_model)
    F = ((rss_r - rss_f) / df_diff) / (rss_f / df_f) if df_diff > 0 and rss_f > 0 else float("nan")
    from scipy.stats import f as f_dist

    p_f = float(1.0 - f_dist.cdf(F, df_diff, df_f)) if np.isfinite(F) else float("nan")
    rows = []
    for name, res in (("M0", m0), ("M1_lake_density", m1)):
        for term in res.params.index:
            ci = res.conf_int().loc[term]
            rows.append(
                {
                    "model": name,
                    "term": str(term),
                    "coef": float(res.params[term]),
                    "std_err": float(res.bse[term]),
                    "t": float(res.tvalues[term]),
                    "p": float(res.pvalues[term]),
                    "ci_low": float(ci[0]),
                    "ci_high": float(ci[1]),
                }
            )
    coef_tbl = pd.DataFrame(rows)
    comp = pd.DataFrame(
        [
            {
                "model": "M0",
                "R2": float(m0.rsquared),
                "adj_R2": float(m0.rsquared_adj),
                "AIC": float(m0.aic),
                "BIC": float(m0.bic),
                "n": int(m0.nobs),
            },
            {
                "model": "M1_lake_density",
                "R2": float(m1.rsquared),
                "adj_R2": float(m1.rsquared_adj),
                "AIC": float(m1.aic),
                "BIC": float(m1.bic),
                "n": int(m1.nobs),
            },
        ]
    )
    comp["delta_R2_vs_M0"] = comp["R2"] - float(m0.rsquared)
    ld_row = coef_tbl[(coef_tbl["model"] == "M1_lake_density") & (coef_tbl["term"] == "lake_density")]
    ld_coef = float(ld_row["coef"].iloc[0]) if len(ld_row) else float("nan")
    ld_p = float(ld_row["p"].iloc[0]) if len(ld_row) else float("nan")
    return {
        "m0": m0,
        "m1": m1,
        "coef_tbl": coef_tbl,
        "comparison": comp,
        "nested_F": F,
        "nested_F_p": p_f,
        "delta_R2": float(m1.rsquared - m0.rsquared),
        "lake_density_coef": ld_coef,
        "lake_density_p": ld_p,
    }


def run_sliding_beta_vs_lake_density(df: pd.DataFrame) -> pd.DataFrame:
    d = df.sort_values("lake_density", ascending=True).reset_index(drop=True)
    n = len(d)
    if n < WINDOW_SIZE:
        raise ValueError(f"n={n} < WINDOW_SIZE={WINDOW_SIZE}; cannot run sliding beta vs lake_density.")
    rows: list[dict[str, Any]] = []
    wid = 0
    start = 0
    while start + WINDOW_SIZE <= n:
        sub = d.iloc[start : start + WINDOW_SIZE]
        log_a = np.log10(sub["A_km2"].to_numpy(float))
        log_rp = np.log10(sub["RP"].to_numpy(float))
        m = np.isfinite(log_a) & np.isfinite(log_rp)
        nv = int(m.sum())
        if nv < MIN_VALID_IN_WINDOW:
            start += STEP_SIZE
            wid += 1
            continue
        o = fit_hacks_ols(log_a[m], log_rp[m])
        sub_m = sub.loc[m]
        rows.append(
            {
                "window_id": wid,
                "start_index": int(start),
                "end_index": int(start + WINDOW_SIZE - 1),
                "n": nv,
                "lake_density_mean": float(sub_m["lake_density"].mean()),
                "lake_density_median": float(sub_m["lake_density"].median()),
                "beta": float(o["h"]),
                "R2": float(o["r2"]),
                "p_slope": float(o["p_slope"]),
                "beta_ci_low": float(o["h_ci95_low"]),
                "beta_ci_high": float(o["h_ci95_high"]),
            }
        )
        start += STEP_SIZE
        wid += 1
    if not rows:
        raise RuntimeError("No sliding windows produced for beta vs lake_density.")
    LOG.info("sliding beta vs lake_density: %d windows", len(rows))
    return pd.DataFrame(rows)


def plot_fig11(beta_df: pd.DataFrame, corr: dict[str, float], png: Path, pdf: Path) -> None:
    import matplotlib.pyplot as plt
    from statsmodels.nonparametric.smoothers_lowess import lowess

    _set_pub_style()
    x = beta_df["lake_density_mean"].to_numpy(float)
    y = beta_df["beta"].to_numpy(float)
    order = np.argsort(x)
    frac = min(0.45, max(0.2, 10.0 / max(len(x), 5)))
    sm = lowess(y, x, frac=frac, it=3, return_sorted=True)

    fig, ax = plt.subplots(figsize=(6.8, 4.6), constrained_layout=True)
    ax.scatter(x, y, s=22, alpha=0.85, c="steelblue", edgecolors="white", linewidths=0.35, zorder=3)
    ax.plot(x[order], y[order], color="0.55", lw=1.0, alpha=0.7, label="Window polyline")
    ax.plot(sm[:, 0], sm[:, 1], color="crimson", lw=2.2, label="LOWESS")
    ax.set_xlabel(r"Mean lake density in window (lake area / $A$, km$^2$ lake per km$^2$ basin)")
    ax.set_ylabel(r"$\beta$ in $\log_{10}(R/P) = \beta \log_{10}(A) + c$")
    ax.set_title("Effect of lake density on runoff-efficiency scaling exponent (sliding windows)")
    ax.text(
        0.03,
        0.97,
        f"Pearson r = {corr.get('pearson_r', float('nan')):.3f}, p = {corr.get('pearson_p', float('nan')):.3g}\n"
        f"Spearman rho = {corr.get('spearman_rho', float('nan')):.3f}, p = {corr.get('spearman_p', float('nan')):.3g}",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.55", alpha=0.92),
    )
    ax.legend(loc="best", fontsize=8)
    fig.savefig(png, bbox_inches="tight", dpi=175)
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def window_level_correlation(beta_df: pd.DataFrame) -> dict[str, float]:
    from scipy.stats import pearsonr, spearmanr

    x = beta_df["lake_density_mean"].to_numpy(float)
    y = beta_df["beta"].to_numpy(float)
    if len(x) < 3:
        return {"pearson_r": float("nan"), "pearson_p": float("nan"), "spearman_rho": float("nan"), "spearman_p": float("nan")}
    rp, pp = pearsonr(x, y)
    rs, ps = spearmanr(x, y)
    return {"pearson_r": float(rp), "pearson_p": float(pp), "spearman_rho": float(rs), "spearman_p": float(ps)}


def build_results_md(
    prep: dict[str, Any],
    lake_meta: dict[str, Any],
    merge_meta: dict[str, Any],
    ols_pack: dict[str, Any],
    win_corr: dict[str, float],
    beta_df: pd.DataFrame,
) -> str:
    ld_p = ols_pack["lake_density_p"]
    ld_c = ols_pack["lake_density_coef"]
    sig = ld_p < 0.05
    if sig and ld_c > 0:
        rp_txt = (
            "At the river record level, **lake_density** is positively associated with **log10(R/P)** after controlling for drainage area and precipitation (linear OLS; see coefficients table). "
            "This is a **statistical association** only."
        )
    elif sig and ld_c < 0:
        rp_txt = (
            "**lake_density** shows a **negative** partial association with **log10(R/P)** in the fitted model. Avoid presuming direction without discussing confounding and data aggregation."
        )
    else:
        rp_txt = (
            "**lake_density** does **not** meet conventional significance in the multivariate OLS for **log10(R/P)**; do not over-interpret lake modulation from this term alone."
        )

    bmin, bmax = float(beta_df["beta"].min()), float(beta_df["beta"].max())
    wc_p = win_corr.get("pearson_p", float("nan"))
    if wc_p < 0.05:
        btxt = (
            f"Across sliding windows sorted by **lake_density**, window-mean **lake_density** vs. local **beta** shows a statistically detectable association (see correlation panel on Fig11). "
            f"**beta** ranges approximately **[{bmin:.4f}, {bmax:.4f}]**."
        )
    else:
        btxt = (
            f"Across windows, the association between mean **lake_density** and local **beta** is **not** strong at conventional levels (p={wc_p:.3g}). Treat LOWESS in Fig11 as **descriptive**."
        )

    lines = [
        "## Modulation by lakes",
        "",
        "### 1. Data and merge",
        f"- River rows after main `clean_for_hacks` and R/P filters: **{prep.get('n_after_rp_filters', '—')}**; after merge with lakes and **lake_density > 0**: **{merge_meta.get('n_after_merge_positive_density', '—')}**.",
        f"- Lake inventory: `{lake_meta.get('lake_excel', '')}`; region column `{lake_meta.get('lake_col_region', '')}`; area column `{lake_meta.get('lake_col_area', '')}`; **{lake_meta.get('lake_provinces', '—')}** province-level groups.",
        f"- River geo column for province: **`{prep.get('river_geo_col', '')}`**.",
        f"- **lake_density** = provincial lake area sum / **A_km2**; **lake_density2** = lake count / **A_km2** (backup proxy).",
        "",
        "### 2. R/P regression (log10_RP ~ log10_A + log10_P + lake_density)",
        "- " + rp_txt,
        f"- **Coefficient (lake_density)**: **{ld_c:.6f}**, **p** = **{ld_p:.4e}**; **ΔR² (M1−M0)** = **{ols_pack['delta_R2']:.5f}**; nested **F** = **{ols_pack['nested_F']:.4f}**, **p** = **{ols_pack['nested_F_p']:.4e}**.",
        "",
        "### 3. Local scaling beta along the lake-density gradient (sliding windows)",
        "- " + btxt,
        "",
        "### 4. Physical interpretation (non-causal, tentative)",
        "- Lakes can alter **storage, evaporation, and routing lags**; any alignment with **R/P** or **beta** must be interpreted with **ecoregion confounding**, **measurement scale**, and **province-level lake aggregation** in mind.",
        "- **Do not** infer causality from these observational regressions.",
        "",
        "### Outputs",
        "- `lake_density_dataset.csv`, `beta_vs_lake_density.csv`, `regression_lake_density_rp.csv`, `lake_modulation_summary.json`, `Fig11_lake_density_vs_beta.*`",
        "",
    ]
    return "\n".join(lines)


def embed_section_45(main_path: Path, section_md: str) -> None:
    if not main_path.exists():
        LOG.warning("Main RESULTS.md not found (%s), skip embed.", main_path)
        return
    start_m = "<!-- AUTO_SECTION_4_5_START -->\n"
    end_m = "<!-- AUTO_SECTION_4_5_END -->\n"
    block = start_m + section_md.strip() + "\n" + end_m
    text = main_path.read_text(encoding="utf-8")
    if start_m in text and end_m in text:
        text = re.sub(re.escape(start_m) + r".*?" + re.escape(end_m), block, text, count=1, flags=re.DOTALL)
    else:
        text = text.rstrip() + "\n\n" + block
    main_path.write_text(text, encoding="utf-8")
    LOG.info("Embedded Section 4.5 in %s", main_path)


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
    d, prep_meta = prepare_river_with_rp(cleaned)
    lake_group, col_reg, col_area, lake_meta = load_lake_table(LAKE_EXCEL_PATH)
    merged, merge_meta = merge_lakes_and_density(d, lake_group)

    out_cols = [
        "province_merge_key",
        "lake_area_sum",
        "lake_count",
        "A_km2",
        "lake_density",
        "lake_density2",
        "P_mm",
        "R_mm",
        "RP",
        "log10_RP",
        "log10_A",
        "log10_P",
    ]
    if "河流编码" in merged.columns:
        out_cols = ["河流编码"] + out_cols
    if prep_meta.get("river_geo_col"):
        out_cols = [prep_meta["river_geo_col"]] + out_cols
    merged[[c for c in out_cols if c in merged.columns]].to_csv(
        OUTPUT_DIR / "lake_density_dataset.csv", index=False, encoding="utf-8-sig"
    )

    ols_pack = fit_ols_lake_models(merged)
    ols_pack["coef_tbl"].to_csv(OUTPUT_DIR / "regression_lake_density_rp.csv", index=False, encoding="utf-8-sig")
    ols_pack["comparison"].to_csv(OUTPUT_DIR / "regression_lake_density_model_compare.csv", index=False, encoding="utf-8-sig")

    beta_df = run_sliding_beta_vs_lake_density(merged)
    beta_df.to_csv(OUTPUT_DIR / "beta_vs_lake_density.csv", index=False, encoding="utf-8-sig")

    win_corr = window_level_correlation(beta_df)
    from statsmodels.nonparametric.smoothers_lowess import lowess

    xb = beta_df["lake_density_mean"].to_numpy(float)
    yb = beta_df["beta"].to_numpy(float)
    frac = min(0.45, max(0.2, 10.0 / max(len(xb), 5)))
    sm = lowess(yb, xb, frac=frac, it=3, return_sorted=True)
    pd.DataFrame({"lake_density_mean": sm[:, 0], "beta_lowess": sm[:, 1]}).to_csv(
        OUTPUT_DIR / "beta_vs_lake_density_lowess.csv", index=False, encoding="utf-8-sig"
    )

    plot_fig11(beta_df, win_corr, OUTPUT_DIR / "Fig11_lake_density_vs_beta.png", OUTPUT_DIR / "Fig11_lake_density_vs_beta.pdf")

    summary = {
        **prep_meta,
        **lake_meta,
        **merge_meta,
        "lake_density_coef": ols_pack["lake_density_coef"],
        "lake_density_pvalue": ols_pack["lake_density_p"],
        "delta_R2_M1_vs_M0": ols_pack["delta_R2"],
        "nested_F_p": ols_pack["nested_F_p"],
        "n_windows_beta": int(len(beta_df)),
        "window_correlation": win_corr,
        "RP_max": RP_MAX,
        "WINDOW_SIZE": WINDOW_SIZE,
        "STEP_SIZE": STEP_SIZE,
    }
    (OUTPUT_DIR / "lake_modulation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    section = build_results_md(prep_meta, lake_meta, merge_meta, ols_pack, win_corr, beta_df)
    (OUTPUT_DIR / "RESULTS_section_4_5.md").write_text(section, encoding="utf-8")
    embed_section_45(MAIN_RESULTS_MD, section)

    LOG.info("4.5 done -> %s", OUTPUT_DIR)
    return summary


def main() -> int:
    try:
        run_all()
        return 0
    except Exception as e:
        LOG.exception("4.5 failed: %s", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
