"""
Step 4. Coupling between river-network structure and runoff efficiency.

This script evaluates whether river-network structure contributes to runoff efficiency after accounting for drainage area and precipitation.

The following model is evaluated:
    log10(R/P) ~ log10(A) + log10(P) + Structure
where:
    R = runoff (mm)
    P = precipitation (mm)
    A = drainage area (km²)

Two structure proxies are considered:
    structure_1 = L / A^0.5
    structure_2 = L / A

The workflow:

1. Load and clean river-network data.
2. Calculate runoff efficiency (R/P).
3. Construct structure metrics.
4. Fit nested regression models.
5. Evaluate the incremental contribution of structure.
6. Quantify standardized effects and multicollinearity.
7. Generate publication-quality figures and summary tables.

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
OUTPUT_DIR = Path(r"F:\Lake\Result")
MAIN_RESULTS_MD = Path(r"F:\Lake\Result\RESULTS.md")

RP_MAX = 1.5

from run_hacks_scaling_analysis import (  # noqa: E402
    clean_for_hacks,
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


def _setup_logger() -> logging.Logger:
    log = logging.getLogger("struct_eff_44")
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


def prepare_structure_efficiency_dataset(cleaned: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    meta: dict[str, Any] = {"module": "structure_efficiency_coupling", "RP_max": RP_MAX}
    n0 = int(len(cleaned))
    meta["n_input_main_clean"] = n0

    pcol = resolve_precipitation_column(cleaned)
    rcol = resolve_runoff_column(cleaned)
    if pcol is None:
        raise ValueError("Precip column not found. Tried: " + ", ".join(COL_PRECIP_MM_CANDIDATES))
    if rcol is None:
        raise ValueError("Runoff column not found. Tried: " + ", ".join(COL_RUNOFF_MM_CANDIDATES))
    meta["precip_column"] = pcol
    meta["runoff_column"] = rcol

    d = cleaned.copy()
    d["P_mm"] = pd.to_numeric(d[pcol], errors="coerce")
    d["R_mm"] = pd.to_numeric(d[rcol], errors="coerce")
    d["A_km2"] = pd.to_numeric(d["A_km2"], errors="coerce")
    d["L_km"] = pd.to_numeric(d["L_km"], errors="coerce")

    drops: list[dict[str, Any]] = []

    def _keep(mask: pd.Series, reason: str) -> None:
        nonlocal d
        before = len(d)
        d = d.loc[mask].copy()
        drops.append({"step": reason, "removed": int(before - len(d)), "remaining": int(len(d))})

    _keep(d["P_mm"].notna() & d["R_mm"].notna() & d["A_km2"].notna() & d["L_km"].notna(), "valid P,R,A,L")
    _keep(d["A_km2"] > 0, "A_gt_0")
    _keep(d["P_mm"] > 0, "P_gt_0")
    _keep(d["R_mm"] > 0, "R_gt_0")
    d["RP"] = d["R_mm"] / d["P_mm"]
    _keep(d["RP"] > 0, "RP_gt_0")
    _keep(d["RP"] <= RP_MAX, f"RP_le_{RP_MAX}")

    d["structure_1"] = d["L_km"] / np.sqrt(d["A_km2"])
    d["structure_2"] = d["L_km"] / d["A_km2"]
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=["structure_1", "structure_2"])

    d["log10_A"] = np.log10(d["A_km2"].astype(float))
    d["log10_P"] = np.log10(d["P_mm"].astype(float))
    d["log10_RP"] = np.log10(d["RP"].astype(float))
    d = d[np.isfinite(d["log10_A"]) & np.isfinite(d["log10_P"]) & np.isfinite(d["log10_RP"])].copy()

    meta["cleaning_steps"] = drops
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
    LOG.info("4.4 prepare: n_in=%d -> n_out=%d; RP mean=%.4f", n0, len(d), meta["RP_stats"]["mean"])

    out_cols = [
        "L_km",
        "A_km2",
        "P_mm",
        "R_mm",
        "RP",
        "log10_RP",
        "log10_A",
        "log10_P",
        "structure_1",
        "structure_2",
    ]
    if "River ID" in d.columns:
        out_cols = ["River ID"] + out_cols
    return d[out_cols].copy(), meta


def _fit_ols(y: pd.Series, X: pd.DataFrame, add_const: bool = True):
    import statsmodels.api as sm

    Xd = sm.add_constant(X) if add_const else X
    return sm.OLS(y, Xd).fit()


def nested_f_test(res_restricted, res_full) -> tuple[float, float, int]:
    """F for comparing nested models: full vs restricted (restricted smaller)."""
    from scipy.stats import f as f_dist

    rss_r = float(res_restricted.ssr)
    rss_f = float(res_full.ssr)
    df_f = int(res_full.df_resid)
    df_diff = int(res_full.df_model - res_restricted.df_model)
    if df_diff <= 0 or rss_f <= 0:
        return float("nan"), float("nan"), df_diff
    F = ((rss_r - rss_f) / df_diff) / (rss_f / df_f)
    pval = float(1.0 - f_dist.cdf(F, df_diff, df_f))
    return F, pval, df_diff


def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    Xc = sm.add_constant(X.to_numpy(dtype=float), has_constant="add")
    names = list(X.columns)
    rows = []
    for i in range(1, Xc.shape[1]):
        rows.append({"variable": names[i - 1], "VIF": float(variance_inflation_factor(Xc, i))})
    return pd.DataFrame(rows)


def extract_coef_table(res, model_name: str) -> pd.DataFrame:
    params = res.params
    bse = res.bse
    tvalues = res.tvalues
    pvalues = res.pvalues
    ci = res.conf_int(alpha=0.05)
    rows = []
    for name in params.index:
        rows.append(
            {
                "model": model_name,
                "term": str(name),
                "coef": float(params[name]),
                "std_err": float(bse[name]),
                "t": float(tvalues[name]),
                "p": float(pvalues[name]),
                "ci_low": float(ci.loc[name, 0]),
                "ci_high": float(ci.loc[name, 1]),
            }
        )
    return pd.DataFrame(rows)


def standardized_coefficients(df: pd.DataFrame, struct_col: str, model_label: str) -> pd.DataFrame:
    """Z-score y and predictors; OLS with intercept (should be near zero)."""
    import statsmodels.api as sm

    cols_y = ["log10_RP"]
    cols_x = ["log10_A", "log10_P", struct_col]
    Z = df[cols_y + cols_x].apply(lambda s: (s - s.mean()) / s.std(ddof=1))
    res = sm.OLS(Z["log10_RP"], sm.add_constant(Z[cols_x])).fit()
    out = extract_coef_table(res, model_label)
    out = out[out["term"] != "const"].copy()
    out.rename(columns={"coef": "std_coef"}, inplace=True)
    return out[["model", "term", "std_coef", "std_err", "t", "p", "ci_low", "ci_high"]]


def run_models(df: pd.DataFrame) -> dict[str, Any]:
    import statsmodels.api as sm

    y = df["log10_RP"]
    X0 = df[["log10_A", "log10_P"]]
    X1 = df[["log10_A", "log10_P", "structure_1"]]
    X1b = df[["log10_A", "log10_P", "structure_2"]]

    m0 = _fit_ols(y, X0)
    m1 = _fit_ols(y, X1)
    m1b = _fit_ols(y, X1b)

    F_nest, p_nest, df_diff = nested_f_test(m0, m1)

    comparison = pd.DataFrame(
        [
            {
                "model": "M0_baseline",
                "formula": "log10_RP ~ log10_A + log10_P",
                "n": int(m0.nobs),
                "R2": float(m0.rsquared),
                "adj_R2": float(m0.rsquared_adj),
                "AIC": float(m0.aic),
                "BIC": float(m0.bic),
            },
            {
                "model": "M1_structure1",
                "formula": "log10_RP ~ log10_A + log10_P + structure_1",
                "n": int(m1.nobs),
                "R2": float(m1.rsquared),
                "adj_R2": float(m1.rsquared_adj),
                "AIC": float(m1.aic),
                "BIC": float(m1.bic),
            },
            {
                "model": "M1_structure2_robustness",
                "formula": "log10_RP ~ log10_A + log10_P + structure_2",
                "n": int(m1b.nobs),
                "R2": float(m1b.rsquared),
                "adj_R2": float(m1b.rsquared_adj),
                "AIC": float(m1b.aic),
                "BIC": float(m1b.bic),
            },
        ]
    )
    comparison["delta_R2_vs_M0"] = comparison["R2"] - float(m0.rsquared)

    coef_m1 = extract_coef_table(m1, "M1_structure1")
    vif_df = compute_vif(df[["log10_A", "log10_P", "structure_1"]])

    std_m1 = standardized_coefficients(df, "structure_1", "M1_structure1_zscore")
    std_m1b = standardized_coefficients(df, "structure_2", "M1_structure2_zscore")

    s1_row = coef_m1[coef_m1["term"] == "structure_1"].iloc[0]

    return {
        "m0": m0,
        "m1": m1,
        "m1b": m1b,
        "comparison": comparison,
        "coef_m1": coef_m1,
        "vif": vif_df,
        "std_m1": std_m1,
        "std_m1b": std_m1b,
        "nested_F": F_nest,
        "nested_F_p": p_nest,
        "nested_df_diff": df_diff,
        "delta_R2": float(m1.rsquared - m0.rsquared),
        "structure1_coef": float(s1_row["coef"]),
        "structure1_p": float(s1_row["p"]),
        "structure1_ci_low": float(s1_row["ci_low"]),
        "structure1_ci_high": float(s1_row["ci_high"]),
    }


def partial_regression_added_variable(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Residuals of log10_RP and structure_1 after regressing out log10_A, log10_P."""
    import statsmodels.api as sm

    y = df["log10_RP"]
    Xc = sm.add_constant(df[["log10_A", "log10_P"]])
    s = df["structure_1"]
    ry = np.asarray(sm.OLS(y, Xc).fit().resid, dtype=float)
    rs = np.asarray(sm.OLS(s, Xc).fit().resid, dtype=float)
    # auxiliary slope (should match m1 coef for structure_1)
    aux = sm.OLS(ry, sm.add_constant(rs)).fit()
    xs_line = np.linspace(np.nanmin(rs), np.nanmax(rs), 100)
    pv = np.asarray(aux.params, dtype=float).ravel()
    ys_line = float(pv[1]) * xs_line + float(pv[0])
    return rs, ry, xs_line, ys_line


def plot_fig9_partial_effect(df: pd.DataFrame, pack: dict[str, Any], png: Path, pdf: Path) -> None:
    import matplotlib.pyplot as plt

    _set_pub_style()
    rs, ry, xs_line, ys_line = partial_regression_added_variable(df)
    m1 = pack["m1"]
    s1 = pack["coef_m1"]
    row = s1[s1["term"] == "structure_1"].iloc[0]

    fig, ax = plt.subplots(figsize=(6.4, 4.6), constrained_layout=True)
    ax.scatter(rs, ry, s=6, alpha=0.12, c="0.25", edgecolors="none", rasterized=True)
    ax.plot(xs_line, ys_line, color="crimson", lw=2.0, label="OLS on partial residuals")
    ax.set_xlabel(r"Residual of $L / A^{0.5}$ after $\log_{10}A$, $\log_{10}P$ (added-variable axis)")
    ax.set_ylabel(r"Residual of $\log_{10}(R/P)$ after $\log_{10}A$, $\log_{10}P$")
    ax.set_title("Partial regression: network-structure proxy vs. runoff efficiency (controlled for scale and P)")
    ax.text(
        0.03,
        0.97,
        f"Full-model coef on structure_1: {row['coef']:.4f}\np = {row['p']:.3g}\n95% CI [{row['ci_low']:.4f}, {row['ci_high']:.4f}]",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.55", alpha=0.92),
    )
    ax.legend(loc="lower right", fontsize=8)
    fig.savefig(png, bbox_inches="tight", dpi=180)
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def plot_fig10_model_compare(pack: dict[str, Any], png: Path, pdf: Path) -> None:
    import matplotlib.pyplot as plt

    _set_pub_style()
    comp = pack["comparison"]
    std1 = pack["std_m1"].copy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 4.4), constrained_layout=True)
    labs = ["M0\nbaseline", "M1\nL/A^0.5", "M1\nL/A"]
    r2s = comp["R2"].tolist()
    cols = ["steelblue", "darkgreen", "peru"]
    ax1.bar(labs, r2s, color=cols, edgecolor="0.3", linewidth=0.6)
    ax1.set_ylabel(r"$R^2$")
    ax1.set_title(r"$R^2$ comparison (nested extension by structure proxy)")
    for i, v in enumerate(r2s):
        ax1.text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=9)

    terms = std1["term"].tolist()
    coefs = std1["std_coef"].tolist()
    ypos = np.arange(len(terms))
    colors = ["steelblue", "peru", "darkgreen"]
    ax2.barh(ypos, coefs, color=colors, edgecolor="0.3", linewidth=0.5)
    ax2.set_yticks(ypos)
    ax2.set_yticklabels([t.replace("structure_1", "L/A^0.5") for t in terms])
    ax2.axvline(0, color="0.4", lw=0.8)
    ax2.set_xlabel("Standardized coefficient (z-scored model)")
    ax2.set_title("Relative importance (M1 with L/A^0.5)")
    fig.savefig(png, bbox_inches="tight", dpi=170)
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def build_results_section_44_md(meta: dict[str, Any], pack: dict[str, Any]) -> str:
    c1 = pack["structure1_coef"]
    p1 = pack["structure1_p"]
    lo, hi = pack["structure1_ci_low"], pack["structure1_ci_high"]
    dR2 = pack["delta_R2"]
    Fv, Fp = pack["nested_F"], pack["nested_F_p"]

    m1b = pack["m1b"]
    s2_coef = float(m1b.params["structure_2"]) if "structure_2" in m1b.params.index else float("nan")
    s2_p = float(m1b.pvalues["structure_2"]) if "structure_2" in m1b.pvalues.index else float("nan")

    vif_max = float(pack["vif"]["VIF"].max()) if len(pack["vif"]) else float("nan")
    vif_note = (
        "Max VIF among log10_A, log10_P, structure_1 is moderate."
        if vif_max < 5
        else "Max VIF suggests possible multicollinearity; interpret coefficients cautiously."
    )

    if p1 < 0.05 and c1 > 0:
        caut = (
            "The estimated coefficient on **structure_1** is **positive** and statistically detectable at conventional levels. "
            "This is **consistent with** (not proof of) more elongated / developed channel geometry relative to basin size being associated with higher **R/P** in log space, **after controlling for drainage area and precipitation**. "
            "Mechanisms (e.g., routing, land cover, data aggregation) are not identified here."
        )
    elif p1 < 0.05 and c1 < 0:
        caut = (
            "The coefficient on **structure_1** is **negative** and statistically detectable. Avoid presuming a positive structural effect; discuss plausible confounding and scale effects."
        )
    else:
        caut = (
            "The coefficient on **structure_1** is **not** statistically strong at conventional levels; **do not** infer a structural coupling from this run alone."
        )

    lines = [
        "## Coupling between river-network structure and runoff efficiency",
        "",
        "### 1. Data cleaning (after main `clean_for_hacks`)",
        f"- Input rows: **{meta.get('n_input_main_clean', '—')}**; final rows: **{meta.get('n_final', '—')}**; removed here: **{meta.get('n_removed_total', '—')}**",
        f"- Precip: `{meta.get('precip_column', '')}`; runoff: `{meta.get('runoff_column', '')}`; filters: `A>0`, `P>0`, `R>0`, `0<RP<={meta.get('RP_max', RP_MAX)}`",
        f"- **RP** (mean / median / std / min / max): **{meta['RP_stats']['mean']:.4f}** / **{meta['RP_stats']['median']:.4f}** / **{meta['RP_stats']['std']:.4f}** / **{meta['RP_stats']['min']:.4f}** / **{meta['RP_stats']['max']:.4f}**",
        "",
        "### 2. Structure proxies",
        r"- **structure_1** = $L / A^{0.5}$ (L in km, A in km$^2$): length relative to the square-root scale of area.",
        r"- **structure_2** = $L / A$: optional contrast proxy.",
        "",
        "### 3. Models",
        "- **M0:** log10(RP) ~ log10(A) + log10(P)",
        "- **M1:** log10(RP) ~ log10(A) + log10(P) + structure_1",
        "",
        "### 4. Model comparison and nested test (M1 vs M0)",
        f"- **R²(M0)** = **{float(pack['m0'].rsquared):.4f}** ; **R²(M1)** = **{float(pack['m1'].rsquared):.4f}** → **ΔR²** = **{dR2:.5f}**",
        f"- **Nested F** (adding structure_1): **F** = **{Fv:.4f}**, **p** = **{Fp:.4e}** (df_diff = **{pack['nested_df_diff']}**)",
        "",
        "### 5. structure_1 in M1",
        f"- Coefficient: **{c1:.5f}**, **p** = **{p1:.4e}**, 95% CI **[{lo:.5f}, {hi:.5f}]**",
        "",
        "### 6. Robustness (structure_2 instead of structure_1)",
        f"- Coefficient on **structure_2**: **{s2_coef:.5f}**, **p** = **{s2_p:.4e}** (same controls). Compare sign/stability with structure_1.",
        "",
        "### 7. Standardized coefficients & VIF",
        "- See **`structure_efficiency_standardized_coefficients.csv`** and **`structure_efficiency_vif.csv`**.",
        f"- {vif_note} (max VIF ≈ **{vif_max:.2f}**).",
        "",
        "### 8. Interpretation (statistical coupling only; not causal)",
        "- " + caut,
        "",
        "### Outputs",
        "- Tables: `structure_efficiency_dataset.csv`, `structure_efficiency_model_comparison.csv`, `structure_efficiency_coefficients.csv` (M1 only), `structure_efficiency_coefficients_m0_m1.csv`, `structure_efficiency_vif.csv`, `structure_efficiency_standardized_coefficients.csv`",
        "- Figures: `Fig9_structure_partial_effect.png`, `Fig10_model_comparison.png` (+ PDF)",
        "",
    ]
    return "\n".join(lines)


def embed_section_44(main_path: Path, section_md: str) -> None:
    if not main_path.exists():
        LOG.warning("Main RESULTS.md not found (%s), skip embed.", main_path)
        return
    start_m = "<!-- AUTO_SECTION_4_4_START -->\n"
    end_m = "<!-- AUTO_SECTION_4_4_END -->\n"
    block = start_m + section_md.strip() + "\n" + end_m
    text = main_path.read_text(encoding="utf-8")
    if start_m in text and end_m in text:
        text = re.sub(re.escape(start_m) + r".*?" + re.escape(end_m), block, text, count=1, flags=re.DOTALL)
    else:
        text = text.rstrip() + "\n\n" + block
    main_path.write_text(text, encoding="utf-8")
    LOG.info("Embedded Section 4.4 in %s", main_path)


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
    df, meta = prepare_structure_efficiency_dataset(cleaned)
    if len(df) < 50:
        raise ValueError("Too few rows after cleaning for stable regression.")

    df.to_csv(OUTPUT_DIR / "structure_efficiency_dataset.csv", index=False, encoding="utf-8-sig")

    pack = run_models(df)
    pack["comparison"].to_csv(OUTPUT_DIR / "structure_efficiency_model_comparison.csv", index=False, encoding="utf-8-sig")
    pack["coef_m1"].to_csv(OUTPUT_DIR / "structure_efficiency_coefficients.csv", index=False, encoding="utf-8-sig")
    pd.concat(
        [extract_coef_table(pack["m0"], "M0_baseline"), pack["coef_m1"]],
        ignore_index=True,
    ).to_csv(OUTPUT_DIR / "structure_efficiency_coefficients_m0_m1.csv", index=False, encoding="utf-8-sig")
    pack["vif"].to_csv(OUTPUT_DIR / "structure_efficiency_vif.csv", index=False, encoding="utf-8-sig")
    std_all = pd.concat([pack["std_m1"], pack["std_m1b"]], ignore_index=True)
    std_all.to_csv(OUTPUT_DIR / "structure_efficiency_standardized_coefficients.csv", index=False, encoding="utf-8-sig")

    plot_fig9_partial_effect(df, pack, OUTPUT_DIR / "Fig9_structure_partial_effect.png", OUTPUT_DIR / "Fig9_structure_partial_effect.pdf")
    plot_fig10_model_compare(pack, OUTPUT_DIR / "Fig10_model_comparison.png", OUTPUT_DIR / "Fig10_model_comparison.pdf")

    meta_out = {
        **meta,
        "delta_R2_M1_vs_M0": pack["delta_R2"],
        "nested_F": pack["nested_F"],
        "nested_F_pvalue": pack["nested_F_p"],
        "structure1_coef": pack["structure1_coef"],
        "structure1_p": pack["structure1_p"],
        "R2_M0": float(pack["m0"].rsquared),
        "R2_M1": float(pack["m1"].rsquared),
        "R2_M1_s2": float(pack["m1b"].rsquared),
    }
    (OUTPUT_DIR / "structure_efficiency_summary.json").write_text(
        json.dumps(meta_out, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    section = build_results_section_44_md(meta, pack)
    (OUTPUT_DIR / "RESULTS_section_4_4.md").write_text(section, encoding="utf-8")
    embed_section_44(MAIN_RESULTS_MD, section)

    LOG.info("4.4 done -> %s", OUTPUT_DIR)
    return meta_out


def main() -> int:
    try:
        run_all()
        return 0
    except Exception as e:
        LOG.exception("4.4 failed: %s", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
