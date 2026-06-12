"""
Step 6. Administrative fragmentation and scaling-law perturbation.

This script evaluates how administrative partitioning alters river-network scaling relationships.
Administrative fragmentation is approximated by splitting each river according to the number of county-level units listed in the geographic attribute field.

For rivers crossing multiple administrative units:
    L_admin = L / n
    A_admin = A / n
where n is the number of administrative segments.

The workflow:

1. Load and clean river-network data.
2. Construct administratively fragmented river records.
3. Compare Hack's law under:
       - natural basin geometry
       - administrative fragmentation
4. Compare runoff-efficiency scaling:
       log10(R/P) ~ log10(A)
5. Quantify changes in scaling exponents and model fit.
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
CLEANED_CSV_PATH: Path | None = None
OUTPUT_DIR = Path(r"F:\Lake\Result")
MAIN_RESULTS_MD = Path(r"F:\Lake\Result\RESULTS.md")


RP_MAX = 1.5

COL_GEO = "流经区县"

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

from run_hacks_scaling_analysis import (  # noqa: E402
    clean_for_hacks,
    fit_hacks_ols,
    load_merged,
    _set_pub_style,
)


def _setup_logger() -> logging.Logger:
    log = logging.getLogger("admin_frag_46")
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


def count_admin_units(x: Any) -> int:
    """Number of county-level segments in 流经区县 (>=1)."""
    if pd.isna(x):
        return 1
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return 1
    parts = re.split(r"[、,，;；]", s)
    parts = [p.strip() for p in parts if p.strip()]
    return max(1, len(parts))


def build_admin_river_dataset(cleaned: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Returns:
      base_natural: one row per river (L, A, RP where available)
      admin_long: expanded rows with L_admin, A_admin
    """
    meta: dict[str, Any] = {}
    if COL_GEO not in cleaned.columns:
        raise ValueError(f"Missing column `{COL_GEO}` in cleaned data; cannot run Section 4.6.")

    d = cleaned.copy()
    d["L_km"] = pd.to_numeric(d["L_km"], errors="coerce")
    d["A_km2"] = pd.to_numeric(d["A_km2"], errors="coerce")
    d = d[d["L_km"].notna() & d["A_km2"].notna() & (d["L_km"] > 0) & (d["A_km2"] > 0)].copy()
    d = d.reset_index(drop=True)
    d["n_admin"] = d[COL_GEO].map(count_admin_units)

    pcol = resolve_precipitation_column(d)
    rcol = resolve_runoff_column(d)
    if pcol and rcol:
        d["P_mm"] = pd.to_numeric(d[pcol], errors="coerce")
        d["R_mm"] = pd.to_numeric(d[rcol], errors="coerce")
        d["RP"] = np.where(
            (d["P_mm"] > 0) & (d["R_mm"] > 0),
            d["R_mm"] / d["P_mm"],
            np.nan,
        )
        d.loc[(d["RP"].notna()) & ((d["RP"] <= 0) | (d["RP"] > RP_MAX)), "RP"] = np.nan
    else:
        d["P_mm"] = np.nan
        d["R_mm"] = np.nan
        d["RP"] = np.nan
        meta["note_rp"] = "Precip/runoff columns not found; RP analysis skipped in outputs."

    meta["n_rivers_L_A"] = int(len(d))
    meta["n_admin_gt1"] = int((d["n_admin"] > 1).sum())
    meta["n_admin_eq1"] = int((d["n_admin"] == 1).sum())

    rows: list[dict[str, Any]] = []
    for pos in range(len(d)):
        r = d.iloc[pos]
        n = int(r["n_admin"])
        la = float(r["L_km"]) / n
        aa = float(r["A_km2"]) / n
        for k in range(n):
            row = {
                "parent_row_id": pos,
                "fragment_index": k,
                "n_admin": n,
                "L_km": float(r["L_km"]),
                "A_km2": float(r["A_km2"]),
                "L_admin": la,
                "A_admin": aa,
                COL_GEO: r[COL_GEO],
            }
            if "河流编码" in d.columns and pd.notna(r.get("河流编码")):
                row["河流编码"] = r["河流编码"]
            if pcol:
                row["P_mm"] = r.get("P_mm", np.nan)
                row["R_mm"] = r.get("R_mm", np.nan)
                row["RP"] = r.get("RP", np.nan)
            rows.append(row)

    admin_long = pd.DataFrame(rows)
    admin_long["log10_A"] = np.log10(admin_long["A_km2"].astype(float))
    admin_long["log10_L"] = np.log10(admin_long["L_km"].astype(float))
    admin_long["log10_A_admin"] = np.log10(admin_long["A_admin"].astype(float))
    admin_long["log10_L_admin"] = np.log10(admin_long["L_admin"].astype(float))
    if admin_long["RP"].notna().any():
        admin_long["log10_RP"] = np.log10(admin_long["RP"].astype(float))

    base_natural = d.copy()
    base_natural["log10_A"] = np.log10(base_natural["A_km2"].astype(float))
    base_natural["log10_L"] = np.log10(base_natural["L_km"].astype(float))
    if pcol and rcol and "RP" in base_natural.columns:
        mrp = base_natural["RP"].notna() & np.isfinite(base_natural["RP"]) & (base_natural["RP"] > 0)
        base_natural.loc[~mrp, "RP"] = np.nan
        base_natural["log10_RP"] = np.where(base_natural["RP"].notna(), np.log10(base_natural["RP"].astype(float)), np.nan)

    meta["n_rows_admin_long"] = int(len(admin_long))
    LOG.info(
        "4.6 admin dataset: rivers=%d, expanded_rows=%d (n_admin>1: %d)",
        len(d),
        len(admin_long),
        meta["n_admin_gt1"],
    )
    return base_natural, admin_long, meta


def _ols_pack(log_a: np.ndarray, log_y: np.ndarray, label: str) -> dict[str, Any]:
    m = np.isfinite(log_a) & np.isfinite(log_y)
    if int(m.sum()) < 10:
        return {"label": label, "n": int(m.sum()), "slope": float("nan"), "intercept": float("nan"), "r2": float("nan"), "p_slope": float("nan")}
    o = fit_hacks_ols(log_a[m], log_y[m])
    return {
        "label": label,
        "n": int(m.sum()),
        "slope": float(o["h"]),
        "intercept": float(o["intercept_c"]),
        "r2": float(o["r2"]),
        "p_slope": float(o["p_slope"]),
        "ci_low": float(o["h_ci95_low"]),
        "ci_high": float(o["h_ci95_high"]),
    }


def run_scaling_comparisons(base: pd.DataFrame, admin_long: pd.DataFrame) -> pd.DataFrame:
    hack_nat = _ols_pack(base["log10_A"].to_numpy(float), base["log10_L"].to_numpy(float), "Hack_natural_L~A")
    hack_adm = _ols_pack(
        admin_long["log10_A_admin"].to_numpy(float),
        admin_long["log10_L_admin"].to_numpy(float),
        "Hack_admin_L_admin~A_admin",
    )
    rows = [hack_nat, hack_adm]
    if "log10_RP" in base.columns and base["log10_RP"].notna().any():
        bn = base.loc[base["log10_RP"].notna()].copy()
        an = admin_long.loc[admin_long["log10_RP"].notna()].copy()
        rp_nat = _ols_pack(bn["log10_A"].to_numpy(float), bn["log10_RP"].to_numpy(float), "RP_natural_logRP~logA")
        rp_adm = _ols_pack(an["log10_A_admin"].to_numpy(float), an["log10_RP"].to_numpy(float), "RP_admin_logRP~logA_admin")
        rows.extend([rp_nat, rp_adm])
    return pd.DataFrame(rows)


def plot_fig13(base: pd.DataFrame, admin_long: pd.DataFrame, hack_nat: dict, hack_adm: dict, png: Path, pdf: Path) -> None:
    import matplotlib.pyplot as plt

    _set_pub_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.5), constrained_layout=True)
    xa = base["log10_A"].to_numpy(float)
    yl = base["log10_L"].to_numpy(float)
    m = np.isfinite(xa) & np.isfinite(yl)
    ax1.scatter(xa[m], yl[m], s=4, alpha=0.12, c="0.25", rasterized=True)
    xs = np.linspace(np.nanmin(xa[m]), np.nanmax(xa[m]), 120)
    ax1.plot(
        xs,
        hack_nat["slope"] * xs + hack_nat["intercept"],
        color="crimson",
        lw=2,
        label=f"h={hack_nat['slope']:.3f}, R²={hack_nat['r2']:.3f}",
    )
    ax1.set_xlabel(r"$\log_{10}$(A, km$^2$) — natural basin")
    ax1.set_ylabel(r"$\log_{10}$(L, km)")
    ax1.set_title("Natural system: Hack's law")
    ax1.legend(loc="lower right", fontsize=8)

    xa2 = admin_long["log10_A_admin"].to_numpy(float)
    yl2 = admin_long["log10_L_admin"].to_numpy(float)
    m2 = np.isfinite(xa2) & np.isfinite(yl2)
    ax2.scatter(xa2[m2], yl2[m2], s=4, alpha=0.12, c="0.35", rasterized=True)
    xs2 = np.linspace(np.nanmin(xa2[m2]), np.nanmax(xa2[m2]), 120)
    ax2.plot(
        xs2,
        hack_adm["slope"] * xs2 + hack_adm["intercept"],
        color="darkgreen",
        lw=2,
        label=f"h={hack_adm['slope']:.3f}, R²={hack_adm['r2']:.3f}",
    )
    ax2.set_xlabel(r"$\log_{10}$(A_admin, km$^2$) — administrative split")
    ax2.set_ylabel(r"$\log_{10}$(L_admin, km)")
    ax2.set_title("Administrative fragmentation: perturbed L–A")
    ax2.legend(loc="lower right", fontsize=8)
    fig.suptitle("Fig13 — Natural vs. administratively fragmented Hack scaling", fontsize=11, y=1.02)
    fig.savefig(png, bbox_inches="tight", dpi=170)
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def plot_fig14(base: pd.DataFrame, admin_long: pd.DataFrame, rp_nat: dict | None, rp_adm: dict | None, png: Path, pdf: Path) -> None:
    import matplotlib.pyplot as plt

    if rp_nat is None or rp_adm is None:
        LOG.warning("Fig14 skipped: no RP data")
        return
    _set_pub_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.5), constrained_layout=True)
    bn = base.loc[base["log10_RP"].notna()].copy()
    xa = bn["log10_A"].to_numpy(float)
    yr = bn["log10_RP"].to_numpy(float)
    m = np.isfinite(xa) & np.isfinite(yr)
    ax1.scatter(xa[m], yr[m], s=4, alpha=0.12, c="steelblue", rasterized=True)
    xs = np.linspace(np.nanmin(xa[m]), np.nanmax(xa[m]), 120)
    ax1.plot(xs, rp_nat["slope"] * xs + rp_nat["intercept"], color="crimson", lw=2, label=f"beta={rp_nat['slope']:.3f}, R²={rp_nat['r2']:.3f}")
    ax1.set_xlabel(r"$\log_{10}$(A)")
    ax1.set_ylabel(r"$\log_{10}$(R/P)")
    ax1.set_title("Natural: runoff-efficiency vs. area")
    ax1.legend(loc="best", fontsize=8)

    an = admin_long.loc[admin_long["log10_RP"].notna()].copy()
    xa2 = an["log10_A_admin"].to_numpy(float)
    yr2 = an["log10_RP"].to_numpy(float)
    m2 = np.isfinite(xa2) & np.isfinite(yr2)
    ax2.scatter(xa2[m2], yr2[m2], s=4, alpha=0.1, c="darkorange", rasterized=True)
    xs2 = np.linspace(np.nanmin(xa2[m2]), np.nanmax(xa2[m2]), 120)
    ax2.plot(xs2, rp_adm["slope"] * xs2 + rp_adm["intercept"], color="darkgreen", lw=2, label=f"beta={rp_adm['slope']:.3f}, R²={rp_adm['r2']:.3f}")
    ax2.set_xlabel(r"$\log_{10}$(A_admin)")
    ax2.set_ylabel(r"$\log_{10}$(R/P) (same basin value per fragment)")
    ax2.set_title("Administrative split: same R/P, fragmented A")
    ax2.legend(loc="best", fontsize=8)
    fig.suptitle("Fig14 — R/P vs. area: natural vs. administrative geometry", fontsize=11, y=1.02)
    fig.savefig(png, bbox_inches="tight", dpi=170)
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def build_results_md(res_tbl: pd.DataFrame, meta: dict[str, Any]) -> str:
    def _get(lab: str) -> dict[str, Any]:
        r = res_tbl[res_tbl["label"] == lab]
        return r.iloc[0].to_dict() if len(r) else {}

    hn, ha = _get("Hack_natural_L~A"), _get("Hack_admin_L_admin~A_admin")
    dh = float(ha.get("slope", np.nan)) - float(hn.get("slope", np.nan)) if hn and ha else float("nan")
    dR2_h = float(ha.get("r2", np.nan)) - float(hn.get("r2", np.nan)) if hn and ha else float("nan")

    rp_n = _get("RP_natural_logRP~logA")
    rp_a = _get("RP_admin_logRP~logA_admin")
    has_rp = bool(rp_n) and bool(rp_a)
    if has_rp:
        db = float(rp_a.get("slope", np.nan)) - float(rp_n.get("slope", np.nan))
        dR2_b = float(rp_a.get("r2", np.nan)) - float(rp_n.get("r2", np.nan))
    else:
        db = dR2_b = float("nan")

    strong = (abs(dR2_h) > 0.01) or (has_rp and abs(dR2_b) > 0.01) or (abs(dh) > 0.03) or (has_rp and abs(db) > 0.03)
    if not strong:
        interp = (
            "Changes in **h**, **R²**, and **beta** between natural and administratively fragmented geometries are **small** in this run. "
            "Treat this subsection as **exploratory**; it can be shortened or omitted in the manuscript if desired."
        )
    else:
        interp = (
            "There are **non-negligible** shifts in fitted **h** and/or **beta** and/or **R²** when moving from basin-scale (L, A) to the equal-split administrative proxy (**L_admin**, **A_admin**). "
            "This is consistent with **geometric fragmentation** perturbing pooled log–log fits; it is **not** a causal statement about administrative boundaries."
        )

    lines = [
        "## Administrative fragmentation",
        "",
        "### 1. Data construction (approximate split)",
        r"- **County tokens** from `流经区县`, split on **、 , ， ; ；**; **n** = number of non-empty segments (minimum 1).",
        r"- If **n > 1**: create **n** rows with **L_admin = L/n**, **A_admin = A/n** (equal partition; **approximate** administrative cut).",
        "- If **n = 1**: one row; **L_admin=L**, **A_admin=A**.",
        f"- Rivers with valid L, A: **{meta.get('n_rivers_L_A', '—')}**; **n_admin>1** for **{meta.get('n_admin_gt1', '—')}** rivers; expanded rows: **{meta.get('n_rows_admin_long', '—')}**.",
        "",
        "### 2. Hack's law (log L ~ log A)",
        f"- **Natural** (basin): **h** ≈ **{hn.get('slope', float('nan')):.4f}**, **R²** ≈ **{hn.get('r2', float('nan')):.4f}**, **n** = **{hn.get('n', '—')}**.",
        f"- **Administrative proxy**: **h_admin** ≈ **{ha.get('slope', float('nan')):.4f}**, **R²** ≈ **{ha.get('r2', float('nan')):.4f}**, **n** = **{ha.get('n', '—')}** (fragment rows).",
        f"- **Δh** = h_admin − h_natural ≈ **{dh:+.4f}**; **ΔR² (Hack)** ≈ **{dR2_h:+.5f}**.",
        "",
        "### 3. Runoff-efficiency scaling (log10(R/P) ~ log10(A))",
    ]
    if has_rp:
        lines += [
            f"- **Natural**: **beta** ≈ **{rp_n.get('slope', float('nan')):.4f}**, **R²** ≈ **{rp_n.get('r2', float('nan')):.4f}**, **n** = **{rp_n.get('n', '—')}**.",
            f"- **Administrative**: same **R/P** per parent river, **x** = **log10(A_admin)**; **beta_admin** ≈ **{rp_a.get('slope', float('nan')):.4f}**, **R²** ≈ **{rp_a.get('r2', float('nan')):.4f}**, **n** = **{rp_a.get('n', '—')}**.",
            f"- **Δbeta** ≈ **{db:+.4f}**; **ΔR² (RP)** ≈ **{dR2_b:+.5f}**.",
        ]
    else:
        lines.append("- RP-based comparison **not available** (missing precip/runoff columns in this table).")

    lines += [
        "",
        "### 4. Interpretation (non-causal)",
        "- " + interp,
        "",
        "### Outputs",
        "- `admin_river_dataset.csv`, `admin_scaling_results.csv`, `Fig13_*`, `Fig14_*`",
        "",
    ]
    return "\n".join(lines)


def embed_section_46(main_path: Path, section_md: str) -> None:
    if not main_path.exists():
        LOG.warning("Main RESULTS.md not found (%s), skip embed.", main_path)
        return
    start_m = "<!-- AUTO_SECTION_4_6_START -->\n"
    end_m = "<!-- AUTO_SECTION_4_6_END -->\n"
    block = start_m + section_md.strip() + "\n" + end_m
    text = main_path.read_text(encoding="utf-8")
    if start_m in text and end_m in text:
        text = re.sub(re.escape(start_m) + r".*?" + re.escape(end_m), block, text, count=1, flags=re.DOTALL)
    else:
        text = text.rstrip() + "\n\n" + block
    main_path.write_text(text, encoding="utf-8")
    LOG.info("Embedded Section 4.6 in %s", main_path)


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
    base_nat, admin_long, meta = build_admin_river_dataset(cleaned)

    admin_long.to_csv(OUTPUT_DIR / "admin_river_dataset.csv", index=False, encoding="utf-8-sig")

    res_tbl = run_scaling_comparisons(base_nat, admin_long)
    res_tbl.to_csv(OUTPUT_DIR / "admin_scaling_results.csv", index=False, encoding="utf-8-sig")

    hack_nat = res_tbl[res_tbl["label"] == "Hack_natural_L~A"].iloc[0].to_dict()
    hack_adm = res_tbl[res_tbl["label"] == "Hack_admin_L_admin~A_admin"].iloc[0].to_dict()
    plot_fig13(base_nat, admin_long, hack_nat, hack_adm, OUTPUT_DIR / "Fig13_natural_vs_admin_Hack.png", OUTPUT_DIR / "Fig13_natural_vs_admin_Hack.pdf")

    rp_nat = res_tbl[res_tbl["label"] == "RP_natural_logRP~logA"]
    rp_adm = res_tbl[res_tbl["label"] == "RP_admin_logRP~logA_admin"]
    if len(rp_nat) and len(rp_adm):
        plot_fig14(
            base_nat,
            admin_long,
            rp_nat.iloc[0].to_dict(),
            rp_adm.iloc[0].to_dict(),
            OUTPUT_DIR / "Fig14_natural_vs_admin_RP_scaling.png",
            OUTPUT_DIR / "Fig14_natural_vs_admin_RP_scaling.pdf",
        )
    else:
        LOG.warning("Fig14 not produced (no RP scaling rows).")

    summary = {**meta}
    for _, r in res_tbl.iterrows():
        summary[r["label"]] = r.to_dict()
    (OUTPUT_DIR / "admin_fragmentation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    section = build_results_md(res_tbl, meta)
    (OUTPUT_DIR / "RESULTS_section_4_6.md").write_text(section, encoding="utf-8")
    embed_section_46(MAIN_RESULTS_MD, section)

    LOG.info("4.6 done -> %s", OUTPUT_DIR)
    return summary


def main() -> int:
    try:
        run_all()
        return 0
    except Exception as e:
        LOG.exception("4.6 failed: %s", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
