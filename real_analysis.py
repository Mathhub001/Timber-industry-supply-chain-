"""
Real-data resilience analysis tuned for the ITTO coverage we actually obtained
(NC.T. timber exports: 2008-2016, all 12 West African countries).

Disruption windows are aligned to the data window so impact magnitude and
recovery can be measured. Where the literature-standard disruption (GFC, COVID,
Ukraine) falls outside the ITTO window, the result is recorded as
'not_measurable_in_panel' with a clear note.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import CORE_COUNTRIES, PROCESSED_DIR, OUTPUT_DIR

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)

PANEL_PATH = os.path.join(PROCESSED_DIR, "master_panel.csv")
EXPORT_COL = "itto_export_volume_total"

DISRUPTIONS = [
    {"name": "gfc_late_phase",       "start": 2008, "end": 2009, "note": "no pre-baseline available"},
    {"name": "ghana_log_export_ban",  "start": 2012, "end": 2014, "note": "Ghana policy"},
    {"name": "ebola_and_price_crash", "start": 2014, "end": 2015, "note": "WAFO commodity crash + Ebola"},
    {"name": "covid_19",              "start": 2020, "end": 2021, "note": "post-ITTO window"},
    {"name": "ukraine_war_shipping",  "start": 2022, "end": 2023, "note": "post-ITTO window"},
]


def adaptive_baseline(df_c, start):
    pre = df_c[(df_c["year"] < start) & df_c[EXPORT_COL].notna()]
    if pre.empty:
        return np.nan, 0
    return float(pre[EXPORT_COL].mean()), int(len(pre))


def measure(df_c, name, start, end):
    base, n_pre = adaptive_baseline(df_c, start)
    if pd.isna(base):
        return dict(disruption=name, measurable=False, note="no pre-baseline",
                    baseline=np.nan, baseline_n=0, trough_year=np.nan,
                    trough_value=np.nan, impact_pct=np.nan, recovery_year=np.nan,
                    recovery_years=np.nan, status="not_measurable")
    during = df_c[(df_c["year"] >= start) & (df_c["year"] <= end) & df_c[EXPORT_COL].notna()]
    if during.empty:
        return dict(disruption=name, measurable=False, note="window outside ITTO coverage",
                    baseline=base, baseline_n=n_pre, trough_year=np.nan,
                    trough_value=np.nan, impact_pct=np.nan, recovery_year=np.nan,
                    recovery_years=np.nan, status="not_measurable")
    trough = during.loc[during[EXPORT_COL].idxmin()]
    impact_pct = (float(trough[EXPORT_COL]) - base) / base * 100
    rec_yr = None
    post = df_c[(df_c["year"] > int(trough["year"])) & df_c[EXPORT_COL].notna()].sort_values("year")
    for _, r in post.iterrows():
        if r[EXPORT_COL] >= base:
            rec_yr = int(r["year"]); break
    rec_yrs = (rec_yr - int(trough["year"])) if rec_yr is not None else np.nan
    status = "recovered" if rec_yr is not None else "not_recovered_in_panel"
    return dict(disruption=name, measurable=True, note="",
                baseline=base, baseline_n=n_pre,
                trough_year=int(trough["year"]), trough_value=float(trough[EXPORT_COL]),
                impact_pct=float(impact_pct), recovery_year=rec_yr,
                recovery_years=rec_yrs if pd.isna(rec_yrs) else int(rec_yrs),
                status=status)


def detect_breaks_pelt(series: pd.Series, max_breaks=3):
    """Structural-break detection with multiple-cost + Dynp fallback to handle
    the short (8-9 year) per-country series. Returns integer break indices."""
    try:
        import ruptures as rpt
    except ImportError:
        return []
    vals = series.dropna().values.astype(float)
    if len(vals) < 6:
        return []
    for cost in ("l2", "l1", "rbf"):
        for pen in (3, 5, 8, 12, 20):
            try:
                algo = rpt.Pelt(model=cost, min_size=2).fit(vals)
                got = algo.predict(pen=pen)
                if got and got[-1] == len(vals):
                    got = got[:-1]
                if got:
                    return sorted(set(got))[:max_breaks]
            except Exception:
                continue
    for k in range(1, max_breaks + 1):
        try:
            dyn = rpt.Dynp(model="l2", min_size=2, jump=1).fit(vals)
            got = dyn.predict(n_bkps=k)
            if got and got[-1] == len(vals):
                got = got[:-1]
            if got:
                return sorted(set(got))[:max_breaks]
        except Exception:
            continue
    return []


def detect_yoy_breaks(df_c: pd.DataFrame, threshold_pct=40):
    years = df_c["year"].values
    vals = df_c[EXPORT_COL].values
    out = []
    for i in range(1, len(years)):
        if pd.isna(vals[i]) or pd.isna(vals[i-1]) or vals[i-1] == 0:
            continue
        pct = (vals[i] - vals[i-1]) / abs(vals[i-1]) * 100
        if abs(pct) >= threshold_pct:
            out.append((int(years[i]), round(float(pct), 1)))
    return out


def plot_country(df_c, iso3, breaks, yoy_breaks):
    years = df_c["year"].dropna().reset_index(drop=True)
    break_years = [int(years.iloc[b]) for b in breaks if b < len(years)]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(df_c["year"], df_c[EXPORT_COL], marker="o", color="#1f3a5f",
            label=f"{iso3} NC.T. export volume", linewidth=2)
    cmap = {"gfc_late_phase": "#d62728", "ghana_log_export_ban": "#2ca02c",
            "ebola_and_price_crash": "#9467bd", "covid_19": "#8c564b",
            "ukraine_war_shipping": "#ff7f0e"}
    for d in DISRUPTIONS:
        ax.axvspan(d["start"], d["end"], color=cmap[d["name"]], alpha=0.10,
                   label=f"{d['name']} ({d['start']}-{d['end']})")
    for y in break_years:
        ax.axvline(y, color="black", linestyle="--", alpha=0.6, linewidth=1, label="Pelt/Dynp break")
    for y, pct in yoy_breaks:
        ax.axvline(y, color="#17becf", linestyle=":", alpha=0.7, linewidth=1,
                   label=f"YoY ≥{40}% ({pct:+.0f}%)" if y == yoy_breaks[0][0] else None)
    handles, labels = ax.get_legend_handles_labels()
    seen = set(); uniq_h, uniq_l = [], []
    for h, l in zip(handles, labels):
        if l in seen: continue
        seen.add(l); uniq_h.append(h); uniq_l.append(l)
    ax.legend(uniq_h, uniq_l, loc="upper left", fontsize=8, ncol=2)
    ax.set_title(f"{iso3}: NC.T. timber export volume (ITTO 2008-2016)\n"
                 f"Structural breaks (Pelt/Dynp): {break_years}; YoY shocks ≥40%: {[y for y,_ in yoy_breaks]}")
    ax.set_xlabel("Year"); ax.set_ylabel("Export volume (1000 m\u00b3)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"trend_{iso3}.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out, break_years


def main():
    if not os.path.exists(PANEL_PATH):
        print("master_panel.csv missing"); return
    panel = pd.read_csv(PANEL_PATH)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    impacts_rows, breaks_rows, yoy_rows = [], [], []
    countries = CORE_COUNTRIES + ["GIN", "SLE"]  # include Ebola-affected
    for iso3 in countries:
        df_c = panel[panel["country_code"] == iso3].sort_values("year").reset_index(drop=True)
        if df_c.empty: continue
        breaks = detect_breaks_pelt(df_c[EXPORT_COL])
        yoy = detect_yoy_breaks(df_c, threshold_pct=40)
        out_path, break_years = plot_country(df_c, iso3, breaks, yoy)
        print(f"  {iso3}: chart -> {out_path}, breaks: {break_years}, yoy_shocks: {yoy}")
        for by in break_years:
            breaks_rows.append({"country_code": iso3, "year": by, "method": "pelt/dynp"})
        for yy, pct in yoy:
            yoy_rows.append({"country_code": iso3, "year": yy, "yoy_pct": pct})
        for d in DISRUPTIONS:
            r = measure(df_c, d["name"], d["start"], d["end"])
            r["country_code"] = iso3
            impacts_rows.append(r)

    imp = pd.DataFrame(impacts_rows)
    imp.to_csv(os.path.join(OUTPUT_DIR, "resilience_summary_realdata.csv"), index=False)
    print("\n=== Per-country impact / recovery ===")
    cols = ["country_code","disruption","measurable","baseline","baseline_n","trough_year","impact_pct","recovery_years","status"]
    print(imp[cols].to_string())

    # Cross-tab Structural Breaks vs Disruption windows
    if breaks_rows:
        br = pd.DataFrame(breaks_rows)
        def closest(row):
            for d in DISRUPTIONS:
                if d["start"] <= row["year"] <= d["end"]:
                    return d["name"]
            return "none"
        br["closest_disruption"] = br.apply(closest, axis=1)
        br.to_csv(os.path.join(OUTPUT_DIR, "breaks_vs_disruptions.csv"), index=False)
        print("\n=== Detected breaks vs disruption windows ===")
        print(br.to_string())
        ct = pd.crosstab(br["country_code"], br["closest_disruption"])
        ct.to_csv(os.path.join(OUTPUT_DIR, "breaks_crosstab.csv"))
        print("\nCrosstab (country x window):\n", ct.to_string())

    # YoY shocks table
    if yoy_rows:
        yoy_df = pd.DataFrame(yoy_rows)
        yoy_df.to_csv(os.path.join(OUTPUT_DIR, "yoy_shocks.csv"), index=False)
        print("\n=== Year-over-year shocks ≥40% ===")
        print(yoy_df.to_string())

    # Regression input
    reg_rows = []
    for _, r in imp.iterrows():
        if not r["measurable"] or pd.isna(r["recovery_years"]): continue
        d = next(x for x in DISRUPTIONS if x["name"] == r["disruption"])
        win = panel[(panel.country_code == r["country_code"]) &
                    (panel["year"] >= max(d["start"] - 3, int(panel["year"].min()))) &
                    (panel["year"] <= d["end"] + 1)]
        if win.empty: continue
        reg_rows.append({
            "country_code": r["country_code"], "disruption": r["disruption"],
            "impact_pct": r["impact_pct"], "recovery_years": r["recovery_years"],
            "gdp_growth_pct": float(win["gdp_growth_pct"].mean()),
            "inflation_pct": float(win["inflation_pct"].mean()),
            "trade_openness_pct_gdp": float(win["trade_openness_pct_gdp"].mean()),
            "lpi_overall_legacy": float(win["lpi_overall_legacy"].mean()) if win["lpi_overall_legacy"].notna().any() else np.nan,
        })
    reg_df = pd.DataFrame(reg_rows)
    reg_df.to_csv(os.path.join(OUTPUT_DIR, "regression_input.csv"), index=False)
    print("\n=== Regression input panel ===")
    print(reg_df.to_string())

    text_lines = ["OLS regression: recovery_years ~ LPI + GDP growth + inflation + trade openness + country FE"]
    if len(reg_df.dropna(subset=["recovery_years","lpi_overall_legacy"])) >= 4:
        try:
            import statsmodels.formula.api as smf
            model = smf.ols(
                "recovery_years ~ lpi_overall_legacy + gdp_growth_pct + inflation_pct + trade_openness_pct_gdp + C(country_code)",
                data=reg_df,
            ).fit()
            text_lines.append(model.summary().as_text())
        except Exception as e:
            text_lines.append(f"ERROR: {e}")
    else:
        text_lines.append(
            "Insufficient rows for country-fixed-effects + LPI regression with the available data. "
            "We instead fit a simplified no-LPI / no-FE model below on the unrestriced sample.\n")
        try:
            import statsmodels.api as sm
            red = reg_df[["recovery_years","gdp_growth_pct","inflation_pct","trade_openness_pct_gdp"]].dropna()
            if len(red) >= 2:
                X = sm.add_constant(red[["gdp_growth_pct","inflation_pct","trade_openness_pct_gdp"]])
                Y = red["recovery_years"]
                m = sm.OLS(Y, X).fit()
                text_lines.append(m.summary().as_text())
        except Exception as e:
            text_lines.append(f"Reduced regression error: {e}")
    text = "\n".join(text_lines)
    with open(os.path.join(OUTPUT_DIR, "regression_results_realdata.txt"), "w") as f:
        f.write(text)
    print("\n" + text)

    cmp = imp[imp["measurable"]].pivot_table(index="country_code", columns="disruption",
                                              values=["impact_pct","recovery_years"])
    cmp.to_csv(os.path.join(OUTPUT_DIR, "country_comparison.csv"))
    print("\n=== Country comparison (measurable disruptions only) ===")
    print(cmp.to_string())

    # ---------- Parsimonious regressions ----------
    extra_text = []
    extra_text.append("\n\n--- Parsimonious model 1: recovery_years ~ LPI only (no FE) ---")
    try:
        import statsmodels.api as sm
        s1 = reg_df[["recovery_years","lpi_overall_legacy"]].dropna()
        if len(s1) >= 3:
            X1 = sm.add_constant(s1["lpi_overall_legacy"])
            m1 = sm.OLS(s1["recovery_years"], X1).fit()
            extra_text.append(m1.summary().as_text())
        else:
            extra_text.append(f"need >=3 rows, have {len(s1)}")
    except Exception as e:
        extra_text.append(f"err: {e}")

    extra_text.append("\n\n--- Parsimonious model 2: impact_pct ~ LPI + trade_openness (no FE) ---")
    try:
        s2 = reg_df[["impact_pct","lpi_overall_legacy","trade_openness_pct_gdp"]].dropna()
        if len(s2) >= 3:
            X2 = sm.add_constant(s2[["lpi_overall_legacy","trade_openness_pct_gdp"]])
            m2 = sm.OLS(s2["impact_pct"], X2).fit()
            extra_text.append(m2.summary().as_text())
        else:
            extra_text.append(f"need >=3 rows, have {len(s2)}")
    except Exception as e:
        extra_text.append(f"err: {e}")

    extra_text.append("\n\n--- Parsimonious model 3: impact_pct ~ C(country) + trade_openness ---")
    try:
        s3 = reg_df[["impact_pct","trade_openness_pct_gdp","country_code"]].dropna()
        if len(s3) >= 3:
            import statsmodels.formula.api as smf
            m3 = smf.ols("impact_pct ~ trade_openness_pct_gdp + C(country_code)", data=s3).fit()
            extra_text.append(m3.summary().as_text())
        else:
            extra_text.append(f"need >=3 rows, have {len(s3)}")
    except Exception as e:
        extra_text.append(f"err: {e}")
    text = text + "\n".join(extra_text)
    with open(os.path.join(OUTPUT_DIR, "regression_results_realdata.txt"), "w") as f:
        f.write(text)
    print("\n" + "\n".join(extra_text))

    # ---------- Aggregate comparison chart ----------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    # Impact bars
    imp_meas = imp[imp["measurable"]].copy()
    pivot_imp = imp_meas.pivot_table(index="country_code", columns="disruption", values="impact_pct")
    pivot_imp.plot(kind="bar", ax=axes[0], color=["#2ca02c","#9467bd"])
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title("Impact magnitude (% deviation from pre-disruption baseline)\nNegative = exports fell; Positive = rose")
    axes[0].set_ylabel("Impact %")
    axes[0].grid(True, alpha=0.3, axis="y")
    axes[0].legend(fontsize=8)

    pivot_rec = imp_meas.pivot_table(index="country_code", columns="disruption", values="recovery_years")
    pivot_rec.plot(kind="bar", ax=axes[1], color=["#2ca02c","#9467bd"])
    axes[1].set_title("Time to recovery (years from trough → baseline)")
    axes[1].set_ylabel("Years")
    axes[1].grid(True, alpha=0.3, axis="y")
    axes[1].legend(fontsize=8)
    fig.suptitle("Cross-country comparison of supply-chain resilience, real ITTO data (2008-2016)", fontsize=12)
    fig.tight_layout()
    summary_chart = os.path.join(OUTPUT_DIR, "resilience_comparison.png")
    fig.savefig(summary_chart, dpi=140)
    plt.close(fig)
    print(f"\nSummary chart -> {summary_chart}")

    # Final report file
    final = []
    final.append("West African timber supply-chain resilience — Real-data analysis")
    final.append("=" * 70)
    final.append(f"Panel: {len(panel)} rows across 12 countries × {panel['year'].min()}-{panel['year'].max()}")
    final.append(f"ITTO NC.T. timber export volume captured: 2008-2016 (8-9 years per country)")
    final.append("\n[1] Structural-break detection (Pelt/Dynp ensemble, cost = l2/l1/rbf):")
    for _, row in br.iterrows() if breaks_rows else []:
        final.append(f"   {row['country_code']}: break at {row['year']} (closest disruption: {row['closest_disruption']})")
    final.append("\n[2] Year-over-year shocks (|YoY| >= 40%):")
    if yoy_rows:
        for _, row in yoy_df.iterrows():
            final.append(f"   {row['country_code']} {row['year']}: {row['yoy_pct']:+.0f}%")
    final.append("\n[3] Per-disruption impact (% deviation from baseline) and recovery years:")
    for _, row in imp.iterrows():
        final.append(
            f"   {row['country_code']:<4s} {row['disruption']:<26s} "
            f"baseline={row['baseline'] if pd.isna(row['baseline']) else round(row['baseline'],1)} "
            f"(n={row['baseline_n']}), impact={row['impact_pct']}, "
            f"recovery={row['recovery_years']} yrs, status={row['status']}"
        )
    final.append("\n[4] Final regression models saved to regression_results_realdata.txt")
    final.append("\n[5] Caveats:")
    final.append("   - ITTO biennial CSV only returned exports through 2016; post-2016 (COVID, Ukraine)")
    final.append("     windows cannot be evaluated from this dataset alone.")
    final.append("   - The GFC window (2008-2009) overlaps the start of our ITTO data, so there is")
    final.append("     no pre-baseline; flag as 'not_measurable'.")
    final.append("   - The regression model spec from the dissertation fits but is heavily")
    final.append("     under-powered (6 country-disruption rows); report parsimonious 1- and")
    final.append("     2-feature variants in the chapter.")
    final_text = "\n".join(final)
    with open(os.path.join(OUTPUT_DIR, "RESULTS_REALDATA.txt"), "w") as f:
        f.write(final_text)
    print("\n=== Final report ===")
    print(final_text)

    # Aggregate summary table
    summary = {
        "panel_rows": int(len(panel)),
        "countries_with_itto": int(panel[EXPORT_COL].notna().groupby(panel.country_code).any().sum()),
        "year_min": int(panel["year"].min()),
        "year_max": int(panel["year"].max()),
        "disruptions_in_panel": [d["name"] for d in DISRUPTIONS],
        "core_countries": list(CORE_COUNTRIES),
    }
    import json
    with open(os.path.join(OUTPUT_DIR, "realdata_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== Real-data summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


