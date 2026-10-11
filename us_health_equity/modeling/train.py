"""Fit the project's statistical models and write results to models/.

1. State-level (ecological) associations between socioeconomic status and outcomes:
   for each year, OLS of an outcome on median household income (per $10k, real dollars)
   and, separately, on % of adults 25+ with a bachelor's degree (per 5 points), plus a
   joint model with both. These describe states, not individuals.
2. Health-spending trend: log-linear OLS of real per-capita national health expenditure
   on year, giving the average annual real growth rate and its confidence interval.

    python -m us_health_equity.modeling.train
"""

import json

from loguru import logger
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import typer

from us_health_equity.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

OUTCOMES = {
    "mmr_coverage": "Kindergarten MMR coverage (%)",
    "any_exemption": "Kindergarten vaccine exemptions (%)",
    "flu_coverage": "Flu vaccination coverage, 6 mo+ (%)",
    "aadr_all_causes": "All-cause mortality (age-adj. per 100k)",
    "aadr_heart_disease": "Heart disease mortality (age-adj. per 100k)",
    "aadr_cancer": "Cancer mortality (age-adj. per 100k)",
    "aadr_unintentional_injuries": "Unintentional injury mortality (age-adj. per 100k)",
    "aadr_diabetes": "Diabetes mortality (age-adj. per 100k)",
    "aadr_chronic_lower_respiratory_disease": "CLRD mortality (age-adj. per 100k)",
    "aadr_stroke": "Stroke mortality (age-adj. per 100k)",
}
PREDICTORS = {
    "income_10k": "Median household income (per $10k, real)",
    "college_5pt": "% with bachelor's degree+ (per 5 pts)",
}


def ses_associations(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.assign(
        income_10k=panel["median_household_income_real"] / 10_000,
        college_5pt=panel["pct_bachelors_plus"] / 5,
    )
    rows = []
    for outcome, outcome_label in OUTCOMES.items():
        for year, g in df.groupby("year"):
            g = g.dropna(subset=[outcome, "income_10k", "college_5pt"])
            if len(g) < 20:
                continue
            fits = {p: smf.ols(f"{outcome} ~ {p}", data=g).fit() for p in PREDICTORS}
            fits["joint"] = smf.ols(f"{outcome} ~ income_10k + college_5pt", data=g).fit()
            for model, fit in fits.items():
                for pred in [p for p in PREDICTORS if p in fit.params]:
                    lo, hi = fit.conf_int().loc[pred]
                    rows.append(
                        {
                            "year": year,
                            "outcome": outcome,
                            "outcome_label": outcome_label,
                            "model": "bivariate" if model != "joint" else "joint",
                            "predictor": pred,
                            "predictor_label": PREDICTORS[pred],
                            "slope": fit.params[pred],
                            "ci_low": lo,
                            "ci_high": hi,
                            "p_value": fit.pvalues[pred],
                            "r": np.sign(fit.params[pred]) * np.sqrt(fit.rsquared)
                            if model != "joint"
                            else np.nan,
                            "r_squared": fit.rsquared,
                            "n_states": int(fit.nobs),
                        }
                    )
    return pd.DataFrame(rows)


def spending_trend(totals: pd.DataFrame) -> dict:
    t = totals.dropna(subset=["nhe_per_capita_real"]).copy()
    t["log_real_pc"] = np.log(t["nhe_per_capita_real"])
    fit = smf.ols("log_real_pc ~ year", data=t).fit()
    lo, hi = fit.conf_int().loc["year"]
    first, last = t.iloc[0], t.iloc[-1]
    years = last["year"] - first["year"]

    def cagr(col: str) -> float:
        return ((last[col] / first[col]) ** (1 / years) - 1) * 100

    return {
        "period": f"{int(first['year'])}-{int(last['year'])}",
        "real_per_capita_annual_growth_pct": (np.exp(fit.params["year"]) - 1) * 100,
        "real_per_capita_annual_growth_ci_pct": [(np.exp(lo) - 1) * 100, (np.exp(hi) - 1) * 100],
        "trend_p_value": fit.pvalues["year"],
        "nominal_total_cagr_pct": cagr("nhe_total_billions"),
        "nominal_per_capita_cagr_pct": cagr("nhe_per_capita"),
        "real_per_capita_cagr_pct": cagr("nhe_per_capita_real"),
        "nominal_total_change_pct": (last["nhe_total_billions"] / first["nhe_total_billions"] - 1)
        * 100,
        "real_per_capita_change_pct": (
            last["nhe_per_capita_real"] / first["nhe_per_capita_real"] - 1
        )
        * 100,
        "pct_gdp_start": first["nhe_pct_gdp"],
        "pct_gdp_end": last["nhe_pct_gdp"],
    }


@app.command()
def main():
    MODELS_DIR.mkdir(exist_ok=True)
    panel = pd.read_csv(PROCESSED_DATA_DIR / "state_panel.csv")
    assoc = ses_associations(panel)
    assoc.to_csv(MODELS_DIR / "ses_associations.csv", index=False)
    logger.info(f"ses_associations.csv: {len(assoc):,} coefficients")

    trend = spending_trend(pd.read_csv(PROCESSED_DATA_DIR / "health_spending_totals.csv"))
    (MODELS_DIR / "spending_trend.json").write_text(json.dumps(trend, indent=2, default=float))
    logger.info(
        f"Real per-capita health spending grew {trend['real_per_capita_annual_growth_pct']:.1f}%/yr "
        f"over {trend['period']} (p={trend['trend_p_value']:.3g})"
    )
    logger.success("Models written.")


if __name__ == "__main__":
    app()
