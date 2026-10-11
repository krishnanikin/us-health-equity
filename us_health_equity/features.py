"""Build the analysis-ready tables in data/processed from the tidy tables in data/interim.

python -m us_health_equity.features
"""

from loguru import logger
import pandas as pd
import typer

from us_health_equity.config import (
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    REAL_DOLLAR_BASE_YEAR,
    START_YEAR,
    STATES,
)

app = typer.Typer()

# BRFSS split its top income band in 2021; these bands are comparable across all years.
INCOME_BANDS = {
    "Less than $15,000": "<$15k",
    "$15,000-$24,999": "$15k-25k",
    "$25,000-$34,999": "$25k-35k",
    "$35,000-$49,999": "$35k-50k",
    "$50,000+": "$50k+",
    "$50,000-$99,999": "$50k+",
    "$100,000-$199,999": "$50k+",
    "$200,000+": "$50k+",
}
INCOME_ORDER = ["<$15k", "$15k-25k", "$25k-35k", "$35k-50k", "$50k+"]
EDUCATION_ORDER = ["Less than H.S.", "H.S. or G.E.D.", "Some post-H.S.", "College graduate"]


def _weighted_mean(df: pd.DataFrame, value: str = "value", weight: str = "sample_size"):
    w = df[weight].fillna(0)
    return (df[value] * w).sum() / w.sum() if w.sum() else df[value].mean()


def brfss_flu_by_ses(brfss: pd.DataFrame) -> pd.DataFrame:
    """Flu-shot coverage (adults 65+) by income band, education and race.

    BRFSS publishes subgroup estimates per state only, so the national figure is the
    median across the 50 states + DC (the same convention CDC uses for its US median).
    For 2021+, the three bands above $50k are pooled with sample-size weights.
    """
    df = brfss[brfss["locationabbr"].isin(STATES)].copy()
    df = df[df["category"] != "Overall"]
    is_inc = df["category"] == "Household Income"
    df.loc[is_inc, "group"] = df.loc[is_inc, "group"].map(INCOME_BANDS)
    df = df.dropna(subset=["group"])
    state_level = (
        df.groupby(["year", "locationabbr", "location", "category", "group"])
        .apply(_weighted_mean, include_groups=False)
        .rename("value")
        .reset_index()
    )
    national = (
        state_level.groupby(["year", "category", "group"])["value"]
        .agg(value="median", n_states="count")
        .reset_index()
        .assign(location="United States (median of states)", locationabbr="US")
    )
    # Drop groups too sparse to summarize nationally (e.g. NHPI is reported by few states).
    national = national[national["n_states"] >= 10]
    out = pd.concat([national, state_level], ignore_index=True)
    order = {g: i for i, g in enumerate(INCOME_ORDER + EDUCATION_ORDER)}
    out["order"] = out["group"].map(order)
    return out.sort_values(["category", "order", "year"])


def mortality_disparities(mort: pd.DataFrame) -> pd.DataFrame:
    """Add each group's age-adjusted rate ratio relative to non-Hispanic White."""
    ref = mort[mort["race_ethnicity"] == "Non-Hispanic White"][["year", "cause", "aadr"]]
    out = mort.merge(
        ref.rename(columns={"aadr": "aadr_nh_white"}), on=["year", "cause"], how="left"
    )
    out["rate_ratio_vs_nh_white"] = (out["aadr"] / out["aadr_nh_white"]).round(3)
    return out.drop(columns="aadr_nh_white")


def state_panel(
    mmr: pd.DataFrame,
    flu_age: pd.DataFrame,
    mort_state: pd.DataFrame,
    ses: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """One row per state-year joining vaccination, mortality and socioeconomic measures."""
    abbr = {v: k for k, v in STATES.items()}
    mmr = mmr[mmr["location"].isin(STATES.values())].assign(
        state=lambda d: d["location"].map(abbr), year=lambda d: d["season_start"]
    )[["state", "year", "mmr_coverage", "any_exemption"]]

    flu = flu_age[
        (flu_age["age_group"] == "6 months+") & flu_age["location"].isin(STATES.values())
    ]
    flu = flu.assign(state=flu["location"].map(abbr), year=flu["season_start"])[
        ["state", "year", "coverage"]
    ].rename(columns={"coverage": "flu_coverage"})

    annual = mort_state[mort_state["period"].str.fullmatch(r"\d{4}") & mort_state["state"].notna()]
    wide = annual.pivot_table(index=["state", "year"], columns="cause", values="aadr")
    wide.columns = [
        f"aadr_{c.lower().replace(' & ', '_').replace(' ', '_').replace('-', '')}"
        for c in wide.columns
    ]
    wide = wide.reset_index()

    cpi = prices.set_index("year")["cpi_u"]
    ses = ses.assign(
        median_household_income_real=lambda d: (
            d["median_household_income"] * cpi[REAL_DOLLAR_BASE_YEAR] / d["year"].map(cpi)
        )
    )
    panel = (
        ses.merge(mmr, on=["state", "year"], how="left")
        .merge(flu, on=["state", "year"], how="left")
        .merge(wide, on=["state", "year"], how="left")
    )
    return panel[panel["year"] >= START_YEAR].sort_values(["state", "year"])


def spending_tables(nhe: pd.DataFrame, prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    cpi = prices.set_index("year")[["cpi_u", "cpi_medical"]]
    deflator = cpi.loc[REAL_DOLLAR_BASE_YEAR, "cpi_u"] / cpi["cpi_u"]

    totals = nhe[nhe["category"] == "Total"].pivot_table(
        index="year", columns="series", values="value"
    )
    totals["nhe_per_capita_real"] = totals["nhe_per_capita"] * deflator
    totals["nhe_total_real_billions"] = totals["nhe_total_billions"] * deflator
    totals["nhe_growth_pct"] = totals["nhe_total_billions"].pct_change() * 100
    totals["gdp_growth_pct"] = totals["gdp_billions"].pct_change() * 100
    totals["cpi_u"] = cpi["cpi_u"]
    totals["cpi_medical"] = cpi["cpi_medical"]
    totals = totals[totals.index >= START_YEAR].reset_index()

    def breakdown(series: str) -> pd.DataFrame:
        df = nhe[nhe["series"] == series].copy()
        df["billions"] = df["value"] / 1000
        df["billions_real"] = df["billions"] * df["year"].map(deflator)
        df = df[df["year"] >= START_YEAR]
        base = df[df["year"] == START_YEAR].set_index("category")["billions"]
        df["pct_change_since_start"] = (df["billions"] / df["category"].map(base) - 1) * 100
        return df.drop(columns=["series", "value"])

    return {
        "health_spending_totals.csv": totals,
        "health_spending_by_payer.csv": breakdown("nhe_by_payer_millions"),
        "health_spending_by_service.csv": breakdown("nhe_by_service_millions"),
    }


@app.command()
def main():
    i, p = INTERIM_DATA_DIR, PROCESSED_DATA_DIR
    p.mkdir(parents=True, exist_ok=True)

    def read(name: str) -> pd.DataFrame:
        return pd.read_csv(i / name)

    prices = read("prices.csv")
    tables = {
        "vax_flu_by_race.csv": read("flu_by_race.csv"),
        "vax_flu_by_age.csv": read("flu_by_age.csv"),
        "vax_flu_65_by_ses.csv": brfss_flu_by_ses(read("brfss_flu_65_by_group.csv")),
        "vax_mmr_kindergarten.csv": read("mmr_kindergarten.csv"),
        "vax_mmr_child_by_ses.csv": read("mmr_nis_child.csv"),
        "mortality_by_race.csv": mortality_disparities(read("mortality_national_by_race.csv")),
        "mortality_by_state.csv": read("mortality_state.csv"),
        "state_panel.csv": state_panel(
            read("mmr_kindergarten.csv"),
            read("flu_by_age.csv"),
            read("mortality_state.csv"),
            read("state_ses.csv"),
            prices,
        ),
        **spending_tables(read("health_spending.csv"), prices),
    }
    for name, df in tables.items():
        df.to_csv(p / name, index=False)
        logger.info(f"{name}: {len(df):,} rows")
    logger.success("Processed tables written.")


if __name__ == "__main__":
    app()
