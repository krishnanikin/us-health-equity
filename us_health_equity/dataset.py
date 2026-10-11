"""Download raw source data (data/raw) and clean it into tidy tables (data/interim).

python -m us_health_equity.dataset download   # fetch everything into data/raw
python -m us_health_equity.dataset process    # tidy raw files into data/interim
"""

import csv
import io
from pathlib import Path
import re
import time
import zipfile

from loguru import logger
import pandas as pd
import requests
from tqdm import tqdm
import typer

from us_health_equity import wonder
from us_health_equity.config import (
    CDC_DATASETS,
    CMS_NHE_URLS,
    FRED_CSV,
    INTERIM_DATA_DIR,
    RAW_DATA_DIR,
    SOCRATA_BASE,
    START_YEAR,
    STATE_ABBR,
    STATES,
)

app = typer.Typer()

HEADERS = {"User-Agent": "us-health-equity research pipeline (python-requests)"}
# FRED's CDN stalls on browser- and python-style agents but serves plain CLI agents.
FRED_HEADERS = {"User-Agent": "curl/8.7.1"}


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def socrata(dataset_id: str, where: str, out: Path, force: bool = False) -> Path:
    """Download rows matching a SoQL WHERE clause from data.cdc.gov as CSV."""
    if out.exists() and not force:
        logger.info(f"cached: {out.name}")
        return out
    url = f"{SOCRATA_BASE}/{dataset_id}.csv"
    resp = requests.get(
        url, params={"$where": where, "$limit": 5_000_000}, headers=HEADERS, timeout=300
    )
    resp.raise_for_status()
    out.write_bytes(resp.content)
    logger.success(f"downloaded {out.name} ({len(resp.content) / 1e6:.1f} MB)")
    return out


def fred(series: str, out_dir: Path, force: bool = False) -> Path:
    out = out_dir / f"{series}.csv"
    if out.exists() and not force:
        return out
    for attempt in range(5):
        try:
            resp = requests.get(FRED_CSV.format(series=series), headers=FRED_HEADERS, timeout=30)
            resp.raise_for_status()
            out.write_bytes(resp.content)
            return out
        except requests.RequestException as err:
            if attempt == 4:
                raise
            logger.warning(f"FRED {series}: {err}; retrying")
            time.sleep(3 * (attempt + 1))
    return out


def cms_nhe(out_dir: Path, force: bool = False) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, url in CMS_NHE_URLS.items():
        marker = out_dir / f"{name}.zip"
        if marker.exists() and not force:
            logger.info(f"cached: CMS NHE {name}")
            continue
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=300)
        resp.raise_for_status()
        marker.write_bytes(resp.content)
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            for member in zf.namelist():
                if member.lower().endswith(".csv"):
                    (out_dir / Path(member).name).write_bytes(zf.read(member))
        logger.success(f"downloaded CMS NHE {name}")


def wonder_queries() -> list[tuple[str, list[str], list[int], str]]:
    """(database, group-by variables, years, cache name) for every WONDER query."""
    race = wonder.RACE_VAR
    yr, hisp, cause = wonder.YEAR, wonder.HISPANIC, wonder.CAUSE_113
    spans = {"D76": list(range(START_YEAR, 2018)), "D158": list(range(2018, 2025))}
    queries = []
    for db, years in spans.items():
        for name, groups in {
            "total": [yr],
            "total_cause": [yr, cause],
            "hisp": [yr, hisp],
            "hisp_cause": [yr, hisp, cause],
            "race": [yr, hisp, race[db]],
            "race_cause": [yr, hisp, race[db], cause],
        }.items():
            queries.append((db, groups, years, f"{db}_{name}"))
    return queries


@app.command()
def download(force: bool = typer.Option(False, help="Re-download even if cached")):
    """Fetch every raw source into data/raw."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    ds = CDC_DATASETS
    first_season = f"{START_YEAR}-{str(START_YEAR + 1)[2:]}"

    # Vaccination ---------------------------------------------------------------
    socrata(
        ds["flu_coverage"],
        f"vaccine='Seasonal Influenza' AND month='5' AND year_season >= '{first_season}' "
        "AND year_season like '%-%' AND geography_type in ('States/Local Areas', "
        "'HHS Regions/National')",
        RAW_DATA_DIR / "cdc_flu_coverage.csv",
        force,
    )
    socrata(
        ds["brfss"],
        f"topic='Flu Shot' AND response='Yes' AND year >= '{START_YEAR}'",
        RAW_DATA_DIR / "cdc_brfss_flu_shot.csv",
        force,
    )
    socrata(
        ds["kindergarten_vax"],
        f"vaccine in ('MMR', 'Exemption') AND year_season >= '{first_season}'",
        RAW_DATA_DIR / "cdc_kindergarten_vax.csv",
        force,
    )
    socrata(
        ds["child_vax"],
        "vaccine='≥1 Dose MMR'",
        RAW_DATA_DIR / "cdc_nis_child_mmr.csv",
        force,
    )

    # Mortality -----------------------------------------------------------------
    socrata(
        ds["leading_causes_state"],
        f"year >= '{START_YEAR}'",
        RAW_DATA_DIR / "nchs_leading_causes_state.csv",
        force,
    )
    socrata(
        ds["vsrr_mortality"],
        "rate_type='Age-adjusted' AND time_period='12 months ending with quarter'",
        RAW_DATA_DIR / "nchs_vsrr_mortality.csv",
        force,
    )
    wonder_dir = RAW_DATA_DIR / "wonder"
    for db, groups, years, name in tqdm(wonder_queries(), desc="CDC WONDER"):
        cache = wonder_dir / f"{name}.xml"
        if force and cache.exists():
            cache.unlink()
        wonder.fetch(db, groups, years, cache)

    # Socioeconomic (state median household income, % bachelor's+) and prices ----
    fred_dir = RAW_DATA_DIR / "fred"
    fred_dir.mkdir(exist_ok=True)
    series = ["CPIAUCSL", "CPIMEDSL", "MEHOINUSA646N"]
    for abbr in STATES:
        series += [f"MEHOINUS{abbr}A646N", f"GCT1502{abbr}"]
    for s in tqdm(series, desc="FRED"):
        fred(s, fred_dir, force)

    # Health spending ------------------------------------------------------------
    cms_nhe(RAW_DATA_DIR / "cms_nhe", force)
    logger.success("All raw data downloaded.")


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

FLU_RACE = {
    "Hispanic": "Hispanic",
    "White, Non-Hispanic": "Non-Hispanic White",
    "Black, Non-Hispanic": "Non-Hispanic Black",
    "Other or Multiple Races, Non-Hispanic": "Non-Hispanic Multiracial/Other",
}

BRFSS_RACE = {
    "White, non-Hispanic": "Non-Hispanic White",
    "Black, non-Hispanic": "Non-Hispanic Black",
    "Hispanic": "Hispanic",
    "Asian, non-Hispanic": "Non-Hispanic Asian",
    "American Indian or Alaskan Native, non-Hispanic": "Non-Hispanic American Indian/Alaska Native",
    "Native Hawaiian or other Pacific Islander, non-Hispanic": (
        "Non-Hispanic Native Hawaiian/Pacific Islander"
    ),
    "Multiracial, non-Hispanic": "Non-Hispanic Multiracial/Other",
    "Other, non-Hispanic": "Non-Hispanic Multiracial/Other",
}

# Leading (#-ranked) causes in the NCHS 113-cause list -> harmonized names.
WONDER_CAUSES = {
    r"^#?Diseases of heart": "Heart disease",
    r"^#?Malignant neoplasms \(C00-C97\)": "Cancer",
    r"^#?Accidents \(unintentional injuries\)": "Unintentional injuries",
    r"^#?Cerebrovascular diseases": "Stroke",
    r"^#?Chronic lower respiratory diseases": "Chronic lower respiratory disease",
    r"^#?Alzheimer disease": "Alzheimer disease",
    r"^#?Diabetes mellitus": "Diabetes",
    r"^#?Nephritis, nephrotic syndrome and nephrosis": "Kidney disease",
    r"^#?Chronic liver disease and cirrhosis": "Chronic liver disease",
    r"^#?Influenza and pneumonia": "Influenza & pneumonia",
    r"^#?Intentional self-harm \(suicide\)": "Suicide",
    r"^#?COVID-19": "COVID-19",
}

STATE_CAUSES = {  # bi63-dtpu cause_name and VSRR cause_of_death -> harmonized
    "All causes": "All causes",
    "Heart disease": "Heart disease",
    "Cancer": "Cancer",
    "Unintentional injuries": "Unintentional injuries",
    "Stroke": "Stroke",
    "CLRD": "Chronic lower respiratory disease",
    "Chronic lower respiratory diseases": "Chronic lower respiratory disease",
    "Alzheimer's disease": "Alzheimer disease",
    "Alzheimer disease": "Alzheimer disease",
    "Diabetes": "Diabetes",
    "Kidney disease": "Kidney disease",
    "Chronic liver disease and cirrhosis": "Chronic liver disease",
    "Influenza and pneumonia": "Influenza & pneumonia",
    "Suicide": "Suicide",
    "COVID-19": "COVID-19",
}


def _season_start(season: pd.Series) -> pd.Series:
    """'2023-24' -> 2023 (the calendar year the school/flu season starts)."""
    return season.str.slice(0, 4).astype(int)


def _ci(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    parts = series.fillna("").str.extract(r"([\d.]+)\s*(?:to|-)\s*([\d.]+)")
    return pd.to_numeric(parts[0], errors="coerce"), pd.to_numeric(parts[1], errors="coerce")


def clean_flu_by_race(raw: Path) -> pd.DataFrame:
    df = pd.read_csv(raw, dtype=str)
    df = df[df["geography"].isin(list(STATES.values()) + ["United States"])]
    df = df[df["dimension_type"] == "Race and Ethnicity"].copy()
    df["race_ethnicity"] = df["dimension"].map(FLU_RACE)
    df["season_start"] = _season_start(df["year_season"])
    df["coverage"] = pd.to_numeric(df["coverage_estimate"], errors="coerce")
    df["lci"], df["uci"] = _ci(df["_95_ci"])
    df["sample_size"] = pd.to_numeric(df["population_sample_size"], errors="coerce")
    return df.rename(columns={"geography": "location", "year_season": "season"})[
        [
            "season",
            "season_start",
            "location",
            "race_ethnicity",
            "coverage",
            "lci",
            "uci",
            "sample_size",
        ]
    ].dropna(subset=["race_ethnicity", "coverage"])


def clean_flu_by_age(raw: Path) -> pd.DataFrame:
    """Overall seasonal flu coverage (all ages, adults, 65+) by state."""
    df = pd.read_csv(raw, dtype=str)
    df = df[df["geography"].isin(list(STATES.values()) + ["United States"])]
    df = df[df["dimension_type"] == "Age"].copy()
    patterns = {
        r"^(≥|>=|Greater than )\s*6 Months": "6 months+",
        r"^(≥|>=|Greater than )\s*18 Years": "18 years+",
        r"^(≥|>=|Greater( than)? )\s*65": "65 years+",
        r"^6 Months - 17 Years$": "6 months-17 years",
    }
    df["age_group"] = None
    for pat, label in patterns.items():
        df.loc[df["dimension"].str.contains(pat, regex=True), "age_group"] = label
    df = df.dropna(subset=["age_group"])
    df["season_start"] = _season_start(df["year_season"])
    df["coverage"] = pd.to_numeric(df["coverage_estimate"], errors="coerce")
    df["lci"], df["uci"] = _ci(df["_95_ci"])
    return df.rename(columns={"geography": "location", "year_season": "season"})[
        ["season", "season_start", "location", "age_group", "coverage", "lci", "uci"]
    ].dropna(subset=["coverage"])


def clean_brfss_flu(raw: Path) -> pd.DataFrame:
    df = pd.read_csv(raw, dtype=str)
    df = df[
        df["break_out_category"].isin(
            ["Overall", "Household Income", "Education Attained", "Race/Ethnicity"]
        )
    ].copy()
    df["value"] = pd.to_numeric(df["data_value"], errors="coerce")
    df["lci"] = pd.to_numeric(df["confidence_limit_low"], errors="coerce")
    df["uci"] = pd.to_numeric(df["confidence_limit_high"], errors="coerce")
    df["sample_size"] = pd.to_numeric(df["sample_size"], errors="coerce")
    df["year"] = df["year"].astype(int)
    df["display_order"] = pd.to_numeric(df["display_order"], errors="coerce")
    is_race = df["break_out_category"] == "Race/Ethnicity"
    df.loc[is_race, "break_out"] = (
        df.loc[is_race, "break_out"].map(BRFSS_RACE).fillna(df.loc[is_race, "break_out"])
    )
    df["location"] = df["locationdesc"].replace(
        {
            "All States and DC (median) **": "US (median of states)",
            "All States and DC (median)": "US (median of states)",
        }
    )
    df = df.rename(columns={"break_out_category": "category", "break_out": "group"})
    # Multiracial + Other collapse onto one label; keep the larger sample.
    df = df.sort_values("sample_size", ascending=False).drop_duplicates(
        ["year", "location", "category", "group"]
    )
    return df[
        [
            "year",
            "location",
            "locationabbr",
            "category",
            "group",
            "value",
            "lci",
            "uci",
            "sample_size",
            "display_order",
        ]
    ].dropna(subset=["value"])


def clean_kindergarten(raw: Path) -> pd.DataFrame:
    df = pd.read_csv(raw, dtype=str)
    df = df[df["geography"].isin(list(STATES.values()) + ["U.S. Median"])].copy()
    df["location"] = df["geography"].replace({"U.S. Median": "United States (median)"})
    df["season_start"] = _season_start(df["year_season"])
    df["value"] = pd.to_numeric(df["coverage_estimate"].str.rstrip("%"), errors="coerce")
    df["metric"] = df["vaccine"].map({"MMR": "mmr_coverage", "Exemption": "any_exemption"})
    df = df[df["dose"].fillna("").isin(["", "Any Exemption"])]
    out = df.pivot_table(
        index=["year_season", "season_start", "location"],
        columns="metric",
        values="value",
        aggfunc="first",
    ).reset_index()
    return out.rename(columns={"year_season": "season"}).rename_axis(columns=None)


def clean_nis_child_mmr(raw: Path) -> pd.DataFrame:
    df = pd.read_csv(raw, dtype=str)
    df = df[df["dimension_type"].isin(["Race and Ethnicity", "Poverty", "Overall", "Age"])].copy()
    # Age rows are by birth year (e.g. "2019"); keep ≥1 dose by 24 months for 2015+ cohorts.
    is_age = df["dimension_type"] == "Age"
    df = df[
        ~is_age
        | (
            (df["dimension"] == "24 Months")
            & (pd.to_numeric(df["year_season"], errors="coerce") >= START_YEAR)
        )
    ]
    df["coverage"] = pd.to_numeric(df["coverage_estimate"], errors="coerce")
    df["lci"], df["uci"] = _ci(df["_95_ci"])
    df["group"] = df["dimension"].replace(FLU_RACE)
    return df.rename(
        columns={"geography": "location", "year_season": "period", "dimension_type": "category"}
    )[["period", "location", "category", "group", "coverage", "lci", "uci"]].dropna(
        subset=["coverage"]
    )


def _race_group(hispanic: str, race: str | None) -> str | None:
    if race is None:
        return {"Hispanic or Latino": "Hispanic"}.get(hispanic)
    if hispanic != "Not Hispanic or Latino":
        return None
    return {
        "White": "Non-Hispanic White",
        "Black or African American": "Non-Hispanic Black",
        "Asian": "Non-Hispanic Asian",
        "Asian or Pacific Islander": "Non-Hispanic Asian",  # bridged race, 2015-2017
        "American Indian or Alaska Native": "Non-Hispanic American Indian/Alaska Native",
        "Native Hawaiian or Other Pacific Islander": (
            "Non-Hispanic Native Hawaiian/Pacific Islander"
        ),
        "More than one race": "Non-Hispanic Multiracial/Other",
    }.get(race)


def _harmonize_cause(label: str) -> str | None:
    for pat, name in WONDER_CAUSES.items():
        if re.search(pat, label):
            return name
    return None


def clean_wonder(wonder_dir: Path) -> pd.DataFrame:
    frames = []
    for db, groups, _years, name in wonder_queries():
        cols = ["year"] + [
            {wonder.HISPANIC: "hispanic", wonder.CAUSE_113: "cause"}.get(g, "race")
            for g in groups[1:]
        ]
        df = wonder.parse((wonder_dir / f"{name}.xml").read_text(), cols)
        df["cause"] = df["cause"].map(_harmonize_cause) if "cause" in df else "All causes"
        if "hispanic" in df:
            race = df["race"] if "race" in df else pd.Series([None] * len(df), index=df.index)
            df["race_ethnicity"] = [_race_group(h, r) for h, r in zip(df["hispanic"], race)]
        else:
            df["race_ethnicity"] = "All races/ethnicities"
        df["source"] = {"D76": "WONDER D76 (bridged race)", "D158": "WONDER D158 (single race)"}[
            db
        ]
        frames.append(df.dropna(subset=["cause", "race_ethnicity"]))
    out = pd.concat(frames, ignore_index=True)
    return out[
        [
            "year",
            "race_ethnicity",
            "cause",
            "deaths",
            "population",
            "crude_rate",
            "aadr",
            "aadr_lci",
            "aadr_uci",
            "aadr_flag",
            "source",
        ]
    ]


def clean_state_mortality(leading: Path, vsrr: Path) -> pd.DataFrame:
    a = pd.read_csv(leading, dtype=str)
    a = (
        a[a["state"].isin(STATES.values())]
        .assign(
            year=lambda d: d["year"].astype(int),
            cause=lambda d: d["cause_name"].map(STATE_CAUSES),
            aadr=lambda d: pd.to_numeric(d["aadr"], errors="coerce"),
            period=lambda d: d["year"].astype(str),
            source="NCHS Leading Causes of Death (final)",
        )
        .rename(columns={"state": "location"})
    )

    v = pd.read_csv(vsrr, dtype=str)
    state_cols = {
        c: c.removeprefix("rate_").replace("_", " ").title()
        for c in v.columns
        if c.startswith("rate_")
        and not c.startswith(("rate_age", "rate_sex", "rate_65"))
        and c != "rate_overall"
    }
    state_cols = {
        c: ("District of Columbia" if n == "District Of Columbia" else n)
        for c, n in state_cols.items()
    }
    v = v.rename(columns={"rate_overall": "United States", **state_cols})
    v = v.melt(
        id_vars=["year_and_quarter", "cause_of_death"],
        value_vars=list(state_cols.values()) + ["United States"],
        var_name="location",
        value_name="aadr",
    )
    v["year"] = v["year_and_quarter"].str.slice(0, 4).astype(int)
    v["quarter"] = v["year_and_quarter"].str.slice(-1).astype(int)
    # Annual figure = 12 months ending Q4; for the latest year keep the latest rolling window.
    latest = v.groupby("year")["quarter"].transform("max")
    v = v[(v["quarter"] == 4) | ((v["year"] == v["year"].max()) & (v["quarter"] == latest))]
    v = v.assign(
        cause=v["cause_of_death"].map(STATE_CAUSES),
        aadr=pd.to_numeric(v["aadr"], errors="coerce"),
        period=v.apply(
            lambda r: (
                str(r["year"]) if r["quarter"] == 4 else f"12 mo. ending {r['year_and_quarter']}"
            ),
            axis=1,
        ),
        source="NCHS VSRR provisional",
    )
    cols = ["year", "period", "location", "cause", "aadr", "source"]
    out = pd.concat([a[cols], v[cols]], ignore_index=True).dropna(subset=["cause", "aadr"])
    out["state"] = out["location"].map(STATE_ABBR)
    return out


def _read_fred(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    df.columns = ["date", "value"]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.assign(year=pd.to_datetime(df["date"]).dt.year).groupby("year")["value"].mean()


def clean_state_ses(fred_dir: Path) -> pd.DataFrame:
    rows = []
    for abbr, name in STATES.items():
        inc = _read_fred(fred_dir / f"MEHOINUS{abbr}A646N.csv").rename("median_household_income")
        edu = _read_fred(fred_dir / f"GCT1502{abbr}.csv").rename("pct_bachelors_plus")
        df = pd.concat([inc, edu], axis=1).reset_index(names="year")
        rows.append(df.assign(state=abbr, location=name))
    out = pd.concat(rows, ignore_index=True)
    return out[out["year"] >= START_YEAR - 1]


def clean_prices(fred_dir: Path) -> pd.DataFrame:
    return pd.concat(
        [
            _read_fred(fred_dir / "CPIAUCSL.csv").rename("cpi_u"),
            _read_fred(fred_dir / "CPIMEDSL.csv").rename("cpi_medical"),
            _read_fred(fred_dir / "MEHOINUSA646N.csv").rename("us_median_household_income"),
        ],
        axis=1,
    ).reset_index(names="year")


def _nhe_rows(path: Path) -> list[list[str]]:
    with open(path, encoding="latin-1", newline="") as fh:
        return list(csv.reader(fh))


def _nhe_years(rows: list[list[str]]) -> list[int]:
    header = next(r for r in rows if len(r) > 1 and r[1].strip() == "1960")
    return [int(x) for x in header[1:] if x.strip().isdigit()]


def _to_float(x: str) -> float:
    try:
        return float(x.replace(",", "").strip())
    except ValueError:
        return float("nan")


def clean_nhe(nhe_dir: Path) -> pd.DataFrame:
    """Long table of national health expenditure series (CMS Office of the Actuary)."""
    records = []

    summary = _nhe_rows(next(nhe_dir.glob("*Summary*.csv")))
    years = _nhe_years(summary)
    section = None
    for row in summary:
        label = row[0].strip() if row else ""
        if not label or len(row) < len(years):
            continue
        if label.startswith(
            ("National Health Expenditures (", "National Health Expenditures as a Percent")
        ):
            section = label
        wanted = {
            "National Health Expenditures (Amount in Billions)": "nhe_total_billions",
            "National Health Expenditures (Per Capita Amount)": "nhe_per_capita",
            "National Health Expenditures as a Percent of Gross Domestic Product (Percent)": "nhe_pct_gdp",
            "U.S. Population1 (Millions)": "population_millions",
            "Gross Domestic Product2  (Amount in Billions)": "gdp_billions",
        }
        key = wanted.get(label)
        if key is None and section == "National Health Expenditures (Per Capita Amount)":
            key = {"Personal Health Care": "personal_health_care_per_capita"}.get(label)
        if key:
            for y, v in zip(years, row[1:]):
                records.append(
                    {"year": y, "series": key, "category": "Total", "value": _to_float(v)}
                )

    detail = _nhe_rows(next(p for p in nhe_dir.glob("NHE*.csv") if "Summary" not in p.name))
    years = _nhe_years(detail)
    payers = {
        "Out of pocket": "Out of pocket",
        "Private Health Insurance": "Private health insurance",
        "Medicare": "Medicare",
        "Medicaid (Title XIX)": "Medicaid & CHIP",
        "CHIP (Title XIX and Title XXI)": "Medicaid & CHIP",
        "Department of Defense": "Other programs (incl. DoD & VA)",
        "Department of Veterans Affairs": "Other programs (incl. DoD & VA)",
        "Other Third Party Payers and Programs": "Other programs (incl. DoD & VA)",
        "Public Health Activity": "Public health",
        "Investment": "Investment",
    }
    services = {
        "Total Hospital Expenditures": "Hospital care",
        "Total Physician and Clinical Expenditures": "Physician & clinical",
        "Total Prescription Drug Expenditures": "Prescription drugs",
        "Total Nursing Care Facilities and Continuing Care Retirement Communities": "Nursing & continuing care",
        "Total Home Health Care Expenditures": "Home health",
        "Total Dental Services Expenditures": "Dental",
        "Total Administration and Total Non-Medical Insurance Expenditures": "Administration & net cost of insurance",
    }
    in_total_block = False
    for row in detail:
        if not row or len(row) < len(years):
            continue
        label = row[0].strip()
        if label == "Total National Health Expenditures":
            in_total_block = True
        elif label == "POPULATION":
            in_total_block = False
        if in_total_block and label in payers:
            for y, v in zip(years, row[1:]):
                records.append(
                    {
                        "year": y,
                        "series": "nhe_by_payer_millions",
                        "category": payers[label],
                        "value": _to_float(v),
                    }
                )
        if label in services:
            for y, v in zip(years, row[1:]):
                records.append(
                    {
                        "year": y,
                        "series": "nhe_by_service_millions",
                        "category": services[label],
                        "value": _to_float(v),
                    }
                )

    df = pd.DataFrame.from_records(records)
    df = df.groupby(["year", "series", "category"], as_index=False)["value"].sum(min_count=1)
    return df[df["year"] >= START_YEAR - 1]


@app.command()
def process():
    """Clean raw downloads into tidy CSVs under data/interim."""
    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    r = RAW_DATA_DIR
    outputs = {
        "flu_by_race.csv": lambda: clean_flu_by_race(r / "cdc_flu_coverage.csv"),
        "flu_by_age.csv": lambda: clean_flu_by_age(r / "cdc_flu_coverage.csv"),
        "brfss_flu_65_by_group.csv": lambda: clean_brfss_flu(r / "cdc_brfss_flu_shot.csv"),
        "mmr_kindergarten.csv": lambda: clean_kindergarten(r / "cdc_kindergarten_vax.csv"),
        "mmr_nis_child.csv": lambda: clean_nis_child_mmr(r / "cdc_nis_child_mmr.csv"),
        "mortality_national_by_race.csv": lambda: clean_wonder(r / "wonder"),
        "mortality_state.csv": lambda: clean_state_mortality(
            r / "nchs_leading_causes_state.csv", r / "nchs_vsrr_mortality.csv"
        ),
        "state_ses.csv": lambda: clean_state_ses(r / "fred"),
        "prices.csv": lambda: clean_prices(r / "fred"),
        "health_spending.csv": lambda: clean_nhe(r / "cms_nhe"),
    }
    for name, build in outputs.items():
        df = build()
        df.to_csv(INTERIM_DATA_DIR / name, index=False)
        logger.info(f"{name}: {len(df):,} rows")
    logger.success("Interim tables written.")


if __name__ == "__main__":
    app()
