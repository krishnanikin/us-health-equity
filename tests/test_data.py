"""Unit tests for parsing/cleaning logic (offline) and processed-table sanity checks."""

import pandas as pd
import pytest

from us_health_equity import dataset, wonder
from us_health_equity.config import PROCESSED_DATA_DIR, START_YEAR

WONDER_XML = """<page><data-table show-all-labels="true">
<r><c l="2015" r="2"/><c l="Hispanic or Latino"/><c v="179,457"/><c v="56,592,793"/>
<c v="317.1" a="c"><l v="(315.6 - 318.6)"/></c><c v="0.7"/>
<c v="525.3" a="c"><l v="(522.8 - 527.9)"/></c><c v="1.3"/></r>
<r><c l="Not Hispanic or Latino"/><c v="2,500,000"/><c v="264,000,000"/>
<c v="947.0" a="c"><l v="(946.0 - 948.0)"/></c><c v="0.6"/>
<c v="Unreliable" a="c"><l v="(1.0 - 2.0)"/></c><c v="0.2"/></r>
</data-table></page>"""


def test_wonder_parse_carries_rowspan_labels():
    df = wonder.parse(WONDER_XML, ["year", "hispanic"])
    assert df["year"].tolist() == [2015, 2015]
    assert df["hispanic"].tolist() == ["Hispanic or Latino", "Not Hispanic or Latino"]
    assert df.loc[0, "deaths"] == 179_457
    assert df.loc[0, "aadr"] == pytest.approx(525.3)
    assert df.loc[0, "aadr_lci"] == pytest.approx(522.8)


def test_wonder_parse_flags_unreliable_rates():
    df = wonder.parse(WONDER_XML, ["year", "hispanic"])
    assert pd.isna(df.loc[1, "aadr"])
    assert df.loc[1, "aadr_flag"] == "Unreliable"


@pytest.mark.parametrize(
    ("hispanic", "race", "expected"),
    [
        ("Hispanic or Latino", None, "Hispanic"),
        ("Not Hispanic or Latino", "White", "Non-Hispanic White"),
        ("Not Hispanic or Latino", "Asian or Pacific Islander", "Non-Hispanic Asian"),
        ("Hispanic or Latino", "White", None),  # Hispanic is reported across all races
        ("Not Stated", None, None),
    ],
)
def test_race_group_harmonization(hispanic, race, expected):
    assert dataset._race_group(hispanic, race) == expected


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("#Diseases of heart (I00-I09,I11,I13,I20-I51)", "Heart disease"),
        ("#Malignant neoplasms (C00-C97)", "Cancer"),
        ("Malignant neoplasm of esophagus (C15)", None),
        ("#Intentional self-harm (suicide) (*U03,X60-X84,Y87.0)", "Suicide"),
    ],
)
def test_cause_harmonization(label, expected):
    assert dataset._harmonize_cause(label) == expected


def test_wonder_queries_cover_study_window():
    years = sorted({y for _, _, ys, _ in dataset.wonder_queries() for y in ys})
    assert years[0] == START_YEAR
    assert years == list(range(years[0], years[-1] + 1))


processed = pytest.mark.skipif(
    not (PROCESSED_DATA_DIR / "state_panel.csv").exists(),
    reason="run `make pipeline` to build processed data",
)


@processed
def test_processed_tables_start_in_study_window():
    for name, col in [
        ("mortality_by_race.csv", "year"),
        ("health_spending_totals.csv", "year"),
        ("vax_mmr_kindergarten.csv", "season_start"),
        ("vax_flu_by_race.csv", "season_start"),
    ]:
        df = pd.read_csv(PROCESSED_DATA_DIR / name)
        assert df[col].min() == START_YEAR, name


@processed
def test_rates_are_plausible():
    mort = pd.read_csv(PROCESSED_DATA_DIR / "mortality_by_race.csv")
    all_cause = mort[
        (mort["cause"] == "All causes") & (mort["race_ethnicity"] == "All races/ethnicities")
    ]
    assert all_cause["aadr"].between(600, 1000).all()
    kg = pd.read_csv(PROCESSED_DATA_DIR / "vax_mmr_kindergarten.csv")
    assert kg["mmr_coverage"].dropna().between(50, 100).all()


@processed
def test_state_panel_is_unique_by_state_year():
    panel = pd.read_csv(PROCESSED_DATA_DIR / "state_panel.csv")
    assert not panel.duplicated(["state", "year"]).any()
    assert panel["state"].nunique() == 51
