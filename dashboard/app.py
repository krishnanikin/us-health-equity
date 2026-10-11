"""US Health Equity dashboard.

    streamlit run dashboard/app.py

Reads the tables produced by `make data features train` (data/processed, models/).
"""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from us_health_equity.config import (
    MAJOR_CAUSES,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RACE_GROUPS,
    REAL_DOLLAR_BASE_YEAR,
    STATES,
)
from us_health_equity.modeling.train import OUTCOMES

st.set_page_config(page_title="US Health Equity", page_icon="📊", layout="wide")

# ---------------------------------------------------------------------------
# Visual system: fixed categorical slots (validated palette), one sequential hue.
# ---------------------------------------------------------------------------
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
RACE_COLORS = {g: SLOTS[i] for i, g in enumerate(RACE_GROUPS)}
RACE_COLORS["All races/ethnicities"] = "#52514e"
BLUE_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]  # ordinal, light -> dark
MUTED, GRID = "#898781", "rgba(137,135,129,0.25)"
TARGET_COLOR = "#d03b3b"


def style(fig: go.Figure, height: int = 380, y_title: str | None = None) -> go.Figure:
    fig.update_layout(
        height=height,
        margin={"l": 8, "r": 8, "t": 40, "b": 24},
        font={"family": "system-ui, -apple-system, Segoe UI, sans-serif", "size": 13},
        legend={"orientation": "h", "yanchor": "top", "y": -0.12, "x": 0, "title": None},
        hoverlabel={"font_size": 12},
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=False, linecolor=MUTED, title=None)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, title=y_title)
    fig.update_traces(
        selector={"type": "scatter", "mode": "lines+markers"},
        line={"width": 2},
        marker={"size": 8},
    )
    return fig


def cell_text(values: np.ndarray) -> np.ndarray:
    """Heatmap labels: two decimals, blank where there is no estimate."""
    v = values.astype(float)
    return np.where(np.isnan(v), "", np.char.mod("%.2f", np.nan_to_num(v)))


def show(fig: go.Figure) -> None:
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@st.cache_data
def load() -> dict:
    p = PROCESSED_DATA_DIR
    names = [
        "vax_flu_by_race",
        "vax_flu_by_age",
        "vax_flu_65_by_ses",
        "vax_mmr_kindergarten",
        "vax_mmr_child_by_ses",
        "mortality_by_race",
        "mortality_by_state",
        "state_panel",
        "health_spending_totals",
        "health_spending_by_payer",
        "health_spending_by_service",
    ]
    data = {n: pd.read_csv(p / f"{n}.csv") for n in names}
    data["assoc"] = pd.read_csv(MODELS_DIR / "ses_associations.csv")
    data["trend"] = json.loads((MODELS_DIR / "spending_trend.json").read_text())
    return data


missing = [
    f
    for f in ["state_panel.csv", "health_spending_totals.csv"]
    if not (PROCESSED_DATA_DIR / f).exists()
]
if missing or not (MODELS_DIR / "spending_trend.json").exists():
    st.error("Processed data not found. Run `make pipeline` from the project root first.")
    st.stop()

D = load()
kg = D["vax_mmr_kindergarten"]
kg_us = kg[kg["location"] == "United States (median)"].sort_values("season_start")
flu_age = D["vax_flu_by_age"]
flu_us = flu_age[(flu_age["location"] == "United States") & (flu_age["age_group"] == "6 months+")]
flu_us = flu_us.sort_values("season_start")
mort = D["mortality_by_race"]
spend = D["health_spending_totals"].sort_values("year")
trend = D["trend"]


# ---------------------------------------------------------------------------
# Header & headline numbers
# ---------------------------------------------------------------------------
st.title("US Health Equity, 2015–present")
st.caption(
    "Vaccination (MMR, flu), major-disease mortality by race/ethnicity, household income "
    "and college attainment, and national healthcare spending. Sources: CDC (SchoolVaxView, "
    "ChildVaxView, FluVaxView, BRFSS, NCHS/WONDER), CMS National Health Expenditure Accounts, "
    "FRED (Census income & education, BLS CPI)."
)

all_cause = mort[(mort["cause"] == "All causes")]
ac_all = all_cause[all_cause["race_ethnicity"] == "All races/ethnicities"].set_index("year")[
    "aadr"
]
bw = all_cause.pivot_table(index="year", columns="race_ethnicity", values="aadr")
bw_ratio = bw["Non-Hispanic Black"] / bw["Non-Hispanic White"]

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric(
    f"Kindergarten MMR, {kg_us.iloc[-1]['season']}",
    f"{kg_us.iloc[-1]['mmr_coverage']:.1f}%",
    f"{kg_us.iloc[-1]['mmr_coverage'] - kg_us.iloc[0]['mmr_coverage']:+.1f} pts vs 2015-16",
)
c2.metric(
    f"Vaccine exemptions, {kg_us.iloc[-1]['season']}",
    f"{kg_us.iloc[-1]['any_exemption']:.1f}%",
    f"{kg_us.iloc[-1]['any_exemption'] - kg_us.iloc[0]['any_exemption']:+.1f} pts vs 2015-16",
    delta_color="inverse",
)
c3.metric(
    f"Flu vaccination 6 mo+, {flu_us.iloc[-1]['season']}",
    f"{flu_us.iloc[-1]['coverage']:.1f}%",
    f"{flu_us.iloc[-1]['coverage'] - flu_us.iloc[0]['coverage']:+.1f} pts vs 2015-16",
)
c4.metric(
    f"All-cause mortality, {ac_all.index.max()}",
    f"{ac_all.iloc[-1]:.0f} /100k",
    f"{(ac_all.iloc[-1] / ac_all.iloc[0] - 1) * 100:+.1f}% vs 2015 (age-adjusted)",
    delta_color="inverse",
)
c5.metric(
    f"Health spending per person, {int(spend.iloc[-1]['year'])}",
    f"${spend.iloc[-1]['nhe_per_capita']:,.0f}",
    f"{trend['real_per_capita_change_pct']:+.1f}% real vs 2015",
    delta_color="off",
)

tabs = st.tabs(
    ["Vaccination", "Mortality", "Income & education", "Healthcare spending", "Data & methods"]
)

# ---------------------------------------------------------------------------
# Vaccination
# ---------------------------------------------------------------------------
with tabs[0]:
    st.subheader("MMR")
    a, b = st.columns(2)
    with a:
        fig = px.line(
            kg_us,
            x="season",
            y="mmr_coverage",
            markers=True,
            title="Kindergarten MMR coverage (US median of states)",
            color_discrete_sequence=[SLOTS[0]],
        )
        fig.add_hline(
            y=95,
            line_dash="dot",
            line_color=TARGET_COLOR,
            annotation_text="95% Healthy People target",
            annotation_position="bottom left",
        )
        show(style(fig, y_title="% vaccinated"))
    with b:
        fig = px.line(
            kg_us,
            x="season",
            y="any_exemption",
            markers=True,
            title="Kindergartners with any vaccine exemption (US median)",
            color_discrete_sequence=[SLOTS[1]],
        )
        show(style(fig, y_title="% exempt"))

    seasons = kg_us["season"].tolist()
    season = st.select_slider("School year (map)", options=seasons, value=seasons[-1])
    m = kg[(kg["season"] == season) & kg["location"].isin(STATES.values())].copy()
    m["state"] = m["location"].map({v: k for k, v in STATES.items()})
    a, b = st.columns([3, 2])
    with a:
        fig = px.choropleth(
            m,
            locations="state",
            locationmode="USA-states",
            scope="usa",
            color="mmr_coverage",
            hover_name="location",
            hover_data={"state": False, "mmr_coverage": ":.1f", "any_exemption": ":.1f"},
            color_continuous_scale=["#cde2fb", "#6da7ec", "#256abf", "#0d366b"],
            labels={"mmr_coverage": "MMR %", "any_exemption": "Exempt %"},
            title=f"Kindergarten MMR coverage by state, {season}",
        )
        fig.update_geos(bgcolor="rgba(0,0,0,0)", showlakes=False)
        show(style(fig, height=420))
    with b:
        below = m[m["mmr_coverage"] < 95].sort_values("mmr_coverage")
        st.markdown(
            f"**{len(below)} of {m['mmr_coverage'].notna().sum()} reporting states** "
            f"were below the 95% target in {season}."
        )
        st.dataframe(
            below[["location", "mmr_coverage", "any_exemption"]].rename(
                columns={"location": "State", "mmr_coverage": "MMR %", "any_exemption": "Exempt %"}
            ),
            hide_index=True,
            height=330,
        )

    st.markdown(
        "**MMR (≥1 dose by 35 months) by household poverty level and race/ethnicity** — "
        "CDC National Immunization Survey–Child, pooled birth cohorts"
    )
    nis = D["vax_mmr_child_by_ses"]
    nis = nis[
        (nis["location"] == "United States")
        & nis["category"].isin(["Poverty", "Race and Ethnicity"])
    ]
    a, b = st.columns(2)
    for col, cat, title in [
        (a, "Poverty", "By poverty level (% of federal poverty level)"),
        (b, "Race and Ethnicity", "By race/ethnicity"),
    ]:
        d = nis[nis["category"] == cat].sort_values(["period", "coverage"])
        fig = px.bar(
            d,
            x="coverage",
            y="group",
            color="period",
            barmode="group",
            orientation="h",
            error_x=d["uci"] - d["coverage"],
            error_x_minus=d["coverage"] - d["lci"],
            title=title,
            color_discrete_sequence=SLOTS[:3],
            labels={"coverage": "% vaccinated", "group": "", "period": "Birth cohorts"},
        )
        fig.update_xaxes(range=[80, 100])
        with col:
            show(style(fig, height=360))

    st.subheader("Influenza")
    a, b = st.columns(2)
    with a:
        fr = D["vax_flu_by_race"]
        fr = fr[fr["location"] == "United States"].sort_values("season_start")
        fig = px.line(
            fr,
            x="season",
            y="coverage",
            color="race_ethnicity",
            markers=True,
            color_discrete_map=RACE_COLORS,
            title="Flu vaccination, everyone 6 months+, by race/ethnicity",
            labels={"race_ethnicity": ""},
        )
        fig.update_layout(hovermode="x unified")
        show(style(fig, height=420, y_title="% vaccinated"))
    with b:
        fa = flu_age[flu_age["location"] == "United States"].sort_values("season_start")
        fig = px.line(
            fa,
            x="season",
            y="coverage",
            color="age_group",
            markers=True,
            title="Flu vaccination by age group",
            color_discrete_sequence=SLOTS,
            labels={"age_group": ""},
        )
        fig.update_layout(hovermode="x unified")
        show(style(fig, height=420, y_title="% vaccinated"))

    st.markdown(
        "**Flu shot in the past year, adults 65+, by household income, education and "
        "race/ethnicity** — BRFSS"
    )
    ses = D["vax_flu_65_by_ses"]
    a, b = st.columns([1, 3])
    with a:
        cat = st.radio(
            "Break out by",
            ["Household Income", "Education Attained", "Race/Ethnicity"],
            key="flu_cat",
        )
        locs = ["United States (median of states)"] + sorted(STATES.values())
        loc = st.selectbox("Geography", locs, key="flu_loc")
    with b:
        d = ses[(ses["category"] == cat) & (ses["location"] == loc)].sort_values(["order", "year"])
        groups = d.sort_values("order")["group"].unique().tolist()
        cmap = (
            RACE_COLORS
            if cat == "Race/Ethnicity"
            else dict(zip(groups, BLUE_RAMP if len(groups) == 5 else BLUE_RAMP[1:]))
        )
        fig = px.line(
            d,
            x="year",
            y="value",
            color="group",
            markers=True,
            category_orders={"group": groups},
            color_discrete_map=cmap,
            labels={"value": "% with flu shot", "group": ""},
            title=f"Flu shot coverage, adults 65+ — {loc}",
        )
        fig.update_layout(hovermode="x unified")
        show(style(fig, height=420, y_title="% with flu shot"))
    st.caption(
        "Income bands above $50k are pooled because BRFSS split its top band in 2021. "
        "National values are the median across the 50 states and DC; groups reported by "
        "fewer than 10 states are omitted."
    )

# ---------------------------------------------------------------------------
# Mortality
# ---------------------------------------------------------------------------
with tabs[1]:
    a, b = st.columns([1, 3])
    with a:
        cause = st.selectbox("Cause of death", MAJOR_CAUSES, key="cause")
        races = st.multiselect(
            "Groups",
            ["All races/ethnicities"] + RACE_GROUPS,
            default=[
                "All races/ethnicities",
                "Hispanic",
                "Non-Hispanic White",
                "Non-Hispanic Black",
                "Non-Hispanic Asian",
                "Non-Hispanic American Indian/Alaska Native",
            ],
        )
        show_ci = st.checkbox("Show 95% confidence intervals", value=False)
    d = mort[(mort["cause"] == cause) & mort["race_ethnicity"].isin(races)].sort_values("year")
    with b:
        fig = px.line(
            d,
            x="year",
            y="aadr",
            color="race_ethnicity",
            markers=True,
            color_discrete_map=RACE_COLORS,
            category_orders={"race_ethnicity": ["All races/ethnicities"] + RACE_GROUPS},
            error_y=(d["aadr_uci"] - d["aadr"]) if show_ci else None,
            error_y_minus=(d["aadr"] - d["aadr_lci"]) if show_ci else None,
            labels={"aadr": "Deaths per 100k (age-adjusted)", "race_ethnicity": ""},
            title=f"{cause}: age-adjusted death rate by race/ethnicity, US",
        )
        fig.add_vline(x=2017.5, line_dash="dot", line_color=MUTED)
        fig.add_annotation(
            x=2017.5,
            y=1,
            yref="paper",
            text="race coding change",
            showarrow=False,
            font={"color": MUTED, "size": 11},
            xanchor="left",
        )
        fig.update_layout(hovermode="x unified")
        show(style(fig, height=440, y_title="Deaths per 100k (age-adjusted)"))

    a, b = st.columns(2)
    with a:
        r = d[d["race_ethnicity"] != "All races/ethnicities"]
        fig = px.line(
            r,
            x="year",
            y="rate_ratio_vs_nh_white",
            color="race_ethnicity",
            markers=True,
            color_discrete_map=RACE_COLORS,
            category_orders={"race_ethnicity": RACE_GROUPS},
            labels={"rate_ratio_vs_nh_white": "Rate ratio", "race_ethnicity": ""},
            title=f"{cause}: rate ratio vs. non-Hispanic White (1.0 = parity)",
        )
        fig.add_hline(y=1, line_color=MUTED)
        fig.update_layout(hovermode="x unified")
        show(style(fig, height=400, y_title="Rate ratio"))
    with b:
        yr = st.select_slider(
            "Year (heatmap)", options=sorted(mort["year"].unique()), value=int(mort["year"].max())
        )
        h = mort[(mort["year"] == yr) & mort["race_ethnicity"].isin(RACE_GROUPS)]
        h = h.pivot_table(index="cause", columns="race_ethnicity", values="rate_ratio_vs_nh_white")
        h = h.reindex(
            index=[c for c in MAJOR_CAUSES if c in h.index],
            columns=[g for g in RACE_GROUPS if g in h.columns],
        )
        z = np.log2(h.values.astype(float))
        fig = go.Figure(
            go.Heatmap(
                z=z,
                x=[c.replace("Non-Hispanic ", "NH ") for c in h.columns],
                y=h.index,
                text=cell_text(h.values),
                texttemplate="%{text}",
                zmid=0,
                zmin=-2,
                zmax=2,
                colorscale=[[0, "#256abf"], [0.5, "#f0efec"], [1, "#e34948"]],
                colorbar={
                    "title": "Ratio",
                    "tickvals": [-2, -1, 0, 1, 2],
                    "ticktext": ["¼×", "½×", "1×", "2×", "4×"],
                },
                hovertemplate="%{y}<br>%{x}: %{text}× NH White<extra></extra>",
            )
        )
        fig.update_layout(title=f"Rate ratio vs. NH White by cause, {yr}")
        fig.update_yaxes(autorange="reversed", gridcolor=None)
        show(style(fig, height=470))
        st.caption("Red = higher mortality than NH White, blue = lower.")

    st.markdown("**State age-adjusted death rates** (latest final + provisional)")
    ms = D["mortality_by_state"]
    periods = sorted(ms["period"].unique(), key=lambda p: p[-7:] if "ending" in p else p)
    a, b = st.columns([1, 3])
    with a:
        per = st.selectbox("Period", periods, index=len(periods) - 1)
        cause_s = st.selectbox(
            "Cause", [c for c in MAJOR_CAUSES if c in ms["cause"].unique()], key="cause_state"
        )
    with b:
        dd = ms[(ms["period"] == per) & (ms["cause"] == cause_s) & ms["state"].notna()]
        fig = px.choropleth(
            dd,
            locations="state",
            locationmode="USA-states",
            scope="usa",
            color="aadr",
            hover_name="location",
            hover_data={"state": False, "aadr": ":.1f"},
            color_continuous_scale=["#cde2fb", "#6da7ec", "#256abf", "#0d366b"],
            labels={"aadr": "per 100k"},
            title=f"{cause_s}, age-adjusted deaths per 100k — {per}",
        )
        fig.update_geos(bgcolor="rgba(0,0,0,0)", showlakes=False)
        show(style(fig, height=420))
    st.caption(
        "State-level final rates by cause are published through 2017 (NCHS) and "
        "provisional rates from 2023 onward (NCHS VSRR); the CDC WONDER API does not "
        "release sub-national data, so 2018–2022 state values are not included."
    )

# ---------------------------------------------------------------------------
# Income & education (state-level associations)
# ---------------------------------------------------------------------------
with tabs[2]:
    st.markdown(
        "Each dot is a state. These are **ecological** associations — they describe states, "
        "not individuals. Individual-level breakdowns by income and education are on the "
        "Vaccination tab (BRFSS flu shots, NIS-Child MMR by poverty)."
    )
    panel = D["state_panel"]
    assoc = D["assoc"]
    a, b, c = st.columns(3)
    outcome = a.selectbox("Outcome", list(OUTCOMES), format_func=OUTCOMES.get)
    pred_label = {
        "median_household_income_real": f"Median household income ({REAL_DOLLAR_BASE_YEAR} $)",
        "pct_bachelors_plus": "% adults 25+ with bachelor's degree or higher",
    }
    xvar = b.radio("Socioeconomic measure", list(pred_label), format_func=pred_label.get)
    years = sorted(panel.dropna(subset=[outcome, xvar])["year"].unique())
    year = c.select_slider("Year", options=years, value=years[-1])

    d = panel[(panel["year"] == year)].dropna(subset=[outcome, xvar])
    slope, intercept = np.polyfit(d[xvar], d[outcome], 1)
    rr = np.corrcoef(d[xvar], d[outcome])[0, 1]
    xs = np.linspace(d[xvar].min(), d[xvar].max(), 50)
    fig = px.scatter(
        d,
        x=xvar,
        y=outcome,
        text="state",
        hover_name="location",
        labels={xvar: pred_label[xvar], outcome: OUTCOMES[outcome]},
        title=f"{OUTCOMES[outcome]} vs. {pred_label[xvar].lower()}, {year} "
        f"(r = {rr:.2f}, n = {len(d)})",
        color_discrete_sequence=[SLOTS[0]],
    )
    fig.update_traces(
        textposition="top center",
        textfont={"size": 10, "color": MUTED},
        marker={"size": 10, "line": {"width": 2, "color": "white"}},
    )
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=intercept + slope * xs,
            mode="lines",
            name="OLS fit",
            line={"color": MUTED, "width": 2, "dash": "dash"},
            hoverinfo="skip",
        )
    )
    fig.update_layout(showlegend=False)
    show(style(fig, height=520, y_title=OUTCOMES[outcome]))

    st.markdown(
        "**How strong is the link each year?** Correlation (r) between each outcome and "
        "each socioeconomic measure across states."
    )
    pk = {"median_household_income_real": "income_10k", "pct_bachelors_plus": "college_5pt"}[xvar]
    s = assoc[(assoc["model"] == "bivariate") & (assoc["predictor"] == pk)]
    s = s.pivot_table(index="outcome_label", columns="year", values="r")
    fig = go.Figure(
        go.Heatmap(
            z=s.values,
            x=s.columns.astype(str),
            y=s.index,
            zmid=0,
            zmin=-1,
            zmax=1,
            text=cell_text(s.values),
            texttemplate="%{text}",
            colorscale=[[0, "#256abf"], [0.5, "#f0efec"], [1, "#e34948"]],
            hovertemplate="%{y}<br>%{x}: r = %{z:.2f}<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed")
    show(style(fig, height=430))
    with st.expander("Regression coefficients (joint model: income + education)"):
        j = assoc[(assoc["model"] == "joint") & (assoc["outcome"] == outcome)]
        st.dataframe(
            j[
                [
                    "year",
                    "predictor_label",
                    "slope",
                    "ci_low",
                    "ci_high",
                    "p_value",
                    "r_squared",
                    "n_states",
                ]
            ].round(3),
            hide_index=True,
        )

# ---------------------------------------------------------------------------
# Healthcare spending
# ---------------------------------------------------------------------------
with tabs[3]:
    t = trend
    st.markdown(
        f"### Yes — US healthcare spending has increased, in nominal and real terms\n"
        f"National health expenditure rose **{t['nominal_total_change_pct']:.0f}%** from "
        f"{t['period'][:4]} to {t['period'][-4:]} (≈{t['nominal_total_cagr_pct']:.1f}% a year). "
        f"After inflation and population growth, spending **per person rose "
        f"{t['real_per_capita_change_pct']:.0f}%**, a trend of "
        f"**{t['real_per_capita_annual_growth_pct']:.1f}% a year** "
        f"(95% CI {t['real_per_capita_annual_growth_ci_pct'][0]:.1f}–"
        f"{t['real_per_capita_annual_growth_ci_pct'][1]:.1f}%, p = {t['trend_p_value']:.1g}). "
        f"Its share of GDP moved from {t['pct_gdp_start']:.1f}% to {t['pct_gdp_end']:.1f}%."
    )
    a, b = st.columns(2)
    with a:
        pc = spend.melt(
            id_vars="year",
            value_vars=["nhe_per_capita", "nhe_per_capita_real"],
            var_name="measure",
            value_name="dollars",
        )
        pc["measure"] = pc["measure"].map(
            {
                "nhe_per_capita": "Nominal dollars",
                "nhe_per_capita_real": f"Real {REAL_DOLLAR_BASE_YEAR} dollars (CPI-U)",
            }
        )
        fig = px.line(
            pc,
            x="year",
            y="dollars",
            color="measure",
            markers=True,
            color_discrete_sequence=SLOTS[:2],
            title="Health spending per person",
            labels={"measure": ""},
        )
        fig.update_layout(hovermode="x unified")
        fig.update_yaxes(tickprefix="$", tickformat=",.0f")
        show(style(fig, y_title="Dollars per person"))
    with b:
        fig = px.line(
            spend,
            x="year",
            y="nhe_pct_gdp",
            markers=True,
            color_discrete_sequence=[SLOTS[0]],
            title="National health expenditure as % of GDP",
        )
        fig.update_yaxes(ticksuffix="%")
        show(style(fig, y_title="% of GDP"))

    a, b = st.columns(2)
    with a:
        g = spend.melt(
            id_vars="year",
            value_vars=["nhe_growth_pct", "gdp_growth_pct"],
            var_name="series",
            value_name="pct",
        )
        g["series"] = g["series"].map(
            {"nhe_growth_pct": "Health spending", "gdp_growth_pct": "GDP"}
        )
        fig = px.bar(
            g,
            x="year",
            y="pct",
            color="series",
            barmode="group",
            color_discrete_sequence=SLOTS[:2],
            title="Annual growth: health spending vs. GDP (nominal)",
            labels={"series": ""},
        )
        fig.update_yaxes(ticksuffix="%")
        show(style(fig, y_title="% change from prior year"))
    with b:
        real = st.toggle(f"Inflation-adjusted ({REAL_DOLLAR_BASE_YEAR} $)", value=True)
        pay = D["health_spending_by_payer"]
        col = "billions_real" if real else "billions"
        order = pay[pay["year"] == pay["year"].max()].sort_values(col, ascending=False)
        fig = px.line(
            pay,
            x="year",
            y=col,
            color="category",
            markers=True,
            category_orders={"category": order["category"].tolist()},
            color_discrete_sequence=SLOTS,
            title="Spending by source of funds",
            labels={"category": ""},
        )
        fig.update_layout(hovermode="x unified")
        fig.update_yaxes(tickprefix="$", ticksuffix="B")
        show(style(fig, y_title="Billions of dollars"))

    sv = D["health_spending_by_service"]
    last = sv[sv["year"] == sv["year"].max()].sort_values("pct_change_since_start")
    fig = px.bar(
        last,
        x="pct_change_since_start",
        y="category",
        orientation="h",
        color_discrete_sequence=[SLOTS[0]],
        hover_data={"billions": ":,.0f"},
        title=f"Growth by type of service, 2015 → {int(sv['year'].max())} (nominal)",
        labels={"pct_change_since_start": "% change since 2015", "category": ""},
        text=last["pct_change_since_start"].map(lambda v: f"{v:+.0f}%"),
    )
    fig.update_traces(textposition="outside", marker_line_width=0)
    fig.update_xaxes(ticksuffix="%")
    show(style(fig, height=360))

# ---------------------------------------------------------------------------
# Data & methods
# ---------------------------------------------------------------------------
with tabs[4]:
    st.markdown("""
| Topic | Source | Coverage | Breakdowns |
|---|---|---|---|
| Kindergarten MMR & exemptions | CDC SchoolVaxView (`ijqb-a7ye`) | 2015-16 → latest school year | State |
| Early-childhood MMR | CDC NIS-Child / ChildVaxView (`fhky-rtsk`) | Birth years 2015-2022; pooled 2014-17 → 2016-19 | Race/ethnicity, poverty level |
| Flu vaccination (6 mo+) | CDC FluVaxView (`vh55-3he6`) | 2015-16 → latest season | Race/ethnicity, age, state |
| Flu shot, adults 65+ | CDC BRFSS (`dttw-5yxu`) | 2015 → latest | Household income, education, race, state |
| National mortality | CDC WONDER D76 (2015-17, bridged race) & D158 (2018+, single race) | 2015 → latest final year | Race/ethnicity × 12 leading causes |
| State mortality | NCHS Leading Causes (`bi63-dtpu`) & VSRR provisional (`489q-934x`) | 2015-17, 2023 → latest | State × cause |
| Income & education | FRED: Census median household income (CPS ASEC), ACS % bachelor's+ | 2015 → latest | State |
| Health spending | CMS National Health Expenditure Accounts | 2015 → 2024 | Payer, service type |
| Inflation | BLS CPI-U and medical-care CPI via FRED | 2015 → latest | — |

**Caveats**
- *Race classification changes in 2018.* WONDER data for 2015-2017 use bridged race (Asian includes Pacific Islander); 2018+ use single race. Multiracial and NHPI groups exist only from 2018.
- *Mortality by income and education.* US death records do not carry household income, and CDC WONDER does not release mortality by education, so mortality is linked to income and college attainment through state-level (ecological) analysis.
- *Age adjustment.* Death rates are age-adjusted to the 2000 US standard population so groups with different age structures are comparable.
- *BRFSS national values* are medians across states; income bands ≥$50k are pooled for comparability across the 2021 questionnaire change.
- *Provisional data* (VSRR, latest flu and school-year estimates) can be revised.
""")
    st.markdown("**Download processed tables**")
    cols = st.columns(4)
    for i, name in enumerate(sorted(p.name for p in PROCESSED_DATA_DIR.glob("*.csv"))):
        cols[i % 4].download_button(
            name, (PROCESSED_DATA_DIR / name).read_bytes(), file_name=name, mime="text/csv"
        )
