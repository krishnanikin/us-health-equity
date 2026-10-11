"""Static report figures (reports/figures) for the README and write-ups.

python -m us_health_equity.plots
"""

from loguru import logger
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import typer

from us_health_equity.config import (
    FIGURES_DIR,
    PROCESSED_DATA_DIR,
    RACE_GROUPS,
    REAL_DOLLAR_BASE_YEAR,
)

app = typer.Typer()

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
RACE_COLORS = dict(zip(RACE_GROUPS, SLOTS, strict=False))
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "lines.linewidth": 2,
        "lines.markersize": 5,
        "legend.frameon": False,
    }
)


def _save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / name, dpi=160)
    plt.close(fig)
    logger.info(f"wrote reports/figures/{name}")


def mmr_figure(kg: pd.DataFrame) -> None:
    us = kg[kg["location"] == "United States (median)"].sort_values("season_start")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(us["season"], us["mmr_coverage"], marker="o", color=SLOTS[0])
    axes[0].axhline(95, ls=":", color="#d03b3b")
    axes[0].text(0, 95.05, "95% target", color="#d03b3b", fontsize=9)
    axes[0].set_title("Kindergarten MMR coverage, US median (%)", loc="left")
    axes[1].plot(us["season"], us["any_exemption"], marker="o", color=SLOTS[1])
    axes[1].set_title("Kindergartners with any exemption, US median (%)", loc="left")
    for ax in axes:
        ax.tick_params(axis="x", rotation=45)
    _save(fig, "mmr_kindergarten.png")


def flu_figure(flu: pd.DataFrame) -> None:
    us = flu[flu["location"] == "United States"].sort_values("season_start")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for group, d in us.groupby("race_ethnicity"):
        ax.plot(d["season"], d["coverage"], marker="o", label=group, color=RACE_COLORS[group])
    ax.set_title("Flu vaccination coverage, 6 months+, by race/ethnicity (%)", loc="left")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(loc="lower left", fontsize=9)
    _save(fig, "flu_by_race.png")


def mortality_figure(mort: pd.DataFrame) -> None:
    d = mort[(mort["cause"] == "All causes") & mort["race_ethnicity"].isin(RACE_GROUPS[:5])]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for group in RACE_GROUPS[:5]:
        g = d[d["race_ethnicity"] == group].sort_values("year")
        ax.plot(g["year"], g["aadr"], marker="o", label=group, color=RACE_COLORS[group])
    ax.axvline(2017.5, ls=":", color=MUTED)
    ax.set_title("All-cause mortality, age-adjusted deaths per 100,000", loc="left")
    ax.legend(fontsize=9, loc="upper left")
    _save(fig, "mortality_by_race.png")


def spending_figure(totals: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(
        totals["year"], totals["nhe_per_capita"], marker="o", color=SLOTS[0], label="Nominal $"
    )
    axes[0].plot(
        totals["year"],
        totals["nhe_per_capita_real"],
        marker="o",
        color=SLOTS[1],
        label=f"Real {REAL_DOLLAR_BASE_YEAR} $",
    )
    axes[0].set_title("Health spending per person", loc="left")
    axes[0].legend()
    axes[1].plot(totals["year"], totals["nhe_pct_gdp"], marker="o", color=SLOTS[0])
    axes[1].set_title("National health expenditure, % of GDP", loc="left")
    _save(fig, "health_spending.png")


@app.command()
def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    p = PROCESSED_DATA_DIR
    mmr_figure(pd.read_csv(p / "vax_mmr_kindergarten.csv"))
    flu_figure(pd.read_csv(p / "vax_flu_by_race.csv"))
    mortality_figure(pd.read_csv(p / "mortality_by_race.csv"))
    spending_figure(pd.read_csv(p / "health_spending_totals.csv"))
    logger.success("Figures written.")


if __name__ == "__main__":
    app()
