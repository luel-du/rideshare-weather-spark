"""Draw the weather-effect chart from the pipeline's result table.

Reads data/output/by_weather_company (written by the notebook) and saves
charts/weather_effect.png. Run from the repository root:

    python scripts/make_charts.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SURFACE, INK, SECONDARY, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e6e5e1"
COLORS = {"Uber": "#2a78d6", "Lyft": "#eb6834"}
WEATHER = ["light rain", "heavy rain", "snow"]
PANELS = [("trips_per_day", "Trips per day"), ("fare_per_mile", "Fare per mile"), ("avg_wait_min", "Wait time")]


def change_vs_dry(df, metric):
    """Percent change of a metric against the same company's dry-day value."""
    wide = df.pivot(index="weather_type", columns="company", values=metric)
    return (wide.loc[WEATHER] / wide.loc["dry"] - 1) * 100


def main():
    df = pd.read_parquet(ROOT / "data" / "output" / "by_weather_company")
    days = df.drop_duplicates("weather_type").set_index("weather_type")["days"]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.2), sharey=True, facecolor=SURFACE)
    width = 0.36

    for ax, (metric, title) in zip(axes, PANELS):
        pct = change_vs_dry(df, metric)
        ax.set_facecolor(SURFACE)
        for i, company in enumerate(COLORS):
            xs = [x + (i - 0.5) * (width + 0.03) for x in range(len(WEATHER))]
            bars = ax.bar(xs, pct[company], width, color=COLORS[company], label=company, zorder=3)
            for bar, value in zip(bars, pct[company]):
                offset = 0.5 if value >= 0 else -0.5
                ax.text(bar.get_x() + bar.get_width() / 2, value + offset, f"{value:+.1f}%",
                        ha="center", va="bottom" if value >= 0 else "top", fontsize=7.5, color=SECONDARY)
        ax.axhline(0, color=MUTED, linewidth=1, zorder=2)
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=INK, pad=10)
        ax.set_xticks(range(len(WEATHER)))
        ax.set_xticklabels([f"{w.capitalize()}\n({days[w]} days)" for w in WEATHER], color=SECONDARY)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:+.0f}%" if v else "0%")
        ax.tick_params(axis="y", colors=MUTED, length=0)
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)

    axes[0].set_ylim(-6, 24)
    axes[0].set_ylabel("Change vs dry days", color=SECONDARY)
    axes[0].legend(frameon=False, loc="upper left", labelcolor=SECONDARY, ncols=2, handlelength=1, columnspacing=1.2)
    fig.suptitle("Heavy rain raises demand, price and waiting time; light rain barely registers",
                 x=0.012, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.012, 0.01, f"NYC Uber and Lyft trips, Jan-Apr 2024, compared with {days['dry']} dry days. "
             "Weather: NOAA Central Park station.", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))

    out = ROOT / "charts" / "weather_effect.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print("saved", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
