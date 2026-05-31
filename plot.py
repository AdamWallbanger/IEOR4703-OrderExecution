import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

tau = [5,10,15,30,60]
P = "30%"
for i in range(len(tau)):
    result_path = "Result_" + str(tau[i]) + "min/"
    filenames = os.listdir(result_path)

    summary = []

    for filename in filenames:
        if filename.endswith(".csv"):
            path = result_path + filename
            df = pd.read_csv(path)

            df = df.dropna(subset=[
                "slippage",
                "benchmark_amount",
                "opti_price",
                "filled_price"
            ]).copy()

            df["slippage_bps"] = (
                df["slippage"] / df["benchmark_amount"].abs()
            ) * 10000

            df["hit"] = np.isclose(
                df["opti_price"],
                df["filled_price"],
                atol=1e-8
            )

            hit_rate = df["hit"].mean()
            avg_slippage_bps = df["slippage_bps"].mean()
            weighted_slippage_bps = (
                df["slippage"].sum()
                / df["benchmark_amount"].abs().sum()
                * 10000
            )

            summary.append({
                "filename": filename,
                "hit_rate": hit_rate,
                "avg_slippage_bps": avg_slippage_bps,
                "weighted_slippage_bps": weighted_slippage_bps
            })

    summary_df = pd.DataFrame(summary)

    plt.figure(figsize=(12, 6))

    summary_df["clean_name"] = (
        summary_df["filename"]
        .str.replace("AIAgent", "", regex=False)
        .str.replace(".csv", "", regex=False)
    )

    # Choose the metric you want to plot
    plot_col = "avg_slippage_bps"
    # plot_col = "weighted_slippage_bps"  # use this instead if preferred

    # Nice shades
    positive_color = "#2E7D32"  # nice green
    negative_color = "#C62828"  # nice red

    colors = [
        positive_color if x >= 0 else negative_color
        for x in summary_df[plot_col]
    ]

    plt.figure(figsize=(12, 6))

    bars = plt.bar(
        summary_df["clean_name"],
        summary_df[plot_col],
        color=colors
    )

    plt.axhline(0, color="black", linestyle="--", linewidth=1)

    # Put slippage number on top of each bar
    for bar, value in zip(bars, summary_df[plot_col]):
        x = bar.get_x() + bar.get_width() / 2
        y = bar.get_height()

        plt.text(
            x,
            y,
            f"{value:.2f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            color=positive_color if value >= 0 else negative_color,
            fontsize=10,
            fontweight="bold"
        )

    # Hit rate as legend
    legend_handles = [
        Patch(
            facecolor=colors[i],
            label=f'{summary_df["clean_name"].iloc[i]} | Hit Rate: {summary_df["hit_rate"].iloc[i]:.2%}'
        )
        for i in range(len(summary_df))
    ]

    plt.legend(
        handles=legend_handles,
        title="Hit Rate",
        loc="upper right",
        frameon=True,
        framealpha=0.9
    )

    plt.xlabel("Future Contract")
    plt.ylabel("Average Slippage (bps)")
    plt.title(
        f"Average Slippage bps\n"
        f"tau = {tau[i]}, P = {P}"
    )

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()