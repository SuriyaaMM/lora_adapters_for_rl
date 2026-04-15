import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as plt_axes
import matplotlib.figure as plt_figure
from typing import List

def draw_metrics(
    eval_lora_df: pd.DataFrame,
    eval_argmax_lora_df: pd.DataFrame,
    lora_df: pd.DataFrame,
    eval_rebase_df: pd.DataFrame,
    eval_argmax_rebase_df: pd.DataFrame,
    rebase_df: pd.DataFrame,
    metric_name: str,
    axes: List[List[plt_axes.Axes]],
    row_index: int = 0,
    do_median: bool = True,
):
    mean_metric_name = metric_name + "_mean"
    std_metric_name = metric_name + "_std"
    median_metric_name = metric_name + "_median"

    eval_lora_total_reward_mean = eval_lora_df[mean_metric_name]
    eval_lora_total_reward_std  = eval_lora_df[std_metric_name]
    eval_rebase_total_reward_mean = eval_rebase_df[mean_metric_name]
    eval_rebase_total_reward_std  = eval_rebase_df[std_metric_name]
    
    eval_argmax_lora_total_reward_mean = eval_argmax_lora_df[mean_metric_name]
    eval_argmax_lora_total_reward_std  = eval_argmax_lora_df[std_metric_name]
    eval_argmax_rebase_total_reward_mean = eval_argmax_rebase_df[mean_metric_name]
    eval_argmax_rebase_total_reward_std  = eval_argmax_rebase_df[std_metric_name]

    train_lora_total_reward_mean = lora_df[mean_metric_name]
    train_lora_total_reward_std  = lora_df[std_metric_name]
    train_rebase_total_reward_mean = rebase_df[mean_metric_name]
    train_rebase_total_reward_std  = rebase_df[std_metric_name]


    axes[row_index][0].plot(eval_lora_total_reward_mean, label="lora")
    axes[row_index][0].fill_between(
        range(len(eval_argmax_lora_total_reward_mean)),
        eval_lora_total_reward_mean - eval_lora_total_reward_std,
        eval_lora_total_reward_mean + eval_lora_total_reward_std,
        alpha=0.1, label="lora"
    )

    axes[row_index][0].plot(eval_rebase_total_reward_mean, label="rebase")
    axes[row_index][0].fill_between(
        range(len(eval_rebase_total_reward_mean)),
        eval_rebase_total_reward_mean - eval_rebase_total_reward_std,
        eval_rebase_total_reward_mean + eval_rebase_total_reward_std,
        alpha=0.1, label="rebase"
    )

    axes[row_index][0].set_title(f"random sampling {metric_name}")
    axes[row_index][0].legend()

    axes[row_index][1].plot(eval_argmax_lora_total_reward_mean, label="lora")
    axes[row_index][1].fill_between(
        range(len(eval_argmax_lora_total_reward_mean)),
        eval_argmax_lora_total_reward_mean - eval_argmax_lora_total_reward_std,
        eval_argmax_lora_total_reward_mean + eval_argmax_lora_total_reward_std,
        alpha=0.1, label="lora"
    )
    axes[row_index][1].plot(eval_argmax_rebase_total_reward_mean, label="rebase")
    axes[row_index][1].fill_between(
        range(len(eval_argmax_rebase_total_reward_mean)),
        eval_argmax_rebase_total_reward_mean - eval_argmax_rebase_total_reward_std,
        eval_argmax_rebase_total_reward_mean + eval_argmax_rebase_total_reward_std,
        alpha=0.1, label="rebase"
    )
    axes[row_index][1].set_title(f"argmax action {metric_name}")
    axes[row_index][1].legend()

    axes[row_index][2].plot(train_lora_total_reward_mean, label="lora")
    axes[row_index][2].fill_between(
        range(len(train_lora_total_reward_mean)),
        train_lora_total_reward_mean - train_lora_total_reward_std,
        train_lora_total_reward_mean + train_lora_total_reward_std,
        alpha=0.1, label="lora"
    )
    axes[row_index][2].plot(train_rebase_total_reward_mean, label="rebase")
    axes[row_index][2].fill_between(
        range(len(train_rebase_total_reward_mean)),
        train_rebase_total_reward_mean - train_rebase_total_reward_std,
        train_rebase_total_reward_mean + train_rebase_total_reward_std,
        alpha=0.1, label="rebase"
    )
    axes[row_index][2].set_title(f"training phase {metric_name}")
    axes[row_index][2].legend()
    

def plot():
    base_df = pd.read_csv("data/base_analytics2.csv")
    eval_base_df = pd.read_csv("data/eval_base_analytics2.csv")
    eval_argmax_base_df = pd.read_csv("data/eval_argmax_base_analytics2.csv")
    rebase_df = pd.read_csv("data/rebase_analytics2.csv")
    eval_rebase_df = pd.read_csv("data/eval_rebase_analytics2.csv")
    eval_argmax_rebase_df = pd.read_csv("data/eval_argmax_rebase_analytics2.csv")
    lora_df = pd.read_csv("data/lora_analytics2.csv")
    eval_lora_df = pd.read_csv("data/eval_lora_analytics2.csv")
    eval_argmax_lora_df = pd.read_csv("data/eval_argmax_lora_analytics2.csv")

    print(eval_base_df.columns)

    fig: plt_figure.Figure
    axes: List[List[plt_axes.Axes]]

    fig, axes = plt.subplots(
        nrows=4, ncols=3,
        figsize=(30, 30)
    )

    draw_metrics(
        eval_lora_df,
        eval_argmax_lora_df,
        lora_df,
        eval_rebase_df,
        eval_argmax_rebase_df,
        rebase_df, "total_reward", axes, 0
    )
    draw_metrics(
        eval_lora_df,
        eval_argmax_lora_df,
        lora_df,
        eval_rebase_df,
        eval_argmax_rebase_df,
        rebase_df, "entropies", axes, 1
    )
    draw_metrics(
        eval_lora_df,
        eval_argmax_lora_df,
        lora_df,
        eval_rebase_df,
        eval_argmax_rebase_df,
        rebase_df, "advantages", axes, 2
    )
    draw_metrics(
        eval_lora_df,
        eval_argmax_lora_df,
        lora_df,
        eval_rebase_df,
        eval_argmax_rebase_df,
        rebase_df, "perplexities", axes, 3
    )

    

    plt.savefig("analysis/lunar_lander_dashboard.png", dpi=300)
if __name__ == "__main__":
    plot()