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


def plot_memory_usage(
    base_df: pd.DataFrame, 
    lora_df: pd.DataFrame, 
    rebase_df: pd.DataFrame, 
    save_path: str = "analysis/memory_usage.png"
):
    """Plot GPU memory usage over training epochs."""
    
    # Check if memory columns exist
    required_cols = ['memory_allocated_mb', 'memory_peak_mb', 'memory_reserved_mb']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot allocated memory
    ax = axes[0]
    if 'memory_allocated_mb' in base_df.columns:
        ax.plot(base_df['memory_allocated_mb'], label='Full Fine-tuning', color='blue', linewidth=2)
    if 'memory_allocated_mb' in lora_df.columns:
        ax.plot(lora_df['memory_allocated_mb'], label='LoRA Fine-tuning', color='green', linewidth=2)
    if 'memory_allocated_mb' in rebase_df.columns:
        ax.plot(rebase_df['memory_allocated_mb'], label='Re-base Fine-tuning', color='orange', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Memory (MB)')
    ax.set_title('Allocated GPU Memory')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot peak memory
    ax = axes[1]
    if 'memory_peak_mb' in base_df.columns:
        ax.plot(base_df['memory_peak_mb'], label='Full Fine-tuning', color='blue', linewidth=2)
    if 'memory_peak_mb' in lora_df.columns:
        ax.plot(lora_df['memory_peak_mb'], label='LoRA Fine-tuning', color='green', linewidth=2)
    if 'memory_peak_mb' in rebase_df.columns:
        ax.plot(rebase_df['memory_peak_mb'], label='Re-base Fine-tuning', color='orange', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Memory (MB)')
    ax.set_title('Peak Allocated GPU Memory')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot reserved memory
    ax = axes[2]
    if 'memory_reserved_mb' in base_df.columns:
        ax.plot(base_df['memory_reserved_mb'], label='Full Fine-tuning', color='blue', linewidth=2)
    if 'memory_reserved_mb' in lora_df.columns:
        ax.plot(lora_df['memory_reserved_mb'], label='LoRA Fine-tuning', color='green', linewidth=2)
    if 'memory_reserved_mb' in rebase_df.columns:
        ax.plot(rebase_df['memory_reserved_mb'], label='Re-base Fine-tuning', color='orange', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Memory (MB)')
    ax.set_title('Reserved GPU Memory')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Memory plot saved to {save_path}")


def plot():
    # Read all CSVs
    base_df = pd.read_csv("data/base_analytics2.csv")
    eval_base_df = pd.read_csv("data/eval_base_analytics2.csv")
    eval_argmax_base_df = pd.read_csv("data/eval_argmax_base_analytics2.csv")
    rebase_df = pd.read_csv("data/rebase_analytics2.csv")
    eval_rebase_df = pd.read_csv("data/eval_rebase_analytics2.csv")
    eval_argmax_rebase_df = pd.read_csv("data/eval_argmax_rebase_analytics2.csv")
    lora_df = pd.read_csv("data/lora_analytics2.csv")
    eval_lora_df = pd.read_csv("data/eval_lora_analytics2.csv")
    eval_argmax_lora_df = pd.read_csv("data/eval_argmax_lora_analytics2.csv")

    # Create main dashboard
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
    print("Dashboard saved to analysis/lunar_lander_dashboard.png")
    
    # Generate memory usage plot
    plot_memory_usage(base_df, lora_df, rebase_df)


if __name__ == "__main__":
    plot()