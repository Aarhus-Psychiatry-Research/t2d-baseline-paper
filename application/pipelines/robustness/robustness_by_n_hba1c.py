from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Optional, Union

from psycop_model_training.model_eval.base_artifacts.plots.base_charts import (
    plot_basic_chart,
)
from psycop_model_training.model_eval.base_artifacts.plots.utils import (
    create_performance_by_input,
)
from psycop_model_training.model_eval.dataclasses import EvalDataset
from sklearn.metrics import roc_auc_score
from t2d_baseline_paper.best_runs import ROBUSTNESS_PATH, best_run
from t2d_baseline_paper.data.load_true_data import load_eval_dataset


def plot_performance_by_n_hba1c(
    eval_dataset: EvalDataset,
    save_path: Optional[Path] = None,
    bins: Sequence[float] = (18, 25, 35, 50, 70),
    bin_continuous_input: Optional[bool] = True,
    metric_fn: Callable = roc_auc_score,
    y_limits: Optional[tuple[float, float]] = (0.5, 1.0),
) -> Union[None, Path]:
    """Plot bar plot of performance (default AUC) by age at time of prediction.

    Args:
        eval_dataset: EvalDataset object
        bins (Sequence[float]): Bins to group by. Defaults to (18, 25, 35, 50, 70, 100).
        bin_continuous_input (bool, optional): Whether to bin input. Defaults to True.
        metric_fn (Callable): Callable which returns the metric to calculate
        save_path (Path, optional): Path to save figure. Defaults to None.
        y_limits (tuple[float, float], optional): y-axis limits. Defaults to (0.5, 1.0).

    Returns:
        Union[None, Path]: Path to saved figure or None if not saved.
    """
    col_name = "eval_hba1c_within_9999_days_count_fallback_nan"

    df = create_performance_by_input(
        eval_dataset=eval_dataset,
        input_values=eval_dataset.custom_columns[col_name],
        input_name=col_name,
        metric_fn=metric_fn,
        bins=bins,
        bin_continuous_input=bin_continuous_input,
    )

    sort_order = sorted(df[f"{col_name}_binned"].unique())

    return plot_basic_chart(
        x_values=df[f"{col_name}_binned"],
        y_values=df["metric"],
        x_title="HbA1c measurements before visit",
        y_title="AUC",
        sort_x=sort_order,
        y_limits=y_limits,
        plot_type=["scatter", "line"],
        bar_count_values=df["n_in_bin"],
        bar_count_y_axis_title="Number of visits",
        save_path=save_path,
    )


def roc_auc_by_n_hba1c():
    eval_ds = load_eval_dataset(
        wandb_group=best_run.wandb_group,
        wandb_run=best_run.model,
        custom_columns=["eval_hba1c_within_9999_days_count_fallback_nan"],
    )

    plot_performance_by_n_hba1c(
        eval_dataset=eval_ds,
        bins=[0, *range(1, 11, 1)],
        save_path=ROBUSTNESS_PATH / "auc_by_n_hba1c.png",
    )


if __name__ == "__main__":
    roc_auc_by_n_hba1c()