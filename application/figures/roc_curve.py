from t2d_baseline_paper.data.load_dev_data import synth_eval_dataset
from t2d_baseline_paper.data.load_true_data import load_eval_dataset
from t2d_baseline_paper.figures.roc_curve import eval_ds_to_roc_plot_spec, plot_auc_roc
from t2d_baseline_paper.globals import PROJECT_ROOT, BestPerformingRuns


def generate_roc_curve(use_synth_data: bool, best_runs: BestPerformingRuns):
    if use_synth_data:
        xgb_spec = eval_ds_to_roc_plot_spec(
            synth_eval_dataset(), legend_title="XGBoost"
        )
        lr_spec = eval_ds_to_roc_plot_spec(
            synth_eval_dataset(noise_to_y_probs=0.5), legend_title="Logistic regression"
        )
    else:
        xgb_spec = eval_ds_to_roc_plot_spec(
            load_eval_dataset(
                wandb_group=best_runs.wandb_group, wandb_run=best_runs.xgboost
            ),
            legend_title="XGBoost",
        )
        lr_spec = eval_ds_to_roc_plot_spec(
            load_eval_dataset(
                wandb_group=best_runs.wandb_group,
                wandb_run=best_runs.logistic_regression,
            ),
            legend_title="Logistic regression",
        )

    OUTPUT_PATH = PROJECT_ROOT / "outputs_for_publishing" / "figures" / "roc_curve.png"
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    return plot_auc_roc(
        specs=[xgb_spec, lr_spec],
        dpi=600,
        save_path=OUTPUT_PATH,
        title=f"ROC curve, {best_runs.lookahead_years} year lookahead",
    )


if __name__ == "__main__":
    plot = generate_roc_curve(use_synth_data=False)
