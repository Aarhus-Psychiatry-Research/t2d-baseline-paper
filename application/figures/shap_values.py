import matplotlib.pyplot as plt
import shap
from psycop_model_training.data_loader.utils import load_and_filter_split_from_cfg
from psycop_model_training.utils.col_name_inference import infer_predictor_col_name

from t2d_baseline_paper.data.load_true_data import load_fullconfig, load_pipe
from t2d_baseline_paper.globals import PROJECT_ROOT, BestPerformingRuns


def generate_shap_value_plot(best_runs: BestPerformingRuns):
    # Load cfg from pickle
    cfg = load_fullconfig(
        wandb_group=best_runs.wandb_group, wandb_run=best_runs.xgboost
    )

    df = load_and_filter_split_from_cfg(cfg=cfg, split="train")

    pred_col_names = infer_predictor_col_name(df)
    X = df[pred_col_names]

    pipe = load_pipe(wandb_group=best_runs.wandb_group, wandb_run=best_runs.xgboost)
    model = pipe["model"]

    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    plt.figure(figsize=fig_size, dpi=dpi)
    shap.plots.beeswarm(shap_values, show=False)

    OUTPUT_PATH = (
        PROJECT_ROOT / "outputs_for_publishing" / "figures" / "shap_beeswarm.png"
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(OUTPUT_PATH)
    plt.close
