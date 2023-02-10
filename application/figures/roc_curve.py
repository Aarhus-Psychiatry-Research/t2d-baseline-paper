from pathlib import Path

from t2d_baseline_paper.data.test_eval_data import synth_eval_dataset
from t2d_baseline_paper.figures.roc_curve import eval_ds_to_roc_plot_spec, plot_auc_roc

if __name__ == "__main__":
    xgb_spec = eval_ds_to_roc_plot_spec(synth_eval_dataset(), "XGBoost")
    lr_spec = eval_ds_to_roc_plot_spec(synth_eval_dataset(noise_to_y_probs=0.5), "Logistic regression")
    
    plot = plot_auc_roc(specs=[xgb_spec, lr_spec], dpi=600, save_path=Path("outputs_for_publishing") / "figures" / "roc_curve.png")
    
    