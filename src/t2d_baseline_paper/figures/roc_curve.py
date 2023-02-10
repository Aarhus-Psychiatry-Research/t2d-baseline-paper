"""AUC ROC curve."""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
from psycop_model_training.model_eval.dataclasses import EvalDataset
from sklearn.metrics import roc_auc_score, roc_curve

from t2d_baseline_paper.data.test_eval_data import synth_eval_dataset


@dataclass
class ROCPlotSpec():
    y: pd.Series
    y_hat_probs: pd.Series
    legend_title: str
    
def eval_ds_to_roc_plot_spec(eval_dataset: EvalDataset, legend_title: str) -> ROCPlotSpec:
    """Convert EvalDataset to ROCPlotSpec."""
    return ROCPlotSpec(
        y=eval_dataset.y,
        y_hat_probs=eval_dataset.y_hat_probs,
        legend_title=legend_title,
    )

def plot_auc_roc(
    specs: Union[ROCPlotSpec, list[ROCPlotSpec]],
    title: str = "ROC-curve",
    fig_size: Optional[tuple] = (5, 5),
    dpi: int = 160,
    save_path: Optional[Path] = None,
) -> Union[None, Path]:
    """Plot AUC ROC curve.

    Args:
        eval_dataset (EvalDataset): Evaluation dataset.
        fig_size (Optional[tuple], optional): figure size. Defaults to None.
        save_path (Optional[Path], optional): path to save figure. Defaults to None.

    Returns:
        Union[None, Path]: None if save_path is None, else path to saved figure.
    """
    plt.figure(figsize=fig_size, dpi=dpi)
    
    for specs in specs:
        fpr, tpr, _ = roc_curve(specs.y, specs.y_hat_probs)
        auc = roc_auc_score(specs.y, specs.y_hat_probs)
        AUC_STR = f"(AUC = {str(round(auc, 3))})"
        
        plt.plot(fpr, tpr, label=f"{specs.legend_title} {AUC_STR}")

    
    plt.legend(loc=4)
    
    plt.title(title)
    plt.xlabel("1 - Specificity")
    plt.ylabel("Sensitivity")

    if save_path is not None:
        plt.savefig(save_path)
    plt.close()

    return save_path


