from pathlib import Path

import polars as pl
from t2d.paper_outputs.model_description.feature_importance.refactored_shap.plot_shap import (
    save_plots_for_top_i_shap_by_variance,
)


def test_plot_top_i_shap(shap_long_df: pl.DataFrame, tmp_path: Path):
    save_plots_for_top_i_shap_by_variance(
        shap_long_df=shap_long_df,
        i=3,
        save_dir=tmp_path,
    )

    assert len(list(tmp_path.glob("*.png"))) == 3
