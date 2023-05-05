# %%
#%load_ext autoreload
# %autoreload 2

# %%
from t2d.paper_outputs.model_description.feature_importance.shap.get_shap_values import (
    get_shap_bundle_for_best_run,
)

long_shap_df = get_shap_bundle_for_best_run(
    n_rows=1_000_000, cache_ver=0.01,
).get_long_shap_df()

# %%
import polars as pl
from t2d.paper_outputs.model_description.feature_importance.shap.shap_table import (
    get_top_i_shap_values_for_printing,
)

table_df = get_top_i_shap_values_for_printing(
    shap_long_df=pl.from_pandas(long_shap_df), i=100,
)
table_df


# %%
from t2d.paper_outputs.config import FIGURES_PATH, OUTPUT_MAPPING

shap_figures_path = FIGURES_PATH / OUTPUT_MAPPING.shap_plots
shap_figures_path.mkdir(exist_ok=True, parents=True)

from t2d.paper_outputs.model_description.feature_importance.shap.plot_shap import (
    save_plots_for_top_i_shap_by_variance,
)

save_plots_for_top_i_shap_by_variance(
    shap_long_df=pl.from_pandas(long_shap_df), i=4, save_dir=shap_figures_path,
)

# %%
