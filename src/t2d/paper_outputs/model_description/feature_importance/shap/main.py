# %%
#%load_ext autoreload
#%autoreload 2

# %%

# %%
from t2d.paper_outputs.model_description.feature_importance.refactored_shap.get_shap_values import (
    get_shap_bundle_for_best_run,
)
from t2d.paper_outputs.model_description.feature_importance.shap.shap_plot import (
    plot_shap_scatter,
)
from t2d.utils.best_runs import Run

shap_values = get_shap_bundle_for_best_run(n_rows=10_000, cache_ver=0.01)

# %%
shap_pd_df = shap_values.get_df()

# %%
import polars as pl

shap_df = pl.from_pandas(shap_pd_df)
shap_aggregated_df = (
    shap_df.with_columns(pl.all().abs().keep_name())
    .std()
    .melt(value_vars=shap_df.columns)
    .sort("value", descending=True)
    .head(100)
)
shap_aggregated_df

# %%
from t2d.utils.feature_name_to_readable import feature_name_to_readable

shap_output_df = shap_aggregated_df.with_columns(
    pl.col("variable").apply(feature_name_to_readable).keep_name(),
    pl.col("value").round(2),
)
shap_output_df

# %%
from t2d.paper_outputs.config import OUTPUT_MAPPING, TABLES_PATH

TABLES_PATH.mkdir(parents=True, exist_ok=True)
shap_output_df.write_csv(TABLES_PATH / f"{OUTPUT_MAPPING.shap_table} - shap_table.csv")

# plot_shap_scatter(shap_values=shap_values, n_to_sample=10_000)

# %%
