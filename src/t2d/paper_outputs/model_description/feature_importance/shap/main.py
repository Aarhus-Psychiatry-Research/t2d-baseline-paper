# %%
#%load_ext autoreload
#%autoreload 2

# %%

# %%
from t2d.paper_outputs.model_description.feature_importance.shap.generate_shap_values import (
    generate_shap_values_for_best_run,
)
from t2d.paper_outputs.model_description.feature_importance.shap.shap_plot import (
    plot_shap_scatter,
)
from t2d.utils.best_runs import Run

shap_values = generate_shap_values_for_best_run(n_rows=10_000, cache_ver=0.01)

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

# plot_shap_scatter(shap_values=shap_values, n_to_sample=10_000)

# %%
