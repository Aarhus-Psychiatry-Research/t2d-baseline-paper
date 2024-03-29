# %%
# %load_ext autoreload
# %autoreload 2

# %%
from t2d.feature_generation.outcome_specification.lab_results import (
    get_first_diabetes_lab_result_above_threshold,
)

df_lab_result = get_first_diabetes_lab_result_above_threshold()

# %%
import pandas as pd
import plotnine as pn

# %%
p = (
    pn.ggplot(df_lab_result, pn.aes(x="timestamp"))
    + pn.geom_histogram(binwidth=30)
    + pn.geom_vline(xintercept=pd.to_datetime("2013-01-01"), color="red")
    + pn.theme_minimal()
    + pn.xlab("Date")
    + pn.ylab("Count")
)

# %%
from t2d.paper_outputs.config import FIGURES_PATH, OUTPUT_MAPPING

FIGURES_PATH.mkdir(parents=True, exist_ok=True)

save_path = (
    FIGURES_PATH
    / f"{OUTPUT_MAPPING.diabetes_incidence_by_time} - incidence_by_time.png"
)

p.save(save_path, dpi=600)

# %%
