# %%
from pathlib import Path

import pandas as pd
from t2d.evaluation.config import TABLES_PATH

# %%
feature_description_path = Path(
    "E:/shared_resources/t2d/feature_sets/psycop_t2d_adminmanber_features_2023_03_22_15_14/feature_set_descriptive_stats/train_feature_descriptive_stats.csv",
)
df = pd.read_csv(feature_description_path)


# %%
# Sort by columns predictor_df, then resolve_multiple, then lookbehin_days, then fallback_strategy
df_sorted = df.sort_values(
    by=["Predictor df", "Resolve multiple", "Lookbehind days", "Fallback strategy"],
)

df_renamed = df_sorted.rename(
    columns={
        "Predictor df": "Predictor",
        "Resolve multiple": "Aggregation method",
        "N unique": "N unique values",
        "50.0-percentile": "Median",
    },
)

# Capitalise the first letter in aggregation method
df_renamed["Aggregation method"] = df_renamed["Aggregation method"].str.capitalize()


df_selected = df_renamed[
    [
        "Predictor",
        "Aggregation method",
        "Lookbehind days",
        "Fallback strategy",
        "N unique values",
        "Mean",
        "1.0-percentile",
        "25.0-percentile",
        "Median",
        "75.0-percentile",
        "99.0-percentile",
    ]
]

# Save to excel
TABLES_PATH.mkdir(parents=True, exist_ok=True)
df_selected.to_excel(TABLES_PATH / "feature_descriptive_stats.xlsx", index=False)
