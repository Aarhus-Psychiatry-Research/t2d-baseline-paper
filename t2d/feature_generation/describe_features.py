"""Main feature generation."""
# %%
import logging
from pathlib import Path

from psycop_feature_generation.application_modules.project_setup import (
    get_project_info,
)
from timeseriesflattener.utils import load_dataset_from_file

log = logging.getLogger()

# %%
project_info = get_project_info(
    project_name="t2d",
)

from t2d.feature_generation.specify_features import FeatureSpecifier

feature_specs = FeatureSpecifier(
    project_info=project_info,
    min_set_for_debug=False,  # Remember to set to False when generating full dataset
).get_feature_specs()

selected_specs = [
    spec
    for spec in feature_specs
    if "pred" in spec.get_col_str() or "outc" in spec.get_col_str()
]

DATASET_FOLDER = Path(
    "E:/shared_resources/t2d/feature_sets/psycop_t2d_adminmanber_features_2023_03_22_15_14",
)
# %%
# %reload_ext autoreload
# %autoreload 2

# %%
dataset = load_dataset_from_file(
    file_path=DATASET_FOLDER
    / "psycop_t2d_adminmanber_features_2023_03_22_15_14_train.parquet",
)

# %%
from psycop_feature_generation.data_checks.flattened.feature_describer import (
    save_feature_descriptive_stats_from_dir,
)

save_feature_descriptive_stats_from_dir(
    feature_set_dir=DATASET_FOLDER,
    feature_specs=selected_specs,  # type: ignore
    file_suffix="parquet",
    splits=["train"],
)


# %%
