import pandas as pd
from psycop_model_training.config_schemas.full_config import FullConfigSchema
from psycop_model_training.preprocessing.post_split.pipeline import (
    create_post_split_pipeline,
)
from psycop_model_training.training.train_and_predict import train_and_predict
from psycop_model_training.utils.col_name_inference import (
    infer_outcome_col_name,
    infer_predictor_col_name,
)

from t2d.evaluation.best_runs import best_run
from t2d.evaluation.figures.shap.generate_shap_values import generate_shap_values
from t2d.evaluation.figures.shap.shap_plot import plot_shap_scatter

if __name__ == "__main__":
    cfg: FullConfigSchema = best_run.cfg

    # Keep only one outcome col
    cfg.preprocessing.pre_split.Config.allow_mutation = True
    cfg.preprocessing.pre_split.keep_only_one_outcome_col = True

    # Disable feature selection
    cfg.preprocessing.post_split.feature_selection.Config.allow_mutation = True
    cfg.preprocessing.post_split.feature_selection.name = None
    cfg.preprocessing.post_split.feature_selection.params = None

    pipe = create_post_split_pipeline(cfg)

    train_col_names = infer_predictor_col_name(df=dataset.train)  # type: ignore # noqa

    get_eval_dataset, pipe = train_and_predict(
        cfg=cfg,
        train_datasets=[dataset.train, dataset.val],  # type: ignore # noqa
        pipe=pipe,
        outcome_col_name="outc_first_diabetes_lab_result_within_1095_days_max_fallback_0_dichotomous",
        train_col_names=train_col_names,
    )

    concatenated_df = pd.concat([dataset.train, dataset.val], ignore_index=True)  # type: ignore # noqa
    feature_cols = infer_predictor_col_name(concatenated_df)
    outcome_cols = infer_outcome_col_name(concatenated_df)

    shap_values = generate_shap_values(
        features=concatenated_df[feature_cols],
        outcome=concatenated_df[outcome_cols],
        pipeline=pipe,  # type: ignore
    )

    plot_shap_scatter(shap_values=shap_values, n_to_sample=10_000)
