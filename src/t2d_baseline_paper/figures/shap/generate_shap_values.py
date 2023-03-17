import pickle

import pandas as pd
import shap
from psycop_model_training.utils.col_name_inference import infer_predictor_col_name
from sklearn.pipeline import Pipeline
from zenml.steps import step


def generate_shap_values(train_df: pd.DataFrame, pipeline: Pipeline) -> bytes:
    pred_col_names = infer_predictor_col_name(train_df)
    features = train_df[pred_col_names]

    model = pipeline["model"]
    explainer = shap.Explainer(model)  # type: ignore
    shap_values = explainer(features)
    return pickle.dumps(shap_values)
