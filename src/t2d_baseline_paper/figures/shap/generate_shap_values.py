import pandas as pd
import shap
from psycop_model_training.utils.col_name_inference import infer_predictor_col_name
from sklearn.pipeline import Pipeline
from zenml.steps import step


import pickle


@step
def generate_shap_values(train_df: pd.DataFrame, pipeline: Pipeline) -> bytes:
    pred_col_names = infer_predictor_col_name(train_df)
    X = train_df[pred_col_names]

    X_subsampled = X.sample(frac=0.11, random_state=42)

    model = pipeline["model"]
    explainer = shap.Explainer(model)
    shap_values = explainer(X_subsampled)
    return pickle.dumps(shap_values)
