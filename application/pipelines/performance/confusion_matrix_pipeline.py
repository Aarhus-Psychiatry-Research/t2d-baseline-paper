import pandas as pd
from sklearn.metrics import confusion_matrix
from t2d_baseline_paper.best_runs import TABLES_PATH, best_run
from t2d_baseline_paper.data.load_true_data import load_eval_dataset


def get_top_fraction(df: pd.DataFrame, col_name: str, fraction: float) -> pd.DataFrame:
    """
    Returns the top N percent of the data sorted by column y in a dataframe df.
    """
    # Calculate the number of rows to select
    num_rows = int(len(df) * fraction)

    # Sort the dataframe by column y and select the top N percent of rows
    sorted_df = df.sort_values(col_name, ascending=False)
    top_fraction = sorted_df.head(num_rows)

    return top_fraction


def confusion_matrix_metrics(
    y_true: pd.Series, y_pred: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Creates a confusion matrix dataframe with PPV, NPV, SENS, and SPEC.
    """
    # Calculate the confusion matrix using sklearn
    cm = confusion_matrix(y_true, y_pred)

    # Extract the TP, FP, TN, and FN values from the confusion matrix
    TP = cm[1, 1]
    FP = cm[0, 1]
    TN = cm[0, 0]
    FN = cm[1, 0]

    # Calculate the PPV, NPV, SENS, and SPEC
    PPV = TP / (TP + FP)
    NPV = TN / (TN + FN)
    SENS = TP / (TP + FN)
    SPEC = TN / (TN + FP)

    # Create the confusion matrix dataframe
    df = pd.DataFrame(
        {"Actual Positive": [TP, FN], "Actual Negative": [FP, TN]},
        index=["Predicted Positive", "Predicted Negative"],
    )

    # Create a separate dataframe for the metrics
    metrics_df = pd.DataFrame(
        {"PPV": [PPV], "NPV": [NPV], "SENS": [SENS], "SPEC": [SPEC]},
    )

    return df, metrics_df


def incidence_by_time_until_outcome_pipeline():
    eval_ds = load_eval_dataset(
        wandb_group=best_run.wandb_group,
        wandb_run=best_run.model,
    )

    df = pd.DataFrame({"y": eval_ds.y, "y_hat_probs": eval_ds.y_hat_probs})

    top_df = get_top_fraction(df=df, col_name="y_hat_probs", fraction=best_run.pos_rate)

    # Set y_hat to 1 for rows in top_df, 0 for others
    prediction_df = df.copy()
    prediction_df["y_hat"] = 0
    prediction_df.loc[top_df.index, "y_hat"] = 1

    conf_matrix, metrics_df = confusion_matrix_metrics(
        y_true=prediction_df["y"],
        y_pred=prediction_df["y_hat"],
    )

    TABLES_PATH.mkdir(parents=True, exist_ok=True)

    # Save the df to a csv file
    conf_matrix.to_csv(TABLES_PATH / "confusion_matrix.csv")
    metrics_df.to_csv(TABLES_PATH / "confusion_matrix_metrics.csv")


if __name__ == "__main__":
    incidence_by_time_until_outcome_pipeline()
