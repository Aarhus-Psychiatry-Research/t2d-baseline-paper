import datetime

import pandas as pd

from t2d.evaluation.best_runs import current_group


def get_all_runs_df() -> pd.DataFrame:
    run_performance_files = current_group.group_dir.glob("*.parquet")

    all_models = pd.concat(
        pd.read_parquet(parquet_file) for parquet_file in run_performance_files
    )

    return all_models


if __name__ == "__main__":
    print("\n\n")
    all_models = get_all_runs_df()

    all_models["roc_auc"] = all_models["roc_auc"].round(2)

    best_models_by_architecture_lookahead = (
        all_models.sort_values("roc_auc", ascending=False)
        .groupby(["model_name", "lookahead_days"])
        .head(1)
        .sort_values(["model_name", "lookahead_days"])
    )[["model_name", "lookahead_days", "roc_auc", "run_name", "timestamp"]]

    best_model_by_auc = all_models.sort_values("roc_auc", ascending=False).head(1)

    now = datetime.datetime.now()

    best_in_last_hour = (
        all_models[all_models["timestamp"] > now - datetime.timedelta(hours=1)]
        .sort_values("roc_auc", ascending=False)
        .head(1)
        .reset_index(drop=True)["roc_auc"][0]
    )
    try:
        best_before_last_hour = (
            all_models[all_models["timestamp"] < now - datetime.timedelta(hours=1)]
            .sort_values("roc_auc", ascending=False)
            .head(1)
            .reset_index(drop=True)["roc_auc"][0]
        )
    except KeyError:
        best_before_last_hour = None

    if best_before_last_hour is not None:
        improvement_over_last_hour = best_in_last_hour - best_before_last_hour
        auroc_improvement_threshold = 0.001
        if improvement_over_last_hour < auroc_improvement_threshold:
            print(
                f"---- READY TO TERMINATE: Improvement of {improvement_over_last_hour} is smaller than threshold of {auroc_improvement_threshold} ----",
            )
        else:
            print(f"AUROC improvement over last hour was {improvement_over_last_hour}")

    first_model_timestamp = all_models.sort_values("timestamp", ascending=True).head(1)[
        "timestamp"
    ][0]

    training_minutes = round((now - first_model_timestamp).total_seconds() / 60)  # type: ignore
    print(f"Model training has been going on for {training_minutes} minutes")

    models_trained_total = len(all_models)
    print(f"In total, {models_trained_total} models have been trained")
    print("\n")

    models_trained_by_architecure_and_lookahead = all_models.groupby(
        ["model_name", "lookahead_days"],
    ).count()["run_name"]

    print(best_models_by_architecture_lookahead)
