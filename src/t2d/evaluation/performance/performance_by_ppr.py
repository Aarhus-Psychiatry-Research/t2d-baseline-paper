import pandas as pd
from psycop.model_evaluation.binary.performance_by_ppr.performance_by_ppr import (
    generate_performance_by_ppr_table,
)
from t2d.evaluation.config import TABLES_PATH, best_run


def output_performance_by_ppr():
    df: pd.DataFrame = generate_performance_by_ppr_table(  # type: ignore
        eval_dataset=best_run.get_eval_dataset(),
        positive_rates=[0.05, 0.04, 0.03, 0.02, 0.01],
    )

    table_path = TABLES_PATH / "performance_by_ppr.xlsx"
    df.to_excel(table_path)

    pass


if __name__ == "__main__":
    output_performance_by_ppr()
