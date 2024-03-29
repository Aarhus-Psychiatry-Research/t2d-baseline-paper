import pandas as pd
from psycop.model_evaluation.binary.performance_by_ppr.performance_by_ppr import (
    generate_performance_by_ppr_table,
)
from t2d.paper_outputs.config import EVAL_RUN, TABLES_PATH


def output_performance_by_ppr():
    eval_dataset = EVAL_RUN.get_eval_dataset()

    df: pd.DataFrame = generate_performance_by_ppr_table(  # type: ignore
        eval_dataset=eval_dataset,
        positive_rates=[0.05, 0.04, 0.03, 0.02, 0.01],
    )

    table_path = TABLES_PATH / "performance_by_ppr.xlsx"
    TABLES_PATH.mkdir(exist_ok=True, parents=True)
    df.to_excel(table_path)


if __name__ == "__main__":
    output_performance_by_ppr()
