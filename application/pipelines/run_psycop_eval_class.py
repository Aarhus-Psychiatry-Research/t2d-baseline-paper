from t2d_baseline_paper.best_runs import best_runs
from t2d_baseline_paper.steps.model_evaluator import (
    evaluate_model,
)

from t2d_baseline_paper.steps.pipeline_loader import pipeline_loader
from t2d_baseline_paper.steps.get_train_split import get_train_split
from t2d_baseline_paper.steps.get_train_split import (
    TrainSplitConf,
)
from zenml.pipelines import pipeline


@pipeline(enable_cache=True)
def base_evaluation_pipeline(
    training_data_loader,
    pipeline_loader,
    model_evaluator,
):
    train_split = training_data_loader()
    pipe = pipeline_loader()
    roc_auc = model_evaluator(pipe=pipe, train_split=train_split)


if __name__ == "__main__":
    BASE_EVAL_PIPELINE_INSTANCE = base_evaluation_pipeline(
        training_data_loader=get_train_split(TrainSplitConf(best_runs=best_runs)),
        pipeline_loader=pipeline_loader(TrainSplitConf(best_runs=best_runs)),
        model_evaluator=evaluate_model(),
    )

    BASE_EVAL_PIPELINE_INSTANCE.run(unlisted=True)
