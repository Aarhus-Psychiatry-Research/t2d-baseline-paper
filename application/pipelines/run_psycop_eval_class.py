from t2d_baseline_paper.best_runs import run
from t2d_baseline_paper.steps.get_train_split import (
    TrainSplitConf,
    get_train_split,
)
from t2d_baseline_paper.steps.model_evaluator import (
    evaluate_model,
)
from t2d_baseline_paper.steps.pipeline_loader import pipeline_loader
from zenml.pipelines import pipeline
from zenml.steps.base_step import BaseStep


@pipeline(enable_cache=True)
def base_evaluation_pipeline(
    training_data_loader: BaseStep,
    pipeline_loader: BaseStep,
    model_evaluator: BaseStep,
):
    train_split = training_data_loader()
    pipe = pipeline_loader()
    model_evaluator(pipe=pipe, train_split=train_split)  # type: ignore


if __name__ == "__main__":
    BASE_EVAL_PIPELINE_INSTANCE = base_evaluation_pipeline(
        training_data_loader=get_train_split(TrainSplitConf(best_runs=run)),
        pipeline_loader=pipeline_loader(TrainSplitConf(best_runs=run)),
        model_evaluator=evaluate_model(),
    )

    BASE_EVAL_PIPELINE_INSTANCE.run(unlisted=True)
