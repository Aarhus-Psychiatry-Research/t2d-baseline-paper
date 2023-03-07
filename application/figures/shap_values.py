from zenml.steps import BaseParameters

from t2d_baseline_paper.globals import BestPerformingRuns


class TrainSplitConf(BaseParameters):
    best_runs: BestPerformingRuns
