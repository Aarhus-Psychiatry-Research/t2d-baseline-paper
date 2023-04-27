from datetime import datetime
from pathlib import Path

from t2d.evaluation.data_loaders.best_runs import Run, RunGroup

date_str = datetime.now().strftime("%Y-%m-%d")
EVALUATION_ROOT = Path(__file__).parent

current_group = RunGroup(name="mameluco-cobblestone")
best_run = Run(
    group=current_group,
    name="airnwhiteback",
    pos_rate=0.03,
)

GENERAL_ARTIFACT_PATH = (
    EVALUATION_ROOT
    / "outputs_for_publishing"
    / date_str
    / f"{best_run.group}"
    / f"{best_run.name}"
)
FIGURES_PATH = GENERAL_ARTIFACT_PATH / "figures"
TABLES_PATH = GENERAL_ARTIFACT_PATH / "tables"
ROBUSTNESS_PATH = FIGURES_PATH / "robustness"
