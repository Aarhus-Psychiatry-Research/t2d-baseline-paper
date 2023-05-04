from datetime import datetime
from pathlib import Path

from t2d.utils.best_runs import Run, RunGroup

########################################
# UPDATE THESE FOR ALTERNATIVE OUTPUTS #
########################################
RUN_GROUP = "mistouching-unwontedness"
BEST_RUN_NAME = "phototonicunelucidated"

date_str = datetime.now().strftime("%Y-%m-%d")
EVALUATION_ROOT = Path(__file__).parent

current_group = RunGroup(name=RUN_GROUP)
best_run = Run(
    group=current_group,
    name=BEST_RUN_NAME,
    pos_rate=0.03,
)

GENERAL_ARTIFACT_PATH = (
    EVALUATION_ROOT
    / "outputs_for_publishing"
    / f"{best_run.group.name}"
    / f"{best_run.name}"
)
FIGURES_PATH = GENERAL_ARTIFACT_PATH / "figures"
TABLES_PATH = GENERAL_ARTIFACT_PATH / "tables"
ESTIMATES_PATH = GENERAL_ARTIFACT_PATH / "estimates"
ROBUSTNESS_PATH = FIGURES_PATH / "robustness"
