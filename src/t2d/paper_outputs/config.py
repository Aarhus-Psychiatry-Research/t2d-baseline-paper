from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import plotnine as pn
from t2d.utils.best_runs import Run, RunGroup

########################################
# UPDATE THESE TO SELECT MODEL OUTPUTS #
########################################
EVALUATION_ROOT = Path(__file__).parent

DEV_GROUP_NAME = "mistouching-unwontedness"
DEVELOPMENT_GROUP = RunGroup(name=DEV_GROUP_NAME)

EVAL_GROUP_NAME = f"{DEV_GROUP_NAME}-eval-on-test"
EVAL_GROUP = RunGroup(name=EVAL_GROUP_NAME)

BEST_POS_RATE = 0.03

RUN_TO_EVAL = Run(
    group=EVAL_GROUP,
    name="peshwashiptrayfuls",
    pos_rate=BEST_POS_RATE,
)

################
# OUTPUT PATHS #
################
date_str = datetime.now().strftime("%Y-%m-%d")

GENERAL_ARTIFACT_PATH = (
    EVALUATION_ROOT
    / "outputs_for_publishing"
    / f"{EVAL_GROUP.name}"
    / f"{RUN_TO_EVAL.name}"
)
FIGURES_PATH = GENERAL_ARTIFACT_PATH / "figures"
TABLES_PATH = GENERAL_ARTIFACT_PATH / "tables"
ESTIMATES_PATH = GENERAL_ARTIFACT_PATH / "estimates"
ROBUSTNESS_PATH = FIGURES_PATH / "robustness"


@dataclass
class OutputMapping:
    diabetes_incidence_by_time: str = "eFigure 3"
    shap_table: str = "eTable 3"
    shap_plots: str = "Figure 3"


OUTPUT_MAPPING = OutputMapping()

PN_THEME = pn.theme_bw() + pn.theme(panel_grid=pn.element_blank())
