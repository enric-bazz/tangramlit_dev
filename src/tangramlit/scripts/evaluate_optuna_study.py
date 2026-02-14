import argparse
import os
import sys
from typing import Optional, Tuple
import pandas as pd
import optuna
import optuna.visualization.matplotlib as ovm
from optuna.importance import (
    FanovaImportanceEvaluator,
    MeanDecreaseImpurityImportanceEvaluator,
    PedAnovaImportanceEvaluator,)

from optuna.trial import TrialState

class CleanTimelineStudy:
    def __init__(self, study):
        # Keep all trials except the broken ones:
        # - RUNNING (should not appear)
        # - missing datetime_complete (aborted / killed / incomplete)
        self._trials = [
            t for t in study.get_trials(deepcopy=False)
            if t.datetime_complete is not None
        ]

    @property
    def trials(self):
        return self._trials

    def get_trials(self, deepcopy=True, states=None):
        trials = self._trials
        if states is not None:
            trials = [t for t in trials if t.state in states]
        return trials

    # Minimal required attributes
    @property
    def directions(self):
        return None

    @property
    def study_name(self):
        return "cleaned_timeline"

    @property
    def user_attrs(self):
        return {}

    @property
    def system_attrs(self):
        return {}


def validate_study_folder(study_folder: str, study_name: str = "tangram_optuna_study") -> str:
    """
    Ensure the study folder contains the Optuna database file.
    Returns the database path.
    """
    db_name = f"{study_name}.db"
    db_path = os.path.join(study_folder, db_name)

    if not os.path.exists(db_path):
        raise FileNotFoundError(
            f"study folder {study_folder} missing expected file: {db_name}"
        )

    return os.path.abspath(db_path)



def compute_and_plot_importances(
    study: optuna.Study,
    ped_baseline_quantile: float = 0.1,
):
    # Evaluators
    eval_fanova = FanovaImportanceEvaluator()
    eval_mdi = MeanDecreaseImpurityImportanceEvaluator()
    eval_ped = PedAnovaImportanceEvaluator(baseline_quantile=ped_baseline_quantile)

    # Compute importances
    imp_fanova = optuna.importance.get_param_importances(study, evaluator=eval_fanova)
    imp_mdi = optuna.importance.get_param_importances(study, evaluator=eval_mdi)
    imp_ped = optuna.importance.get_param_importances(study, evaluator=eval_ped)

    # Merge all param names
    all_params = sorted(set(imp_fanova) | set(imp_mdi) | set(imp_ped))

    # Build dataframe
    df = pd.DataFrame(index=["FANOVA", "MDI", "PED_ANOVA"], columns=all_params)
    for p in all_params:
        df.loc["FANOVA", p] = imp_fanova.get(p, float("nan"))
        df.loc["MDI", p] = imp_mdi.get(p, float("nan"))
        df.loc["PED_ANOVA", p] = imp_ped.get(p, float("nan"))

    # Figures with the corresponding evaluator
    fig_fanova = ovm.plot_param_importances(study, evaluator=eval_fanova)
    fig_mdi = ovm.plot_param_importances(study, evaluator=eval_mdi)
    fig_ped = ovm.plot_param_importances(study, evaluator=eval_ped)

    return df, (fig_fanova, fig_mdi, fig_ped)




def main(argv: Optional[list] = None):
    p = argparse.ArgumentParser(description="Save optuna.visualization plots for study evaluation.")
    p.add_argument('study_folder', help="Path to folder containing the Optuna DB file")
    p.add_argument('--study-name', default='tangram_optuna_study', help='Optuna study name')
    args = p.parse_args(argv)

    # Normalize folder
    study_folder = os.path.abspath(args.study_folder)

    # Make study output folder
    os.makedirs(os.path.join(study_folder, args.study_name), exist_ok=True)


    if not os.path.isdir(study_folder):
        print(f"study folder not found: {study_folder}")
        sys.exit(1)

    study_name = args.study_name  # must be a plain name

    # Validate DB exists and get DB absolute path
    db_path = validate_study_folder(study_folder, study_name)

    # Build storage URL
    storage = f"sqlite:///{db_path}"

    # Load study
    study = optuna.load_study(study_name=study_name, storage=storage)

    # importance
    df, figs = compute_and_plot_importances(study)
    df.to_excel(f"{study_folder}/{study_name}/{study_name}_param_importance.xlsx")

    for label, ax in zip(["anova", "mdi", "ped"], figs):
        fig = ax.figure
        fig.savefig(f"{study_folder}/{study_name}/{study_name}_param_importance_{label}.png",
                dpi=200, bbox_inches="tight")

    # history
    hist_ax = ovm.plot_optimization_history(study)
    hist_fig = hist_ax.figure
    hist_fig.savefig(f"{study_folder}/{study_name}/{study_name}_history.png",
                    dpi=200, bbox_inches="tight")

    # intermediate values
    inter_ax = ovm.plot_intermediate_values(study)
    if inter_ax.get_legend():
        inter_ax.get_legend().remove()
    inter_fig = inter_ax.figure
    inter_fig.savefig(f"{study_folder}/{study_name}/{study_name}_intermediate_values.png",
                    dpi=200, bbox_inches="tight")

    # timeline
    # timeline_study = CleanTimelineStudy(study)
    # time_ax = ovm.plot_timeline(timeline_study)
    time_ax = ovm.plot_timeline(study)
    time_fig = time_ax.figure
    time_fig.savefig(f"{study_folder}/{study_name}/{study_name}_timeline.png",
                    dpi=200, bbox_inches="tight")
    # edf
    edf_ax = ovm.plot_edf(study)
    edf_fig = edf_ax.figure
    edf_fig.savefig(f"{study_folder}/{study_name}/{study_name}_edf.png",
                    dpi=200, bbox_inches="tight")

    print(f"Saved images to: {study_folder}")


if __name__ == '__main__':
    main()
