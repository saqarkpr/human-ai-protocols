"""Data loading and the information split between the AI and the clinician.

Data: real PhysioNet/CinC Challenge 2012 ICU stays (Set A and Set B), the same
files used in icu-mortality-trustworthy-eval. See data/PROVENANCE.md.

The central design choice of this project lives here: the AI and the
clinician do NOT see the same information.

* The AI sees the charted vitals and labs (heart rate, blood pressure,
  lactate, creatinine, ...), but NOT the neurological exam. All Glasgow Coma
  Scale (GCS) columns are removed, and so are SAPS-I and SOFA, because both
  severity scores contain a GCS component and would leak it back in.
* The simulated clinician forms a judgment from a few bedside cues, led by
  the neurological exam (GCS), plus age, lowest blood pressure, highest
  respiratory rate and ventilation status.

This information asymmetry is a modelling assumption, not a fact about any
real hospital. It is the mechanism through which human-AI complementarity
can arise at all, and scripts/run_all.py also runs an ablation where the
asymmetry is removed, to show what happens without it.
"""
import gzip
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LABEL = "In-hospital_death"
ID = "recordid"
# Leakage columns, verified in icu-mortality-trustworthy-eval/scripts/check_leakage.py
LEAKAGE = ["Survival", "Length_of_stay"]
SEVERITY_SCORES = ["SAPS-I", "SOFA"]  # both contain a GCS component
CLINICIAN_CUES = [
    "GCS_lowest", "GCS_last", "GCS_first",   # neurological exam
    "Age", "MAP_lowest", "RespRate_highest", "MechVentLast8Hour",
]


def load(set_name):
    path = DATA_DIR / f"PhysionetChallenge2012-set-{set_name}.csv.gz"
    with gzip.open(path, "rt") as f:
        return pd.read_csv(f)


def gcs_columns(df):
    return [c for c in df.columns if c.startswith("GCS_")]


def ai_features(df, include_neuro=False):
    """Feature matrix the AI is allowed to see."""
    drop = [LABEL, ID] + LEAKAGE
    if not include_neuro:
        drop += gcs_columns(df) + SEVERITY_SCORES
    return df.drop(columns=[c for c in drop if c in df.columns])


def clinician_cues(df):
    return df[CLINICIAN_CUES].copy()


def label(df):
    return df[LABEL].astype(int).values


def severity_tercile(df, cuts=None):
    """Observed severity (SAPS-I) in three bins. Used only by the causal
    analysis, as the variable a hospital would use when deciding who gets
    senior review. Cut points are fixed on Set A and reused on Set B."""
    s = df["SAPS-I"].astype(float)
    if cuts is None:
        cuts = np.nanquantile(s, [1 / 3, 2 / 3])
    s = s.fillna(np.nanmedian(s))
    return np.digitize(s, cuts), cuts


def split_set_a(df, frac_opt=0.3, seed=0):
    """Set A is split into a training part (fit the AI and the clinician's
    cue model) and an optimisation part (estimate protocol costs for the
    LP). Set B is never touched until final evaluation."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    n_opt = int(frac_opt * len(df))
    return df.iloc[idx[n_opt:]].reset_index(drop=True), df.iloc[idx[:n_opt]].reset_index(drop=True)
