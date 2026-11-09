"""Neurological / general intensive care.

Real PhysioNet/CinC Challenge 2012 ICU stays (Set A, Set B), see
data/icu/PROVENANCE.md.

Information split. The AI sees charted vitals and laboratory values but NOT
the neurological examination. All Glasgow Coma Scale (GCS) columns are
removed, and so are SAPS-I and SOFA, because both contain a GCS component.
The simulated clinician judges from bedside cues led by the GCS, plus age,
lowest mean arterial pressure, highest respiratory rate and ventilation.
"""
import gzip
from pathlib import Path

import numpy as np
import pandas as pd

from ..protocols import Costs
from .base import Domain

DATA = Path(__file__).resolve().parents[2] / "data" / "icu"
LABEL, ID = "In-hospital_death", "recordid"
LEAKAGE = ["Survival", "Length_of_stay"]  # verified in icu-mortality-trustworthy-eval
SCORES = ["SAPS-I", "SOFA"]
CUES = ["GCS_lowest", "GCS_last", "GCS_first", "Age", "MAP_lowest",
        "RespRate_highest", "MechVentLast8Hour"]


class ICUDomain(Domain):
    def _load(self, s):
        with gzip.open(DATA / f"PhysionetChallenge2012-set-{s}.csv.gz", "rt") as f:
            return pd.read_csv(f)

    def splits(self):
        if not self._cache:
            a, b = self._load("a"), self._load("b")
            idx = np.random.default_rng(self.seed).permutation(len(a))
            n_opt = int(0.3 * len(a))
            self._cache.update(train=a.iloc[idx[n_opt:]].reset_index(drop=True),
                               opt=a.iloc[idx[:n_opt]].reset_index(drop=True), eval=b)
        return self._cache

    def ai_features(self, df, private=False):
        drop = [LABEL, ID] + LEAKAGE
        if not private:
            drop += [c for c in df.columns if c.startswith("GCS_")] + SCORES
        return df.drop(columns=[c for c in drop if c in df.columns])

    def cues(self, df):
        return df[CUES].copy()

    def label(self, df):
        return df[LABEL].astype(int).values

    def severity(self, df):
        return df["SAPS-I"].astype(float).values


ICU = ICUDomain(name="icu", costs=Costs(fn=5.0, fp=1.0), case="patient",
                positive="in-hospital death",
                human_private="neurological examination (Glasgow Coma Scale)")
