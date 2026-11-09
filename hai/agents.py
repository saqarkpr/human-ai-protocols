"""The agents of the simulation.

AIAgent
    A real model (gradient-boosted trees or logistic regression) trained on
    real ICU data. Its output is a mortality risk.

ClinicianAgent
    A *simulated* clinician. Its judgment follows the lens model of human
    judgment (Brunswik; Hammond): a person combines a few observable cues
    roughly linearly, plus inconsistency noise. Concretely:

        own_logit = cue_logit(patient) + Normal(0, sigma)

    where cue_logit is a logistic model on the bedside cues (fit once on the
    training data, standing in for clinical experience) and sigma is the
    clinician's inconsistency. Lower sigma means a more reliable clinician.

    When the clinician sees the AI's advice, it revises by weight-on-advice
    (the standard "advice taking" model from judgment research):

        final_logit = (1 - w) * own_logit + w * ai_logit

    Two parameters capture how the protocol shapes reliance:
      * w_first: weight when the AI advice is shown BEFORE the clinician
        forms a view (anchoring, higher w)
      * w_blind: weight when the clinician commits to a view first and
        only then sees the AI (lower w)

Everything about the clinician is an assumption. The sensitivity analysis
in scripts/run_all.py varies sigma and w to show which conclusions depend
on these numbers and which do not.
"""
from dataclasses import dataclass

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

EPS = 1e-6


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class AIAgent:
    def __init__(self, kind="gbt_cal", seed=0):
        self.kind = kind
        if kind == "gbt_cal":
            # Main AI. Gradient-boosted trees with isotonic calibration fit by
            # 5-fold cross-validation inside the training data. Calibration
            # matters here because every agent decides by thresholding risk.
            self.model = CalibratedClassifierCV(
                HistGradientBoostingClassifier(random_state=seed, max_iter=200),
                method="isotonic", cv=5)
        elif kind == "gbt":
            # Same trees without the calibration step (used as a "model update").
            self.model = HistGradientBoostingClassifier(random_state=seed, max_iter=200)
        elif kind == "logreg":
            self.model = Pipeline([
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=3000)),
            ])
        else:
            raise ValueError(kind)

    def fit(self, X, y):
        self.columns = list(X.columns)
        self.model.fit(X, y)
        return self

    def risk(self, X):
        return self.model.predict_proba(X[self.columns])[:, 1]


class CueModel:
    """Logistic model on bedside cues: the clinician's 'experience'."""

    def __init__(self):
        self.model = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000)),
        ])

    def fit(self, cues, y):
        self.columns = list(cues.columns)
        self.model.fit(cues, y)
        return self

    def cue_logit(self, cues):
        return logit(self.model.predict_proba(cues[self.columns])[:, 1])


@dataclass(frozen=True)
class ClinicianAgent:
    name: str
    sigma: float      # judgment inconsistency (logit scale)
    w_first: float    # weight on AI advice if advice is shown first
    w_blind: float    # weight on AI advice if the clinician commits first

    def judge(self, cue_logit, rng):
        return cue_logit + rng.normal(0.0, self.sigma, size=len(cue_logit))

    def revise(self, own_logit, ai_logit, advice_first):
        w = self.w_first if advice_first else self.w_blind
        return (1 - w) * own_logit + w * ai_logit


# Default agents used throughout. A senior clinician is more consistent and
# relies less on the AI; a junior anchors more strongly when shown advice first.
JUNIOR = ClinicianAgent("junior", sigma=1.0, w_first=0.6, w_blind=0.3)
SENIOR = ClinicianAgent("senior", sigma=0.5, w_first=0.4, w_blind=0.2)
