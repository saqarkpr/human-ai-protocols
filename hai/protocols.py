"""Interaction protocols as compositions of simple routing actions.

Decision task: flag a patient as high mortality risk (so they get escalated
care) or not. Costs are asymmetric. Missing a patient who dies (false
negative) costs C_FN = 5, flagging a survivor (false positive) costs
C_FP = 1. With a calibrated risk p, the cost-minimising rule is to flag when
p >= C_FP / (C_FP + C_FN) = 1/6, and every agent uses that threshold.

A World holds, for one simulation seed, every quantity any protocol could
need: the AI's risk, the junior's and senior's own (pre-advice) judgments,
and the true outcome. All protocols are evaluated on the SAME world
(common random numbers), so differences between protocols are paired and
not blurred by different noise draws.
"""
from dataclasses import dataclass

import numpy as np

from .agents import JUNIOR, SENIOR, logit

C_FN = 5.0
C_FP = 1.0
THRESHOLD = C_FP / (C_FP + C_FN)
T_LOGIT = float(logit(np.array([THRESHOLD]))[0])


@dataclass
class World:
    y: np.ndarray
    ai_logit: np.ndarray
    junior_own: np.ndarray
    senior_own: np.ndarray
    junior: object = JUNIOR
    senior: object = SENIOR


def make_world(y, ai_risk, cue_logit, seed, junior=JUNIOR, senior=SENIOR):
    rng = np.random.default_rng(seed)
    return World(
        y=np.asarray(y),
        ai_logit=logit(ai_risk),
        junior_own=junior.judge(cue_logit, rng),
        senior_own=senior.judge(cue_logit, rng),
        junior=junior,
        senior=senior,
    )


# ---------------------------------------------------------------------------
# Atomic actions. Each returns (flag, junior_used, senior_used), one entry
# per patient.
# ---------------------------------------------------------------------------

def _out(flag, j, s):
    n = len(flag)
    return flag.astype(bool), np.full(n, j, float) if np.isscalar(j) else j.astype(float), \
        np.full(n, s, float) if np.isscalar(s) else s.astype(float)


def ai_auto(w):
    return _out(w.ai_logit >= T_LOGIT, 0, 0)


def junior_alone(w):
    return _out(w.junior_own >= T_LOGIT, 1, 0)


def junior_advice_first(w):
    return _out(w.junior.revise(w.junior_own, w.ai_logit, advice_first=True) >= T_LOGIT, 1, 0)


def junior_blind_first(w):
    return _out(w.junior.revise(w.junior_own, w.ai_logit, advice_first=False) >= T_LOGIT, 1, 0)


def senior_alone(w):
    return _out(w.senior_own >= T_LOGIT, 0, 1)


def senior_blind_first(w):
    return _out(w.senior.revise(w.senior_own, w.ai_logit, advice_first=False) >= T_LOGIT, 0, 1)


def escalate_on_disagreement(w):
    """Junior commits to a view, then sees the AI. If junior and AI agree,
    that is the decision. If they disagree, a senior (who also commits
    first, then sees the AI) makes the call. Disagreement is used as the
    signal of an unreliable case."""
    d_j = w.junior_own >= T_LOGIT
    d_a = w.ai_logit >= T_LOGIT
    d_s = w.senior.revise(w.senior_own, w.ai_logit, advice_first=False) >= T_LOGIT
    disagree = d_j != d_a
    flag = np.where(disagree, d_s, d_j)
    return _out(flag, 1, disagree)


ACTIONS = {
    "ai_auto": ai_auto,
    "junior_alone": junior_alone,
    "junior_advice_first": junior_advice_first,
    "junior_blind_first": junior_blind_first,
    "escalate_on_disagreement": escalate_on_disagreement,
    "senior_blind_first": senior_blind_first,
    "senior_alone": senior_alone,
}
# Actions that keep a human in the loop (used by the oversight constraint).
HUMAN_ACTIONS = {a for a in ACTIONS if a != "ai_auto"}


def confidence_deferral(w, band):
    """AI decides on its own when its risk is far from the threshold;
    cases inside the uncertainty band (in logit units) go to the junior,
    blind-first."""
    near = np.abs(w.ai_logit - T_LOGIT) < band
    fa, ja, sa = ai_auto(w)
    fj, jj, sj = junior_blind_first(w)
    return _out(np.where(near, fj, fa), np.where(near, jj, ja), np.where(near, sj, sa))


def apply_policy(w, cell, assignment, seed=0):
    """Apply a cell-wise randomised policy (from the LP). `cell[i]` is the
    cell of patient i, `assignment[c]` maps action name -> probability."""
    rng = np.random.default_rng(10_000 + seed)
    u = rng.random(len(w.y))
    outs = {a: f(w) for a, f in ACTIONS.items()}
    flag = np.zeros(len(w.y), bool)
    j = np.zeros(len(w.y))
    s = np.zeros(len(w.y))
    chosen = np.empty(len(w.y), dtype=object)
    for c, probs in assignment.items():
        idx = np.where(cell == c)[0]
        names = list(probs)
        cum = np.cumsum([max(probs[a], 0.0) for a in names])
        cum = cum / cum[-1]
        pick = np.minimum(np.searchsorted(cum, u[idx], side="right"), len(names) - 1)
        for k, a in enumerate(names):
            sel = idx[pick == k]
            flag[sel], j[sel], s[sel] = outs[a][0][sel], outs[a][1][sel], outs[a][2][sel]
            chosen[sel] = a
    return flag, j, s, chosen


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def patient_cost(flag, y):
    return np.where((y == 1) & ~flag, C_FN, 0.0) + np.where((y == 0) & flag, C_FP, 0.0)


def summarise(flag, j, s, y, human=None):
    tp = np.sum(flag & (y == 1))
    fn = np.sum(~flag & (y == 1))
    tn = np.sum(~flag & (y == 0))
    fp = np.sum(flag & (y == 0))
    out = {
        "cost_per_100": 100 * patient_cost(flag, y).mean(),
        "sensitivity": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "junior_reviews_pct": 100 * j.mean(),
        "senior_reviews_pct": 100 * s.mean(),
    }
    if human is not None:
        out["human_in_loop_pct"] = 100 * human.mean()
    return out
