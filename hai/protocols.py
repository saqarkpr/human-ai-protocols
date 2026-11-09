"""Interaction protocols as compositions of simple routing actions.

The decision is binary (flag the case or not). Costs are asymmetric and
domain specific. A missed positive (a death in the ICU, an intrusion in
cybersecurity) costs C_FN, a false alarm costs C_FP. With a calibrated risk
p the cost-minimising rule is to flag when p >= C_FP / (C_FP + C_FN), and
every agent uses that threshold.

A World holds, for one simulation seed, everything any protocol could need.
These are the AI's risk, the junior's and senior's own (pre-advice)
judgments, the true outcome and the domain costs. All protocols are
evaluated on the SAME world (common random numbers), so differences between
protocols are paired.
"""
from dataclasses import dataclass

import numpy as np

from .agents import JUNIOR, SENIOR, logit


@dataclass(frozen=True)
class Costs:
    fn: float
    fp: float

    @property
    def threshold(self):
        return self.fp / (self.fp + self.fn)

    @property
    def t_logit(self):
        return float(np.log(self.threshold / (1 - self.threshold)))


@dataclass
class World:
    y: np.ndarray
    ai_logit: np.ndarray
    junior_own: np.ndarray
    senior_own: np.ndarray
    costs: Costs
    junior: object = JUNIOR
    senior: object = SENIOR

    @property
    def t(self):
        return self.costs.t_logit


def make_world(y, ai_risk, cue_logit, seed, costs, junior=JUNIOR, senior=SENIOR):
    rng = np.random.default_rng(seed)
    return World(y=np.asarray(y), ai_logit=logit(ai_risk),
                 junior_own=junior.judge(cue_logit, rng),
                 senior_own=senior.judge(cue_logit, rng),
                 costs=costs, junior=junior, senior=senior)


# ---------------------------------------------------------------------------
# Atomic actions. Each returns (flag, junior_used, senior_used) per case.
# ---------------------------------------------------------------------------

def _out(flag, j, s):
    n = len(flag)
    j = np.full(n, float(j)) if np.isscalar(j) else np.asarray(j, float)
    s = np.full(n, float(s)) if np.isscalar(s) else np.asarray(s, float)
    return np.asarray(flag, bool), j, s


def ai_auto(w):
    return _out(w.ai_logit >= w.t, 0, 0)


def junior_alone(w):
    return _out(w.junior_own >= w.t, 1, 0)


def junior_advice_first(w):
    return _out(w.junior.revise(w.junior_own, w.ai_logit, advice_first=True) >= w.t, 1, 0)


def junior_blind_first(w):
    return _out(w.junior.revise(w.junior_own, w.ai_logit, advice_first=False) >= w.t, 1, 0)


def senior_alone(w):
    return _out(w.senior_own >= w.t, 0, 1)


def senior_blind_first(w):
    return _out(w.senior.revise(w.senior_own, w.ai_logit, advice_first=False) >= w.t, 0, 1)


def disagreement(w):
    """Junior's committed (blind) decision differs from the AI's decision."""
    return (w.junior_own >= w.t) != (w.ai_logit >= w.t)


def escalate_on_disagreement(w):
    """Junior commits to a view, then sees the AI. If they agree, that is the
    decision. If they disagree, a senior (blind-first) makes the call."""
    d_j = w.junior_own >= w.t
    d_s = w.senior.revise(w.senior_own, w.ai_logit, advice_first=False) >= w.t
    dis = disagreement(w)
    return _out(np.where(dis, d_s, d_j), 1, dis)


ACTIONS = {
    "ai_auto": ai_auto,
    "junior_alone": junior_alone,
    "junior_advice_first": junior_advice_first,
    "junior_blind_first": junior_blind_first,
    "escalate_on_disagreement": escalate_on_disagreement,
    "senior_blind_first": senior_blind_first,
    "senior_alone": senior_alone,
}
HUMAN_ACTIONS = {a for a in ACTIONS if a != "ai_auto"}


def confidence_deferral(w, band):
    """AI decides alone when its risk is far from the threshold; cases inside
    the uncertainty band (logit units) go to the junior, blind-first."""
    near = np.abs(w.ai_logit - w.t) < band
    fa, ja, sa = ai_auto(w)
    fj, jj, sj = junior_blind_first(w)
    return _out(np.where(near, fj, fa), np.where(near, jj, ja), np.where(near, sj, sa))


def apply_policy(w, cell, assignment, seed=0):
    """Apply a cell-wise randomised policy (from the LP). `cell[i]` is the
    cell of case i, `assignment[c]` maps action name -> probability."""
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

def case_cost(flag, y, costs):
    return np.where((y == 1) & ~flag, costs.fn, 0.0) + np.where((y == 0) & flag, costs.fp, 0.0)


def summarise(flag, j, s, y, costs, human=None):
    tp = np.sum(flag & (y == 1))
    fn = np.sum(~flag & (y == 1))
    tn = np.sum(~flag & (y == 0))
    fp = np.sum(flag & (y == 0))
    out = {
        "cost_per_100": 100 * case_cost(flag, y, costs).mean(),
        "sensitivity": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "junior_reviews_pct": 100 * j.mean(),
        "senior_reviews_pct": 100 * s.mean(),
    }
    if human is not None:
        out["human_in_loop_pct"] = 100 * np.mean(human)
    return out
