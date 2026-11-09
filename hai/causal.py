"""Causal analysis of interaction logs (pgmpy).

Question: does sending a case to senior review (instead of the status-quo
"junior sees the AI advice first") lower the expected decision cost, and
by how much?

A hospital that only has observational logs faces a confounding problem.
Under a realistic status-quo policy, seniors are called more often for
sicker patients and for cases where the AI is uncertain. Those are the
harder cases, so a naive comparison of outcomes in senior-reviewed vs
junior-reviewed cases mixes the effect of the review with the difficulty
of the cases that were sent to it.

Because this is a simulation, the ground truth is known: every patient can
be run through BOTH arms with the same random draws, which gives the true
average effect. That lets us check whether the causal model recovers it.

Causal graph (discrete variables):

    Sev  -> Unc            sicker patients make the AI less certain
    Sev  -> Review <- Unc  the status-quo routing policy
    Sev  -> Out
    Unc  -> Out
    Review -> Out

Sev    observed severity (SAPS-I tercile, 0/1/2)
Unc    AI uncertain (risk within a band around the decision threshold)
Review 1 = senior review (blind-first), 0 = junior with AI advice first
Out    0 = correct decision, 1 = false alarm (cost 1), 2 = missed death (cost 5)

pgmpy derives the adjustment set {Sev, Unc} from the graph and answers the
interventional query P(Out | do(Review)) by back-door adjustment. Expected
cost under each intervention is 1 * P(Out=1) + 5 * P(Out=2).
"""
import numpy as np
import pandas as pd
from pgmpy.inference import CausalInference
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.parameter_estimator import DiscreteMLE

from .protocols import C_FN, C_FP, T_LOGIT, junior_advice_first, senior_blind_first

EDGES = [("Sev", "Unc"), ("Sev", "Review"), ("Unc", "Review"),
         ("Sev", "Out"), ("Unc", "Out"), ("Review", "Out")]

# Status-quo probability of senior review given (Sev, Unc).
STATUS_QUO_POLICY = {
    (0, 0): 0.05, (0, 1): 0.25,
    (1, 0): 0.15, (1, 1): 0.45,
    (2, 0): 0.35, (2, 1): 0.75,
}


def _outcome(flag, y):
    return np.where((y == 1) & ~flag, 2, np.where((y == 0) & flag, 1, 0))


def _cost(out):
    return np.where(out == 2, C_FN, np.where(out == 1, C_FP, 0.0))


def simulate_log(world, sev, band=0.75, seed=0):
    rng = np.random.default_rng(20_000 + seed)
    unc = (np.abs(world.ai_logit - T_LOGIT) < band).astype(int)
    p_rev = np.array([STATUS_QUO_POLICY[(s, u)] for s, u in zip(sev, unc)])
    review = (rng.random(len(sev)) < p_rev).astype(int)
    out_j = _outcome(junior_advice_first(world)[0], world.y)
    out_s = _outcome(senior_blind_first(world)[0], world.y)
    out = np.where(review == 1, out_s, out_j)
    log = pd.DataFrame({"Sev": sev, "Unc": unc, "Review": review, "Out": out})
    truth = {
        "cost": float(100 * (_cost(out_s).mean() - _cost(out_j).mean())),
        "missed_death": float(100 * ((out_s == 2).mean() - (out_j == 2).mean())),
    }
    return log, truth


def estimate(log):
    cost = _cost(log.Out.values)
    r1 = log.Review.values == 1
    naive = {"cost": float(100 * (cost[r1].mean() - cost[~r1].mean())),
             "missed_death": float(100 * ((log.Out[r1] == 2).mean() - (log.Out[~r1] == 2).mean()))}
    model = DiscreteBayesianNetwork(EDGES)
    model.fit(log, estimator=DiscreteMLE())
    ci = CausalInference(model)
    adj = ci.get_minimal_adjustment_set("Review", "Out")
    p = {}
    for r in (0, 1):
        q = ci.query(variables=["Out"], do={"Review": r}, show_progress=False)
        states = q.state_names["Out"]
        p[r] = {int(s): float(v) for s, v in zip(states, q.values)}
    exp_cost = {r: C_FP * p[r].get(1, 0) + C_FN * p[r].get(2, 0) for r in (0, 1)}
    backdoor = {"cost": 100 * (exp_cost[1] - exp_cost[0]),
                "missed_death": 100 * (p[1].get(2, 0) - p[0].get(2, 0))}
    return {"naive": naive, "backdoor": backdoor, "adjustment_set": sorted(adj) if adj else []}
