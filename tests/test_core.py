"""Sanity checks: run with `python -m pytest -q`."""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hai import symbolic as S  # noqa: E402
from hai.agents import JUNIOR, SENIOR  # noqa: E402
from hai.optimise import solve  # noqa: E402
from hai.protocols import ACTIONS, Costs, World, apply_policy, case_cost  # noqa: E402

COSTS = Costs(fn=5.0, fp=1.0)


def toy_world(n=500, seed=0, junior=JUNIOR, costs=COSTS):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.15).astype(int)
    ai = rng.normal(0, 1.5, n) + 1.5 * y - 1.5
    return World(y=y, ai_logit=ai, junior_own=ai + rng.normal(0, 1, n),
                 senior_own=ai + rng.normal(0, .5, n), costs=costs, junior=junior, senior=SENIOR)


def test_threshold_is_cost_optimal():
    assert abs(COSTS.threshold - 1 / 6) < 1e-12
    assert abs(Costs(fn=10, fp=1).threshold - 1 / 11) < 1e-12


def test_ai_auto_uses_no_humans():
    _, j, s = ACTIONS["ai_auto"](toy_world())
    assert j.sum() == 0 and s.sum() == 0


def test_zero_weight_on_advice_equals_junior_alone():
    w = toy_world(junior=replace(JUNIOR, w_first=0.0))
    assert np.array_equal(ACTIONS["junior_advice_first"](w)[0], ACTIONS["junior_alone"](w)[0])


def test_full_weight_on_advice_equals_ai():
    w = toy_world(junior=replace(JUNIOR, w_first=1.0))
    assert np.array_equal(ACTIONS["junior_advice_first"](w)[0], ACTIONS["ai_auto"](w)[0])


def test_escalation_only_calls_senior_on_disagreement():
    w = toy_world()
    _, _, s = ACTIONS["escalate_on_disagreement"](w)
    disagree = (w.junior_own >= w.t) != (w.ai_logit >= w.t)
    assert np.array_equal(s.astype(bool), disagree)


def test_case_cost_values():
    flag = np.array([True, False, True, False])
    y = np.array([1, 1, 0, 0])
    assert case_cost(flag, y, COSTS).tolist() == [0, 5, 1, 0]


def test_lp_respects_constraints():
    names = list(ACTIONS)
    rng = np.random.default_rng(1)
    K = 5
    cost = rng.random((K, len(names)))
    sen = np.zeros((K, len(names)))
    sen[:, names.index("senior_alone")] = 1
    sen[:, names.index("senior_blind_first")] = 1
    jun = np.zeros_like(sen)
    for a in ["junior_alone", "junior_advice_first", "junior_blind_first", "escalate_on_disagreement"]:
        jun[:, names.index(a)] = 1
    n = np.full(K, 1 / K)
    out = solve(names, cost, sen, jun, n, senior_budget=0.1, junior_budget=0.5,
                min_oversight=0.5, no_auto_cells=[K - 1])
    x = np.array([[out["assignment"][k].get(a, 0) for a in names] for k in range(K)])
    assert out["status"] == "optimal"
    assert np.allclose(x.sum(1), 1, atol=1e-4)
    assert (n[:, None] * x * sen).sum() <= 0.1 + 1e-4
    assert (n[:, None] * x * jun).sum() <= 0.5 + 1e-4
    assert x[K - 1, names.index("ai_auto")] < 1e-4
    human = np.array([a != "ai_auto" for a in names])
    assert (n[:, None] * x[:, human]).sum() >= 0.5 - 1e-4


def test_apply_policy_follows_assignment():
    w = toy_world(n=20000)
    cell = np.zeros(len(w.y), int)
    _, _, _, chosen = apply_policy(w, cell, {0: {"ai_auto": 0.3, "junior_alone": 0.7}})
    assert abs(np.mean(chosen == "ai_auto") - 0.3) < 0.02


# --- symbolic (thesis method) ------------------------------------------------

def toy_bank(n=400):
    ws = [toy_world(n=n, seed=s) for s in range(3)]
    risk = 1 / (1 + np.exp(-ws[0].ai_logit))
    top = risk >= np.quantile(risk, 0.8)
    return S.make_bank(ws, risk, top, np.zeros(n, bool), case_cost), ws


def test_grammar_incomplete_and_complete():
    with pytest.raises(S.Incomplete):
        S.expand([0, 1])               # rule started, condition threshold given, no route yet
    p = S.expand([1, 0])               # "always: ai_auto"
    assert p["rules"] == [] and p["default"] == "ai_auto"
    p = S.expand([0, 3, 4, 1, 1])      # if disagree: senior_blind_first; else junior_advice_first
    assert p["rules"][0] == (("dis",), "senior_blind_first") and p["default"] == "junior_advice_first"


def test_always_route_matches_action():
    bank, ws = toy_bank()
    for k, route in enumerate(S.ROUTES):
        p = S.expand([1, k])
        _, cost, _, _, _ = S.evaluate(p, np.array([]), bank)
        ref = np.array([case_cost(ACTIONS[route](w)[0], w.y, w.costs) for w in ws])
        assert np.allclose(cost, ref)


def test_disagree_rule_commits_junior():
    bank, _ = toy_bank()
    p = S.expand([0, 3, 4, 1, 0])      # if disagree: senior; else ai_auto
    _, _, jun, _, committed = S.evaluate(p, np.array([]), bank)
    assert committed.all() and (jun == 1).all()


def test_governance_violation_detected():
    bank, _ = toy_bank()
    lim = {"senior": 1.0, "junior": 1.0, "oversight": 0.0}
    _, m = S.score(S.expand([1, 0]), np.array([]), bank, lim)   # automate everything
    assert m["auto_in_top_risk_pct"] == 100 and not m["feasible"]


def test_simplify_never_worse():
    bank, _ = toy_bank()
    lim = {"senior": 0.2, "junior": 0.6, "oversight": 0.3}
    p = S.expand([0, 0, 1, 1, 1, 0, 2, 0, 1, 1])
    loss, th = S.fit_thresholds(p, bank, lim)
    p2, th2 = S.simplify(p, th, bank, lim)
    assert S.score(p2, th2, bank, lim)[0] <= loss + 1e-12


def test_audit_condition_samples_requested_fraction():
    bank, _ = toy_bank(n=5000)
    p = S.expand([0, 5, 4, 1, 0])      # if random audit(w_0): senior_blind_first; else ai_auto
    assert p["rules"][0][0][0] == "audit"
    assigned, _, _, sen, _ = S.evaluate(p, np.array([0.2]), bank)
    assert abs(sen.mean() - 0.2) < 0.02
