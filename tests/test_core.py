"""Sanity checks: run with `python -m pytest -q`."""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hai.agents import JUNIOR, SENIOR, logit  # noqa: E402
from hai.optimise import solve  # noqa: E402
from hai.protocols import (ACTIONS, T_LOGIT, THRESHOLD, World, apply_policy,  # noqa: E402
                           patient_cost)


def toy_world(n=500, seed=0, junior=JUNIOR):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.15).astype(int)
    ai = rng.normal(0, 1.5, n) + 1.5 * y - 1.5
    return World(y=y, ai_logit=ai, junior_own=ai + rng.normal(0, 1, n),
                 senior_own=ai + rng.normal(0, .5, n), junior=junior, senior=SENIOR)


def test_threshold_is_cost_optimal():
    assert abs(THRESHOLD - 1 / 6) < 1e-12
    assert abs(T_LOGIT - logit(np.array([1 / 6]))[0]) < 1e-12


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
    disagree = (w.junior_own >= T_LOGIT) != (w.ai_logit >= T_LOGIT)
    assert np.array_equal(s.astype(bool), disagree)


def test_patient_cost_values():
    flag = np.array([True, False, True, False])
    y = np.array([1, 1, 0, 0])
    assert patient_cost(flag, y).tolist() == [0, 5, 1, 0]


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
