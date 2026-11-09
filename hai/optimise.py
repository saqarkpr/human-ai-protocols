"""Protocol optimisation as a linear program (CVXPY).

Patients are grouped into cells by the AI's risk decile (something known
before any human looks at the case). For every cell k and routing action a
the simulation on the optimisation split gives:

    cost[k, a]    expected decision cost per patient
    senior[k, a]  expected senior reviews per patient
    junior[k, a]  expected junior reviews per patient

The policy x[k, a] is the probability of routing a patient in cell k to
action a. The LP is

    minimise    sum_k n_k sum_a x[k,a] cost[k,a]
    subject to  sum_a x[k,a] = 1,  x >= 0                    (valid policy)
                sum_k n_k sum_a x[k,a] senior[k,a] <= B_s    (senior capacity)
                sum_k n_k sum_a x[k,a] junior[k,a] <= B_j    (junior capacity)
                sum_k n_k sum_{a human} x[k,a]     >= H_min  (human oversight)
                x[k, ai_auto] = 0 for the top risk deciles    (governance: no
                                 automated decision for the highest-risk patients)
                x[k, ai_auto] = 0 where the AI is not shown to be calibrated
                                 in that cell                  (epistemic: automate
                                 only where reliability is evidenced)

The dual variable of the senior-capacity constraint is reported: it says
how much expected cost would fall per extra unit of senior capacity, which
is the kind of quantity a hospital could actually act on.
"""
import cvxpy as cp
import numpy as np

from .protocols import ACTIONS, HUMAN_ACTIONS, patient_cost


def risk_cells(ai_risk, cuts=None, n_cells=10):
    """Cells = AI risk quantile bins. Calibrated (isotonic) risks contain ties,
    so duplicate cut points are dropped and the number of cells can be < n_cells."""
    if cuts is None:
        cuts = np.unique(np.quantile(ai_risk, np.linspace(0, 1, n_cells + 1)[1:-1]))
        # digitize with right=True puts ties on the lower side; drop cuts that
        # would leave an empty bin
        cell = np.digitize(ai_risk, cuts, right=True)
        keep = [k for k in range(len(cuts)) if np.any(cell == k + 1)]
        cuts = cuts[keep]
    return np.digitize(ai_risk, cuts, right=True), cuts


def cell_statistics(worlds, cell, n_cells):
    """Average cost/resource use per (cell, action) over simulation seeds."""
    names = list(ACTIONS)
    cost = np.zeros((n_cells, len(names)))
    sen = np.zeros_like(cost)
    jun = np.zeros_like(cost)
    for w in worlds:
        for a_i, a in enumerate(names):
            flag, j, s = ACTIONS[a](w)
            pc = patient_cost(flag, w.y)
            for k in range(n_cells):
                m = cell == k
                cost[k, a_i] += pc[m].mean()
                sen[k, a_i] += s[m].mean()
                jun[k, a_i] += j[m].mean()
    r = len(worlds)
    return names, cost / r, sen / r, jun / r


def calibration_gap_per_cell(ai_risk, y, cell, n_cells):
    return np.array([abs(ai_risk[cell == k].mean() - y[cell == k].mean()) for k in range(n_cells)])


def solve(names, cost, sen, jun, n_frac, senior_budget, junior_budget=1.0,
          min_oversight=0.0, no_auto_cells=(), solver="CLARABEL"):
    K, A = cost.shape
    x = cp.Variable((K, A), nonneg=True)
    n = n_frac.reshape(-1, 1)
    human = np.array([1.0 if a in HUMAN_ACTIONS else 0.0 for a in names])
    auto = names.index("ai_auto")
    cons = [cp.sum(x, axis=1) == 1]
    c_sen = cp.sum(cp.multiply(n, cp.multiply(x, sen))) <= senior_budget
    cons += [c_sen,
             cp.sum(cp.multiply(n, cp.multiply(x, jun))) <= junior_budget,
             cp.sum(cp.multiply(n, x @ np.diag(human))) >= min_oversight]
    for k in no_auto_cells:
        cons.append(x[k, auto] == 0)
    prob = cp.Problem(cp.Minimize(cp.sum(cp.multiply(n, cp.multiply(x, cost)))), cons)
    prob.solve(solver=solver)
    if x.value is None:  # infeasible: the constraints cannot all hold at once
        return {"status": prob.status, "expected_cost_per_100": None,
                "assignment": None, "senior_shadow_price": None}
    xv = np.clip(x.value, 0, None)
    xv = xv / xv.sum(axis=1, keepdims=True)
    assignment = {k: {names[a]: float(xv[k, a]) for a in range(A) if xv[k, a] > 1e-4}
                  for k in range(K)}
    return {
        "status": prob.status,
        "expected_cost_per_100": 100 * float(prob.value),
        "assignment": assignment,
        "senior_shadow_price": float(c_sen.dual_value),
    }
