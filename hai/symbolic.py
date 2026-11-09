"""Grammar-guided symbolic reinforcement learning for protocol synthesis.

This module transfers the method of the author's M.Sc. thesis
(Symbolic-RL-Fredholm-Solver: "Solving Second-Kind Fredholm Integral
Equations Using a Combination of Reinforcement Learning and Context-Free
Grammars") from mathematical expressions to human-AI interaction protocols.

                    thesis                         | this module
  ------------------------------------------------ | -------------------------------------------
  context-free grammar of expressions              | context-free grammar of protocols
  terminals x and learnable constants w_i          | conditions on AI risk, disagreement, AI
                                                   | uncertainty, with learnable thresholds w_i
  DQN picks production rules, state = padded       | same (5 actions, MLP 128-128, replay buffer,
  action sequence                                  | target network)
  inner LBFGS fit of the constants                 | inner coordinate search over the thresholds
                                                   | (the protocol cost is not differentiable)
  fitness = residual of the integral equation      | fitness = simulated decision cost + penalties
                                                   | for violated constraints
  reward = 1 / (fitness + 1e-5)                    | same
  output: interpretable closed-form solution       | output: interpretable, auditable protocol
  compared with PINN / PI-KAN (black box)          | compared with the LP optimum (randomised,
                                                   | not human-readable)

Grammar (every non-terminal reads the next action modulo its number of choices):

    <protocol> ::= <rule> ; <protocol>            (choice 0, at most MAX_RULES rules)
                 | else <route>                   (choice 1)
    <rule>     ::= if <cond> then <route>
    <cond>     ::= <cond> and <cond>              (0, depth limited)
                 | risk >= w_i                    (1)
                 | risk <  w_i                    (2)
                 | disagree                       (3)  junior's committed view differs from AI
                 | uncertain(w_i)                 (4)  |logit(risk) - threshold| < w_i
                 | audit(w_i)                     (5)  a random fraction w_i of cases (audit sampling)
    <route>    ::= ai_auto | junior_advice_first | junior_blind_first
                 | escalate_on_disagreement | senior_blind_first

Semantics. The protocol is a decision list. A case takes the route of the
first rule whose condition holds, otherwise the default route. Evaluating
`disagree` requires the junior to commit to an independent view, so it costs
a junior review for every case that reaches that rule, and a junior who has
already committed can no longer be shown the advice first (advice-first then
behaves as blind-first).
"""
import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

torch.set_num_threads(1)  # the network is tiny; one thread is fastest and avoids contention

from .protocols import ACTIONS

ROUTES = ["ai_auto", "junior_advice_first", "junior_blind_first",
          "escalate_on_disagreement", "senior_blind_first"]
N_ACTIONS = 6  # thesis: 5 production choices; one more here for the audit condition
MAX_RULES = 3
MAX_COND_DEPTH = 2
STATE_SIZE = 30          # thesis: fixed-length padded action sequence
INVALID_PENALTY = 100.0  # thesis: Invalid_Expr_Penalty
REG = 1e-5               # thesis: reward = 1 / (loss + 1e-5)


class Incomplete(Exception):
    pass


# ---------------------------------------------------------------------------
# Grammar expansion: action sequence -> program (nested tuples)
# ---------------------------------------------------------------------------

def expand(actions):
    a = list(actions)
    counter = [0]

    def take(n):
        if not a:
            raise Incomplete
        return a.pop(0) % n

    def route():
        return ROUTES[take(len(ROUTES))]

    def cond(depth):
        c = take(6)
        if c == 0 and depth < MAX_COND_DEPTH:
            return ("and", cond(depth + 1), cond(depth + 1))
        if c in (0, 1):
            counter[0] += 1
            return ("ge", counter[0] - 1)
        if c == 2:
            counter[0] += 1
            return ("lt", counter[0] - 1)
        if c == 3:
            return ("dis",)
        counter[0] += 1
        if c == 4:
            return ("unc", counter[0] - 1)
        return ("audit", counter[0] - 1)

    rules = []
    while True:
        c = take(2)
        if c == 0 and len(rules) < MAX_RULES:
            k = cond(0)
            rules.append((k, route()))
        else:
            default = route()
            break
    return {"rules": rules, "default": default, "n_params": counter[0]}


def param_kinds(program):
    kinds = {}

    def walk(c):
        if c[0] == "and":
            walk(c[1]), walk(c[2])
        elif c[0] in ("ge", "lt"):
            kinds[c[1]] = "risk"
        elif c[0] == "unc":
            kinds[c[1]] = "band"
        elif c[0] == "audit":
            kinds[c[1]] = "frac"
    for c, _ in program["rules"]:
        walk(c)
    return [kinds[i] for i in range(program["n_params"])]


def uses_dis(c):
    return c[0] == "dis" or (c[0] == "and" and (uses_dis(c[1]) or uses_dis(c[2])))


def to_text(program, theta=None):
    def p(i, kind):
        if theta is None:
            return f"w_{i}"
        return f"{theta[i]:.3g}" if kind in ("risk", "frac") else f"{theta[i]:.2f}"

    def c_txt(c):
        if c[0] == "and":
            return f"({c_txt(c[1])} and {c_txt(c[2])})"
        if c[0] == "ge":
            return f"AI risk >= {p(c[1], 'risk')}"
        if c[0] == "lt":
            return f"AI risk < {p(c[1], 'risk')}"
        if c[0] == "dis":
            return "junior disagrees with AI"
        if c[0] == "audit":
            return f"random audit (fraction {p(c[1], 'frac')})"
        return f"AI uncertain (band {p(c[1], 'band')})"
    lines = []
    for k, (c, r) in enumerate(program["rules"]):
        lines.append(f"{'if' if k == 0 else 'elif'} {c_txt(c)}: {r}")
    lines.append(f"{'else' if program['rules'] else 'always'}: {program['default']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Fast evaluation on precomputed simulation worlds
# ---------------------------------------------------------------------------

@dataclass
class Bank:
    """Everything needed to score any program on a set of worlds."""
    risk: np.ndarray        # (n,)
    dist: np.ndarray        # (n,) |ai_logit - threshold|
    dis: np.ndarray         # (R, n) bool
    cost: np.ndarray        # (R, A, n) per-route case cost
    jun: np.ndarray         # (R, A, n)
    sen: np.ndarray         # (R, A, n)
    cost_blind_jr: np.ndarray  # (R, n) cost of junior_blind_first (for committed juniors)
    top_mask: np.ndarray    # (n,) governance: highest-risk cells, no automation allowed
    unreliable_mask: np.ndarray  # (n,) epistemic: AI calibration not evidenced
    risk_grid: np.ndarray
    band_grid: np.ndarray
    u: np.ndarray           # (R, n) uniform draws for audit sampling
    frac_grid: np.ndarray


def make_bank(worlds, risk, top_mask, unreliable_mask, case_cost):
    R, n = len(worlds), len(risk)
    cost = np.zeros((R, len(ROUTES), n))
    jun = np.zeros_like(cost)
    sen = np.zeros_like(cost)
    dis = np.zeros((R, n), bool)
    cbj = np.zeros((R, n))
    for r, w in enumerate(worlds):
        for a, name in enumerate(ROUTES):
            f, j, s = ACTIONS[name](w)
            cost[r, a], jun[r, a], sen[r, a] = case_cost(f, w.y, w.costs), j, s
        dis[r] = (w.junior_own >= w.t) != (w.ai_logit >= w.t)
        cbj[r] = cost[r, ROUTES.index("junior_blind_first")]
    return Bank(risk=risk, dist=np.abs(worlds[0].ai_logit - worlds[0].t), dis=dis, cost=cost,
                jun=jun, sen=sen, cost_blind_jr=cbj, top_mask=top_mask,
                unreliable_mask=unreliable_mask,
                risk_grid=np.unique(np.quantile(risk, np.linspace(0.05, 0.95, 19))),
                band_grid=np.array([0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]),
                u=np.random.default_rng(31_337).random((R, n)),
                frac_grid=np.array([0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]))


def _cond_mask(c, bank, theta):
    if c[0] == "and":
        return _cond_mask(c[1], bank, theta) & _cond_mask(c[2], bank, theta)
    if c[0] == "ge":
        return np.broadcast_to(bank.risk >= theta[c[1]], bank.dis.shape)
    if c[0] == "lt":
        return np.broadcast_to(bank.risk < theta[c[1]], bank.dis.shape)
    if c[0] == "dis":
        return bank.dis
    if c[0] == "audit":
        return bank.u < theta[c[1]]
    return np.broadcast_to(bank.dist < theta[c[1]], bank.dis.shape)


def evaluate(program, theta, bank):
    """Return per-world arrays: assigned route, case cost, junior use, senior use."""
    R, n = bank.dis.shape
    assigned = np.full((R, n), -1)
    committed = np.zeros((R, n), bool)
    for c, route in program["rules"]:
        free = assigned == -1
        if uses_dis(c):
            committed |= free
        hit = free & _cond_mask(c, bank, theta)
        assigned[hit] = ROUTES.index(route)
    assigned[assigned == -1] = ROUTES.index(program["default"])
    ri = np.arange(R)[:, None]
    ci = np.arange(n)[None, :]
    cost = bank.cost[ri, assigned, ci]
    adv = ROUTES.index("junior_advice_first")
    cost = np.where(committed & (assigned == adv), bank.cost_blind_jr, cost)
    jun = np.maximum(bank.jun[ri, assigned, ci], committed)
    sen = bank.sen[ri, assigned, ci]
    return assigned, cost, jun, sen, committed


COMPLEXITY = 0.002  # per condition node: prefer the simpler of two equally good protocols


def n_nodes(program):
    def cnt(c):
        return 1 + (cnt(c[1]) + cnt(c[2]) if c[0] == "and" else 0)
    return sum(cnt(c) for c, _ in program["rules"])


def score(program, theta, bank, limits, lam=10.0):
    """Loss = mean decision cost per case + lam * constraint violations
    + a small complexity penalty per condition node (Occam's razor, as in
    symbolic regression)."""
    assigned, cost, jun, sen, committed = evaluate(program, theta, bank)
    auto = assigned == ROUTES.index("ai_auto")
    human = (~auto) | committed
    m = {
        "cost_per_100": 100 * cost.mean(),
        "senior_pct": 100 * sen.mean(),
        "junior_pct": 100 * jun.mean(),
        "human_in_loop_pct": 100 * human.mean(),
        "auto_in_top_risk_pct": 100 * auto[:, bank.top_mask].mean() if bank.top_mask.any() else 0.0,
        "auto_in_unreliable_pct": 100 * auto[:, bank.unreliable_mask].mean()
        if bank.unreliable_mask.any() else 0.0,
    }
    viol = (max(0, m["senior_pct"] / 100 - limits["senior"]) + max(0, m["junior_pct"] / 100 - limits["junior"])
            + max(0, limits["oversight"] - m["human_in_loop_pct"] / 100)
            + m["auto_in_top_risk_pct"] / 100 + m["auto_in_unreliable_pct"] / 100)
    m["violation"] = viol
    m["feasible"] = viol < 1e-9
    m["n_conditions"] = n_nodes(program)
    return cost.mean() + lam * viol + COMPLEXITY * n_nodes(program), m


def fit_thresholds(program, bank, limits, sweeps=2):
    """Inner optimisation of the learnable thresholds (thesis: LBFGS on the
    constants). Coordinate search over a grid, because the cost is piecewise
    constant in the thresholds."""
    kinds = param_kinds(program)
    init = {"risk": np.median(bank.risk_grid), "band": 0.75, "frac": 0.1}
    grids = {"risk": bank.risk_grid, "band": bank.band_grid, "frac": bank.frac_grid}
    theta = np.array([init[k] for k in kinds])
    best, _ = score(program, theta, bank, limits)
    for _ in range(sweeps):
        for i, k in enumerate(kinds):
            for v in grids[k]:
                t = theta.copy()
                t[i] = v
                s, _ = score(program, t, bank, limits)
                if s < best - 1e-12:
                    best, theta = s, t
    return best, theta


class Fitness:
    """Cached fitness of an action sequence (thesis: `fitness(action_sequence)`)."""

    def __init__(self, bank, limits):
        self.bank, self.limits, self.cache, self.evals = bank, limits, {}, 0

    def __call__(self, actions):
        try:
            prog = expand(actions)
        except Incomplete:
            return INVALID_PENALTY, None, None, False
        key = to_text(prog)
        if key not in self.cache:
            self.evals += 1
            loss, theta = fit_thresholds(prog, self.bank, self.limits)
            self.cache[key] = (loss, prog, theta)
        loss, prog, theta = self.cache[key]
        return loss, prog, theta, True


# ---------------------------------------------------------------------------
# DQN (architecture and update rule as in the thesis)
# ---------------------------------------------------------------------------

class DQN(nn.Module):
    def __init__(self, state_size=STATE_SIZE, action_size=N_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_size, 128), nn.ReLU(),
                                 nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, action_size))

    def forward(self, x):
        return self.net(x)


def pad(state):
    s = [a + 1 for a in state][-STATE_SIZE:]   # +1 so that padding (0) differs from action 0
    return s + [0] * (STATE_SIZE - len(s))


def search(bank, limits, episodes=600, max_steps=STATE_SIZE, seed=0, use_dqn=True,
           gamma=0.99, lr=1e-3, batch=64, eps_start=1.0, eps_end=0.05, target_every=15):
    """Run DQN-guided (use_dqn=True) or uniformly random (use_dqn=False)
    grammar search. Returns the best program found, its thresholds and a log."""
    rng = random.Random(seed)
    torch.manual_seed(seed)
    fit = Fitness(bank, limits)
    policy, target = DQN(), DQN()
    target.load_state_dict(policy.state_dict())
    opt = optim.Adam(policy.parameters(), lr=lr)
    memory = deque(maxlen=10_000)
    best = (np.inf, None, None)
    curve = []
    for ep in range(episodes):
        eps = eps_end + (eps_start - eps_end) * max(0.0, 1 - ep / (0.7 * episodes))
        state = []
        for step in range(max_steps):
            s = pad(state)
            if not use_dqn or rng.random() < eps:
                a = rng.randrange(N_ACTIONS)
            else:
                with torch.no_grad():
                    a = int(policy(torch.tensor([s], dtype=torch.float32)).argmax(1))
            state = state + [a]
            loss, prog, theta, complete = fit(state)
            reward = 1.0 / (loss + REG)  # incomplete programs get loss = INVALID_PENALTY, as in the thesis
            done = complete or step == max_steps - 1
            if complete and loss < best[0]:
                best = (loss, prog, theta)
            if use_dqn:
                memory.append((s, a, reward, pad(state), float(done)))
                if len(memory) >= batch:
                    b = rng.sample(memory, batch)
                    S, A, Rw, S2, D = map(lambda x: torch.tensor(x, dtype=torch.float32), zip(*b))
                    q = policy(S).gather(1, A.long().unsqueeze(1)).squeeze(1)
                    with torch.no_grad():
                        tgt = Rw + gamma * target(S2).max(1)[0] * (1 - D)
                    l = nn.functional.mse_loss(q, tgt)
                    opt.zero_grad()
                    l.backward()
                    opt.step()
            if done:
                break
        if use_dqn and ep % target_every == 0:
            target.load_state_dict(policy.state_dict())
        curve.append(best[0])
    return {"loss": best[0], "program": best[1], "theta": best[2],
            "text": to_text(best[1], best[2]) if best[1] else None,
            "best_curve": curve, "fitness_evaluations": fit.evals}


# ---------------------------------------------------------------------------
# Simplification of a found program (for readability; never makes it worse)
# ---------------------------------------------------------------------------

def _flatten(c):
    return _flatten(c[1]) + _flatten(c[2]) if c[0] == "and" else [c]


def simplify(program, theta, bank, limits):
    """Merge redundant conjuncts (risk >= a and risk >= b -> risk >= max(a, b)),
    then drop any rule whose removal does not increase the loss."""
    new_theta, rules = [], []

    def param(v):
        new_theta.append(float(v))
        return len(new_theta) - 1

    for c, route in program["rules"]:
        atoms = _flatten(c)
        ge = [theta[a[1]] for a in atoms if a[0] == "ge"]
        lt = [theta[a[1]] for a in atoms if a[0] == "lt"]
        unc = [theta[a[1]] for a in atoms if a[0] == "unc"]
        aud = [theta[a[1]] for a in atoms if a[0] == "audit"]
        parts = []
        if ge:
            parts.append(("ge", param(max(ge))))
        if lt:
            parts.append(("lt", param(min(lt))))
        if unc:
            parts.append(("unc", param(min(unc))))
        if aud:
            parts.append(("audit", param(min(aud))))
        if any(a[0] == "dis" for a in atoms):
            parts.append(("dis",))
        cond = parts[0]
        for p in parts[1:]:
            cond = ("and", cond, p)
        rules.append((cond, route))
    prog = {"rules": rules, "default": program["default"], "n_params": len(new_theta)}
    th = np.array(new_theta)
    best, _ = score(prog, th, bank, limits)
    changed = True
    while changed and prog["rules"]:
        changed = False
        for k in range(len(prog["rules"])):
            cand = {**prog, "rules": prog["rules"][:k] + prog["rules"][k + 1:]}
            s, _ = score(cand, th, bank, limits)
            if s <= best + 1e-12:
                prog, best, changed = cand, s, True
                break
    # if the last rule already catches every remaining case, it becomes the default
    while prog["rules"] and not uses_dis(prog["rules"][-1][0]):
        R, n = bank.dis.shape
        free = np.ones((R, n), bool)
        for c, _ in prog["rules"][:-1]:
            free &= ~_cond_mask(c, bank, th) if not uses_dis(c) else free
        last = prog["rules"][-1]
        if np.all(_cond_mask(last[0], bank, th)[free]):
            cand = {**prog, "rules": prog["rules"][:-1], "default": last[1]}
            if score(cand, th, bank, limits)[0] <= best + 1e-12:
                prog, best = cand, score(cand, th, bank, limits)[0]
                continue
        break
    return prog, th
