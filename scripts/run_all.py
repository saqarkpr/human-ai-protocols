"""Run every experiment for one or both domains and write results/<domain>/.

    python scripts/run_all.py --domain icu
    python scripts/run_all.py --domain cyber
    python scripts/run_all.py --domain all          # both (about 15 minutes on CPU)

Stages (per domain)
  1. Agents. Train the AI (without the human's private information) and the
     human cue model on the training split.
  2. Fixed protocols on the held-out evaluation split (30 simulation seeds).
  3. Optimal routing by linear program (CVXPY) on the optimisation split,
     evaluated on the held-out split, plus a sweep over the senior budget.
  4. Symbolic protocol synthesis with the thesis method (grammar + DQN), against
     a random-search control with the same number of episodes, 5 seeds each.
  5. Sensitivity map of complementarity (clinician inconsistency x reliance).
  6. Ablation. The AI also receives the human's private information.
  7. Stability under model updates (logistic regression, uncalibrated GBT).
  8. Causal analysis of confounded interaction logs (pgmpy back-door).
"""
import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kendalltau, wilcoxon
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hai import causal, symbolic  # noqa: E402
from hai.agents import AIAgent, CueModel, JUNIOR, SENIOR, logit  # noqa: E402
from hai.domains import DOMAINS  # noqa: E402
from hai.optimise import calibration_gap_per_cell, cell_statistics, risk_cells, solve  # noqa: E402
from hai.protocols import (ACTIONS, HUMAN_ACTIONS, apply_policy, case_cost,  # noqa: E402
                           confidence_deferral, make_world, summarise)

R_OPT, R_EVAL, N_BOOT = 20, 30, 2000
N_CELLS = 10
LIMITS = {"senior": 0.10, "junior": 0.50, "oversight": 0.50}
CALIB_GAP_MAX = 0.05
TOP_RISK_NO_AUTO = 2
SYM_SEEDS, SYM_EPISODES, SYM_WORLDS = 5, 800, 10
SOLO = ["ai_auto", "junior_alone", "senior_alone"]


def ece(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return float(sum(abs(p[idx == b].mean() - y[idx == b].mean()) * np.mean(idx == b)
                     for b in range(bins) if np.any(idx == b)))


def boot_ci(x, n=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    m = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)])
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def setup(D, ai_kind="gbt_cal", private=False):
    s = D.splits()
    ai = AIAgent(ai_kind).fit(D.ai_features(s["train"], private), D.label(s["train"]))
    cues = CueModel().fit(D.cues(s["train"]), D.label(s["train"]))
    parts = {}
    for name in ("opt", "eval"):
        df = s[name]
        parts[name] = {"df": df, "y": D.label(df), "ai_risk": ai.risk(D.ai_features(df, private)),
                       "cue_logit": cues.cue_logit(D.cues(df))}
    return parts


def worlds(D, part, n, offset, junior=JUNIOR, senior=SENIOR):
    return [make_world(part["y"], part["ai_risk"], part["cue_logit"], offset + r, D.costs, junior, senior)
            for r in range(n)]


def pick_band(ws, target=0.30):
    return float(np.quantile(np.abs(ws[0].ai_logit - ws[0].t), target))


def fixed_protocols(band):
    p = dict(ACTIONS)
    p["confidence_deferral"] = lambda w: confidence_deferral(w, band)
    return p


def evaluate_protocols(ws, protos):
    out, per_case = {}, {}
    for name, f in protos.items():
        rows, pc = [], np.zeros(len(ws[0].y))
        for w in ws:
            flag, j, s = f(w)
            rows.append(summarise(flag, j, s, w.y, w.costs))
            pc += case_cost(flag, w.y, w.costs)
        out[name] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        out[name]["cost_seed_sd"] = float(np.std([r["cost_per_100"] for r in rows]))
        per_case[name] = pc / len(ws)
    return out, per_case


def cells_and_masks(parts):
    opt, ev = parts["opt"], parts["eval"]
    cell_opt, cuts = risk_cells(opt["ai_risk"], n_cells=N_CELLS)
    cell_ev, _ = risk_cells(ev["ai_risk"], cuts=cuts)
    K = len(cuts) + 1
    gap = calibration_gap_per_cell(opt["ai_risk"], opt["y"], cell_opt, K)
    no_auto = sorted(set(range(K - TOP_RISK_NO_AUTO, K)) | set(np.where(gap > CALIB_GAP_MAX)[0].tolist()))
    top = list(range(K - TOP_RISK_NO_AUTO, K))
    unreliable = sorted(set(np.where(gap > CALIB_GAP_MAX)[0].tolist()) - set(top))
    return dict(cell_opt=cell_opt, cell_ev=cell_ev, cuts=cuts, K=K, gap=gap, no_auto=no_auto,
                top=top, unreliable=unreliable)


def optimise_and_eval(parts, opt_ws, eval_ws, senior_budget, cm):
    names, cost, sen, jun = cell_statistics(opt_ws, cm["cell_opt"], cm["K"])
    n_frac = np.bincount(cm["cell_opt"], minlength=cm["K"]) / len(cm["cell_opt"])
    sol = solve(names, cost, sen, jun, n_frac, senior_budget, LIMITS["junior"], LIMITS["oversight"],
                cm["no_auto"])
    if sol["assignment"] is None:
        return sol, None, None
    rows, pc = [], np.zeros(len(parts["eval"]["y"]))
    for r, w in enumerate(eval_ws):
        flag, j, s, chosen = apply_policy(w, cm["cell_ev"], sol["assignment"], seed=r)
        rows.append(summarise(flag, j, s, w.y, w.costs, np.isin(chosen, list(HUMAN_ACTIONS))))
        pc += case_cost(flag, w.y, w.costs)
    return sol, {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}, pc / len(eval_ws)


def subset(part, idx):
    return {"df": part["df"].iloc[idx].reset_index(drop=True), "y": part["y"][idx],
            "ai_risk": part["ai_risk"][idx], "cue_logit": part["cue_logit"][idx]}


def evidence_experiment(D, parts, opt_ws):
    """Where should the evidence used to design a protocol come from?
    The held-out split is halved. Half E1 plays the role of evidence collected
    under deployment conditions, half E2 is the test. The LP is optimised once
    on the in-distribution optimisation split and once on E1, and both are
    evaluated on E2, next to fixed protocols."""
    ev = parts["eval"]
    idx = np.random.default_rng(123).permutation(len(ev["y"]))
    e1, e2 = subset(ev, idx[: len(idx) // 2]), subset(ev, idx[len(idx) // 2:])
    ws1, ws2 = worlds(D, e1, R_OPT, 7000), worlds(D, e2, R_EVAL, 8000)
    out = {}
    pcs = {}
    for name, p, w_opt in [("in_distribution_evidence", {"opt": parts["opt"], "eval": e2}, opt_ws),
                           ("deployment_evidence", {"opt": e1, "eval": e2}, ws1)]:
        cmx = cells_and_masks(p)
        sol, m, pc = optimise_and_eval(p, w_opt, ws2, LIMITS["senior"], cmx)
        out[name] = {"status": sol["status"], "eval": m, "no_auto_cells": cmx["no_auto"],
                     "assignment": sol["assignment"]}
        pcs[name] = pc
    fm, fpc = evaluate_protocols(ws2, {k: ACTIONS[k] for k in ["ai_auto", "escalate_on_disagreement",
                                                               "senior_blind_first"]})
    out["fixed_on_test_half"] = {k: v["cost_per_100"] for k, v in fm.items()}
    if all(v is not None for v in pcs.values()):
        d = 100 * (pcs["deployment_evidence"] - pcs["in_distribution_evidence"])
        out["deployment_minus_in_distribution"] = float(d.mean())
        out["deployment_minus_in_distribution_ci95"] = boot_ci(d)
        for k in ["ai_auto", "escalate_on_disagreement"]:
            for name in pcs:
                dd = 100 * (pcs[name] - fpc[k])
                out[f"{name}_minus_{k}"] = float(dd.mean())
                out[f"{name}_minus_{k}_ci95"] = boot_ci(dd)
    out["ai_missed_positive_rate"] = {
        "opt": float(np.mean(parts["opt"]["ai_risk"][parts["opt"]["y"] == 1] < D.costs.threshold)),
        "eval": float(np.mean(ev["ai_risk"][ev["y"] == 1] < D.costs.threshold))}
    return out


def complementarity(m, team, solo):
    return m[team]["cost_per_100"] - min(m["ai_auto"]["cost_per_100"], m[solo]["cost_per_100"])


def comp_block(m):
    return {"junior_advice_first": complementarity(m, "junior_advice_first", "junior_alone"),
            "junior_blind_first": complementarity(m, "junior_blind_first", "junior_alone"),
            "senior_blind_first": complementarity(m, "senior_blind_first", "senior_alone")}


def make_symbolic_bank(D, part, ws, cells, cell_name):
    cell = cells[cell_name]
    return symbolic.make_bank(ws, part["ai_risk"], np.isin(cell, cells["top"]),
                              np.isin(cell, cells["unreliable"]), case_cost)


def run_domain(name):
    D = DOMAINS[name]
    out_dir = ROOT / "results" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    res = {"domain": name, "case": D.case, "positive": D.positive, "human_private": D.human_private,
           "costs": {"fn": D.costs.fn, "fp": D.costs.fp, "threshold": D.costs.threshold},
           "config": {"R_opt": R_OPT, "R_eval": R_EVAL, "limits": LIMITS, "calib_gap_max": CALIB_GAP_MAX,
                      "top_risk_deciles_no_auto": TOP_RISK_NO_AUTO, "junior": JUNIOR.__dict__,
                      "senior": SENIOR.__dict__, "symbolic": {"seeds": SYM_SEEDS, "episodes": SYM_EPISODES,
                                                              "worlds": SYM_WORLDS}}}
    s = D.splits()
    res["sizes"] = {k: int(len(v)) for k, v in s.items()}
    res["prevalence"] = {k: float(D.label(v).mean()) for k, v in s.items()}

    # 1. agents
    parts = setup(D)
    ev = parts["eval"]
    opt_ws, eval_ws = worlds(D, parts["opt"], R_OPT, 0), worlds(D, ev, R_EVAL, 1000)
    res["agents_eval"] = {
        "ai_auroc": float(roc_auc_score(ev["y"], ev["ai_risk"])), "ai_ece": ece(ev["ai_risk"], ev["y"]),
        "human_cue_model_auroc": float(roc_auc_score(ev["y"], ev["cue_logit"])),
        "junior_own_auroc": float(np.mean([roc_auc_score(w.y, w.junior_own) for w in eval_ws])),
        "senior_own_auroc": float(np.mean([roc_auc_score(w.y, w.senior_own) for w in eval_ws])),
        "corr_ai_vs_human_cues": float(np.corrcoef(logit(ev["ai_risk"]), ev["cue_logit"])[0, 1]),
    }
    print(name, "agents", res["agents_eval"])

    # 2. fixed protocols
    band = pick_band(opt_ws)
    metrics, per_case = evaluate_protocols(eval_ws, fixed_protocols(band))
    # best individual agent = cheapest of the three solo protocols (AI, junior, senior)
    ref = min(SOLO, key=lambda k: metrics[k]["cost_per_100"])
    for k in metrics:
        d = 100 * (per_case[k] - per_case["ai_auto"])
        metrics[k]["diff_vs_ai_auto"] = float(d.mean())
        metrics[k]["diff_vs_ai_auto_ci95"] = boot_ci(d)
        d = 100 * (per_case[k] - per_case[ref])
        metrics[k]["diff_vs_best_individual"] = float(d.mean())
        metrics[k]["diff_vs_best_individual_ci95"] = boot_ci(d)
    res["best_individual"] = ref
    res["confidence_deferral_band_logit"] = band
    res["fixed_protocols"] = metrics
    res["complementarity"] = comp_block(metrics)

    # 3. LP
    cm = cells_and_masks(parts)
    sol, lp_m, lp_pc = optimise_and_eval(parts, opt_ws, eval_ws, LIMITS["senior"], cm)
    for other in dict.fromkeys(["escalate_on_disagreement", "confidence_deferral", "ai_auto", ref]):
        d = 100 * (lp_pc - per_case[other])
        lp_m[f"diff_vs_{other}"] = float(d.mean())
        lp_m[f"diff_vs_{other}_ci95"] = boot_ci(d)
    m_opt, _ = evaluate_protocols(opt_ws, {k: ACTIONS[k] for k in ["escalate_on_disagreement", "ai_auto"]})
    res["lp"] = {"solution": sol, "eval": lp_m, "n_cells": cm["K"], "no_auto_cells": cm["no_auto"],
                 "calibration_gap_per_cell": cm["gap"].tolist(), "cell_upper_bounds": cm["cuts"].tolist(),
                 "cohort_reference": {"escalate_opt": m_opt["escalate_on_disagreement"]["cost_per_100"],
                                      "escalate_eval": metrics["escalate_on_disagreement"]["cost_per_100"],
                                      "ai_auto_opt": m_opt["ai_auto"]["cost_per_100"],
                                      "ai_auto_eval": metrics["ai_auto"]["cost_per_100"]}}
    sweep = []
    for b in [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]:
        s_b, m_b, _ = optimise_and_eval(parts, opt_ws, eval_ws, b, cm)
        sweep.append({"senior_budget": b, "status": s_b["status"],
                      "expected_opt": s_b["expected_cost_per_100"],
                      "eval_cost": m_b["cost_per_100"] if m_b else None,
                      "eval_senior_pct": m_b["senior_reviews_pct"] if m_b else None})
    res["senior_budget_sweep"] = sweep
    print(name, "LP eval", round(lp_m["cost_per_100"], 2), "vs", ref, round(metrics[ref]["cost_per_100"], 2))

    res["evidence_experiment"] = evidence_experiment(D, parts, opt_ws)
    print(name, "evidence", {k: v for k, v in res["evidence_experiment"].items() if "assignment" not in str(k)})

    # 4. symbolic synthesis (thesis method)
    bank_opt = make_symbolic_bank(D, parts["opt"], opt_ws[:SYM_WORLDS], cm, "cell_opt")
    bank_ev = make_symbolic_bank(D, ev, eval_ws, cm, "cell_ev")
    sym = {"dqn": [], "random": []}
    for method in ("dqn", "random"):
        for seed in range(SYM_SEEDS):
            r = symbolic.search(bank_opt, LIMITS, episodes=SYM_EPISODES, seed=seed, use_dqn=(method == "dqn"))
            prog, th = symbolic.simplify(r["program"], r["theta"], bank_opt, LIMITS)
            loss_opt, m_opt_s = symbolic.score(prog, th, bank_opt, LIMITS)
            _, m_ev = symbolic.score(prog, th, bank_ev, LIMITS)
            _, cost_ev, _, _, _ = symbolic.evaluate(prog, th, bank_ev)
            pc = cost_ev.mean(0)
            sym[method].append({"method": method, "seed": seed, "loss_opt": float(loss_opt), "opt": m_opt_s, "eval": m_ev,
                                "text": symbolic.to_text(prog, th), "fitness_evaluations": r["fitness_evaluations"],
                                "best_curve": r["best_curve"],
                                "diff_vs_lp": float(100 * (pc - lp_pc).mean()),
                                "diff_vs_lp_ci95": boot_ci(100 * (pc - lp_pc)),
                                "diff_vs_ai_auto": float(100 * (pc - per_case["ai_auto"]).mean()),
                                "diff_vs_ai_auto_ci95": boot_ci(100 * (pc - per_case["ai_auto"])),
                                "diff_vs_best_individual": float(100 * (pc - per_case[ref]).mean()),
                                "diff_vs_best_individual_ci95": boot_ci(100 * (pc - per_case[ref]))})
            print(name, method, seed, round(loss_opt, 4), round(m_ev["cost_per_100"], 2), m_ev["feasible"])
    dl = np.array([x["loss_opt"] for x in sym["dqn"]])
    rl = np.array([x["loss_opt"] for x in sym["random"]])
    best = min(sym["dqn"] + sym["random"], key=lambda x: x["loss_opt"])
    sym["summary"] = {
        "dqn_loss_opt_mean": float(dl.mean()), "random_loss_opt_mean": float(rl.mean()),
        "dqn_better_seeds": int(np.sum(dl < rl - 1e-12)), "ties": int(np.sum(np.abs(dl - rl) <= 1e-12)),
        "wilcoxon_p": float(wilcoxon(dl, rl).pvalue) if np.any(np.abs(dl - rl) > 1e-12) else 1.0,
        "selected_text": best["text"], "selected_eval": best["eval"], "selected_seed_method":
            f"{best['method']} seed {best['seed']}",
        "selected_diff_vs_lp": best["diff_vs_lp"], "selected_diff_vs_lp_ci95": best["diff_vs_lp_ci95"],
        "selected_diff_vs_ai_auto": best["diff_vs_ai_auto"],
        "selected_diff_vs_ai_auto_ci95": best["diff_vs_ai_auto_ci95"],
        "selected_diff_vs_best_individual": best["diff_vs_best_individual"],
        "selected_diff_vs_best_individual_ci95": best["diff_vs_best_individual_ci95"],
        "eval_cost_dqn_mean": float(np.mean([x["eval"]["cost_per_100"] for x in sym["dqn"]])),
        "eval_cost_random_mean": float(np.mean([x["eval"]["cost_per_100"] for x in sym["random"]])),
        "feasible_eval_dqn": int(sum(x["eval"]["feasible"] for x in sym["dqn"])),
        "feasible_eval_random": int(sum(x["eval"]["feasible"] for x in sym["random"])),
    }
    res["symbolic"] = sym
    print(name, "symbolic", {k: v for k, v in sym["summary"].items() if k != "selected_text"})
    print(sym["summary"]["selected_text"])

    # 5. sensitivity
    sigmas, wgrid = [0.25, 0.5, 1.0, 1.5, 2.0], [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    grid = np.zeros((len(wgrid), len(sigmas)))
    for i, wv in enumerate(wgrid):
        for j, sg in enumerate(sigmas):
            jr = replace(JUNIOR, sigma=sg, w_first=wv)
            gws = worlds(D, ev, 10, 5000, jr, SENIOR)
            m, _ = evaluate_protocols(gws, {"ai_auto": ACTIONS["ai_auto"], "junior_alone": ACTIONS["junior_alone"],
                                            "team": ACTIONS["junior_advice_first"]})
            grid[i, j] = m["team"]["cost_per_100"] - min(m["ai_auto"]["cost_per_100"], m["junior_alone"]["cost_per_100"])
    res["sensitivity"] = {"sigmas": sigmas, "weights_on_advice": wgrid, "team_minus_best_individual": grid.tolist()}

    # 6. ablation
    parts_p = setup(D, private=True)
    m_p, _ = evaluate_protocols(worlds(D, parts_p["eval"], R_EVAL, 1000), fixed_protocols(band))
    res["ablation_ai_gets_private_info"] = {
        "ai_auroc": float(roc_auc_score(parts_p["eval"]["y"], parts_p["eval"]["ai_risk"])),
        "corr_ai_vs_human_cues": float(np.corrcoef(logit(parts_p["eval"]["ai_risk"]),
                                                   parts_p["eval"]["cue_logit"])[0, 1]),
        "complementarity": comp_block(m_p), "protocol_costs": {k: v["cost_per_100"] for k, v in m_p.items()}}

    # 7. model updates
    res["model_updates"] = {}
    names = sorted(metrics)
    for kind in ["logreg", "gbt"]:
        pu = setup(D, kind)
        wu = worlds(D, pu["eval"], R_EVAL, 1000)
        mu, _ = evaluate_protocols(wu, fixed_protocols(pick_band(worlds(D, pu["opt"], 1, 0))))
        tau, p_tau = kendalltau([metrics[n]["cost_per_100"] for n in names], [mu[n]["cost_per_100"] for n in names])
        res["model_updates"][kind] = {
            "ai_auroc": float(roc_auc_score(pu["eval"]["y"], pu["eval"]["ai_risk"])),
            "ai_ece": ece(pu["eval"]["ai_risk"], pu["eval"]["y"]),
            "protocol_costs": {k: v["cost_per_100"] for k, v in mu.items()},
            "complementarity": comp_block(mu), "kendall_tau_vs_main": float(tau), "kendall_p": float(p_tau)}

    # 8. causal
    sev, cuts = D.severity_bins(parts["opt"]["df"])
    sev_ev, _ = D.severity_bins(ev["df"], cuts)
    n_levels = len(cuts) + 1
    rows = []
    for r, w in enumerate(eval_ws):
        log, truth = causal.simulate_log(w, sev_ev, n_levels, seed=r)
        est = causal.estimate(log, D.costs)
        rows.append({"true": truth, **est, "senior_share": float(log.Review.mean())})
    res["causal"] = {"adjustment_set": rows[0]["adjustment_set"], "runs": len(rows), "severity_levels": n_levels,
                     "senior_share_mean": float(np.mean([r["senior_share"] for r in rows])), "per_run": rows}
    for q in ["cost", "missed_death"]:
        tr = np.array([r["true"][q] for r in rows])
        nv = np.array([r["naive"][q] for r in rows])
        bd = np.array([r["backdoor"][q] for r in rows])
        res["causal"][q] = {"true_mean": float(tr.mean()), "naive_mean": float(nv.mean()),
                            "backdoor_mean": float(bd.mean()),
                            "naive_abs_error_mean": float(np.abs(nv - tr).mean()),
                            "backdoor_abs_error_mean": float(np.abs(bd - tr).mean()),
                            "naive_wrong_sign_runs": int(np.sum(np.sign(nv) != np.sign(tr))),
                            "backdoor_wrong_sign_runs": int(np.sum(np.sign(bd) != np.sign(tr)))}
    print(name, "causal", res["causal"]["cost"])

    res["runtime_s"] = time.time() - t0
    (out_dir / "results.json").write_text(json.dumps(res, indent=2, default=float))
    figures(res, out_dir)
    return res


def figures(res, out):
    plt.rcParams.update({"font.size": 9})
    title = {"icu": "ICU (PhysioNet 2012)", "cyber": "Cybersecurity (NSL-KDD)"}[res["domain"]]
    m = res["fixed_protocols"]
    order = sorted(m, key=lambda k: m[k]["cost_per_100"])
    vals = [m[k]["cost_per_100"] for k in order]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.barh(order, vals, xerr=[m[k]["cost_seed_sd"] for k in order], color="#4C72B0")
    ax.axvline(res["lp"]["eval"]["cost_per_100"], color="#C44E52", ls="--", label="optimised (LP)")
    ax.axvline(res["symbolic"]["summary"]["selected_eval"]["cost_per_100"], color="#55A868", ls=":",
               lw=2, label="symbolic protocol (thesis method)")
    ax.invert_yaxis()
    ax.set_xlim(min(vals) * 0.85, max(vals) * 1.05)
    ax.set_xlabel(f"decision cost per 100 {res['case']}s (held-out, mean ± sd over seeds)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "protocol_costs.png", dpi=160)
    plt.close(fig)

    s = res["sensitivity"]
    g = np.array(s["team_minus_best_individual"])
    fig, ax = plt.subplots(figsize=(5, 3.8))
    lim = np.abs(g).max()
    im = ax.imshow(g, cmap="RdBu_r", vmin=-lim, vmax=lim, origin="lower", aspect="auto")
    ax.set_xticks(range(len(s["sigmas"])), s["sigmas"])
    ax.set_yticks(range(len(s["weights_on_advice"])), s["weights_on_advice"])
    ax.set_xlabel("human inconsistency sigma (logit)")
    ax.set_ylabel("weight on AI advice w")
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            ax.text(j, i, f"{g[i, j]:+.1f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, label="team cost - best individual\n(blue = complementarity)")
    ax.set_title(f"{title}: where complementarity exists")
    fig.tight_layout()
    fig.savefig(out / "complementarity_map.png", dpi=160)
    plt.close(fig)

    sw = [r for r in res["senior_budget_sweep"] if r["eval_cost"] is not None]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot([100 * r["senior_budget"] for r in sw], [r["expected_opt"] for r in sw], "o--", label="LP expected (optimisation split)")
    ax.plot([100 * r["senior_budget"] for r in sw], [r["eval_cost"] for r in sw], "s-", label="realised (held-out)")
    ax.set_xlabel("senior review budget (% of cases)")
    ax.set_ylabel(f"cost per 100 {res['case']}s")
    ax.set_title(title)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "senior_budget_frontier.png", dpi=160)
    plt.close(fig)

    c = res["causal"]["per_run"]
    tr, nv, bd = ([r["true"]["cost"] for r in c], [r["naive"]["cost"] for r in c],
                  [r["backdoor"]["cost"] for r in c])
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.scatter(tr, nv, label="naive comparison", alpha=.7)
    ax.scatter(tr, bd, label="back-door (pgmpy do-query)", alpha=.7)
    lo, hi = min(tr + nv + bd), max(tr + nv + bd)
    ax.plot([lo, hi], [lo, hi], "k:", lw=1, label="perfect estimate")
    ax.axhline(0, color="grey", lw=.5)
    ax.set_xlabel(f"true effect of senior review\n(change in cost per 100 {res['case']}s)")
    ax.set_ylabel("estimated effect from logs")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(out / "causal_estimates.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.2))
    for method, col in (("dqn", "#C44E52"), ("random", "#8172B2")):
        curves = np.array([x["best_curve"] for x in res["symbolic"][method]])
        curves = np.minimum(curves, 2.0)
        mu = 100 * curves.mean(0)
        ax.plot(mu, color=col, label=f"{'DQN (thesis method)' if method == 'dqn' else 'random grammar search'}")
        ax.fill_between(range(curves.shape[1]), 100 * curves.min(0), 100 * curves.max(0), color=col, alpha=.15)
    ax.axhline(res["lp"]["solution"]["expected_cost_per_100"], color="k", ls="--", lw=1, label="LP optimum (opt split)")
    ax.set_ylim(res["lp"]["solution"]["expected_cost_per_100"] * 0.95, res["lp"]["solution"]["expected_cost_per_100"] * 1.35)
    ax.set_xlabel("episode")
    ax.set_ylabel("best loss found (x100)")
    ax.set_title(f"{title}: symbolic protocol search")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "symbolic_search.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="all", choices=["icu", "cyber", "all"])
    a = ap.parse_args()
    for d in (["icu", "cyber"] if a.domain == "all" else [a.domain]):
        run_domain(d)
    print("done")
