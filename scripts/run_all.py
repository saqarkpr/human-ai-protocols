"""Run every experiment and write results/ and figures.

    python scripts/run_all.py            # full run (about 2 to 4 minutes on CPU)

Stages
  1. Train the AI (real ICU data, no neurological exam) and the clinician cue model.
  2. Evaluate fixed interaction protocols on held-out Set B (30 simulation seeds).
  3. Optimise a routing policy with a linear program on the optimisation split,
     then evaluate it on Set B. Sweep the senior-review budget.
  4. Sensitivity: where does complementarity exist, as a function of the
     clinician's inconsistency and weight on AI advice?
  5. Ablation: give the AI the neurological exam too (no information asymmetry).
  6. Stability under a model update: swap the AI from gradient-boosted trees to
     logistic regression and check whether the protocol ranking survives.
  7. Causal: naive vs back-door estimate of the effect of senior review from
     confounded observational logs, against the simulated ground truth.
"""
import json
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kendalltau
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hai import data as D  # noqa: E402
from hai.agents import AIAgent, CueModel, JUNIOR, SENIOR, logit  # noqa: E402
from hai.protocols import (ACTIONS, HUMAN_ACTIONS, T_LOGIT, apply_policy,  # noqa: E402
                           confidence_deferral, make_world, patient_cost, summarise)
from hai.optimise import calibration_gap_per_cell, cell_statistics, risk_cells, solve  # noqa: E402
from hai import causal  # noqa: E402

RES = ROOT / "results"
RES.mkdir(exist_ok=True)
R_OPT, R_EVAL, N_BOOT = 20, 30, 2000
N_CELLS = 10
SENIOR_BUDGET, JUNIOR_BUDGET, MIN_OVERSIGHT = 0.10, 0.50, 0.50
CALIB_GAP_MAX = 0.05
TOP_RISK_NO_AUTO = 2


def ece(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return float(sum(abs(p[idx == b].mean() - y[idx == b].mean()) * np.mean(idx == b)
                     for b in range(bins) if np.any(idx == b)))


def boot_ci(x, n=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    m = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)])
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def setup(ai_kind="gbt", include_neuro=False):
    A, B = D.load("a"), D.load("b")
    A_tr, A_opt = D.split_set_a(A)
    ai = AIAgent(ai_kind).fit(D.ai_features(A_tr, include_neuro), D.label(A_tr))
    cues = CueModel().fit(D.clinician_cues(A_tr), D.label(A_tr))
    parts = {}
    for name, df in [("opt", A_opt), ("eval", B)]:
        parts[name] = {
            "df": df,
            "y": D.label(df),
            "ai_risk": ai.risk(D.ai_features(df, include_neuro)),
            "cue_logit": cues.cue_logit(D.clinician_cues(df)),
        }
    return parts


def worlds(part, n, offset, junior=JUNIOR, senior=SENIOR):
    return [make_world(part["y"], part["ai_risk"], part["cue_logit"], offset + r, junior, senior)
            for r in range(n)]


def pick_band(opt_worlds, target=0.30):
    """Uncertainty band (logit units) so that about `target` of cases are deferred."""
    d = np.abs(opt_worlds[0].ai_logit - T_LOGIT)
    return float(np.quantile(d, target))


def fixed_protocols(band):
    p = {a: f for a, f in ACTIONS.items()}
    p["confidence_deferral"] = lambda w: confidence_deferral(w, band)
    return p


def evaluate_protocols(ws, protos):
    """Mean metrics over seeds + per-patient cost averaged over seeds."""
    out, per_patient = {}, {}
    for name, f in protos.items():
        rows, pc = [], np.zeros(len(ws[0].y))
        for w in ws:
            flag, j, s = f(w)
            rows.append(summarise(flag, j, s, w.y))
            pc += patient_cost(flag, w.y)
        out[name] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        out[name]["cost_seed_sd"] = float(np.std([r["cost_per_100"] for r in rows]))
        per_patient[name] = pc / len(ws)
    return out, per_patient


def optimise_and_eval(parts, opt_ws, eval_ws, senior_budget):
    opt, ev = parts["opt"], parts["eval"]
    cell_opt, cuts = risk_cells(opt["ai_risk"], n_cells=N_CELLS)
    cell_ev, _ = risk_cells(ev["ai_risk"], cuts=cuts)
    K = len(cuts) + 1
    names, cost, sen, jun = cell_statistics(opt_ws, cell_opt, K)
    gap = calibration_gap_per_cell(opt["ai_risk"], opt["y"], cell_opt, K)
    no_auto = sorted(set(range(K - TOP_RISK_NO_AUTO, K)) |
                     set(np.where(gap > CALIB_GAP_MAX)[0].tolist()))
    n_frac = np.bincount(cell_opt, minlength=K) / len(cell_opt)
    sol = solve(names, cost, sen, jun, n_frac, senior_budget, JUNIOR_BUDGET,
                MIN_OVERSIGHT, no_auto)
    info = {"no_auto_cells": no_auto, "calibration_gap_per_cell": gap.tolist(),
            "n_cells": K, "cell_share_opt_split": n_frac.tolist(),
            "cell_risk_upper_bounds": cuts.tolist()}
    if sol["assignment"] is None:
        return sol, None, None, info
    rows, pc = [], np.zeros(len(ev["y"]))
    for r, w in enumerate(eval_ws):
        flag, j, s, chosen = apply_policy(w, cell_ev, sol["assignment"], seed=r)
        human = np.isin(chosen, list(HUMAN_ACTIONS))
        rows.append(summarise(flag, j, s, w.y, human))
        pc += patient_cost(flag, w.y)
    ev_metrics = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    return sol, ev_metrics, pc / len(eval_ws), info


def complementarity(metrics, team, solo_human):
    best = min(metrics["ai_auto"]["cost_per_100"], metrics[solo_human]["cost_per_100"])
    return metrics[team]["cost_per_100"] - best


def main():
    results = {"config": {"R_opt": R_OPT, "R_eval": R_EVAL, "n_cells": N_CELLS,
                          "senior_budget": SENIOR_BUDGET, "junior_budget": JUNIOR_BUDGET,
                          "min_oversight": MIN_OVERSIGHT, "calib_gap_max": CALIB_GAP_MAX,
                          "top_risk_deciles_no_auto": TOP_RISK_NO_AUTO,
                          "junior": JUNIOR.__dict__, "senior": SENIOR.__dict__}}

    # ---- 1. agents -------------------------------------------------------
    parts = setup("gbt_cal")
    ev = parts["eval"]
    opt_ws, eval_ws = worlds(parts["opt"], R_OPT, 0), worlds(ev, R_EVAL, 1000)
    results["agents_on_set_b"] = {
        "ai_auroc": float(roc_auc_score(ev["y"], ev["ai_risk"])),
        "ai_ece": ece(ev["ai_risk"], ev["y"]),
        "clinician_cue_model_auroc": float(roc_auc_score(ev["y"], ev["cue_logit"])),
        "junior_own_auroc_mean": float(np.mean([roc_auc_score(w.y, w.junior_own) for w in eval_ws])),
        "senior_own_auroc_mean": float(np.mean([roc_auc_score(w.y, w.senior_own) for w in eval_ws])),
        "corr_ai_vs_cue_logit": float(np.corrcoef(logit(ev["ai_risk"]),
                                                  ev["cue_logit"])[0, 1]),
        "prevalence": float(ev["y"].mean()),
    }
    print("agents:", json.dumps(results["agents_on_set_b"], indent=1))

    # ---- 2. fixed protocols ---------------------------------------------
    band = pick_band(opt_ws)
    protos = fixed_protocols(band)
    metrics, per_patient = evaluate_protocols(eval_ws, protos)
    best_solo_junior = per_patient["ai_auto"] if metrics["ai_auto"]["cost_per_100"] <= \
        metrics["junior_alone"]["cost_per_100"] else per_patient["junior_alone"]
    for name in metrics:
        d = 100 * (per_patient[name] - best_solo_junior)
        metrics[name]["diff_vs_best_of_ai_or_junior"] = float(d.mean())
        metrics[name]["diff_ci95"] = boot_ci(d)
    results["confidence_deferral_band_logit"] = band
    results["fixed_protocols_set_b"] = metrics
    results["complementarity"] = {
        "junior_advice_first": complementarity(metrics, "junior_advice_first", "junior_alone"),
        "junior_blind_first": complementarity(metrics, "junior_blind_first", "junior_alone"),
        "senior_blind_first": complementarity(metrics, "senior_blind_first", "senior_alone"),
    }
    print("protocols:")
    for k, v in sorted(metrics.items(), key=lambda kv: kv[1]["cost_per_100"]):
        print(f"  {k:26s} cost/100={v['cost_per_100']:.2f}  sens={v['sensitivity']:.3f} "
              f"spec={v['specificity']:.3f}  jr%={v['junior_reviews_pct']:.0f} sr%={v['senior_reviews_pct']:.1f}"
              f"  diff={v['diff_vs_best_of_ai_or_junior']:+.2f} {np.round(v['diff_ci95'], 2)}")

    # ---- 3. optimisation -------------------------------------------------
    sol, lp_metrics, lp_pc, lp_info = optimise_and_eval(parts, opt_ws, eval_ws, SENIOR_BUDGET)
    d = 100 * (lp_pc - per_patient["escalate_on_disagreement"])
    lp_metrics["diff_vs_escalate_on_disagreement"] = float(d.mean())
    lp_metrics["diff_vs_escalate_ci95"] = boot_ci(d)
    d3 = 100 * (lp_pc - per_patient["confidence_deferral"])
    lp_metrics["diff_vs_confidence_deferral"] = float(d3.mean())
    lp_metrics["diff_vs_confidence_deferral_ci95"] = boot_ci(d3)
    d2 = 100 * (lp_pc - best_solo_junior)
    lp_metrics["diff_vs_best_of_ai_or_junior"] = float(d2.mean())
    lp_metrics["diff_ci95"] = boot_ci(d2)
    results["optimised_policy"] = {"lp": {k: v for k, v in sol.items()},
                                   "set_b": lp_metrics, **lp_info}
    print("LP status", sol["status"], "expected(opt split)", round(sol["expected_cost_per_100"], 2),
          "set B", round(lp_metrics["cost_per_100"], 2), "shadow", round(sol["senior_shadow_price"], 3))
    print("LP set B metrics", json.dumps(lp_metrics, indent=1))
    # Is the gap between LP-expected and realised cost optimism, or is Set B just
    # a harder cohort? Compare a fixed protocol's cost on both cohorts.
    m_opt, _ = evaluate_protocols(opt_ws, {"escalate_on_disagreement": ACTIONS["escalate_on_disagreement"],
                                           "ai_auto": ACTIONS["ai_auto"]})
    results["optimised_policy"]["cohort_reference"] = {
        "escalate_on_disagreement_opt_split": m_opt["escalate_on_disagreement"]["cost_per_100"],
        "escalate_on_disagreement_set_b": metrics["escalate_on_disagreement"]["cost_per_100"],
        "ai_auto_opt_split": m_opt["ai_auto"]["cost_per_100"],
        "ai_auto_set_b": metrics["ai_auto"]["cost_per_100"],
    }

    sweep = []
    for b in [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]:
        s_b, m_b, _, _ = optimise_and_eval(parts, opt_ws, eval_ws, b)
        sweep.append({"senior_budget": b, "expected_opt_split": s_b["expected_cost_per_100"],
                      "set_b_cost": m_b["cost_per_100"] if m_b else None,
                      "set_b_senior_pct": m_b["senior_reviews_pct"] if m_b else None,
                      "shadow_price": s_b["senior_shadow_price"], "status": s_b["status"]})
    results["senior_budget_sweep"] = sweep

    # ---- 4. sensitivity --------------------------------------------------
    sigmas = [0.25, 0.5, 1.0, 1.5, 2.0]
    ws_grid = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    grid = np.zeros((len(ws_grid), len(sigmas)))
    for i, wv in enumerate(ws_grid):
        for j, sg in enumerate(sigmas):
            jr = replace(JUNIOR, sigma=sg, w_first=wv)
            gws = [make_world(ev["y"], ev["ai_risk"], ev["cue_logit"], 5000 + r, jr, SENIOR)
                   for r in range(10)]
            m, _ = evaluate_protocols(gws, {"ai_auto": ACTIONS["ai_auto"],
                                            "junior_alone": ACTIONS["junior_alone"],
                                            "team": ACTIONS["junior_advice_first"]})
            grid[i, j] = m["team"]["cost_per_100"] - min(m["ai_auto"]["cost_per_100"],
                                                         m["junior_alone"]["cost_per_100"])
    results["sensitivity"] = {"sigmas": sigmas, "weights_on_advice": ws_grid,
                              "team_minus_best_individual": grid.tolist()}

    # ---- 5. ablation: no information asymmetry ---------------------------
    parts_n = setup("gbt_cal", include_neuro=True)
    ev_n = parts_n["eval"]
    ws_n = worlds(ev_n, R_EVAL, 1000)
    m_n, _ = evaluate_protocols(ws_n, fixed_protocols(band))
    results["ablation_ai_sees_neuro_exam"] = {
        "ai_auroc": float(roc_auc_score(ev_n["y"], ev_n["ai_risk"])),
        "corr_ai_vs_cue_logit": float(np.corrcoef(logit(ev_n["ai_risk"]),
                                                  ev_n["cue_logit"])[0, 1]),
        "complementarity": {
            "junior_advice_first": complementarity(m_n, "junior_advice_first", "junior_alone"),
            "junior_blind_first": complementarity(m_n, "junior_blind_first", "junior_alone"),
            "senior_blind_first": complementarity(m_n, "senior_blind_first", "senior_alone"),
        },
        "protocol_costs": {k: v["cost_per_100"] for k, v in m_n.items()},
    }

    # ---- 6. stability under model updates ------------------------------
    results["model_updates"] = {}
    names = sorted(metrics)
    for kind in ["logreg", "gbt"]:
        parts_u = setup(kind)
        ws_u = worlds(parts_u["eval"], R_EVAL, 1000)
        m_u, _ = evaluate_protocols(ws_u, fixed_protocols(pick_band(worlds(parts_u["opt"], 1, 0))))
        _, lp_u, _, _ = optimise_and_eval(parts_u, worlds(parts_u["opt"], R_OPT, 0), ws_u, SENIOR_BUDGET)
        tau, p_tau = kendalltau([metrics[n]["cost_per_100"] for n in names],
                                [m_u[n]["cost_per_100"] for n in names])
        results["model_updates"][kind] = {
            "ai_auroc": float(roc_auc_score(parts_u["eval"]["y"], parts_u["eval"]["ai_risk"])),
            "ai_ece": ece(parts_u["eval"]["ai_risk"], parts_u["eval"]["y"]),
            "protocol_costs": {k: v["cost_per_100"] for k, v in m_u.items()},
            "optimised_cost": lp_u["cost_per_100"] if lp_u else None,
            "complementarity": {
                "junior_advice_first": complementarity(m_u, "junior_advice_first", "junior_alone"),
                "junior_blind_first": complementarity(m_u, "junior_blind_first", "junior_alone"),
                "senior_blind_first": complementarity(m_u, "senior_blind_first", "senior_alone"),
            },
            "kendall_tau_ranking_vs_main": float(tau), "kendall_p": float(p_tau),
        }

    # ---- 7. causal -------------------------------------------------------
    sev, _ = D.severity_tercile(ev["df"], cuts=D.severity_tercile(parts["opt"]["df"])[1])
    rows = []
    for r, w in enumerate(eval_ws):
        log, truth = causal.simulate_log(w, sev, seed=r)
        est = causal.estimate(log)
        rows.append({"true": truth, **est, "senior_share": float(log.Review.mean())})
    results["causal"] = {"adjustment_set": rows[0]["adjustment_set"], "runs": len(rows),
                         "senior_share_mean": float(np.mean([r["senior_share"] for r in rows])),
                         "per_run": rows}
    for q in ["cost", "missed_death"]:
        tr = np.array([r["true"][q] for r in rows])
        nv = np.array([r["naive"][q] for r in rows])
        bd = np.array([r["backdoor"][q] for r in rows])
        results["causal"][q] = {
            "true_mean": float(tr.mean()), "naive_mean": float(nv.mean()),
            "backdoor_mean": float(bd.mean()),
            "naive_abs_error_mean": float(np.abs(nv - tr).mean()),
            "backdoor_abs_error_mean": float(np.abs(bd - tr).mean()),
            "naive_wrong_sign_runs": int(np.sum(np.sign(nv) != np.sign(tr))),
            "backdoor_wrong_sign_runs": int(np.sum(np.sign(bd) != np.sign(tr))),
        }
    print("causal:", {k: v for k, v in results["causal"].items() if k != "per_run"})

    (RES / "results.json").write_text(json.dumps(results, indent=2, default=float))
    figures(results)
    print("done")


def figures(res):
    plt.rcParams.update({"font.size": 9})
    # protocol costs
    m = res["fixed_protocols_set_b"]
    order = sorted(m, key=lambda k: m[k]["cost_per_100"])
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    vals = [m[k]["cost_per_100"] for k in order]
    err = [m[k]["cost_seed_sd"] for k in order]
    ax.barh(order, vals, xerr=err, color="#4C72B0")
    ax.axvline(res["optimised_policy"]["set_b"]["cost_per_100"], color="#C44E52", ls="--",
               label="optimised (LP)")
    ax.invert_yaxis()
    ax.set_xlim(30, max(vals) + 3)
    ax.set_xlabel("decision cost per 100 patients (Set B, mean ± sd over seeds)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(RES / "protocol_costs.png", dpi=160)

    # sensitivity heatmap
    s = res["sensitivity"]
    g = np.array(s["team_minus_best_individual"])
    fig, ax = plt.subplots(figsize=(5, 3.8))
    lim = np.abs(g).max()
    im = ax.imshow(g, cmap="RdBu_r", vmin=-lim, vmax=lim, origin="lower", aspect="auto")
    ax.set_xticks(range(len(s["sigmas"])), s["sigmas"])
    ax.set_yticks(range(len(s["weights_on_advice"])), s["weights_on_advice"])
    ax.set_xlabel("clinician inconsistency sigma (logit)")
    ax.set_ylabel("weight on AI advice w")
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            ax.text(j, i, f"{g[i, j]:+.1f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, label="team cost − best individual\n(blue = complementarity)")
    ax.set_title("Where does human-AI complementarity exist?")
    fig.tight_layout()
    fig.savefig(RES / "complementarity_map.png", dpi=160)

    # budget sweep
    sw = [r for r in res["senior_budget_sweep"] if r["set_b_cost"] is not None]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot([100 * r["senior_budget"] for r in sw], [r["expected_opt_split"] for r in sw], "o--",
            label="LP expected (optimisation split)")
    ax.plot([100 * r["senior_budget"] for r in sw], [r["set_b_cost"] for r in sw], "s-",
            label="realised on held-out Set B")
    ax.set_xlabel("senior review budget (% of patients)")
    ax.set_ylabel("decision cost per 100 patients")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RES / "senior_budget_frontier.png", dpi=160)

    # causal
    c = res["causal"]
    tr = [r["true"]["cost"] for r in c["per_run"]]
    nv = [r["naive"]["cost"] for r in c["per_run"]]
    bd = [r["backdoor"]["cost"] for r in c["per_run"]]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.scatter(tr, nv, label="naive comparison", alpha=.7)
    ax.scatter(tr, bd, label="back-door (pgmpy do-query)", alpha=.7)
    lo, hi = min(tr + nv + bd), max(tr + nv + bd)
    ax.plot([lo, hi], [lo, hi], "k:", lw=1, label="perfect estimate")
    ax.axhline(0, color="grey", lw=.5)
    ax.set_xlabel("true effect of senior review\n(change in cost per 100 patients)")
    ax.set_ylabel("estimated effect from logs")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(RES / "causal_estimates.png", dpi=160)


if __name__ == "__main__":
    main()
