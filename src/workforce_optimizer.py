"""Part 3 - Workforce planning.
(a) Operator-task model (LightGBM) -> P(high productivity) for every operator/workstation pair.
(b) Linear programming (scipy HiGHS) assigns operators to stations maximizing total probability
    under skill constraints.
(c) Overtime / incentive planner: cheapest plan whose predicted output reaches the target.
"""
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from .config import MODELS_DIR, SEED, SHIFT_MINUTES
from .data import make_X
from .productivity_model import _lgbm, forecast_units

OPF = ["experience_months", "skill_level", "task_smv", "task_complexity", "hours_worked", "recent_efficiency"]


def generate_operator_task_data(n=4000, seed=SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "experience_months": rng.integers(1, 120, n), "skill_level": rng.integers(1, 6, n),
        "task_smv": rng.uniform(0.2, 1.5, n).round(3), "task_complexity": rng.integers(1, 6, n),
        "hours_worked": rng.uniform(0, 8, n).round(1), "recent_efficiency": rng.uniform(0.3, 1.0, n).round(3)})
    z = (-1.0 + 0.012 * df.experience_months + 0.45 * (df.skill_level - df.task_complexity)
         + 3 * (df.recent_efficiency - 0.65) - 0.1 * (df.hours_worked - 4) - 0.5 * df.task_smv)
    df["high_productivity"] = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    return df


def train_operator_model() -> dict:
    df = generate_operator_task_data()
    Xtr, Xte, ytr, yte = train_test_split(df[OPF], df["high_productivity"], test_size=0.2, random_state=SEED)
    m = _lgbm().fit(Xtr, ytr)
    joblib.dump(m, MODELS_DIR / "operator_clf.joblib")
    return {"accuracy": round(accuracy_score(yte, m.predict(Xte)), 4)}


def make_demo_roster(n_ops=14, n_st=10, seed=SEED):
    rng = np.random.default_rng(seed + 1)
    ops = pd.DataFrame({"operator": [f"OP{i+1:02d}" for i in range(n_ops)],
                        "experience_months": rng.integers(3, 100, n_ops),
                        "skill_level": rng.integers(2, 6, n_ops),
                        "recent_efficiency": rng.uniform(0.5, 0.95, n_ops).round(2)})
    comp = rng.integers(1, 5, n_st)
    sts = pd.DataFrame({"station": [f"ST{i+1:02d}" for i in range(n_st)],
                        "task_smv": rng.uniform(0.3, 1.2, n_st).round(2), "task_complexity": comp,
                        "min_skill": np.maximum(1, comp - 1)})
    return ops, sts


def score_matrix(model, ops, sts, hours_worked=2.0):
    rows = [{"experience_months": o.experience_months, "skill_level": o.skill_level,
             "task_smv": s.task_smv, "task_complexity": s.task_complexity,
             "hours_worked": hours_worked, "recent_efficiency": o.recent_efficiency}
            for o in ops.itertuples() for s in sts.itertuples()]
    P = model.predict_proba(pd.DataFrame(rows)[OPF])[:, 1]
    return P.reshape(len(ops), len(sts))


def assign_operators(P, feasible):
    """LP: max sum P_ij x_ij ; each station gets exactly 1 operator; each operator <= 1 station."""
    n_op, n_st = P.shape
    A_eq = np.zeros((n_st, n_op * n_st)); A_ub = np.zeros((n_op, n_op * n_st))
    for j in range(n_st):
        A_eq[j, j::n_st] = 1
    for i in range(n_op):
        A_ub[i, i * n_st:(i + 1) * n_st] = 1
    bounds = [(0, 1 if feasible[i, j] else 0) for i in range(n_op) for j in range(n_st)]
    res = linprog(-P.flatten(), A_ub=A_ub, b_ub=np.ones(n_op), A_eq=A_eq, b_eq=np.ones(n_st),
                  bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError("No feasible assignment under skill constraints: " + res.message)
    return res.x.reshape(n_op, n_st).round().astype(int)


def optimal_assignment(ops, sts):
    model = joblib.load(MODELS_DIR / "operator_clf.joblib")
    P = score_matrix(model, ops, sts)
    feas = ops["skill_level"].values[:, None] >= sts["min_skill"].values[None, :]
    try:
        X = assign_operators(P, feas)
    except RuntimeError:
        X = assign_operators(P, np.ones_like(feas, dtype=bool))
    rows = [{"station": sts.station[j], "operator": ops.operator[i], "skill": ops.skill_level[i],
             "min_skill": sts.min_skill[j], "p_high_productivity": round(P[i, j], 3)}
            for i, j in zip(*np.nonzero(X))]
    return pd.DataFrame(rows).sort_values("station").reset_index(drop=True)


def plan_overtime_incentive(reg, base: dict, target_units: float,
                            ot_cost_per_min=0.5, incentive_cost_per_worker=1.0,
                            shift_minutes=SHIFT_MINUTES):
    """Grid-search overtime (min) x incentive (currency) -> cheapest plan reaching target_units."""
    grid = [dict(base, over_time=ot, incentive=inc)
            for ot in range(0, 7201, 600) for inc in range(0, 101, 10)]
    g = pd.DataFrame(grid)
    prod = reg.predict(make_X(g))
    g["pred_productivity"] = prod.round(3)
    g["pred_units"] = forecast_units(prod, g["no_of_workers"], g["smv"], g["over_time"], shift_minutes).round()
    g["cost"] = g["over_time"] * ot_cost_per_min + g["incentive"] * g["no_of_workers"] * incentive_cost_per_worker
    ok = g[g["pred_units"] >= target_units]
    best = ok.sort_values("cost").iloc[0] if len(ok) else g.sort_values("pred_units", ascending=False).iloc[0]
    return best[["over_time", "incentive", "pred_productivity", "pred_units", "cost"]].to_dict(), bool(len(ok))
