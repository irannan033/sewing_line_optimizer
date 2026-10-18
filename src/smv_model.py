"""Part 1 - SMV (Standard Minute Value) estimation + total assembly-time prediction.

* Decision tree & random forest regressors predict SMV of one operation from its attributes.
* An MLP (small deep network) predicts total order assembly time under a learning curve.
Replace the synthetic generators with your own time-study CSV (same column names) for real use.
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from .config import MODELS_DIR, SEED

OP_CAT = ["garment_type", "machine_type", "fabric_type"]
OP_NUM = ["stitch_count", "seam_length_cm", "difficulty", "num_plies"]


def generate_operation_data(n=2500, seed=SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    machine = rng.choice(["single_needle", "overlock", "flatlock", "bartack", "manual"], n)
    fabric = rng.choice(["cotton", "denim", "knit", "polyester"], n)
    df = pd.DataFrame({
        "garment_type": rng.choice(["shirt", "trouser", "tshirt", "jacket"], n),
        "machine_type": machine, "fabric_type": fabric,
        "stitch_count": rng.integers(20, 600, n), "seam_length_cm": rng.uniform(5, 150, n).round(1),
        "difficulty": rng.integers(1, 6, n), "num_plies": rng.integers(1, 5, n)})
    mbase = pd.Series(machine).map({"single_needle": 0.0, "overlock": 0.1, "flatlock": 0.2,
                                    "bartack": 0.15, "manual": 0.4}).values
    fab = pd.Series(fabric).map({"cotton": 1.0, "denim": 1.2, "knit": 0.95, "polyester": 1.05}).values
    smv = (0.1 + 0.0012 * df.stitch_count + 0.006 * df.seam_length_cm + mbase + 0.03 * df.num_plies)
    smv = smv * (1 + 0.12 * (df.difficulty - 1)) * fab + rng.normal(0, 0.03, n)
    df["smv"] = smv.clip(lower=0.05).round(3)
    return df


def generate_order_data(n=2000, seed=SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_ops = rng.integers(8, 40, n)
    sum_smv = n_ops * rng.uniform(0.3, 1.2, n)
    qty = rng.integers(100, 8000, n)
    workers = rng.integers(10, 60, n)
    eff = 0.5 + 0.4 * (1 - np.exp(-qty / 800))            # learning curve
    total = sum_smv * qty / (workers * eff) * rng.normal(1, 0.03, n)
    return pd.DataFrame({"n_operations": n_ops, "sum_smv": sum_smv.round(3), "order_qty": qty,
                         "n_workers": workers, "mean_difficulty": rng.uniform(1, 5, n).round(2),
                         "total_minutes": total})


def train_smv_models(df=None) -> dict:
    df = generate_operation_data() if df is None else df
    X, y = df[OP_CAT + OP_NUM], df["smv"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)
    out = {}
    for name, m in {"decision_tree": DecisionTreeRegressor(max_depth=10, random_state=SEED),
                    "random_forest": RandomForestRegressor(300, random_state=SEED, n_jobs=-1)}.items():
        pre = ColumnTransformer([("c", OneHotEncoder(handle_unknown="ignore"), OP_CAT)], remainder="passthrough")
        pipe = Pipeline([("pre", pre), ("m", m)]).fit(Xtr, ytr)
        p = pipe.predict(Xte)
        out[name] = {"R2": round(r2_score(yte, p), 4), "MSE": round(mean_squared_error(yte, p), 5)}
        if name == "random_forest":
            joblib.dump(pipe, MODELS_DIR / "smv_rf.joblib")

    # Deep network for total assembly time
    od = generate_order_data()
    feats = ["n_operations", "sum_smv", "order_qty", "n_workers", "mean_difficulty"]
    Xtr, Xte, ytr, yte = train_test_split(od[feats], od["total_minutes"], test_size=0.2, random_state=SEED)
    mlp = TransformedTargetRegressor(
        regressor=Pipeline([("s", StandardScaler()),
                            ("m", MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=800, early_stopping=True, random_state=SEED))]),
        func=np.log1p, inverse_func=np.expm1).fit(Xtr, ytr)
    out["mlp_total_time"] = {"MAPE_percent": round(100 * mean_absolute_percentage_error(yte, mlp.predict(Xte)), 2)}
    joblib.dump(mlp, MODELS_DIR / "total_time_mlp.joblib")
    return out


def predict_smv(operation: dict) -> float:
    pipe = joblib.load(MODELS_DIR / "smv_rf.joblib")
    return float(pipe.predict(pd.DataFrame([operation]))[0])
