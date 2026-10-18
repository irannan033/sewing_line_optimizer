"""Part 2 - Team productivity: LightGBM classifier (hit target?) + RF regressor (actual productivity)
and end-of-day output forecast."""
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from .config import MODELS_DIR, SEED, SHIFT_MINUTES
from .data import CAT, make_X

try:
    from lightgbm import LGBMClassifier
except ImportError:  # graceful fallback
    from sklearn.ensemble import HistGradientBoostingClassifier as LGBMClassifier


def _pre():
    return ColumnTransformer([("c", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT)],
                             remainder="passthrough")


def _lgbm():
    try:
        return LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=SEED, verbose=-1)
    except TypeError:
        return LGBMClassifier(random_state=SEED)


def train_classifier(df) -> dict:
    X = make_X(df)
    y = (df["actual_productivity"] >= df["targeted_productivity"]).astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    cands = {"lightgbm": _lgbm(),
             "random_forest": RandomForestClassifier(300, random_state=SEED, n_jobs=-1),
             "logistic_regression": LogisticRegression(max_iter=2000)}
    res = {}
    for name, m in cands.items():
        pipe = Pipeline([("pre", _pre()), ("m", m)]).fit(Xtr, ytr)
        p = pipe.predict(Xte)
        res[name] = {"accuracy": round(accuracy_score(yte, p), 4), "f1": round(f1_score(yte, p), 4)}
        if name == "lightgbm":
            joblib.dump(pipe, MODELS_DIR / "productivity_clf.joblib")
    return res


def train_regressor(df) -> dict:
    X, y = make_X(df), df["actual_productivity"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)
    pipe = Pipeline([("pre", _pre()), ("m", RandomForestRegressor(400, random_state=SEED, n_jobs=-1))]).fit(Xtr, ytr)
    p = pipe.predict(Xte)
    joblib.dump(pipe, MODELS_DIR / "productivity_reg.joblib")
    return {"R2": round(r2_score(yte, p), 4), "MSE": round(mean_squared_error(yte, p), 5)}


def load_models():
    return (joblib.load(MODELS_DIR / "productivity_clf.joblib"),
            joblib.load(MODELS_DIR / "productivity_reg.joblib"))


def forecast_units(productivity, workers, smv, over_time=0, shift_minutes=SHIFT_MINUTES):
    """Units produced = efficiency * available minutes / SMV."""
    return productivity * (workers * shift_minutes + over_time) / smv


def predict_scenario(clf, reg, scenario: dict) -> dict:
    df = pd.DataFrame([scenario])
    prod = float(reg.predict(make_X(df))[0])
    prob = float(clf.predict_proba(make_X(df))[0, 1])
    return {"predicted_productivity": round(prod, 3), "prob_hit_target": round(prob, 3),
            "forecast_units": round(float(forecast_units(prod, scenario["no_of_workers"],
                                                         scenario["smv"], scenario["over_time"])))}
