"""Data loading + synthetic data generators.

If the real UCI dataset exists at data/raw/garments_worker_productivity.csv it is used.
Otherwise a synthetic dataset with the SAME columns is generated so the whole project runs
out of the box.
"""
import numpy as np
import pandas as pd
from .config import RAW_CSV, SEED

CAT = ["quarter", "department", "day", "team"]
NUM = ["targeted_productivity", "smv", "wip", "over_time", "incentive",
       "idle_time", "idle_men", "no_of_style_change", "no_of_workers"]


def generate_garment_data(n: int = 1500, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    date = pd.to_datetime("2015-01-01") + pd.to_timedelta(rng.integers(0, 59, n), unit="D")
    dept = rng.choice(["sewing", "finishing"], n, p=[0.6, 0.4])
    smv = np.where(dept == "sewing", rng.uniform(15, 40, n), rng.uniform(2, 6, n)).round(2)
    workers = np.where(dept == "sewing", rng.integers(30, 60, n), rng.integers(8, 16, n)).astype(float)
    target = rng.choice([0.6, 0.7, 0.75, 0.8, 0.85], n)
    over_time = rng.choice([0, 0, 1440, 2880, 4080, 6960], n)
    incentive = np.where(rng.random(n) < 0.5, 0, rng.choice([23, 30, 40, 50, 60, 75, 100], n))
    idle_time = np.where(rng.random(n) < 0.03, rng.uniform(2, 300, n), 0.0)
    idle_men = np.where(idle_time > 0, rng.integers(1, 30, n), 0)
    style = rng.choice([0, 1, 2], n, p=[0.8, 0.12, 0.08])
    wip = np.where(dept == "sewing", rng.integers(200, 2000, n), np.nan)
    actual = (0.55 + 0.35 * target + 0.0012 * incentive - 0.000008 * over_time
              - 0.0006 * idle_time - 0.04 * style - 0.0015 * (workers - 30) * (dept == "sewing")
              - 0.05 * (dept == "finishing") + rng.normal(0, 0.06, n))
    actual = np.clip(actual, 0.2, 1.2)
    df = pd.DataFrame({
        "date": date, "department": dept,
        "team": rng.integers(1, 13, n), "targeted_productivity": target, "smv": smv,
        "wip": wip, "over_time": over_time, "incentive": incentive, "idle_time": idle_time,
        "idle_men": idle_men, "no_of_style_change": style, "no_of_workers": workers,
        "actual_productivity": actual.round(4)})
    df["day"] = df["date"].dt.day_name()
    df["quarter"] = "Quarter" + (np.minimum((df["date"].dt.day - 1) // 7 + 1, 5)).astype(str)
    return df


def clean_garment_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["department"] = df["department"].str.strip().replace({"sweing": "sewing"})
    df["wip"] = df["wip"].fillna(0)
    df["team"] = df["team"].astype(str)
    return df.dropna(subset=["actual_productivity"]).reset_index(drop=True)


def load_garment_data():
    """Returns (df, source_name)."""
    if RAW_CSV.exists():
        return clean_garment_data(pd.read_csv(RAW_CSV)), "UCI real dataset"
    return clean_garment_data(generate_garment_data()), "synthetic dataset"


def make_X(df: pd.DataFrame) -> pd.DataFrame:
    X = df[CAT + NUM].copy()
    X["team"] = X["team"].astype(str)
    return X
