"""Streamlit dashboard:  streamlit run app/dashboard.py"""
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import pandas as pd
import streamlit as st

from src.config import OUTPUTS_DIR, MODELS_DIR
from src.productivity_model import load_models, predict_scenario
from src.workforce_optimizer import make_demo_roster, optimal_assignment, plan_overtime_incentive

st.set_page_config(page_title="Sewing Line Optimizer", layout="wide")
st.title("Sewing Line Optimization & Workforce Planning")

if not (MODELS_DIR / "productivity_reg.joblib").exists():
    st.error("Models not found. Run:  python -m src.train_all"); st.stop()
clf, reg = load_models()

tab1, tab2, tab3, tab4 = st.tabs(["Model metrics", "Forecast & overtime plan", "Operator assignment", "Live vision"])

with tab1:
    m = OUTPUTS_DIR / "metrics.json"
    st.json(json.loads(m.read_text())) if m.exists() else st.info("Run train_all first.")

with tab2:
    c = st.columns(4)
    dept = c[0].selectbox("Department", ["sewing", "finishing"])
    smv = c[1].number_input("SMV", 1.0, 60.0, 26.0)
    workers = c[2].number_input("Workers", 5, 100, 40)
    target = c[3].slider("Target productivity", 0.4, 1.0, 0.8)
    c = st.columns(4)
    ot = c[0].number_input("Overtime (min)", 0, 10000, 0, step=300)
    inc = c[1].number_input("Incentive", 0, 200, 0, step=10)
    wip = c[2].number_input("WIP", 0, 5000, 900)
    style = c[3].selectbox("Style changes", [0, 1, 2])
    base = dict(quarter="Quarter1", department=dept, day="Monday", team="5", targeted_productivity=target,
                smv=smv, wip=wip, over_time=ot, incentive=inc, idle_time=0, idle_men=0,
                no_of_style_change=style, no_of_workers=workers)
    res = predict_scenario(clf, reg, base)
    st.metric("Predicted productivity", res["predicted_productivity"])
    st.metric("P(hit target)", res["prob_hit_target"])
    st.metric("Forecast end-of-day units", res["forecast_units"])
    goal = st.number_input("Target units for planner", 0, 100000, int(res["forecast_units"] * 1.2))
    if st.button("Find cheapest overtime + incentive plan"):
        plan, ok = plan_overtime_incentive(reg, dict(base, over_time=0, incentive=0), goal)
        st.success("Target reachable" if ok else "Target not reachable - best effort shown")
        st.json(plan)

with tab3:
    if not (MODELS_DIR / "operator_clf.joblib").exists():
        st.info("Train models first.")
    else:
        ops, sts = make_demo_roster()
        st.write("Operators"); st.dataframe(ops)
        st.write("Workstations"); st.dataframe(sts)
        st.write("Optimal assignment (LP)"); st.dataframe(optimal_assignment(ops, sts))

with tab4:
    f = OUTPUTS_DIR / "vision_stats.json"
    if st.button("Refresh") or True:
        if f.exists():
            d = json.loads(f.read_text())
            a, b, c3 = st.columns(3)
            a.metric("Line output", d["line_output"]); b.metric("Bottleneck", d["bottleneck"] or "-")
            c3.metric("Forecast end-of-shift", d["forecast_end_of_shift"]["mean"])
            st.dataframe(pd.DataFrame(d["stations"]))
            st.bar_chart(pd.DataFrame(d["stations"]).set_index("station")["utilization"])
        else:
            st.info("Start the monitor:  python -m src.vision.realtime_monitor --demo")
