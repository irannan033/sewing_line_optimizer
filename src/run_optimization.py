"""Demo of operator assignment + overtime/incentive planning:  python -m src.run_optimization"""
import pandas as pd
from .productivity_model import load_models, predict_scenario
from .workforce_optimizer import make_demo_roster, optimal_assignment, plan_overtime_incentive


def main():
    pd.set_option("display.width", 120)
    clf, reg = load_models()
    ops, sts = make_demo_roster()
    print("=== Optimal operator -> workstation assignment (LightGBM + LP) ===")
    print(optimal_assignment(ops, sts).to_string(index=False))

    base = dict(quarter="Quarter1", department="sewing", day="Monday", team="5",
                targeted_productivity=0.8, smv=26.0, wip=900, over_time=0, incentive=0,
                idle_time=0, idle_men=0, no_of_style_change=0, no_of_workers=40)
    print("\n=== End-of-day forecast for the base scenario ===")
    print(predict_scenario(clf, reg, base))

    target = 1.25 * predict_scenario(clf, reg, base)["forecast_units"]
    plan, reached = plan_overtime_incentive(reg, base, target)
    print(f"\n=== Cheapest overtime/incentive plan for target {target:.0f} units "
          f"({'target reachable' if reached else 'target NOT reachable - best effort'}) ===")
    print(plan)


if __name__ == "__main__":
    main()
