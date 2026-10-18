"""Train every model and save metrics:  python -m src.train_all"""
import json
import warnings
warnings.filterwarnings("ignore")
from .config import OUTPUTS_DIR
from .data import load_garment_data
from .smv_model import train_smv_models
from .productivity_model import train_classifier, train_regressor
from .workforce_optimizer import train_operator_model


def main():
    df, src = load_garment_data()
    print(f"[data] using {src}: {len(df)} rows")
    metrics = {"data_source": src,
               "smv": train_smv_models(),
               "productivity_classifier": train_classifier(df),
               "productivity_regressor": train_regressor(df),
               "operator_model": train_operator_model()}
    (OUTPUTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print("\nModels saved in /models, metrics in /outputs/metrics.json")


if __name__ == "__main__":
    main()
