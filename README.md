# Sewing Line Optimization & Workforce Planning
Machine Learning + Real-Time Image Processing for garment assembly lines.

## 1. What this project does

| Module | File | Technique | Purpose |
|---|---|---|---|
| SMV estimation | `src/smv_model.py` | Decision Tree, Random Forest, MLP (deep NN) | Predict Standard Minute Value per operation and total assembly time (no subjective engineer estimate) |
| Productivity prediction | `src/productivity_model.py` | LightGBM classifier (vs RF, LogReg), RF regressor | Will a team hit its target? Forecast end-of-day output |
| Workforce planning | `src/workforce_optimizer.py` | LightGBM + Linear Programming (SciPy HiGHS) | Assign best operators to workstations under skill constraints; find cheapest overtime + incentive plan |
| Real-time forecast | `src/forecast.py` | Particle filter | Track live production rate, forecast end-of-shift output with uncertainty |
| Real-time vision | `src/vision/realtime_monitor.py` | OpenCV MOG2 background subtraction | Per-station cycle count, cycle time, utilization, idle %, bottleneck detection from camera/video |
| Dashboard | `app/dashboard.py` | Streamlit | Metrics, what-if forecasting, assignment, live vision stats |

### Architecture
```
 Camera / video ──► realtime_monitor.py ──► cycles, cycle time, utilization, bottleneck
                                   │                         │
                                   ▼                         ▼
                        particle filter (forecast.py)   outputs/vision_stats.json ──► dashboard
 Factory CSV (UCI) ──► data.py ──► productivity_model.py ──► forecast / P(hit target)
 Time-study data   ──► smv_model.py ──► SMV + total assembly time
 Operator records  ──► workforce_optimizer.py ──► LightGBM scores ─► LP assignment, overtime/incentive plan
```

## 2. Setup in VS Code (step by step)
1. Install **Python 3.10 - 3.12** and **VS Code** (+ the *Python* extension).
2. Unzip the project, then in VS Code: **File > Open Folder** > `sewing_line_optimizer`.
3. Open the terminal (`Ctrl+``) and create a virtual environment:
   ```bash
   python -m venv .venv
   # Windows:  .venv\Scripts\activate        macOS/Linux:  source .venv/bin/activate
   pip install -r requirements.txt
   ```
4. `Ctrl+Shift+P` > **Python: Select Interpreter** > choose `.venv`.

## 3. Run it
```bash
python -m src.train_all                      # trains all models, writes outputs/metrics.json
python -m src.run_optimization               # assignment + forecast + overtime/incentive plan
python -m src.vision.realtime_monitor --demo # real-time vision on synthetic video (press q to quit)
streamlit run app/dashboard.py               # dashboard (run the monitor in another terminal for live tab)
python -m pytest                             # tests
```
Always run from the project root (the `-m` form). In VS Code you can also use **Run and Debug**
(F5): launch configs for all of the above are in `.vscode/launch.json`.

### Real camera / video
```bash
python -m src.vision.realtime_monitor --source 0 --stations 4                 # webcam
python -m src.vision.realtime_monitor --source line.mp4 --stations 6 --select-rois
```
* `--select-rois` lets you draw a box around each workstation (ENTER confirms each, ESC finishes).
  Order stations in **process flow order**: the last one is treated as line output.
* Mount the camera overhead / at an angle where each operator's hands are in their ROI; keep lighting steady.
* Tune `thr`, `gap_s`, `min_active_s` in `StationTracker` for your garment type and camera.

## 4. Use real data (important)
By default the project generates **synthetic data** so it runs immediately. Synthetic metrics are *not*
comparable with published results. For meaningful results:
1. **Productivity**: download the UCI *Productivity Prediction of Garment Employees* dataset
   (https://archive.ics.uci.edu/dataset/597/productivity+prediction+of+garment+employees),
   save as `data/raw/garments_worker_productivity.csv`. It is detected automatically.
   (Columns: date, quarter, department, day, team, targeted_productivity, smv, wip, over_time,
   incentive, idle_time, idle_men, no_of_style_change, no_of_workers, actual_productivity.)
2. **SMV**: replace `generate_operation_data()` with your time-study CSV having columns
   `garment_type, machine_type, fabric_type, stitch_count, seam_length_cm, difficulty, num_plies, smv`
   and pass it to `train_smv_models(df)`.
3. **Operators**: replace `generate_operator_task_data()` with your operator-task history
   (`experience_months, skill_level, task_smv, task_complexity, hours_worked, recent_efficiency, high_productivity`).

The figures quoted in the literature (R2 0.77-0.86 for SMV trees/forests, 78.3% LightGBM accuracy,
4.69% DNN error, R2 0.963 with IoT + random forest) come from specific datasets and papers. Do not expect
identical numbers; report what YOUR data gives, with cross-validation.

## 5. How each part works

**SMV (smv_model.py)** - one-hot encodes categorical attributes (garment, machine, fabric), regresses SMV
with a depth-limited decision tree and a 300-tree random forest; reports R2 and MSE. A small MLP
(64-32, log-target) predicts total order time from operations count, summed SMV, quantity, workers - this
captures the non-linear learning curve; reports MAPE.

**Productivity (productivity_model.py)** - label = `actual >= target`. LightGBM, Random Forest and Logistic
Regression are compared. A Random Forest regressor predicts productivity, and
`units = productivity x (workers x shift_min + overtime_min) / SMV` gives end-of-day output.

**Workforce LP (workforce_optimizer.py)** - LightGBM gives `P(high productivity)` for each operator/station pair.
Binary assignment variables `x_ij` maximize `sum P_ij x_ij` s.t. each station has exactly one operator, each
operator at most one station, and `x_ij = 0` when skill < required. The assignment LP is totally unimodular,
so HiGHS returns integral solutions.

**Overtime/incentive planner** - evaluates a grid of overtime x incentive values through the regressor, picks
the cheapest plan that meets the target units (cost rates are parameters).

**Vision (realtime_monitor.py)** - MOG2 per ROI > shadow removal > morphological opening > motion ratio.
Hysteresis state machine turns motion bursts into cycles. Outputs live JSON every second and a CSV summary.

**Particle filter (forecast.py)** - 500 particles over the production rate (units/s), Poisson-like likelihood,
systematic resampling; end-of-shift forecast = current + rate x remaining time, with P10/P90 band.

## 6. Folder layout
```
sewing_line_optimizer/
├── app/dashboard.py
├── src/{config,data,smv_model,productivity_model,workforce_optimizer,forecast,train_all,run_optimization}.py
├── src/vision/realtime_monitor.py
├── tests/test_smoke.py
├── data/raw/            <- put UCI CSV here
├── models/  outputs/    <- generated
├── .vscode/{launch,settings}.json
└── requirements.txt
```

## 7. Extension ideas
* **Better vision**: swap motion detection for YOLOv8 (`pip install ultralytics`) person/pose detection per
  ROI, or an action-recognition model to classify sewing vs. idle vs. material handling.
* **IoT edge**: run the monitor on a Raspberry Pi/Jetson; publish stats via MQTT; use counters/ESP32 sensors on
  machines as additional features for the forecast model (as in the R2 = 0.963 IoT setup).
* **Hyper-parameter tuning**: Optuna + cross-validation; SHAP for feature importance on LightGBM.
* **Line balancing**: use per-station `avg_cycle_s` to compute takt time and move tasks off the bottleneck.
* **Privacy**: monitoring workers has legal/ethical implications - inform workers, use aggregate station-level
  metrics, and comply with local labour and data-protection rules.

## 8. Troubleshooting
* `ModuleNotFoundError: src` - run from project root with `python -m ...`.
* `cv2.imshow` error on servers - add `--no-display`.
* LightGBM install fails - the code falls back to scikit-learn HistGradientBoosting.
* Dashboard says models not found - run `python -m src.train_all` first.
