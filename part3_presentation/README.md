# Surgery Schedule Optimizer - Web App

A Flask web application for surgery duration prediction and OR schedule optimization.

---

## Requirements

**Python packages:**
```
flask pandas numpy joblib scikit-learn xgboost
```

**Required files (must exist before running):**
```
part3_presentation/artifacts/best_model.joblib        trained XGBoost pipeline
part3_presentation/artifacts/feature_columns.json    list of 34 feature column names
part2_optimization/optimizer.py                       greedy scheduling algorithm
part3_presentation/feature_engineering.py            custom transformer definitions
```

`best_model.joblib` and `feature_columns.json` are the artifacts from the best-performing model selected in Part 1 (XGBoost, chosen after comparing multiple models via cross-validated MAE). They are provided pre-trained - no retraining is needed to run the app.

**Run the app:**
```
python part3_presentation/app.py
```
Then open `http://localhost:3000` in your browser.

---

## Tab 1 - Schedule Optimizer

**Upload:** A CSV file similar to `surgeries.csv`  with columns `index`, `start_time`, `end_time` representing individual surgeries for a single day.

**Output:**
- **Results Summary** - total cost, number of anesthetists used, average shift length, total overtime hours.
- **Gantt Chart (All Anesthetists)** - color-coded timeline of all surgeries grouped by anesthetist. Each color represents a different operating room. Hover over a block for details.
- **Per-Anesthetist Table** - select a specific anesthetist from the dropdown to see their individual schedule as a sorted table with room colors matching the Gantt chart.

---

## Tab 2 - Duration Predictions

**Upload:** A CSV file similar to `surgeries to predict.csv` with columns:
`Surgery Type`, `Anesthesia Type`, `Age`, `BMI`, `DoctorID`, `AnaesthetistID`

**Output:**
- **Accuracy Metrics** - model MAE (±minutes), MAPE (%), accuracy within ±15 min, and improvement over manual ±25 min estimates.
- **Business Impact Analysis** - compares current manual scheduling against the AI model. Four metrics are calculated:

  | Metric | Formula |
  |---|---|
  | **Surgeries Per Month** | `[(day_min - AI_MAE×surgeries_per_day) − (day_min - manual_buffer×surgeries_per_day)] ÷ avg_surgery_min × 22 days` - capacity recovered by using tighter AI buffers instead of manual estimates |
  | **Annual Revenue** | `extra_surgeries_per_month × 12 × revenue_per_surgery` |
  | **Overtime Savings** | `(manual_overrun_rate − model_overrun_rate) × surgeries_per_day × work_days × cost_per_overrun` - reduction in overrun incidents multiplied by cost per incident |
  | **Total Annual Benefit** | `Annual Revenue + Overtime Savings + Idle Time Savings` where idle time savings = `idle_value_per_OR_per_day × work_days × num_ORs` (fixed, independent of the model) |

  The only value that comes from your uploaded data is the **AI MAE** (model prediction error). All other inputs are taken from the Business Settings tab.

---

## Tab 3 - Business Settings

No upload required. Configure the economic parameters used in the Business Impact calculation:

| Field | What it represents |
|---|---|
| Operating Day (min) | Total OR time available per day |
| Average Surgery (min) | Typical procedure duration |
| Manual Buffer (±min) | Current surgeon scheduling buffer - the "before AI" baseline |
| Revenue per Surgery ($) | Net revenue per procedure |
| Manual / Model Overrun Rate | Fraction of surgeries that run over schedule |
| Cost per Overrun ($) | Cost per overrun incident |
| Idle Time Value per OR/day ($) | Value of unused OR time |
| Number of ORs | Total operating rooms in your facility |
| Work Days per Year | Annual operating days |

Click **Save Configuration** to apply. If a prediction has already been run, the Business Impact section updates immediately with the new numbers. Settings reset to defaults if the server restarts.
