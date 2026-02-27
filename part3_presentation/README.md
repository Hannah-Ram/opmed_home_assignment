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
- **Business Impact Analysis** - compares current manual scheduling against the AI model:

  | Metric | Formula |
  |---|---|
  | **Extra Surgeries Per Day** | `(∣Manual MAE∣ − ∣AI MAE∣) × surgeries_per_day ÷ avg_surgery_min` — total time freed across all daily surgeries, converted back to surgery slots |
  | **Extra Surgeries Per Month** | `extra_surgeries_per_day × work_days_per_month` |
  | **Total Annual Benefit** | `extra_surgeries_per_day × work_days_per_year × revenue_per_surgery` |

  `avg_surgery_min` is computed automatically from the **Duration in Minutes** column of your uploaded file. `Manual MAE` is the typical prediction error of the current manual scheduling method (minutes). All other inputs are taken from the Business Settings tab.

---

## Tab 3 - Business Settings

No upload required. Configure the economic parameters used in the Business Impact calculation (accessible via the ⚙ Business Settings button on the Duration Predictions tab):

| Field | Default | What it represents |
|---|---|---|
| Surgeries per Day | 30 | Average number of surgeries performed per OR per day |
| ORs per Department | 4 | Number of operating rooms in a typical department |
| Departments | 3 | Number of departments in your facility - multiplied by ORs per Department to get total facility ORs |
| Work Days per Month | 22 | Operating days per month; annual figures are derived automatically as × 12 |
| Revenue per Surgery ($) | 15,000 | Average net revenue per additional surgery unlocked by the AI improvement |
| Manual MAE (minutes) | 25 | Typical prediction error of the current manual scheduling method - the "before AI" baseline |

**Note:** Average Surgery Duration is **not** a configurable field — it is computed automatically from the `Duration in Minutes` column of your uploaded file.

Click **Save Configuration** to apply. Click **Reset to Defaults** to restore all values above. If a prediction has already been run, the Business Impact section updates immediately with the new numbers. Settings reset to defaults if the server restarts.
