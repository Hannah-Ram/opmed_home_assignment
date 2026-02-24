# Opmed - Operating Room Optimization System

A comprehensive solution for predicting surgery duration and optimizing operating room scheduling to minimize labor costs and improve resource utilization.

---

## System Requirements

- **OS**: Windows
- **Python Version**: 3.10.4

---

## Dependencies

```
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
joblib
colorcet
Flask (for Part 3 web interface)
plotly
```

---

## Part 1: Surgery Duration Prediction Model

Predicts surgery duration using machine learning models trained on historical surgical data including patient demographics, procedure type, and anesthesia method. The model employs comprehensive feature engineering combined with cross-validated ensemble techniques including XGBoost, Random Forest, and Gradient Boosting. This enables accurate OR scheduling and reduces operating room idle time and delay cascades.

---

## Part 2: Schedule Optimizer

Assigns surgeries to operating rooms and anesthesiologists using a greedy heuristic algorithm that minimizes total labor costs while respecting resource constraints (20 rooms max, 12-hour shift limits). The optimizer sorts surgeries by start time and greedily assigns each to the lowest-numbered available room and lowest-cost existing anesthesiologist (or creates a new one), with cost calculations accounting for minimum 5-hour paid shifts and overtime penalties.


