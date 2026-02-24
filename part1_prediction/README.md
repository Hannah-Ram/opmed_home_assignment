# Part 1 - Surgery Duration Prediction Model

## Overview

Predict surgery duration (in minutes) based on patient and procedural characteristics to optimize operating room (OR) scheduling and reduce delays.

---

## Analysis & Feature Relationships

### Key Findings from Exploratory Data Analysis

- Surgery Type is the Strongest Duration Predictor - different procedure types have clearly different typical durations and variability, dominating other features.
- Patient features: Patient age is roughly normally distributed around ~50, while BMI is centered around the mid-20s with a slight right tail toward higher values.
- Age shows at most a mild relationship with duration, while BMI has little clear effect overall, most variation is explained by procedure type rather than patient demographics.
 
---

## Model Training & Development

### Methodology

#### Data Preprocessing
1. **Outlier Removal**: 
- Removed physiologically impossible BMI values (<10)
- The data is fairly evenly distributed across surgery types and anesthesia types, so no re-sampling was needed.
- The dataset contains no missing (null) values, so no imputation was required.
2. **Feature Engineering**: 
   - Interaction terms (Age × BMI, Surgery Type × Age, Surgery Type × BMI)
   - Polynomial features (Age², BMI²)
   - Bucketed features (Age groups, BMI categories)
   - Log transformation of BMI for non-linearity

3. **Feature Encoding**:
   - **One-Hot Encoding**: Surgery Type, Anesthesia Type
   - **Frequency Encoding**: Doctor ID, Anesthetist ID (count in training fold)
   - **Target Encoding** (Mean Duration): Surgery Type mean duration per value
   - **Fold-Safe**: All encoding done separately per cross-validation fold (prevents leakage)

#### Model Selection & Comparison

Four model families were evaluated:

| Model | Rationale | Pros | Cons |
|-------|-----------|------|------|
| **Random Forest** | Handles non-linearity, feature interactions naturally | Robust, fast training, interpretable | Can overfit, slower inference |
| **Gradient Boosting** | Sequential learning, captures complex patterns | Excellent generalization, handles outliers | Slower tuning, Black-box |
| **XGBoost** | Modern boosting with regularization | Fast, scalable, strong on interactions | Hyperparameter-sensitive |
| **Ridge Regression** | Linear baseline for comparison | Simple, interpretable, fast | Limited capturing non-linearity |


### Cross-Validation Strategy

**GroupKFold (n_splits=5)**: 
- Surgeries grouped by Doctor ID
- Same surgeon never straddles fold boundary
- Simulates prediction on new surgeries from new doctors
- More realistic generalization assessment

**Results by Model**:

| Model | MAE (min) | MAPE (%) | Acc ±15min (%) |
|-------|-----------|----------|----------------|
| Random Forest | 15.85 | 18.96 | 49.48 |
| Gradient Boosting | 15.87 | 19.00 | 49.75 |
| XGBoost | 15.74 | 18.92 | 49.99 |
| Ridge | 15.94 | 19.81 | 49.54 |


### *Best Performing Model*: XGBoost
---


### Feature Importance (XGBoost - Top Predictors)

1. **Surgery Type 2** (44.6%) - Specific procedure category is the dominant predictor
2. **Surgery Mean Duration** (27.7%) - Average duration for surgery type
3. **Surgery Type 0** (21.6%) - Another major procedure category
4. **Surgery Type 0 × Age Interaction** (1.8%) - Complex procedures affected by patient age
5. **Age 60+ Category** (1.3%) - Senior patients show distinct patterns

**Key Insight**: Surgery classification alone explains 93.9% of duration variation (44.6% + 27.7% + 21.6%); patient factors contribute minimally. This confirms procedure type is the overwhelming predictor of surgery duration.

---

## Business Impact Quantification

### Current State vs. ML Model

### 1. How many more surgeries could be scheduled per OR per month?

#### What we know
- Current scheduling uses surgeon estimates with ~±25 minutes average error.
- Our best model achieved **MAE = 15.74 minutes**.
- Average error improvement is **~9.26 minutes per surgery** (25 − 15.74).
- From the solution optimizer schedule for **2023-04-25**:
  - Total surgeries scheduled that day: **114**
  - We assume **20 ORs** are available, so this is **~5.7 surgeries per OR per day** (114 / 20)
  - Average scheduled surgery duration based on the surgeries to predict is **115 minutes** per case

#### Assumptions
- The day in in the optimizer solution is representative of a typical operating day.
- Better predictions allow reducing the “safety buffer” added to each scheduled case.
- We quantify two scenarios:
  - **Conservative:** reduce buffer by half the improvement (~4.6 min per surgery)
  - **Optimistic:** reduce buffer by the full improvement (~9.3 min per surgery)
- We assume there are 20 working days per month

#### Calculation
1) **Surgeries per OR per month**
- Surgeries per OR per day ≈ 5.7
- Surgeries per OR per month = 5.7 × 20  
  - **114 surgeries per OR per month**

2) **Minutes freed per OR per month**
- Conservative: 114 × 4.6 ≈ **528 minutes** freed (~8.8 hours)
- Optimistic: 114 × 9.3 ≈ **1056 minutes** freed (~17.6 hours)

3) **Extra surgeries per OR per month**
- Extra surgeries ≈ (minutes freed) / (average case block)
- Average case block from the optimizer schedule ≈ 115 minutes
- Conservative: 528 / 115 ≈ **4 extra surgeries per OR per month**
- Optimistic: 1056 / 115 ≈ **9 extra surgeries per OR per month**

#### Hospital-wide impact (20 ORs)
- Conservative: 4 × 20 ≈ **80 extra surgeries per month**
- Optimistic: 9 × 20 ≈ **180 extra surgeries per month**


### 2. What's the estimated reduction in overtime costs if delays decrease?

#### Given
- Current scheduling error (surgeon estimates): 25 minutes average error per surgery
- Model error (MAE): 15.74 minutes
- Error reduction per surgery: 25 − 15.74 = 9.26 minutes
- From `solution.csv`: 114 surgeries scheduled in the day
- Optimizer output (baseline): total overtime hours = 27.25
- Overtime cost premium formula includes: 0.5 × overtime_hours - based on the optimizer formula: max(5, shift_duration) + 0.5 × max(0, shift_duration - 9) 

#### Assumption
We assume overtime happens because small delays accumulate across the day and push the schedule later overall.
Not all reduced error turns into less overtime, because some delays are absorbed (gaps, cancellations, faster cases, etc.).

So we assume only a fraction `p` of the error reduction becomes actual overtime reduction.
- Conservative: p = 0.10 (10%)
- Moderate:    p = 0.25 (25%)
- Optimistic:  p = 0.50 (50%)

#### Step 1: Total delay minutes reduced per day
Delay minutes reduced per day = (error reduction per surgery) × (surgeries per day)
= 9.26 × 114 = 1055.64 minutes

#### Step 2: Convert to overtime hours reduced
Overtime hours reduced = p × (delay minutes reduced per day / 60)

- If p = 0.10: 0.10 × (1055.64 / 60) = 1.76 hours
- If p = 0.25: 0.25 × (1055.64 / 60) = 4.40 hours
- If p = 0.50: 0.50 × (1055.64 / 60) = 8.80 hours

#### Step 3: New overtime hours estimate
New overtime hours = 27.25 - overtime hours reduced

- p = 0.10: 27.25 - 1.76 = 25.49 hours
- p = 0.25: 27.25 - 4.40 = 22.85 hours
- p = 0.50: 27.25 - 8.80 = 18.45 hours

#### Step 4: Convert overtime reduction to cost savings (premium portion)
From the cost function, the overtime premium is: 0.5 × overtime_hours
So savings from reduced overtime is: 0.5 × (overtime hours reduced)

- p = 0.10: 0.5 × 1.76 = 0.88 cost-units saved
- p = 0.25: 0.5 × 4.40 = 2.20 cost-units saved
- p = 0.50: 0.5 × 8.80 = 4.40 cost-units saved

---

## Technical Details

### Requirements
- Python 3.8+

Python packages:
- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib
- seaborn
- joblib

### Files
- `surgeries to predict.csv`: Input data containing patient and procedural features used for prediction
- `analysis.ipynb`: Exploratory data analysis - distributions, correlations, and feature relationships
- `model_training.ipynb`: Full model training pipeline - preprocessing, cross-validation, hyperparameter 

