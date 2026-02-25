#!/usr/bin/env python3
"""Flask web app for surgery schedule optimizer."""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import io
import json
from datetime import datetime
import sys
import os
import joblib

# Get the directory of the current file 
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to get to Opmed folder
parent_dir = os.path.dirname(current_dir)
# Add part_2 folder to path
part_2_dir = os.path.join(parent_dir, 'part2_optimization')
sys.path.insert(0, part_2_dir)
# Add folder to path (for feature_engineering)
sys.path.insert(0, current_dir)

try:
    # Import optimizer functions
    import optimizer
    assign_rooms = optimizer.assign_rooms
    greedy_assign_anesthetists = optimizer.greedy_assign_anesthetists
    compute_metrics = optimizer.compute_metrics
    OPTIMIZER_AVAILABLE = True
except Exception as e:
    print(f"Warning: Could not import optimizer: {e}", flush=True)
    OPTIMIZER_AVAILABLE = False

# Make FrequencyEncoder and GroupMeanEncoder available in __main__ so that
# joblib can unpickle the pipeline (which was pickled with __main__ references).
try:
    from feature_engineering import FrequencyEncoder, GroupMeanEncoder
    import __main__ as _main_module
    _main_module.FrequencyEncoder = FrequencyEncoder
    _main_module.GroupMeanEncoder = GroupMeanEncoder
except Exception as e:
    print(f"Warning: Could not import feature_engineering: {e}", flush=True)

# Load model artifacts once at startup
_artifacts_dir = os.path.join(current_dir, 'artifacts')
try:
    MODEL = joblib.load(os.path.join(_artifacts_dir, 'best_model.joblib'))
    with open(os.path.join(_artifacts_dir, 'feature_columns.json')) as _f:
        FEATURE_COLUMNS = json.load(_f)
    MODEL_AVAILABLE = True
    print(f"Model loaded. Feature columns: {len(FEATURE_COLUMNS)}", flush=True)
except Exception as e:
    print(f"Warning: Could not load model artifacts: {e}", flush=True)
    MODEL = None
    FEATURE_COLUMNS = []
    MODEL_AVAILABLE = False

# Custom JSON Encoder
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

app = Flask(__name__, static_folder='static')
app.json_encoder = NumpyEncoder

# Preload config to avoid lazy-load issues
BUSINESS_CONFIG = {
    'operating_day_minutes': 480,
    'average_surgery_minutes': 110,
    'work_days_per_year': 250,
    'revenue_per_surgery': 500,
    'manual_buffer_minutes': 25,
    'overrun_rate_manual': 0.05,
    'overrun_rate_model': 0.025,
    'cost_per_overrun': 35,
    'idle_time_value_per_or_per_day': 47.50,
    'num_operating_rooms': 12,
}

# Session config
session_config = {}

def get_config():
    """Get merged config."""
    merged = BUSINESS_CONFIG.copy()
    merged.update(session_config)
    return merged


def build_features(df_raw, expected_cols):
    """Reproduce feature preprocessing from model_training.ipynb.

    Steps mirror the notebook exactly:
      1. Drop Unnamed: 0 if present.
      2. Filter rows where BMI < 10.
      3. Keep surgery_type_raw for GroupMeanEncoder.
      4. One-hot encode Surgery Type (surgery_*) and Anesthesia Type
         (anesthesia_*, drop_first=True).
      5. Engineered features: age_bmi, age_sq, bmi_sq, log1p_bmi.
      6. Age and BMI bucket dummies.
      7. Surgery x patient interactions: surgery_*_x_age, surgery_*_x_bmi.
      8. Reindex to expected_cols with fill_value=0.
    """
    df = df_raw.copy()

    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    df = df[df['BMI'] >= 10].reset_index(drop=True)

    surgery_type_raw = df['Surgery Type'].copy()

    surgery_dummies = pd.get_dummies(df['Surgery Type'], prefix='surgery', dtype=int)
    anesthesia_dummies = pd.get_dummies(
        df['Anesthesia Type'], prefix='anesthesia', drop_first=True, dtype=int
    )

    df = pd.concat(
        [df.drop(columns=['Surgery Type', 'Anesthesia Type']), surgery_dummies, anesthesia_dummies],
        axis=1,
    )
    df['surgery_type_raw'] = surgery_type_raw

    if 'Duration in Minutes' in df.columns:
        df = df.drop(columns=['Duration in Minutes'])

    df['age_bmi'] = df['Age'] * df['BMI']
    df['age_sq'] = df['Age'] ** 2
    df['bmi_sq'] = df['BMI'] ** 2
    df['log1p_bmi'] = np.log1p(df['BMI'])

    age_dummies = pd.get_dummies(
        pd.cut(df['Age'], bins=[0, 30, 45, 60, np.inf],
               labels=['age_lt30', 'age_30_45', 'age_45_60', 'age_60plus']),
        dtype=int,
    )
    bmi_dummies = pd.get_dummies(
        pd.cut(df['BMI'], bins=[0, 18.5, 25, 30, 35, np.inf],
               labels=['bmi_under18', 'bmi_18_25', 'bmi_25_30', 'bmi_30_35', 'bmi_35plus']),
        dtype=int,
    )
    df = pd.concat([df, age_dummies, bmi_dummies], axis=1)

    surgery_cols = [c for c in df.columns if c.startswith('surgery_') and c != 'surgery_type_raw']
    for col in surgery_cols:
        df[f'{col}_x_age'] = df[col] * df['Age']
        df[f'{col}_x_bmi'] = df[col] * df['BMI']

    df = df.reindex(columns=expected_cols, fill_value=0)
    return df

@app.route('/')
def index():
    """Serve main page."""
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health():
    """Health check."""
    return jsonify({"status": "ok"}), 200

@app.route('/predict-duration', methods=['POST'])
def predict_duration():
    """Predict surgery durations using the trained XGBoost model."""
    try:
        if not MODEL_AVAILABLE:
            return jsonify({"error": "Model artifacts not loaded. Ensure artifacts/best_model.joblib and artifacts/feature_columns.json exist."}), 500

        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        if not file or file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # Read CSV
        stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
        df = pd.read_csv(stream)

        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])

        # Duration in Minutes is optional
        required_feature_cols = ['Surgery Type', 'Anesthesia Type', 'Age', 'BMI', 'DoctorID', 'AnaesthetistID']
        missing = [c for c in required_feature_cols if c not in df.columns]
        if missing:
            return jsonify({"error": f"Missing columns: {', '.join(missing)}"}), 400

        has_duration = 'Duration in Minutes' in df.columns

        # Build feature matrix (handles BMI filter internally)
        X = build_features(df, FEATURE_COLUMNS)
        if len(X) == 0:
            return jsonify({"error": "No valid records after filtering BMI < 10"}), 400

        # Real model inference; clamp minimum to 5 minutes
        predicted_durations = np.maximum(MODEL.predict(X).astype(float), 5.0)

        # Align original df with filtered rows (same BMI >= 10 filter)
        df_clean = df[df['BMI'] >= 10].reset_index(drop=True)

        # Compute metrics only when ground-truth durations are present
        manual_mae = 25.0
        manual_acc_15 = 52.0
        if has_duration:
            actual_durations = df_clean['Duration in Minutes'].values.astype(float)
            mae = float(np.mean(np.abs(predicted_durations - actual_durations)))
            mape = float(np.mean(np.abs((predicted_durations - actual_durations) / (actual_durations + 1e-6))) * 100)
            acc_15 = float(np.mean(np.abs(predicted_durations - actual_durations) <= 15) * 100)
        else:
            actual_durations = None
            mae = 15.74   # known 5-fold CV MAE from training
            mape = 18.92
            acc_15 = 49.99

        # Build per-row predictions list
        predictions = []
        for i, pred in enumerate(predicted_durations):
            actual = float(actual_durations[i]) if actual_durations is not None else None
            error = float(abs(pred - actual)) if actual is not None else None
            surgery_type = int(df_clean['Surgery Type'].iloc[i])
            predictions.append({
                'index': int(i),
                'surgery_type': surgery_type,
                'actual': float(round(actual, 1)) if actual is not None else None,
                'predicted': float(round(pred, 1)),
                'error': float(round(error, 1)) if error is not None else None,
                'manual_error': 25,
                'time_saved': float(round(max(0, 25 - error), 1)) if error is not None else None,
            })

        # Business impact
        config = get_config()
        surgeries_per_day = config['operating_day_minutes'] / config['average_surgery_minutes']
        model_buffer = round(mae, 1)
        current_buffer = config['manual_buffer_minutes']

        current_overhead = surgeries_per_day * current_buffer
        model_overhead = surgeries_per_day * model_buffer
        surgeries_gained_daily = (
            (config['operating_day_minutes'] - model_overhead) / config['average_surgery_minutes']
            - (config['operating_day_minutes'] - current_overhead) / config['average_surgery_minutes']
        )
        surgeries_gained_yearly = surgeries_gained_daily * 22 * 12
        additional_revenue_yearly = surgeries_gained_yearly * config['revenue_per_surgery']
        overruns_prevented_yearly = (
            surgeries_per_day * config['overrun_rate_manual']
            - surgeries_per_day * config['overrun_rate_model']
        ) * config['work_days_per_year']
        overtime_savings_yearly = overruns_prevented_yearly * config['cost_per_overrun']
        idle_savings_yearly = (
            config['idle_time_value_per_or_per_day']
            * config['work_days_per_year']
            * config['num_operating_rooms']
        )

        business_impact = {
            'surgeries_gained_daily': float(round(surgeries_gained_daily, 2)),
            'surgeries_gained_monthly': float(round(surgeries_gained_daily * 22, 2)),
            'surgeries_gained_yearly': int(round(surgeries_gained_yearly, 0)),
            'additional_revenue_yearly': int(round(additional_revenue_yearly, 0)),
            'overtime_savings_yearly': int(round(overtime_savings_yearly, 0)),
            'idle_time_savings_yearly': int(round(idle_savings_yearly, 0)),
            'total_annual_benefit': int(round(additional_revenue_yearly + overtime_savings_yearly + idle_savings_yearly, 0)),
        }

        # Per-surgery-type stats (only when actuals are available)
        surgery_stats = {}
        if has_duration:
            for stype in df_clean['Surgery Type'].unique():
                mask = (df_clean['Surgery Type'] == stype).values
                actual_type = actual_durations[mask]
                pred_type = predicted_durations[mask]
                surgery_stats[int(stype)] = {
                    'count': int(mask.sum()),
                    'avg_actual': float(round(actual_type.mean(), 1)),
                    'avg_predicted': float(round(pred_type.mean(), 1)),
                    'avg_error': float(round(np.mean(np.abs(pred_type - actual_type)), 1)),
                }

        improvement = float(round((manual_mae - mae) / manual_mae * 100, 1)) if mae < manual_mae else 0.0

        return jsonify({
            'success': True,
            'metrics': {
                'mae': float(round(mae, 2)),
                'mape': float(round(mape, 2)),
                'acc_15': float(round(acc_15, 1)),
                'manual_mae': float(manual_mae),
                'manual_acc_15': float(round(manual_acc_15, 1)),
                'improvement': improvement,
            },
            'predictions': predictions,
            'business_impact': business_impact,
            'surgery_stats': surgery_stats,
        })

    except Exception as e:
        return jsonify({"error": str(e)[:200]}), 500

@app.route('/optimize', methods=['POST'])
def optimize():
    """Optimize schedule using greedy assignment."""
    try:
        if not OPTIMIZER_AVAILABLE:
            return jsonify({"error": "Optimizer module not available"}), 500
        
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['file']
        if not file or file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Read CSV
        stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
        df = pd.read_csv(stream)
        
        # Check required columns (supports different naming conventions)
        surgeries = []
        for idx, row in df.iterrows():
            try:
                surgery_id = str(row.get('index', row.get('Index', idx)))
                start_time = pd.to_datetime(row.get('start_time', row.get('Start', row.get('start'))))
                end_time = pd.to_datetime(row.get('end_time', row.get('End', row.get('end'))))
                surgeries.append({
                    'id': surgery_id,
                    'start': start_time,
                    'end': end_time
                })
            except Exception as e:
                continue
        
        if not surgeries:
            return jsonify({"error": "No valid surgeries found in file"}), 400
        
        # Sort by start time
        surgeries.sort(key=lambda x: x['start'])
        
        # Assign rooms
        room_ids = assign_rooms(surgeries, num_rooms=20)
        
        # Assign anesthetists
        anest_ids, anesthetists = greedy_assign_anesthetists(surgeries)
        
        # Compute metrics
        metrics = compute_metrics(surgeries, anest_ids, anesthetists)
        
        # Build gantt_data for frontend visualization
        gantt_data = []
        for s, anest_id, room_id in zip(surgeries, anest_ids, room_ids):
            start = s['start']
            end = s['end']
            duration_h = (end - start).total_seconds() / 3600.0
            gantt_data.append({
                'surgery_id': s['id'],
                'anesthetist': f'anesthetist - {anest_id}',
                'room': f'room - {room_id}',
                'start': start.isoformat(sep=' '),
                'end': end.isoformat(sep=' '),
                'duration_h': round(duration_h, 2)
            })
        
        return jsonify({
            'success': True,
            'metrics': metrics,
            'gantt_data': gantt_data,
            'num_results': len(gantt_data)
        })
    
    except Exception as e:
        return jsonify({"error": str(e)[:200]}), 500

@app.route('/config', methods=['GET'])
def get_config_endpoint():
    """Get config."""
    descriptions = {
        'operating_day_minutes': 'Total minutes available for surgeries per operating day',
        'average_surgery_minutes': 'Average duration of a surgery in minutes',
        'work_days_per_year': 'Number of working days per year',
        'revenue_per_surgery': 'Revenue generated per surgery',
        'manual_buffer_minutes': 'Buffer time added manually between surgeries',
        'overrun_rate_manual': 'Percentage of surgeries that exceed manual buffer time estimates',
        'overrun_rate_model': 'Percentage of surgeries that exceed model predictions',
        'cost_per_overrun': 'Cost incurred per surgery overrun',
        'idle_time_value_per_or_per_day': 'Value of idle time per operating room per day',
        'num_operating_rooms': 'Total number of operating rooms available',
    }
    return jsonify({'success': True, 'config': get_config(), 'descriptions': descriptions})

@app.route('/config', methods=['POST'])
def update_config():
    """Update config."""
    try:
        data = request.get_json()
        for key, value in data.items():
            session_config[key] = value
        return jsonify({'success': True, 'config': get_config()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, port=3000, use_reloader=False, threaded=True)
