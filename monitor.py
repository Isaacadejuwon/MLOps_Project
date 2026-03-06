import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

print("Loading baseline data...")
# 1. Load the "normal" data we saved during training
reference_data = pd.read_csv("model_dir/baseline_data.csv")

# 2. Simulate "Current" Production Data 
# We are making a copy and aggressively altering it to simulate drift.
# Let's pretend median incomes skyrocketed and house ages dropped.
print("Simulating drifted production data...")
current_data = reference_data.copy()
current_data['MedInc'] = current_data['MedInc'] * 1.5  # Incomes go up 50%
current_data['HouseAge'] = current_data['HouseAge'] * 0.5 # Houses are newer

# 3. Generate the Drift Report
print("Analyzing statistical data drift...")
# We use Evidently's built-in DataDriftPreset to check every single column
drift_report = Report(metrics=[DataDriftPreset()])
drift_report.run(reference_data=reference_data, current_data=current_data)

# 4. Save the output
drift_report.save_html("drift_report.html")
print("Complete! Open 'drift_report.html' in your web browser to view the dashboard.")