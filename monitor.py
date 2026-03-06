import pandas as pd
from scipy.stats import ks_2samp

print("Loading baseline data...")
reference_data = pd.read_csv("model_dir/baseline_data.csv")

print("Simulating drifted production data...")
current_data = reference_data.copy()
# Aggressively alter the data to simulate a shifting housing market
current_data['MedInc'] = current_data['MedInc'] * 1.5  # Incomes go up 50%
current_data['HouseAge'] = current_data['HouseAge'] * 0.5 # Houses get newer

print("\n--- REAL-TIME DATA DRIFT REPORT ---")
drift_count = 0
features_to_monitor = ['MedInc', 'HouseAge', 'AveRooms', 'Population']

for column in features_to_monitor:
    # The KS test calculates the difference between the two distributions
    stat, p_value = ks_2samp(reference_data[column], current_data[column])
    
    # A p-value under 0.05 means the data has definitively drifted
    if p_value < 0.05:
        print(f"⚠️ DRIFT DETECTED: '{column}' (p-value: {p_value:.4f})")
        drift_count += 1
    else:
        print(f"✅ STABLE: '{column}'")

print("-" * 35)
if drift_count > 0:
    print(f"CRITICAL: {drift_count} feature(s) drifted.")
    print("ACTION: Triggering automated MLOps retraining pipeline...")
else:
    print("System healthy. No retraining needed.")