import os
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import mlflow
import joblib

# 1. LOAD THE DATA (Using Pandas and Scikit-Learn)
print("Loading California Housing dataset...")
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Split the data: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. SET UP TRACKING (Using MLflow)
# This creates a logical grouping for our experiments
mlflow.set_experiment("california_housing_rf")

# Start a tracking run
with mlflow.start_run():
    # Define our "hyperparameters" (settings for the algorithm)
    n_estimators = 50
    max_depth = 5
    
    # Log these settings to MLflow so we have a record of this exact build
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    
    # 3. TRAIN THE MODEL (Using Scikit-Learn)
    print(f"Training model with n_estimators={n_estimators}, max_depth={max_depth}...")
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    
    # 4. EVALUATE THE MODEL
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    
    # Log the resulting error rate to MLflow
    mlflow.log_metric("mse", mse)
    print(f"Model training complete! Mean Squared Error: {mse:.4f}")
    
    # 5. CREATE THE ARTIFACT (Using Joblib)
    os.makedirs("model_dir", exist_ok=True)
    artifact_path = "model_dir/model.joblib"
    joblib.dump(model, artifact_path)
    print(f"Success: Model artifact saved to {artifact_path}")