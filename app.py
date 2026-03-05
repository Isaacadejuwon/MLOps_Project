from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# 1. Initialize the web application
app = FastAPI(title="California Housing ML API")

# 2. Load the build artifact (the model) into memory when the server starts
model = joblib.load("model_dir/model.joblib")

# 3. Define the exact JSON structure the API expects
class HouseFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

# 4. Create the prediction endpoint
@app.post("/predict")
def predict_price(features: HouseFeatures):
    # Convert the incoming JSON payload into a format the model understands
    # We use pandas to keep the feature names aligned with how it was trained
    input_df = pd.DataFrame([features.dict()])
    
    # Feed the data to the model to get a prediction
    prediction = model.predict(input_df)
    
    # Return the result as a standard JSON response
    return {"predicted_price_in_100k": float(prediction[0])}