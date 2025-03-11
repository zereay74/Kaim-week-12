
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import os

from model_loader import load_models
from preprocess import preprocess_test_data  # Ensure this function is defined as described

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load test data (assumed to be stored as a pickle to preserve types)
test_df = pd.read_pickle("../week 12 data/Data/rossmann-store-sales/merged_test_data.pkl")  # Ensure this file has the 'Date' column

# Load models and scaler from disk
ml_model, lstm_model, scaler = load_models()

@app.get("/predict")
def predict():
    try:
        # Preprocess test data; ensure Date column is preserved in original_df
        X_test, original_df = preprocess_test_data(test_df, scaler)
        
        # Generate predictions
        ml_predictions = ml_model.predict(X_test)
        lstm_predictions = lstm_model.predict(np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1)))
        
        # Create a DataFrame with the Date and predictions
        response_df = original_df[['Date']].copy()
        response_df["ML_Model_Predictions"] = ml_predictions
        response_df["DL_Model_Predictions"] = lstm_predictions.flatten()
        
        # Convert Date column to string
        response_df["Date"] = response_df["Date"].astype(str)
        
        # Return as a list of JSON records
        return JSONResponse(content=response_df.to_dict(orient="records"))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)


# display in react dash board commandsy
## npx create-react-app sales-dashboard