from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import numpy as np
import tensorflow as tf
import pickle
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler
import json
import os
from pathlib import Path
import logging
import joblib

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Road Condition Prediction API")

# Get the base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Mount static files
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "api" / "static")), name="static")

# Templates
templates = Jinja2Templates(directory=str(BASE_DIR / "api" / "templates"))

# Global variables for models
lstm_model = None
random_forest_model = None
gru_model = None

# Define input features
FEATURES = [
    'acc_x_dashboard_left', 'acc_y_dashboard_left', 'acc_z_dashboard_left',
    'acc_x_dashboard_right', 'acc_y_dashboard_right', 'acc_z_dashboard_right',
    'gyro_x_dashboard_left', 'gyro_y_dashboard_left', 'gyro_z_dashboard_left'
]

class SensorData(BaseModel):
    readings: list[list[float]]  # List of sensor readings

def load_models():
    global lstm_model, random_forest_model, gru_model
    
    try:
        # Load LSTM model (.keras format)
        lstm_path = BASE_DIR / "api" / "models" / "lstm_road_condition_model_optimized.keras"
        if lstm_path.exists():
            try:
                lstm_model = tf.keras.models.load_model(str(lstm_path), compile=False)
                lstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                logger.info("LSTM model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading LSTM model: {str(e)}")
                lstm_model = None
        
        # Load Random Forest model (pickle format)
        rf_path = BASE_DIR / "api" / "models" / "random_forest_model.pkl"
        if rf_path.exists():
            with open(rf_path, 'rb') as f:
                random_forest_model = pickle.load(f)
                logger.info("Random Forest model loaded successfully")
        
        # Load GRU model (pickle format)
        gru_path = BASE_DIR / "api" / "models" / "gru_model.pkl"
        if gru_path.exists():
            with open(gru_path, 'rb') as f:
                gru_model = pickle.load(f)
                logger.info("GRU model loaded successfully")
        
        # Log which models are available
        logger.info("\nModel Loading Status:")
        logger.info(f"LSTM Model: {'Available' if lstm_model is not None else 'Not Available'}")
        logger.info(f"Random Forest Model: {'Available' if random_forest_model is not None else 'Not Available'}")
        logger.info(f"GRU Model: {'Available' if gru_model is not None else 'Not Available'}")
    
    except Exception as e:
        logger.error(f"Error during model loading: {str(e)}")

# Load models at startup
load_models()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "lstm_available": lstm_model is not None,
        "rf_available": random_forest_model is not None,
        "gru_available": gru_model is not None
    })

@app.post("/predict/lstm")
async def predict_lstm(data: SensorData):
    if lstm_model is None:
        raise HTTPException(status_code=503, detail="LSTM model is not available")
    
    try:
        # Prepare data for LSTM
        X = np.array(data.readings)
        
        # Check input shape
        if X.shape != (20, 9):
            raise HTTPException(
                status_code=400, 
                detail=f"Expected input shape (20, 9), got {X.shape}"
            )
        
        # Normalize data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Reshape for LSTM (add batch dimension)
        X_reshaped = X_scaled.reshape(1, 20, 9)
        
        # Make prediction
        prediction = lstm_model.predict(X_reshaped)
        
        # Get road type and confidence
        road_types = ['Asphalt', 'Cobblestone', 'Dirt']
        predicted_class = np.argmax(prediction[0])
        confidence = float(prediction[0][predicted_class])
        
        # Get probabilities for all classes
        probabilities = {
            road_type: float(prob) 
            for road_type, prob in zip(road_types, prediction[0])
        }
        
        return {
            "road_type": road_types[predicted_class],
            "confidence": confidence,
            "probabilities": probabilities
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/rf")
async def predict_random_forest(data: SensorData):
    if random_forest_model is None:
        raise HTTPException(status_code=503, detail="Random Forest model is not available")
    
    try:
        # Prepare data
        X = np.array(data.readings)
        
        # Normalize data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Make prediction
        prediction = random_forest_model.predict(X_scaled)
        probabilities = random_forest_model.predict_proba(X_scaled)
        
        # Get road type
        road_types = ['Asphalt', 'Cobblestone', 'Dirt']
        predicted_class = prediction[0]
        
        return {
            "road_type": road_types[predicted_class],
            "confidence": float(probabilities[0][predicted_class])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/gru")
async def predict_gru(data: SensorData):
    if gru_model is None:
        raise HTTPException(status_code=503, detail="GRU model is not available")
    
    try:
        # Prepare data for GRU
        X = np.array(data.readings)
        
        # Check input shape
        if X.shape != (20, 9):
            raise HTTPException(
                status_code=400, 
                detail=f"Expected input shape (20, 9), got {X.shape}"
            )
        
        # Normalize data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Reshape for GRU (add batch dimension)
        X_reshaped = X_scaled.reshape(1, 20, 9)
        
        # Make prediction
        prediction = gru_model.predict(X_reshaped)
        
        # Get road type and confidence
        road_types = ['Asphalt', 'Cobblestone', 'Dirt']
        predicted_class = np.argmax(prediction[0])
        confidence = float(prediction[0][predicted_class])
        
        # Get probabilities for all classes
        probabilities = {
            road_type: float(prob) 
            for road_type, prob in zip(road_types, prediction[0])
        }
        
        return {
            "road_type": road_types[predicted_class],
            "confidence": confidence,
            "probabilities": probabilities
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 