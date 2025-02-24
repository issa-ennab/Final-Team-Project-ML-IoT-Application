# Road Condition Prediction API

This API provides endpoints for predicting road conditions using trained machine learning models. It supports LSTM, GRU, and Random Forest models for predictions based on vehicle sensor data.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the API:
```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

## API Endpoints

### Web Interface
- `GET /`: Web interface for uploading sensor data and getting predictions

### Prediction Endpoints
- `POST /predict/lstm`: Get predictions using the LSTM model
- `POST /predict/rf`: Get predictions using the Random Forest model
- `POST /prdecit/grul`: Get predictions using the GRU model

## Input Data Format

The API expects sensor data in JSON format:

```json
{
    "readings": [
        [acc_x_left, acc_y_left, acc_z_left, acc_x_right, acc_y_right, acc_z_right, gyro_x, gyro_y, gyro_z],
        // ... more readings (20 timesteps for LSTM)
    ]
}
```

### Required Features:
- `acc_x_dashboard_left`: Left dashboard X-axis acceleration
- `acc_y_dashboard_left`: Left dashboard Y-axis acceleration
- `acc_z_dashboard_left`: Left dashboard Z-axis acceleration
- `acc_x_dashboard_right`: Right dashboard X-axis acceleration
- `acc_y_dashboard_right`: Right dashboard Y-axis acceleration
- `acc_z_dashboard_right`: Right dashboard Z-axis acceleration
- `gyro_x_dashboard_left`: Left dashboard X-axis gyroscope
- `gyro_y_dashboard_left`: Left dashboard Y-axis gyroscope
- `gyro_z_dashboard_left`: Left dashboard Z-axis gyroscope

### Model Requirements:
- LSTM model requires 20 timesteps of sensor data
- Random Forest model can work with a single timestep
- GRU: requires mulitiple steps of sensor data

## Response Format

```json
{
    "road_type": "Asphalt|Cobblestone|Dirt",
    "confidence": 0.95  // Prediction confidence between 0 and 1
}
```


## Models

The API uses two trained models:
1. LSTM Model: Optimized for sequence-based prediction
2. Random Forest Model: For instant predictions based on current readings
3. Gru Model: Fast and accurate predictions with lower computational overhead compared to LSTM.

All models were trained on a dataset of vehicle sensor readings collected from various road conditions. 