import tensorflow as tf
from keras.losses import MeanAbsoluteError  # Explicitly import MAE


def load_models():
    ml_model_path = "../models/best_ml_model_20250308_124935.pkl"
    lstm_model_path = "../models/best_lstm_model_20250308_133927.h5"
    import joblib
    scaler = joblib.load("../models/scaler.pkl") 

    # Load ML Model
    import joblib
    with open(ml_model_path, 'rb') as file:
        ml_model = joblib.load(file)

    # Register custom loss function
    custom_objects = {"mae": MeanAbsoluteError()}  

    # Load LSTM Model with custom_objects
    lstm_model = tf.keras.models.load_model(lstm_model_path, custom_objects=custom_objects)

    return ml_model, lstm_model, scaler
