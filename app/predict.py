import numpy as np
import pandas as pd

def make_predictions(ml_model, lstm_model, X_test, original_test_df):
    """ Generate predictions using ML and LSTM models """
    predictions_ml = ml_model.predict(X_test)
    predictions_dl = lstm_model.predict(np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1)))

    # Store predictions in DataFrame
    results = original_test_df[['Date']].copy()  # Keep Date for plotting
    results['ML_Model_Predictions'] = predictions_ml
    results['DL_Model_Predictions'] = predictions_dl.flatten()

    return results
