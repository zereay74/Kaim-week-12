import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load scaler (Assumes you saved it when training the models)

def preprocess_test_data(df, scaler):
    # Convert Date column to datetime if needed
    df['Date'] = pd.to_datetime(df['Date'])
    df['Weekday'] = df['Date'].dt.weekday
    df['Is_Weekend'] = df['Weekday'].apply(lambda x: 1 if x >= 5 else 0)
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['Is_Beginning_Month'] = (df['Day'] <= 10).astype(int)
    df['Is_Mid_Month'] = ((df['Day'] > 10) & (df['Day'] <= 20)).astype(int)
    df['Is_End_Month'] = (df['Day'] > 20).astype(int)
    
    # Holiday features, etc.
    df['Days_To_Next_Holiday'] = df.groupby('StateHoliday')['Date'].transform(
        lambda x: (x - x.shift(-1)).dt.days.abs()
    )
    df['Days_After_Last_Holiday'] = df.groupby('StateHoliday')['Date'].transform(
        lambda x: (x - x.shift(1)).dt.days
    )
    
    df.fillna(0, inplace=True)
    
    # Only drop 'Id' if present (we keep Date)
    if 'Id' in df.columns:
        df.drop(columns=['Id'], inplace=True)
    
    # Select numeric columns for scaling (Date remains untouched in original_df)
    numeric_columns = df.select_dtypes(include=['number']).columns
    scaled_data = scaler.transform(df[numeric_columns])
    
    return scaled_data, df



'''
def extract_features(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Weekday'] = df['Date'].dt.weekday
    df['Is_Weekend'] = df['Weekday'].apply(lambda x: 1 if x >= 5 else 0)
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['Is_Beginning_Month'] = (df['Day'] <= 10).astype(int)
    df['Is_Mid_Month'] = ((df['Day'] > 10) & (df['Day'] <= 20)).astype(int)
    df['Is_End_Month'] = (df['Day'] > 20).astype(int)

    # Handling holidays
    df['Days_To_Next_Holiday'] = df.groupby('StateHoliday')['Date'].transform(
        lambda x: (x - x.shift(-1)).dt.days.abs()
    )
    df['Days_After_Last_Holiday'] = df.groupby('StateHoliday')['Date'].transform(
        lambda x: (x - x.shift(1)).dt.days
    )

    df.fillna(0, inplace=True)  # Handling NaNs
    return df


def preprocess_test_data(test_df):
    """ Preprocess the test data to match training data """
    test_df = extract_features(test_df)  # Apply feature extraction

    # Stop dropping 'Date' to allow plotting
    test_df.drop(columns=['Id'], inplace=True)

    # Select only numeric columns for scaling
    numeric_columns = test_df.select_dtypes(include=['number']).columns

    # Handle infinite and NaN values
    test_df[numeric_columns] = test_df[numeric_columns].replace([np.inf, -np.inf], np.nan)
    test_df.fillna(0, inplace=True)

    # Scale numeric features
    test_scaled = scaler.transform(test_df[numeric_columns])
    
    return test_scaled, test_df  # Return both scaled and original for plotting
'''