
import os
import sys
import logging
import pandas as pd
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from tqdm import tqdm  # Import tqdm for progress bar
from sklearn.inspection import permutation_importance

# Set up logging
log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'logs.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)


class SalesDataProcessor:
    def __init__(self, train_path, test_path, store_path):
        """
        Initialize the SalesDataProcessor with file paths.
        """
        self.train_path = train_path
        self.test_path = test_path
        self.store_path = store_path
        self.df_train = None
        self.df_test = None
        self.df_store = None

    def load_data(self):
        """
        Load the train, test, and store datasets.
        """
        try:
            self.df_train = pd.read_csv(self.train_path, parse_dates=['Date'])
            self.df_test = pd.read_csv(self.test_path, parse_dates=['Date'])
            self.df_store = pd.read_csv(self.store_path)
            logging.info("Data loaded successfully!")
        except Exception as e:
            logging.error(f"Error loading data: {e}")

    def merge_data(self):
        """
        Merge store data with train and test datasets on 'Store' column.
        Returns the merged train and test DataFrames.
        """
        try:
            # Merge store data with train
            self.df_train = self.df_train.merge(self.df_store, on='Store', how='left')
            # Merge store data with test
            self.df_test = self.df_test.merge(self.df_store, on='Store', how='left')

            # Drop 'Customers' from train since it's not in test
            if 'Customers' in self.df_train.columns:
                self.df_train.drop(columns=['Customers'], inplace=True)

            logging.info("Merging completed successfully!")
            return self.df_train, self.df_test  # Return merged datasets
        except Exception as e:
            logging.error(f"Error during merging: {e}")
            return None, None

    def check_missing_values(self, df):
            """ Check for missing values in dataset """
            missing_info = df.isnull().sum()
            missing_percentage = (missing_info / len(df)) * 100
            return pd.DataFrame({
                'Column': df.columns,
                'Missing Values': missing_info,
                'Missing Percentage': missing_percentage,
                'Data Type':df.dtypes
            }).reset_index(drop=True)
    
    def drop_null_values(self, df):
        """
        Drop rows with null values from the given DataFrame and save the cleaned dataset.
        Args:
            df (pd.DataFrame): DataFrame to clean.
            save_path (str): Path to save the cleaned DataFrame.
        Returns:
            Modified DataFrame (nulls removed permanently).
        """
        try:
            before_drop = len(df)
            df.dropna(inplace=True)
            after_drop = len(df)

            dropped_rows = before_drop - after_drop
            logging.info(f"Dropped {dropped_rows} rows.")

            #return df
        except Exception as e:
            logging.error(f"Error dropping null values: {e}")
            return None
# Example Usage:
# processor = SalesDataProcessor("train.csv", "test.csv", "store.csv")
# processor.load_data()
# train_df, test_df = processor.merge_data()
# print(processor.check_missing_values("train"))
# print(processor.check_missing_values("test"))


# Define custom tags for models (instead of using sklearn's Tags)
MODEL_TAGS = {
    'RandomForest': 'Regressor',
    'DecisionTree': 'Regressor',
    'LSTM': 'Deep Learning Model'
}

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SalesPredictionPipeline:
    def __init__(self, train_df, test_df):
        self.train_df = train_df
        self.test_df = test_df
        self.scaler = StandardScaler()
        self.best_ml_model = None
        self.best_dl_model = None
    
    def extract_features(self, df):
        df['Date'] = pd.to_datetime(df['Date'])
        df['Weekday'] = df['Date'].dt.weekday
        df['Is_Weekend'] = df['Weekday'].apply(lambda x: 1 if x >= 5 else 0)
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df['Is_Beginning_Month'] = (df['Day'] <= 10).astype(int)
        df['Is_Mid_Month'] = ((df['Day'] > 10) & (df['Day'] <= 20)).astype(int)
        df['Is_End_Month'] = (df['Day'] > 20).astype(int)
        
        df['Days_To_Next_Holiday'] = df.groupby('StateHoliday')['Date'].transform(lambda x: (x - x.shift(-1)).dt.days.abs())
        df['Days_After_Last_Holiday'] = df.groupby('StateHoliday')['Date'].transform(lambda x: (x - x.shift(1)).dt.days)
        
        df.fillna(0, inplace=True)  # Handling NaNs
        return df
    
    def preprocess_data(self):
        logging.info("Extracting features for train and test datasets.")
        
        # Extract features for both train and test
        self.train_df = self.extract_features(self.train_df)
        self.test_df = self.extract_features(self.test_df)
        
        # Drop unnecessary columns
        self.train_df.drop(columns=['Date'], inplace=True)
        self.test_df.drop(columns=['Date', 'Id'], inplace=True)
        
        # Separate target variable 'Sales' from features
        X = self.train_df.drop(columns=['Sales'])
        y = self.train_df['Sales']
        
        # Select only numeric columns for scaling
        numeric_columns = X.select_dtypes(include=['number']).columns
        
        # Ensure there are no NaN or infinite values
        X[numeric_columns] = X[numeric_columns].replace([np.inf, -np.inf], np.nan)
        X.fillna(0, inplace=True)
        
        # Split the data into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale the data using StandardScaler only on numeric columns
        X_train = self.scaler.fit_transform(X_train[numeric_columns])
        X_val = self.scaler.transform(X_val[numeric_columns])
        X_test = self.scaler.transform(self.test_df[numeric_columns])
        
        return X_train, X_val, y_train, y_val, X_test

    
    def train_models(self, X_train, X_val, y_train, y_val):
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'DecisionTree': DecisionTreeRegressor(random_state=42)
        }

        best_model = None
        best_loss = float('inf')

        logging.info("Training ML models...")

        # Get correct feature names from the numeric columns before scaling
        feature_names = self.train_df.drop(columns=['Sales']).select_dtypes(include=['number']).columns.tolist()

        for name, model in models.items():
            logging.info(f"Training {name}...")

            model.fit(X_train, y_train)
            predictions = model.predict(X_val)

            mae = mean_absolute_error(y_val, predictions)
            mse = mean_squared_error(y_val, predictions)

            logging.info(f"{name} - MAE: {mae}, MSE: {mse}")

            if mae < best_loss:  # Selecting best model based on MAE
                best_loss = mae
                best_model = model

        self.best_ml_model = best_model
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_filename = f'best_ml_model_{timestamp}.pkl'
        joblib.dump(best_model, model_filename)
        logging.info(f"Best ML model saved as {model_filename}!")

        # Feature Importance Display
        def display_feature_importance(model, feature_names):
            if hasattr(model, "feature_importances_"):  # For tree-based models
                importances = model.feature_importances_
                if len(feature_names) != len(importances):
                    logging.warning("Feature names and importance values have different lengths!")
                    return
                
                importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
                importance_df = importance_df.sort_values(by="Importance", ascending=False)
                logging.info("\nFeature Importance:\n")
                logging.info(importance_df)

        # Display feature importance for the best model
        display_feature_importance(best_model, feature_names)

    def train_lstm(self, X_train, X_val, y_train, y_val):
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))

        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
            Dropout(0.2),
            LSTM(50),
            Dense(1)
        ])

        model.compile(optimizer='adam', loss='mae')

        logging.info("Training LSTM...")
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=32, verbose=1)

        predictions = model.predict(X_val)

        mae = mean_absolute_error(y_val, predictions)
        mse = mean_squared_error(y_val, predictions)

        logging.info(f"LSTM - MAE: {mae}, MSE: {mse}")

        self.best_dl_model = model
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_filename = f'best_lstm_model_{timestamp}.h5'
        model.save(model_filename)
        logging.info(f"Best LSTM model saved as {model_filename}!")

    
    def load_model(self, model_filename):
        if model_filename.endswith('.pkl'):
            self.best_ml_model = joblib.load(model_filename)
            logging.info(f"ML model loaded from {model_filename}")
        elif model_filename.endswith('.h5'):
            self.best_dl_model = tf.keras.models.load_model(model_filename)
            logging.info(f"DL model loaded from {model_filename}")
    
    def predict(self, X_test):
        logging.info("Generating predictions...")
        predictions_ml = self.best_ml_model.predict(X_test)
        predictions_dl = self.best_dl_model.predict(np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1)))
        
        # Create a dataframe to view predictions
        results = pd.DataFrame({
            'ML_Model_Predictions': predictions_ml,
            'DL_Model_Predictions': predictions_dl.flatten()  # Flatten to match shape
        })
        
        return results

  

# Usage Example:
# train_df = pd.read_csv('cleaned_train.csv')
# test_df = pd.read_csv('cleaned_test.csv')
# pipeline = SalesPredictionPipeline(train_df, test_df)
# X_train, X_val, y_train, y_val, X_test = pipeline.preprocess_data()
# pipeline.train_models(X_train, X_val, y_train, y_val)
# pipeline.train_lstm(X_train, X_val, y_train, y_val)
# pipeline.load_model('best_ml_model_20230308_103212.pkl')  # Example to load the model
# predictions = pipeline.predict(X_test)
# pipeline.plot_predictions(X_test, y_val)
