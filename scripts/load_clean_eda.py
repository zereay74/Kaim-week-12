import pandas as pd
import numpy as np
import logging
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import LabelEncoder
import os
import sys

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
logger = logging.getLogger(__name__)

class DataLoader:
    """Loads and merges Rossmann sales data."""
    def __init__(self, train_path, test_path, store_path):
        self.train_path = train_path
        self.test_path = test_path
        self.store_path = store_path
    
    def load_data(self):
        logging.info("Loading datasets...")
        train = pd.read_csv(self.train_path, parse_dates=['Date'])
        test = pd.read_csv(self.test_path, parse_dates=['Date'])
        store = pd.read_csv(self.store_path)
        
        logging.info("Merging store data with train and test...")
        train = train.merge(store, on='Store', how='left')
        test = test.merge(store, on='Store', how='left')
        
        return train, test, store

class DataCleaner:
    """Handles missing values and outliers."""
    def __init__(self, df):
        self.df = df

    def check_missing_values(self):
        """
        Check for missing values in each column.
        :return: DataFrame with columns, missing value count, and percentage.
        """
        logger.info("Checking for missing values in the DataFrame.")
        missing_info = self.df.isnull().sum()
        missing_percentage = (missing_info / len(self.df)) * 100
        logger.info("Missing values check completed.")
        return pd.DataFrame({
            'Column': self.df.columns,
            'Missing Values': missing_info,
            'Missing Percentage': missing_percentage,
            'Data Type': self.df.dtypes
        }).reset_index(drop=True)

    def fill_missing_values(self):
        logging.info("Handling missing values...")
        self.df.fillna({
            'CompetitionDistance': self.df['CompetitionDistance'].median(),
            'CompetitionOpenSinceYear': self.df['CompetitionOpenSinceYear'].mode()[0],
            'CompetitionOpenSinceMonth': self.df['CompetitionOpenSinceMonth'].mode()[0],
            'Promo2SinceYear': self.df['Promo2SinceYear'].mode()[0],
            'Promo2SinceWeek': self.df['Promo2SinceWeek'].mode()[0],
            'PromoInterval': self.df['PromoInterval'].mode()[0],
            'Open': self.df['Open'].mode()[0]
        }, inplace=True)
        return self.df
    
    def remove_outliers(self):
        logging.info("Removing outliers...")
        self.df = self.df[(np.abs(stats.zscore(self.df['Sales'])) < 3)]
        return self.df

class FeatureEngineer:
    """Creates new features for analysis."""
    def __init__(self, df):
        self.df = df
    
    def extract_date_features(self):
        logging.info("Extracting date features...")
        self.df['Year'] = self.df['Date'].dt.year
        self.df['Month'] = self.df['Date'].dt.month
        self.df['WeekOfYear'] = self.df['Date'].dt.isocalendar().week
        self.df['DayOfWeek'] = self.df['Date'].dt.dayofweek
        return self.df
    
    def encode_categorical(self):
        logging.info("Encoding categorical variables...")
        le = LabelEncoder()
        self.df['StateHoliday'] = le.fit_transform(self.df['StateHoliday'].astype(str))
        self.df['StoreType'] = le.fit_transform(self.df['StoreType'].astype(str))
        self.df['Assortment'] = le.fit_transform(self.df['Assortment'].astype(str))
        return self.df
    
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Configure logging
logging.basicConfig(filename='logs/data_exploration.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

class DataVisualizer:
    def __init__(self, train_df, test_df):
        self.train_df = train_df
        self.test_df = test_df
        logging.info("DataVisualizer initialized with training and test data.")
    
    def plot_promotion_distribution(self):
        plt.figure(figsize=(10, 5))
        sns.histplot(self.train_df['Promo'], label='Train', kde=False, color='blue', alpha=0.6)
        sns.histplot(self.test_df['Promo'], label='Test', kde=False, color='red', alpha=0.6)
        plt.legend()
        plt.title("Promotion Distribution in Train and Test Sets")
        plt.xlabel("Promo")
        plt.ylabel("Count")
        plt.show()
        logging.info("Plotted promotion distribution.")
    
    def plot_sales_during_holidays(self):
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='StateHoliday', y='Sales', data=self.train_df)
        plt.title("Sales Behavior Before, During, and After Holidays")
        plt.xlabel("State Holiday")
        plt.ylabel("Sales")
        plt.show()
        logging.info("Plotted sales behavior during holidays.")
    
    def plot_seasonal_sales(self):
        self.train_df['Month'] = pd.to_datetime(self.train_df['Date']).dt.month
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Month', y='Sales', data=self.train_df)
        plt.title("Seasonal Purchase Behaviors")
        plt.xlabel("Month")
        plt.ylabel("Sales")
        plt.show()
        logging.info("Plotted seasonal sales behavior.")
    
    def plot_sales_vs_customers(self):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='Customers', y='Sales', data=self.train_df, alpha=0.5)
        plt.title("Correlation between Sales and Customers")
        plt.xlabel("Number of Customers")
        plt.ylabel("Sales")
        plt.show()
        logging.info("Plotted sales vs customers correlation.")
    
    def plot_promo_effect_on_sales(self):
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Promo', y='Sales', data=self.train_df)
        plt.title("Effect of Promo on Sales")
        plt.xlabel("Promo")
        plt.ylabel("Sales")
        plt.show()
        logging.info("Plotted effect of promo on sales.")
    
    def plot_customer_behavior_store_opening(self):
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Open', y='Customers', data=self.train_df)
        plt.title("Customer Behavior During Store Openings and Closures")
        plt.xlabel("Store Open (1=Yes, 0=No)")
        plt.ylabel("Customers")
        plt.show()
        logging.info("Plotted customer behavior during store openings.")
    
    def plot_weekday_sales_effect(self):
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='DayOfWeek', y='Sales', data=self.train_df)
        plt.title("Effect of Weekday Openings on Weekend Sales")
        plt.xlabel("Day of the Week")
        plt.ylabel("Sales")
        plt.show()
        logging.info("Plotted weekday sales effect.")
    
    def run_all_visualizations(self):
        self.plot_promotion_distribution()
        self.plot_sales_during_holidays()
        self.plot_seasonal_sales()
        self.plot_sales_vs_customers()
        self.plot_promo_effect_on_sales()
        self.plot_customer_behavior_store_opening()
        self.plot_weekday_sales_effect()
        logging.info("Completed all visualizations.")
''' 
# Example usage
def main():
    loader = DataLoader('train.csv', 'test.csv', 'store.csv')
    train, test = loader.load_data()
    
    cleaner = DataCleaner(train)
    train = cleaner.fill_missing_values()
    train = cleaner.remove_outliers()
    
    fe = FeatureEngineer(train)
    train = fe.extract_date_features()
    train = fe.encode_categorical()
    
    viz = DataVisualizer(train)
    viz.plot_sales_distribution()
    viz.plot_sales_trend()
    viz.plot_sales_by_promo()
    viz.plot_sales_by_holiday()

if __name__ == "__main__":
    main()
'''