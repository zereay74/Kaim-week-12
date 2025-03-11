# Scripts

This folder contains Python scripts that automate various parts of the Rossmann Sales Forecasting project workflow. These scripts are essential for data loading, cleaning, exploratory data analysis, and running the end-to-end prediction pipeline.

## Included Scripts

- **load_clean_eda.py**:  
  - Loads raw data from CSV files.
  - merges train, test, and store dataframes, then performs EDA
  - Cleans the data by handling missing values, outliers, and basic feature engineering.
  - Performs initial exploratory data analysis (EDA) and generates summary insights.

- **feature_and_prediction.py**:  
  - Implements the complete sales prediction pipeline.
  - merges train, test, and store dataframes then removes null values for the sake of reducing computation (we can impute missing values using appropriate strategy)
  - Extracts and engineers features from the data (including date-based, holiday-related, and promotional features).
  - Preprocesses the data (scaling numeric columns, handling missing values, etc.).
  - Trains predictive models (Random Forest and LSTM) for forecasting store sales.
  - Generates predictions using the trained models and exposes them for serving.

*(Additional scripts may be added as the project evolves.)*

## How to Use

1. **implement it in jupyter notebook (already imported and implemented):**
   ```bash
   cd scripts
