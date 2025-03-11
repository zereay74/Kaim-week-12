# Rossmann Sales Forecasting - Data-Driven Decision Making!

## Project Overview
This project aims to build an end-to-end machine learning solution to predict sales six weeks ahead for Rossmann stores. The solution incorporates:
- **Data Cleaning & Preprocessing**: Handling missing values, outliers, and performing feature engineering (including extracting Date-based features, holiday effects, and promotion statuses).
- **Exploratory Data Analysis (EDA)**: Investigating customer purchasing behavior, promotional impact, and seasonal trends to derive key insights.
- **Machine Learning Modeling**: Developing predictive models for sales forecasting using Decision Tree, Random Forest and LSTM networks, where the Random Forest model performed exceptionally well.
- **Model Serving API Call**: Exposing the prediction results via a FastAPI (or Flask) model serving API, which is then consumed by a React dashboard.
- **Logging & Monitoring**: Keeping track of model performance and pipeline integrity.
- **CI/CD & MLOps**: Automating workflows using GitHub Actions for continuous integration, testing, and deployment.

## Folder Structure
```
├── .github/workflows     # GitHub Actions workflows for CI/CD
├── .vscode               # VS Code settings and extensions
├── app                   # FastAPI and React app
│   ├── sales_dashboard/  # React app files
│   ├── main.py           # Main Python app for FastAPI
│   ├── model_loader.py   # Loads ML (pkl) and LSTM (h5) models and scaler
│   ├── predict.py        # Prediction logic and API endpoint
│   ├── preprocess.py     # Data preprocessing functions
├── logs                  # Logs for monitoring outputs
│   ├── logs.log
├── notebooks             # Jupyter notebooks for data processing & insights
│   ├── Task_1_Exploration_of_Customer_Purchasing_Behavior.ipynb
│   ├── Task_2_Prediction_of_Store_Sales.ipynb
├── scripts               # Python scripts for automation (data load, clean, EDA)
│   ├── load_clean_eda.py # merges train, test, and store dataframes, then performs EDA
│   ├── feature_and_prediction.py # # merges train, test, and store dataframes then removes null values for the sake of reducing computation (we can impute missing values using appropriate strategy)
├── tests                 # Unit tests for data validation & pipeline integrity
├── .gitignore            # Ignore unnecessary files
├── README.md             # Project documentation
├── requirements.txt      # Dependencies
```

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/zereay74/Kaim-week-12.git
   cd Kaim-week-12
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Data Loading & Cleaning
Run the notebook Task_1_Exploration_of_Customer_Purchasing_Behavior.ipynb

### 2. Exploratory Data Analysis (EDA)
Run the notebook Task_1_Exploration_of_Customer_Purchasing_Behavior.ipynb

### 3. Model Training & Prediction
- **Train Models:**  
  Open the notebook  Task_2_Prediction_of_Store_Sales.ipynb  and follow steps
- **Load Models & Predict:**  
  Open the notebook  Task_2_Prediction_of_Store_Sales.ipynb # remember to export the merged test dataframe to pkl for later use in predictions.
### 4. Model Serving API & Dashboard
- **Backend API:**  
  For the mergerd, cleaned, and exported test dataframe The FastAPI (or Flask) app exposes the predictions via an endpoint (`/predict`). It loads the preprocessed test data, applies the saved scaler, and returns predictions (with Date values preserved for visualization).
  in the terminal and change dir (cd app ) to app then run command to see predictions
  ```bash
  uvicorn main:app --reload 
  ```
- **React Dashboard:**  
  The React app fetches the prediction data from the API, displaying it in both a table and an integrated plot (showing Date vs. Model Predictions). The dashboard allows date-range filtering for improved visibility.
  open additional terminal without closing previous one then change directory to (app/sales_dashboard) to see the predictions the graph is shown at the end of the page.
  ```bash
  npm start 
  ```

## Key Features
✅ **Automated Data Cleaning**: Handles missing values, outliers, and performs robust feature engineering.  
✅ **Exploratory Analysis**: Provides insights into customer behavior, promotional impact, and seasonal sales trends.  
✅ **Scalable ML Pipeline**: Combines traditional ML (Random Forest) and deep learning (LSTM) for accurate sales forecasting.  
✅ **Real-Time Model Serving**: Exposes predictions via an API for integration with modern dashboards.  
✅ **Interactive React Dashboard**: Visualizes predictions through tables and plots with date filtering capabilities.  
✅ **Logging & Monitoring**: Tracks pipeline execution and model performance.  
✅ **CI/CD Integration**: Utilizes GitHub Actions to automate testing and deployment workflows.

## Requirements
- Python 3.8+
- Pandas, NumPy
- Matplotlib, Seaborn, SciPy
- Scikit-learn
- TensorFlow
- FastAPI, Uvicorn (or Flask, if preferred)
- React (Create React App or Vite)
- Axios, Recharts (for the React frontend)

Install dependencies with:
```bash
pip install -r requirements.txt
```

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-name`).
5. Open a pull request.

## License
This project is licensed under the MIT License.

---

This README outlines the entire project workflow—from data preprocessing and EDA to model training, API serving, and dashboard visualization—providing a complete picture of how Rossmann Pharmaceuticals can leverage data-driven insights for sales forecasting.

Enjoy exploring and enhancing the project! 🚀