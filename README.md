# NYC Airbnb Price Prediction & Analytics (Semester Project)

## Overview
This project analyzes the 2019 New York City Airbnb dataset to uncover trends in rental prices, popularity, and location. It features a comprehensive Jupyter Notebook for data science workflows and a Streamlit Dashboard for interactive exploration.

## Directory Structure
*   `nyc_airbnb_analysis.ipynb`: The core analysis notebook (Cleaning, EDA, ML).
*   `app.py`: Interactive Streamlit Dashboard.
*   `AB_NYC_2019.csv`: Dataset (downloaded automatically).
*   `nyc_airbnb_model.pkl`: Trained Machine Learning model (Random Forest).
*   `requirements.txt`: Python dependencies.

## Key Features
1.  **Exploratory Data Analysis**:
    *   Geospatial mapping of listings.
    *   Price distribution analysis.
    *   Borough-wise comparisons.
2.  **Machine Learning**:
    *   Price Prediction model using **Random Forest Regressor**.
    *   Feature Importance analysis.
3.  **Dashboard**:
    *   Filter listings by Borough, Room Type, and Price.
    *   Interactive Map.
    *   Real-time Price Predictor form.

## How to Run

### 1. Setup
Install requirements:
```bash
pip install -r requirements.txt
```

### 2. Run Analysis & Train Model
Execute the notebook or the generator script to prepare the data and model:
```bash
python generate_project_notebook.py
# Or open nyc_airbnb_analysis.ipynb and Run All
```

### 3. Launch Dashboard
Start the web application:
```bash
streamlit run app.py
```
