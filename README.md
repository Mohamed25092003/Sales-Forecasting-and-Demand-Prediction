# Sales Forecasting and Demand Prediction

This repository contains a machine learning project focused on predicting sales and demand for different product categories. The project includes a detailed Jupyter notebook with data exploration, preprocessing, visualizations, and model training using XGBoost.

## Project Overview

The notebook in this branch contains the following steps:

### 1. Dataset Exploration

* Initial exploration of the dataset to understand its structure and key metrics.
* Focused analysis on **units sold** and **total revenue**.

### 2. Preprocessing

* Data cleaning and handling of missing values.
* Feature engineering to enhance model performance and capture seasonal or categorical trends.

### 3. Data Visualization

* Visualizations using Matplotlib and Seaborn to uncover sales trends and demand patterns.
* Comparison across different product categories, especially fruits and vegetables.

### 4. Modeling

* Applied Random Forest and XGBoost regressors.
* Final XGBoost models selected for:

  * Total revenue prediction.
  * Units sold prediction.
  * Specific demand prediction for **fruits** and **vegetables**.

### 5. Model Evaluation and Interpretation

* Models evaluated using metrics like RMSE, MAE, and R² score.
* Feature importance plotted for deeper insights into what drives sales.

## Link to Interactive Notebook

You can access the interactive version of the notebook (with full outputs) [here](https://colab.research.google.com/drive/18yufPDnhCvpQnZrP1WShxb0hdPEX7_OG?usp=drive_link).

**Note**: The notebook in this branch is cleaned of outputs for better version control. Use the link above for the full interactive version.

## How to Use

Clone this repository:

```bash
git clone https://github.com/YourUsername/Sales-Forecasting-and-Demand-Prediction
```

Ensure you have the required libraries installed:

```bash
pip install -r requirements.txt
```

Open the `Sales-Forecasting-and-Demand-Prediction.ipynb` file in Jupyter Notebook or Google Colab.

You can use the two XGBoost models for prediction:

* `XGBoost_Fruits_Model.pkl`: For fruit demand prediction.
* `XGBoost_Vegetables_Model.pkl`: For vegetable demand prediction.

## Files in this Branch

* `Sales-Forecasting-and-Demand-Prediction.ipynb`: Main notebook with data analysis and model development.
* `XGBoost_Fruits_Model.pkl`: Trained XGBoost model for fruits.
* `XGBoost_Vegetables_Model.pkl`: Trained XGBoost model for vegetables.
* `.gitattributes`: Handles large file tracking via Git LFS if needed.

---

