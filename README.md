# Sales Forecasting and Demand Prediction

This repository contains a machine learning project focused on predicting sales and demand for different product categories. The project includes a detailed Jupyter notebook with data exploration, preprocessing, visualizations, and model training using **XGBoost**.

## Branch Overview

This branch includes the following files:

- **`Sales-Forecasting-and-Demand-Prediction.ipynb`**: A Jupyter notebook that walks through the entire process, from exploratory data analysis to model training and evaluation.
- **`XGBoost_Beverages_Model.pkl`**: Pickle file for the trained XGBoost model for predicting sales of beverages.
- **`XGBoost_Meat_Model.pkl`**: Pickle file for the trained XGBoost model for predicting sales of meat.
- **`.gitattributes`**: Git attributes file to handle large files such as the pickle files efficiently.

## Project Overview

The notebook in this branch contains the following steps:

### 1. **Dataset Exploration**
   - Quick exploration of the dataset to understand the structure and distribution of data.
   - Filtering data to focus on the "units sold" feature for analysis.

### 2. **Preprocessing**
   - Basic preprocessing steps such as data cleaning and handling missing values.
   - Feature engineering to extract valuable insights and features from the data.

### 3. **Data Visualization**
   - Extensive visualizations with Plotly focusing on the units sold.
   - Creating interactive and static plots to understand trends and patterns in the data.

### 4. **Modeling**
   - Experimentation with **Random Forest** and **XGBoost** models.
   - Both models were applied to two filtered item types in the dataset: **beverages** and **meat**.
   - XGBoost was selected as the final model for both item types.

### 5. **Model Evaluation and Interpretation**
   - Evaluating the models using appropriate metrics.
   - Visualizations to interpret the results and feature importance from the XGBoost model.

## Link to Interactive Notebook

You can access the interactive notebook (with all outputs) [here](YOUR_NOTEBOOK_URL_HERE).

> **Note**: The notebook in this branch has been cleaned of outputs for version control, but you can view and interact with the notebook directly through the provided link.

## How to Use

1. Clone this repository:
   ```bash
   git clone https://github.com/Mohamed25092003/Sales-Forecasting-and-Demand-Prediction

2. Ensure you have the required libraries installed:
    ```bash
    pip install -r requirements.txt

3. Open Sales-Forecasting-and-Demand-Prediction.ipynb in your Jupyter Notebook or Google Colab environment.

4. You can use the two XGBoost models (XGBoost_Beverages_Model.pkl and XGBoost_Meat_Model.pkl) for prediction tasks.

## Files in this Branch
	- Sales-Forecasting-and-Demand-Prediction.ipynb: The main Jupyter notebook with detailed analysis and model training.

	- XGBoost_Beverages_Model.pkl: The trained XGBoost model for beverages.

	- XGBoost_Meat_Model.pkl: The trained XGBoost model for meat.

	- .gitattributes: Git attributes file to handle large files.
