# 📈 Sales Forecasting and Demand Prediction App

This Streamlit-based web app predicts **Units Sold** and **Total Revenue** for various item types based on uploaded CSV files. It uses pre-trained XGBoost models to make predictions for individual item categories or for all items at once.

---

## 🚀 Features

- 📊 Predict **units sold** for a selected product category.
- 💵 Predict **total revenue** when "All" is selected.
- 📁 Upload your own CSV and get downloadable predictions.
- ✅ Clean UI powered by Streamlit.
- 🧠 Models trained using XGBoost and joblib/pickle.

---

## 📁 Project Structure

```
.
├── dataset/                  # Sample or input data
│   └── sales_Depi.csv
├── models/                  # Pre-trained models
│   ├── model_all_revenue.pkl
│   ├── model_all_UNITS.pkl
│   └── ... other models ..
|── / 
├── sales_forecasting_app.py # Main Streamlit application
├── Dockerfile               # Docker setup
├── requirement.txt          # Python dependencies
└── README.md                # You are here
```

---

## ⚙️ Installation

Make sure you have **Python 3.8+** and **pip** installed.

### 1. Clone the repo

```bash
git clone https://github.com/your-username/sales-forecasting-app.git
cd sales-forecasting-app
```

### 2. Install dependencies

```bash
pip install -r requirement.txt
```

> 🔧 If using conda:  
> `conda create -n sales-env python=3.8 && conda activate sales-env && pip install -r requirement.txt`

---

## ▶️ Running the App

```bash
streamlit run sales_forecasting_app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 🐳 Running with Docker (Optional)

### 1. Build the image

```bash
docker build -t sales-forecasting-app .
```

### 2. Run the container

```bash
docker run -p 8501:8501 sales-forecasting-app
```

---

## 📤 Using the App

1. Choose an item category (or "All").
2. Upload a CSV file with the required columns.
3. View predictions directly in the browser.
4. Download the results as a new CSV file.

---

## 🧪 Models Info

Each model is stored in the `models/` directory and contains:
- A trained XGBoost model
- A list of feature names (`feature_names`) used during training

For `"all"` selection:
- `model_all_UNITS.pkl`: Predicts Units Sold
- `model_all_revenue.pkl`: Predicts Total Revenue

---

## 📌 Dependencies

- streamlit
- pandas
- numpy
- scikit-learn
- xgboost
- joblib / pickle

> All specified in `requirement.txt`

---

## 📋 Required CSV Format

Your input CSV file must include the following columns **with exact names**:

```
Region, Country, Item Type, Sales Channel, Order Priority, Order Date,
Order ID, Ship Date, Units Sold, Unit Price, Unit Cost,
Total Revenue, Total Cost, Total Profit
```
📊 For additional visualizations and analysis, check out our [Colab notebook](https://colab.research.google.com/drive/1FPQU-gWH4ManVfXh89WLf3mH5MN07baM?usp=sharing).

---

## 👥 Contributors

<p align="left">
  <a href="https://github.com/Mohamed25092003">
    <img src="https://avatars.githubusercontent.com/Mohamed25092003?v=4" width="64" height="64" alt="@Mohamed25092003"/>
  </a>
  <a href="https://github.com/OmarAbdElSalam32">
    <img src="https://avatars.githubusercontent.com/OmarAbdElSalam32?v=4" width="64" height="64" alt="@OmarAbdElSalam32"/>
  </a>
  <a href="https://github.com/mahmoudmenisy">
    <img src="https://avatars.githubusercontent.com/mahmoudmenisy?v=4" width="64" height="64" alt="@mahmoudmenisy"/>
  </a>
  <a href="https://github.com/nassar24">
    <img src="https://avatars.githubusercontent.com/nassar24?v=4" width="64" height="64" alt="@nassar24"/>
  </a>
  <a href="https://github.com/ahmedbenbellaa">
    <img src="https://avatars.githubusercontent.com/ahmedbenbellaa?v=4" width="64" height="64" alt="@ahmedbenbellaa"/>
  </a>
  <a href="https://github.com/mariam-hamdi26">
    <img src="https://avatars.githubusercontent.com/mariam-hamdi26?v=4" width="64" height="64" alt="@mariam-hamdi26"/>
  </a>

