# Forecasting Asset Class Return and Volatility Using LSTM and Temporal Fusion Transformer

Final Year Project – John Wallace  
Trinity College Dublin (2025)  

---

## 🧠 Overview

This project investigates the predictive capabilities of deep learning models — specifically **LSTM** and **Temporal Fusion Transformer (TFT)** — for forecasting **monthly return and volatility** across three key asset classes:

- **Bonds** (Bloomberg U.S. Aggregate Bond Index)
- **Equities** (S&P 500 Index)
- **Gold** (Gold Futures Spot Price)

The forecasts are then used to dynamically allocate capital using a **Mean-Variance Optimization (MVO)** strategy and benchmarked against a traditional 60/40 portfolio.

---

## 📁 Project Structure

ASSET ALLOCATION PUBLIC/
├── Allocation/
│ └── MVO_creation.ipynb # Portfolio optimization (MVO)
│
├── models/
│ ├── hyperparameter search logic/
│ │ ├── LSTM_Param_Template.py
│ │ ├── tensor_search.py
│ │ └── TFT_Param_Template.py
│ ├── Input Variables (DUMMY)/
│ │ └── dummy_variables.csv # Sample input variables
│ └── TFT + LSTM training scripts/
│ ├── Bond_return_training.py
│ └── LSTM_Train_Template.py
│
├── Portfolio simulation/
│ └── Backtesting.ipynb # Portfolio strategy simulation
│
├── Predictions/
│ ├── LSTM Predictions/
│ │ ├── Bond_Test_Return_LSTM_Predictions.csv
│ │ ├── Bond_Test_Vol_LSTM_Predictions.csv
│ │ ├── Equity_Test_Return_LSTM_Predictions.csv
│ │ ├── Equity_Test_Vol_LSTM_Predictions.csv
│ │ ├── Gold_Test_Return_LSTM_Predictions.csv
│ │ └── Gold_Test_Vol_LSTM_Predictions.csv
│ ├── TFT Predictions/
│ │ ├── LSTM_predict.py
│ │ └── TFT_predict.py
│ └── metrics.ipynb # Accuracy & interval evaluation
│
├── Disclaimer.md # Project assumptions & scope
├── FYP_Project_2025.pdf # Final thesis document
├── README.md # This file
└── thesis graph.png # Visual comparison of portfolios



---

## 🎯 Project Objectives

- 📈 Forecast **1-month ahead return and volatility** for Bonds, Equities, and Gold.
- 📊 Generate **quantile predictions** (5%, 50%, 95%) for uncertainty-aware forecasting.
- ⚙️ Train and tune **LSTM** and **TFT** models using structured macroeconomic and asset features.
- 🧠 Evaluate with standard and interval-based metrics.
- 💼 Construct dynamic portfolios using **MVO** based on forecasted outputs.
- 📉 Benchmark performance vs. a traditional 60/40 portfolio.

---

## 📊 Model Features

### Input Variables:
- CPI (actual & forecast)
- PMI (actual)
- Unemployment Rate
- 1-Year and 10-Year T-Bill Rates
- Prior month’s asset return and volatility

### Quantile Output Targets:
- 5th percentile (Predicted_5%)
- 50th percentile (Predicted_50%) — Median
- 95th percentile (Predicted_95%)

---

## 🧪 Model Performance

| Asset  | Target     | Best Model | R²     | MAE    | RMSE   | Coverage | Interval Score |
|--------|------------|------------|--------|--------|--------|----------|----------------|
| Bond   | Return     | **TFT**    | 0.86   | 0.022  | 0.029  | 91%      | 0.068          |
| Bond   | Volatility | LSTM       | 0.71   | 0.011  | 0.017  | 93%      | 0.055          |
| Equity | Return     | **TFT**    | 0.88   | 0.031  | 0.042  | 90%      | 0.075          |
| Equity | Volatility | **TFT**    | 0.76   | 0.012  | 0.018  | 91%      | 0.058          |
| Gold   | Return     | LSTM       | 0.72   | 0.034  | 0.049  | 88%      | 0.082          |
| Gold   | Volatility | **TFT**    | 0.69   | 0.013  | 0.020  | 90%      | 0.061          |

> Evaluation conducted using rolling prediction on post-training test sets (July 2024 – Feb 2025).

---

## 📈 Portfolio Performance (Backtest)

| Metric             | MVO Portfolio | 60/40 Benchmark |
|--------------------|---------------|-----------------|
| Total Return        | **13.00%**    | 8.08%           |
| Annualized Return   | **23.33%**    | 14.24%          |
| Sharpe Ratio        | **0.9028**    | 0.5236          |
| Max Drawdown        | -7.70%        | -9.12%          |

✅ The MVO portfolio outperformed across all key metrics, demonstrating the practical utility of monthly ML-based forecasts.

---

## 🧭 How to Run This Project (Workflow)

This project is designed to be run **modularly** through Python scripts and Jupyter notebooks.

### Step 1: Hyperparameter Tuning (Optional)
- Navigate to `/models/hyperparameter search logic/`
- Use `LSTM_Param_Template.py` or `TFT_Param_Template.py` with `tensor_search.py` to explore optimal parameters using Optuna.

### Step 2: Model Training
- Go to `/models/TFT + LSTM training scripts/`
- Run `Bond_return_training.py` or `LSTM_Train_Template.py` manually.
- Select the appropriate asset and target (return or volatility) before execution.

### Step 3: Rolling Prediction
- Go to `/Predictions/TFT Predictions/`
- Run `TFT_predict.py` or `LSTM_predict.py` to generate rolling forecasts for each asset class.
- Predictions will be saved to CSV files in `LSTM Predictions/` or similar.

### Step 4: Evaluation & Metrics
- Open `Predictions/metrics.ipynb` to compute:
  - MAE, RMSE, R²
  - Coverage rate and interval scoring for confidence bands

### Step 5: Portfolio Construction
- Open `/Allocation/MVO_creation.ipynb` to generate asset weights using forecasted return/volatility and apply a risk tolerance.
- Run `/Portfolio simulation/Backtesting.ipynb` to simulate portfolio performance over time and compare against a 60/40 benchmark.

---

## 🧰 Tools Used

- Python 3.10+
- PyTorch / PyTorch Lightning
- PyTorch Forecasting
- Optuna
- Matplotlib, Pandas, NumPy, Scikit-learn

---

## 📄 Thesis & Documentation

- 📘 Final write-up: [`FYP_Project_2025.pdf`](./FYP_Project_2025.pdf)
- 📉 Portfolio graph: [`thesis graph.png`](./thesis%20graph.png)

---

## ⚠️ Disclaimer

This project is for academic use only. It does not constitute financial advice or guarantee real-world investment performance.

---

## 👤 Author

**John Wallace**  
Trinity College Dublin – Final Year, 2025  
[LinkedIn](www.linkedin.com/in/jay-wallace-73b1ab1b5) *(optional)*
