# Quantum vs Classical Option Pricing Simulator

A Streamlit-based web app comparing classical Geometric Brownian Motion (GBM) and Quantum Stochastic Differential Equation (QSDE) models for European option pricing. Integrates real-time market data, quantum-generated randomness, and interactive 3D visualization to explore volatility, profit/loss, and model accuracy.



## Overview

This project investigates whether quantum-enhanced stochastic models offer advantages in option pricing versus traditional methods like Black-Scholes with GBM. It simulates both models under the same conditions and provides visual insights into differences in:

- Expected returns
- Option prices
- Model volatility
- Profit and loss under stress

---

## Features

- **GBM vs Quantum Model** side-by-side
- **Interactive 3D Volatility Surfaces**
- **Monte Carlo Simulations** with configurable input
- **Real-time stock data** via Yahoo Finance (`yfinance`)
- **Quantum randomness injection** using Qiskit
- **Run replay & saving system**
- Built with:
  - `Streamlit`, `Plotly`, `Qiskit`, `yfinance`, `numpy`, `scikit-learn`

---

## How It Works

- **Classical GBM** is simulated using the standard log-normal SDE.
- **QSDE model** applies quantum circuits to generate bias and simulate paths with quantum drift.
- **Monte Carlo simulations** are run for both models and visualized using `plotly`.
- **Volatility surfaces** show how different inputs affect option prices.
- Users can toggle between **manual inputs** or **live market data** for pricing realism.

---

## Installation

```bash
# 1. Clone the repo
git clone 
cd

# 2. Create environment
python -m venv venv
source venv/bin/activate  or .\venv\Scripts\activate on Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
