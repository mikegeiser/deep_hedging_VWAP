# Deep Hedging the Volume-Weighted Average Price in Order-Driven Markets

This repository implements a full VWAP risk deep-hedging pipeline ...
It includes:
- Event simulation using a parametric model (Santa Fe LOB) calibrated on real LOBSTER data 
- Market feature extraction from simulated event sequences
- Deep Hedging modelling and training
- Forward pass statistics for training and policy analysis

This code accompanies the master thesis:

> We study volume-weighted average price execution risk in order-driven markets within the Deep Hedging framework. A Santa Fe limit order book model, calibrated to LOBSTER intraday data for four NASDAQ stocks, generates market scenarios.
> The agent trades through a liquidity provider that returns execution prices for market orders and exogenous fill rules for limit orders, ignoring market impact. Policies are represented by a system of neural networks that map current market features and previous actions to a market order size and a vector of limit order placements.
> Training minimizes a convex risk measure of terminal wealth using TensorFlow on simulated paths.
> The learned policies are stock-specific and economically interpretable: tight-spread books use early market orders and shallow limit order posting before shifting passive; spread-out books rely on purely passive deep placement; intermediate structures favor mid-level passive accumulation.
> Deep Hedging thus emerges as a viable approach in order-driven markets, with microstructure conditions shaping optimal market and limit order allocation.

**Remark 1:** We provide the Santa Fe model parameters for four stocks: CSCO, INTC, PCLN, TSLA. In the explanations below, &lt;STOCK&gt; means any of these names.

**Remark 2:** Raw LOBSTER data is not included due to licensing restrictions. Simulated events and features stored here represent a reduced dataset due to GitHub storage limits.

---

# Folder Structure

```
C:\deep_hedging_VWAP\
│
├── data\
│   ├── CSCO\
│   │   └── CSCO_santa_fe_param.csv
│   ├── INTC\
│   │   └── INTC_santa_fe_param.csv
│   ├── PCLN\
│   │   └── PCLN_santa_fe_param.csv
│   └── TSLA\
│       ├── TSLA_market_events.h5
│       ├── TSLA_market_features.h5
│       ├── TSLA_market_features_best.weights.h5
│       ├── TSLA_market_features_last.weights.h5
│       ├── TSLA_market_features_forward_stats.h5
│       ├── TSLA_market_features_history.json
│       ├── TSLA_santa_fe_param.csv
│       ├── TSLA_santa_fe_intensities.png
│       └── TSLA_santa_fe_queues.png
│
├── deep_hedging\
│   ├── deep_hedging_param.py
│   ├── export_forward_stats_market_features.py
│   ├── extract_market_features.py
│   ├── loss_function.py
│   ├── model_2.py
│   ├── nn_architecture.py
│   └── train_agent_2.py
│
├── lob_simulator\
│   ├── __init__.py
│   ├── lob.py
│   ├── plot_santa_fe_params.py
│   ├── santa_fe_model.py
│   ├── santa_fe_param.py
│   ├── simulate_events.py
│   └── util_lob.py
│
└── training_results\
    ├── plot_forward_stats_market_features.py
    ├── plot_training_history_market_features.py
    ├── CSCO\
    ├── INTC\
    ├── PCLN\
    └── TSLA\
        ├── TSLA_avg_delta_trajectory.png
        ├── TSLA_avg_policy_bars.png
        ├── TSLA_avg_terminal_delta_boxplot.png
        ├── TSLA_avg_training_loss.png
        └── TSLA_avg_wealth_kde.png
```

---

# Data Files

**Inside `data/<STOCK>/` you will find:**

- **<STOCK>_santa_fe_param.csv** – Calibration parameters for the Santa Fe LOB model  
- **<STOCK>_market_events.h5** – Simulated LOB event sequences  
- **<STOCK>_market_features.h5** – Extracted market snapshots  
- **<STOCK>_market_features_best.weights.h5** – Best model weights (early stopping)  
- **<STOCK>_market_features_last.weights.h5** – Last epoch weights  
- **<STOCK>_market_features_history.json** – Training loss values  
- **<STOCK>_market_features_forward_stats.h5** – Stats from full forward pass  

---

# Plot Files

- **<STOCK>_santa_fe_intensities.png** – λ(k) and ρ(k) curves  
- **<STOCK>_santa_fe_queues.png** – Initial bid/ask queue depths  
- **<STOCK>_avg_wealth_kde.png** – KDE terminal wealth distribution  
- **<STOCK>_avg_delta_trajectory.png** – Mean delta trajectory  
- **<STOCK>_terminal_delta_boxplot.png** – Terminal delta boxplot  
- **<STOCK>_avg_policy_bars.png** – Policy φ and θ₁…θ_L averages  
- **<STOCK>_avg_training_loss.png** – Loss across epochs  

---

# Parameter Tuning

**Santa Fe Model Parameters (<STOCK>_santa_fe_param.csv):**
- Order book geometry: K, K_trunc, L  
- Event intensities: γ, λ(k), ρ(k)  
- Volume distribution parameters: μ_M, σ_M, μ_L, σ_L, μ_C, σ_C  
- Initial LOB state: S₀, ε₀, a₀(k), b₀(k)  
- Tick size and scaling constants  

**Deep Hedging Parameters (deep_hedging_param.py):**
- time_window (hours)  
- N: number of decision steps  
- num_layers and num_neurons: architecture size  
- Training hyperparameters: learning rate, batch size, epochs  
- Gradient aggregation  

---

# Execution Pipeline

## Steps 1–2  
Run from:
```
C:\deep_hedging_VWAP\lob_simulator\
```
1. Plot calibration
```
python plot_santa_fe_params.py --stock <STOCK>
```
2. Simulate events
```
python simulate_events.py --stock <STOCK> --num-paths 250000 --n-jobs 20 --chunk-size 1000
```

## Steps 3–5  
Run from:
```
C:\deep_hedging_VWAP\deep_hedging\
```
3. Extract features
```
python extract_market_features.py --stock <STOCK> --n-jobs 20 --chunk-size 1000
```
4. Train agent
```
python train_agent_2.py --stock <STOCK> --accum-steps 5
```
5. Export forward stats
```
python export_forward_stats_market_features.py --stock <STOCK>
```

## Steps 6–7  
Run from:
```
C:\deep_hedging_VWAP\training_results\
```
6. Plot training loss
```
python plot_training_history_market_features.py --stock <STOCK>
```
7. Plot analysis
```
python plot_forward_stats_market_features.py --stock <STOCK>
```

---

# End of README

