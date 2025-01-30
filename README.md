# **McGill-FIAM Quant Asset Management Project**

This repository contains a **quantitative investment strategy** pipeline developed for the **McGill-FIAM Hackathon**. The end-to-end project showcases:

1. **Data Preprocessing**  
2. **Feature Engineering & Selection**  
3. **Machine Learning Predictor** (returns alpha model)  
4. **Portfolio Construction** (Black-Litterman, Hierarchical Risk Parity)  
5. **Backtesting & Performance Evaluation**  

Additionally, it integrates **causal discovery** techniques and **LLM-driven fundamental features** (from 10-K filings) for richer alpha signals.


Slides presented at McGill-FIAM: http://bit.ly/40GVIlI


---

## **Project Highlights**

1. **ML-Based Alpha Generation**  
   - We use a Bagging Random Forest (with advanced feature engineering) to predict stock excess returns.
   
2. **Causal Discovery**  
   - Uses **AVICI** (pretrained causal discovery model) to identify potential causal links among financial factors.

3. **Portfolio Optimization**  
   - Implements **Hierarchical Risk Parity (HRP)** and **Black-Litterman** for robust asset allocation.
   - Multiple strategies exploring Carhart factors, custom alpha signals, and different optimization methods.

4. **NLP & Fundamental Analysis**  
   - Zero-shot language model (LLaMA) to extract **risk factor scores**, **readability**, and **sentiment** from SEC 10-K filings.
   - Integrates these textual features into factor and alpha models.

5. **Comprehensive Backtesting**  
   - Rolling-window approach from **2010–2023**.  
   - Reports Sharpe Ratio, Information Ratio, max drawdown, log-loss, confusion matrices, etc.  
   - Final results in `06-Backtesting/BlackLitermann-HRP/HRP_backtest.png` showcasing multiple strategies:
     - Carhart + HRP
     - Alpha-signal + HRP (no BL)
     - Alpha-signal + HRP + Black-Litterman

---

## **Repository Structure**

Below is a high-level breakdown (see `scripts/directory_tree.py` for full detail):

```
mcgill_fiam
├─ 01-Data_Preprocessing
│   ├─ preprocessing_code.py
│   ├─ factors_theme.json
├─ 02-Feature_Engineering
│   └─ feature_engineering_code.py
├─ 03-Feature_Importance
│   ├─ feature_importance_code.py
│   ├─ feature_selection_code.py
│   └─ top_100_features.json
├─ 04-Predictor
│   ├─ train_AlphaSignals.py
│   └─ predictions_performance.py
├─ 05-Asset_Allocation
│   ├─ strategy_3/... 
│   ├─ strategy_10/... 
│   ├─ ...
│   └─ original_hrp.py
├─ 06-Backtesting
│   ├─ backtester_parallel.py
│   ├─ backtest_stats.py
│   ├─ BlackLitermann-HRP/
│   │   ├─ HRP_backtest.png ← **Final backtest chart**
│   └─ ...
├─ 0X-Causal_discovery
│   ├─ discovery.py
│   └─ Final_Features.json
├─ 0L-CoTZeroShotFeatures
│   ├─ create_dataset.py
│   └─ llama-3.2-3B-Instruct-Inference-*.py
├─ objects/
│   ├─ (Intermediate data, model outputs, predictions, performance metrics, etc.)
├─ raw_data/
│   ├─ factor_char_list.csv
│   ├─ hackathon_sample_v2.csv
│   └─ mkt_ind.csv
├─ notebooks/
│   ├─ (Assorted exploratory & debugging Jupyter notebooks)
├─ requirements.txt
├─ README.md (this file)
└─ ...
```

---

## **Installation & Setup**

1. **Clone** this repository:
   ```bash
   git clone https://github.com/theAayushbajaj/mcgill_fiam.git
   cd mcgill_fiam
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   - Python 3.7+ recommended
   - If you plan on running large language models locally, ensure you have **GPU support** or suitable hardware.

3. **Data Files**  
   - The main data files (`hackathon_sample_v2.csv`, `mkt_ind.csv`, etc.) should reside in `raw_data` or `raw_data_v3`.
   - Additional references:
     - **Factor definitions**: `factor_char_list.csv`  
     - **Market index**: `mkt_ind.csv`
     - **SEC 10-Ks** for fundamental NLP in `0L-CoTZeroShotFeatures/assets/` or `datasets/`.

---

## **Running the Pipeline**

Below is the typical end-to-end workflow:

1. **Data Preprocessing**  
   ```bash
   cd 01-Data_Preprocessing
   python preprocessing_code.py
   cd ..
   ```

2. **Feature Engineering**  
   ```bash
   cd 02-Feature_Engineering
   python feature_engineering_code.py
   cd ..
   ```

3. **Feature Importance & Selection**  
   ```bash
   cd 03-Feature_Importance
   python feature_importance_code.py
   python feature_selection_code.py
   cd ..
   ```

4. **(Optional) Causal Discovery**  
   ```bash
   cd 0X-Causal_discovery
   python discovery.py
   cd ..
   ```
   - Produces a causal DAG using the **AVICI** model.  
   - Helps refine final feature sets.

5. **Predictor Training**  
   ```bash
   cd 04-Predictor
   python train_AlphaSignals.py
   # Produces out-of-sample predictions
   cd ..
   ```
   - Rolling-window Bagging RF  
   - Generates monthly predictions in `objects/predictions_*.csv`

6. **Asset Allocation**  
   - Multiple strategies exist in `05-Asset_Allocation`.  
   - By default, the backtester calls a “main.py” from your chosen strategy folder.  
   - E.g., `strategy_3/main.py` or `strategy_12/main.py`.

7. **Backtesting**  
   ```bash
   cd 06-Backtesting
   python backtester_parallel.py
   cd ..
   ```
   - Aggregates predictions, builds a monthly portfolio, and tracks performance.  
   - Final results (Sharpe ratio, alpha, drawdowns, etc.) stored in `objects/` and plots in `06-Backtesting/...`.

8. **(Optional) NLP on 10-K Filings**  
   ```bash
   cd 0L-CoTZeroShotFeatures
   python create_dataset.py
   python llama-3.2-3B-Instruct-Inference-RISK_FACTORS.py
   python llama-3.2-3B-Instruct-Inference-READABILITY-SCORE.py
   python llama-3.2-3B-Instruct-Inference-SENTIMENT-SCORES.py
   cd ..
   ```
   - Integrates fundamental signals into your pipeline or factor set.

---

## **Key Outputs**

- **Backtest Results**  
  - `06-Backtesting/BlackLitermann-HRP/HRP_backtest.png`:  
    - Shows **three** final portfolios:
      1. **Carhart + HRP**  
      2. **Alpha signals + HRP (no BL)**  
      3. **Alpha signals + Black-Litterman + HRP**  

- **Model Predictions & Performance**  
  - `objects/predictions_*.csv` : ML predictions for each monthly test set  
  - `objects/performance_summary.csv` : Overall classification metrics (accuracy, F1, etc.)  
  - `objects/Trading_Stats.pkl` : Dictionary with final trading performance stats  
  - `objects/TradingLog_Stats.pkl` : Detailed log-level stats (hit-miss, average trade ret, etc.)

- **Intermediate Files**  
  - `objects/prices.csv`, `objects/signals.csv`, `objects/market_caps.csv`, etc.  
  - `objects/X_DATASET.csv` & `objects/Y_DATASET.csv` : Aggregated features & targets  
  - `objects/FULL_stacked_data.csv` : Full feature-target-time panel  

---

## **Project Flow Diagram**

```
          Raw Data
             |
   (1) Data Preprocessing
             |
   (2) Feature Engineering ----> (3) Feature Importance & Causal Discovery
             |                         |
             v                         |
     (4) ML Alpha Model  <-------------
             |
   (5) Portfolio Construction & Optimization (HRP / BL)
             |
   (6) Backtesting & Evaluation
             |
         Results & Plots
```

---

## **Additional Notes**

- **Causal Discovery**:  
  The pipeline can incorporate causality-based feature selection. Ensure `avici` is installed if you run `discovery.py`.

- **Performance Tips**:
  - For large data or heavy ML training, consider parallel execution / HPC.
  - For LLM-based scripts, ensure GPU availability or use smaller models.

- **References**  
  - **Advances in Financial Machine Learning** by Marcos López de Prado  
  - **Black-Litterman** (1992) <br/>
  - **Carhart 4-Factor Model** (Carhart, 1997)

<!-- ---

## **Contributors**
- **[Your Name]** (primary author)
- **[Team Members / Collaborators]**
- **[McGill-FIAM Hackathon participants / Mentors]**

For questions or suggestions, please reach out via `[Your Email]` or open an Issue in this repository. -->

<!-- ---

## **License**
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. If you are using any proprietary data (e.g., from WRDS, 10-K filings, etc.), please consult the respective data provider’s terms and conditions. -->

---

**Thank you for using the McGill-FIAM Quant Asset Management Project!** Feel free to open issues or pull requests to improve the codebase. Happy hacking and investing!