# 🏏 Cricket Player Rating & Transfer Value System (V2 Production Engine)

A context-aware machine learning architecture and dual-mode market valuation engine designed for professional T20 and IPL cricket analytics. Built in strict alignment with the **Cricket Player Rating and Transfer Value System** specification and calibrated using real-world match datasets (Cricsheet T20s, Opta ball-by-ball records, and Arena benchmarks).

---

## 🚀 Key Improvements Over Raw Stats & Arena v1

| Limitation in Legacy / v1 Formula | How this Machine Learning System Solves It |
|---|---|
| **Cumulative Total Bias** (raw runs & wickets reward match volume) | Uses rate metrics (runs/inn, strike rate, dot ball %, wickets/match) and **credibility shrinkage** $c = n / (n + k)$. |
| **No Venue Context** (flat pitch vs green top treated identically) | **Venue scoring factor** and **bowling difficulty index** dynamically normalize runs and economy. |
| **No Opponent Adjustment** (runs against weak attack equal elite attack) | Computes pre-match **opponent bowler difficulty** and **opponent batter difficulty** indices (0–100 scale). |
| **Data Leakage Risk** | **Strict Walk-Forward Temporal Order:** Match $N$ strictly uses historical data from matches $1 \dots N-1$. |
| **Arbitrary Caps & Ties** | **CatBoost Regressors + Normal CDF Scaling** produce smooth, unbounded, non-linear rating distributions. |
| **Monetary Valuation** | **Dual-Mode Engine:** Transparent **7-Factor Rating Valuation Index** ($x^{2.15}$ curve) + **Supervised CatBoost Regressor** on $\log(1 + \text{Valuation})$. |

---

## 🏛️ System Architecture

```
                                  +-------------------------------------------------------------+
                                  |                 1. Ball-by-Ball Match Ingestion             |
                                  |           (Cricsheet JSON & Opta Feeds: 24,845 Deliveries)  |
                                  +-------------------------------------------------------------+
                                                                 |
                                                                 v
                                  +-------------------------------------------------------------+
                                  |              2. Leak-Free Walk-Forward Engine               |
                                  |   - Prior career accumulators & rolling form (L3, L5, L10)  |
                                  |   - Venue run-scoring and bowling difficulty factors        |
                                  |   - Opposition bowler and batter strength indices           |
                                  +-------------------------------------------------------------+
                                                                 |
                                                                 v
                                  +-------------------------------------------------------------+
                                  |               3. CatBoost ML Rating Engine                  |
                                  |   - CatBoost Batting Performance Regressor (RMSE: 17.58)     |
                                  |   - CatBoost Bowling Performance Regressor (RMSE: 16.06)    |
                                  |   - Dynamic Role-Aware Weighting & Form Factor Multiplier   |
                                  +-------------------------------------------------------------+
                                                                 |
                                                                 v
                                  +-------------------------------------------------------------+
                                  |              4. Dual-Mode Transfer Value Engine             |
                                  |   - 7-Factor Market Index with Role Multipliers & x^2.15    |
                                  |   - Supervised CatBoost Valuation Regressor (₹0.20-30.00 Cr)|
                                  +-------------------------------------------------------------+
                                                                 |
                                                                 v
                                  +-------------------------------------------------------------+
                                  |              5. Analytics & Dashboard Delivery              |
                                  |   - Interactive Streamlit Dashboard (`app/streamlit_app.py`)|
                                  |   - Multi-Sheet Master Excel (`exported_workbooks/`)        |
                                  +-------------------------------------------------------------+
```

---

## 🎯 Rating Methodology & Role Weighting

### 1. Dynamic Role-Aware Composite
$$\text{Overall Rating} = (W_{\text{bat}} \times R_{\text{bat}}) + (W_{\text{bowl}} \times R_{\text{bowl}})$$

- **Batsman:** 90% Batting, 10% Bowling
- **Bowler:** 10% Batting, 90% Bowling
- **All-Rounder:** 55% Batting, 45% Bowling
- **Wicketkeeper:** 85% Batting, 15% Bowling
- **Unknown:** 50% Batting, 50% Bowling

### 2. Recent Form Factor Multiplier
$$\text{Form Factor} = \text{clip}\left(\frac{\text{Recent 5 Matches Output} + 1}{\text{Career Avg per 5 Matches} + 1}, 0.75, 1.25\right)$$
$$\text{Current Rating} = \text{clip}\left(\text{Predicted Rating} \times 0.90 + 0.10 \times (\text{Form Factor} \times \text{Predicted Rating}), 0, 100\right)$$

### 3. Competitive Rating Bands
- **Elite (90.0 – 100.0):** Top-tier match winners and global superstars.
- **World Class (80.0 – 89.9):** Core franchise pillars and proven high performers.
- **International (70.0 – 79.9):** Consistent top-tier league performers.
- **Domestic Star (60.0 – 69.9):** Reliable role players and domestic standouts.
- **Squad Player (< 60.0):** Backup options and emerging players.

---

## 💰 Transfer Value Model (Dual-Mode)

### Mode 1: 7-Factor Rating-Based Valuation Index
$$\text{Index} = \left(0.50 S_{\text{perf}} + 0.17 S_{\text{form}} + 0.10 S_{\text{context}} + 0.08 S_{\text{consistency}} + 0.05 S_{\text{experience}} + 0.05 S_{\text{scarcity}} + 0.05 S_{\text{availability}}\right) \times M_{\text{role}}$$

**Role Multipliers ($M_{\text{role}}$):**
- All-Rounder: **1.18x** (Highest market premium due to dual utility)
- Wicketkeeper: **1.08x** (Specialist role scarcity)
- Bowler: **1.02x** (Specialist bowling demand)
- Batsman: **1.00x** (Base market benchmark)
- Unknown: **0.92x**

**Non-Linear Monetary Conversion Curve:**
$$\text{Estimated Transfer Value (₹ Cr)} = ₹0.20 + (₹30.00 - ₹0.20) \times \left(\frac{\text{Index}}{100}\right)^{2.15}$$

### Mode 2: Supervised CatBoost Regressor
- Fits CatBoost regression on $\log(1 + \text{Valuation})$ calibrated against marquee IPL franchise auction anchors.
- Exponential back-transformation: $Y = \exp(y) - 1$, clipped to $[₹0.20, ₹30.00]\text{ Cr}$.

---

## 📂 Project Structure

```
cricket-player-rating-transfer-system/
├── app/
│   └── streamlit_app.py                       # Modern Interactive Dashboard
├── data/
│   └── processed/
│       ├── active_players_ratings_valuations.csv  # Master evaluated dataset (2,357 players)
│       └── player_walkforward_features.csv        # 24,845 walk-forward feature rows
├── exported_workbooks/
│   └── arena_player_rating_and_valuation_calculated.xlsx # 4-sheet multi-tab Excel
├── models/
│   ├── batting_catboost_model.cbm             # CatBoost Batting Regressor
│   ├── bowling_catboost_model.cbm             # CatBoost Bowling Regressor
│   ├── transfer_value_catboost_model.cbm      # Supervised Valuation Regressor
│   └── full_pipeline_report.json              # Full evaluation metrics & distribution
├── src/
│   ├── ingestion.py                           # Ball-by-ball Cricsheet & Opta parser
│   ├── context_features.py                    # Walk-forward leak-free feature builder
│   ├── rating_engine.py                       # CatBoost rating engine & role aggregator
│   ├── valuation_engine.py                    # 7-factor index & supervised valuation
│   ├── arena_v1_compatibility.py              # Arena v1 formula calculator for benchmarks
│   └── pipeline.py                            # End-to-end pipeline orchestrator
├── run_pipeline.py                            # Pipeline execution script
├── test_system.py                             # Verification test suite
└── README.md
```

---

## ⚡ Quickstart

### 1. Run Unit & Component Tests
```bash
python test_system.py
```

### 2. Execute Full Pipeline
```bash
python run_pipeline.py --matches 1200
```

### 3. Launch the Interactive Dashboard
```bash
streamlit run app/streamlit_app.py
```
