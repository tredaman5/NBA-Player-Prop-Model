# 🏀 NBA Player Prop Edge Model

A fully automated end-to-end NBA player prop betting system that:

-   Pulls live sportsbook player props daily\
-   Generates pre-game predictions using machine learning\
-   Converts predictions into probabilistic edges\
-   Calculates expected value (EV) and optimal bet sizing\
-   Displays results in both CLI format and a Streamlit dashboard

------------------------------------------------------------------------

## 🚀 What This Project Does

This project builds a complete quantitative workflow for NBA player
props:

1.  Fetch live odds (Points / Rebounds / Assists)\
2.  Build slate features for upcoming games\
3.  Predict player performance using trained ML models\
4.  Convert projections → probabilities (Normal distribution
    assumption)\
5.  Compare model probability vs sportsbook implied probability\
6.  Calculate edge, expected value, and fractional Kelly bet sizing\
7.  Display ranked bets in CLI and dashboard format

------------------------------------------------------------------------

## 🧠 Modeling Approach

### Feature Engineering

Models use:

-   Rolling per-minute production (3 / 5 / 10 game windows)
-   Predicted minutes model
-   Team rolling statistics
-   Rest days & back-to-back indicators
-   Home / Away splits

### Prediction Flow

Historical Data → Feature Engineering → Model Training\
↓\
Slate Feature Generation\
↓\
Model Predictions\
↓\
Probability Conversion (Normal(mu, sigma))\
↓\
Edge + EV + Kelly Bet Sizing

------------------------------------------------------------------------

## 📊 Edge Calculation

Edge = Model Probability − Implied Probability

Expected value per \$1 wagered is computed using American odds.

Bet sizing uses:

-   Full Kelly Fraction
-   Fractional Kelly (default 0.25x)
-   Minimum edge threshold (default 2%)

------------------------------------------------------------------------

## 🗂 Project Structure

    NBA-Player-Prop-Model/
    │
    ├── app.py
    ├── data/
    │   └── processed/
    ├── models_artifacts/
    ├── src/
    │   ├── betting/
    │   ├── odds/
    │   ├── slate/
    │   └── ui/

------------------------------------------------------------------------

## ⚙️ Installation

Clone the repository:

    git clone https://github.com/yourusername/NBA-Player-Prop-Model.git
    cd NBA-Player-Prop-Model

Install dependencies:

    pip install -r requirements.txt

Key libraries:

-   pandas
-   numpy
-   scikit-learn
-   nba_api
-   requests
-   streamlit

------------------------------------------------------------------------

## 🔑 API Key Setup

This project uses The Odds API.

PowerShell:

    $env:ODDS_API_KEY="YOUR_KEY_HERE"

Do not commit your API key.

------------------------------------------------------------------------

## 🏃 Daily Run (One Command)

    python -m src.ui.run_daily --date 2026-02-13

This runs the full pipeline: - Fetch props - Build slate - Generate
predictions - Calculate edge & bet sizing - Output results

------------------------------------------------------------------------

## 📈 Launch Dashboard

    streamlit run app.py

Dashboard Features:

-   Filter by prop type
-   Edge threshold slider
-   EV filter
-   Sort by edge / EV / bet size
-   Confidence tiers (A/B/C/D)
-   Clean card layout with NBA-style branding

------------------------------------------------------------------------

## 📌 Sample Model Performance

  Prop       MAE    RMSE
  ---------- ------ ------
  Points     2.78   3.78
  Rebounds   1.49   2.03
  Assists    1.20   1.67

Models outperform rolling average baselines.

------------------------------------------------------------------------

## 🔒 Disclaimer

This project is for educational and research purposes only.

It is not financial advice and does not guarantee profitability.

Sports betting involves financial risk.

------------------------------------------------------------------------

## 👤 Author

Tre Smith\
Computer Science --- University of South Carolina

Interests: - Sports analytics - Quantitative modeling - Machine learning
systems - Financial engineering
