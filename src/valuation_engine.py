"""
Transfer Value & Market Valuation Engine.
Implements the dual-mode valuation architecture from Cricket Player Rating & Transfer Value System:
1. 7-Factor Rating-Based Valuation Index Mode with non-linear power curve (x^2.15, ₹0.20 Cr – ₹30.00 Cr).
2. Supervised CatBoost Transfer Value Regressor on log-transformed targets.
"""

import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


class CricketTransferValuationEngine:
    MIN_VALUE_CR = 0.20   # Base auction price floor in INR Crore
    MAX_VALUE_CR = 30.00  # Marquee franchise auction ceiling in INR Crore
    POWER_EXPONENT = 2.15 # Auction dynamics exponential elasticity

    ROLE_MULTIPLIERS = {
        'All-Rounder': 1.18,
        'All-rounder': 1.18,
        'Wicketkeeper': 1.08,
        'Bowler': 1.02,
        'Batsman': 1.00,
        'Batter': 1.00,
        'Unknown': 0.92
    }

    # Reference IPL marquee auction & retention valuation anchors (for supervised calibration)
    AUCTION_BENCHMARKS = [
        {'player': 'Mitchell Starc', 'role': 'Bowler', 'rating': 93.5, 'transfer_value_cr': 24.75},
        {'player': 'Pat Cummins', 'role': 'All-Rounder', 'rating': 91.8, 'transfer_value_cr': 20.50},
        {'player': 'Virat Kohli', 'role': 'Batsman', 'rating': 96.2, 'transfer_value_cr': 21.00},
        {'player': 'Heinrich Klaasen', 'role': 'Wicketkeeper', 'rating': 94.0, 'transfer_value_cr': 23.00},
        {'player': 'Jasprit Bumrah', 'role': 'Bowler', 'rating': 97.4, 'transfer_value_cr': 18.00},
        {'player': 'Hardik Pandya', 'role': 'All-Rounder', 'rating': 90.5, 'transfer_value_cr': 16.35},
        {'player': 'Rishabh Pant', 'role': 'Wicketkeeper', 'rating': 95.0, 'transfer_value_cr': 27.00},
        {'player': 'Shreyas Iyer', 'role': 'Batsman', 'rating': 89.2, 'transfer_value_cr': 26.75},
        {'player': 'Rashid Khan', 'role': 'All-Rounder', 'rating': 96.8, 'transfer_value_cr': 18.00},
        {'player': 'Rohit Sharma', 'role': 'Batsman', 'rating': 92.0, 'transfer_value_cr': 16.30},
        {'player': 'Glenn Maxwell', 'role': 'All-Rounder', 'rating': 86.5, 'transfer_value_cr': 14.25},
        {'player': 'Nicholas Pooran', 'role': 'Wicketkeeper', 'rating': 91.2, 'transfer_value_cr': 21.00},
        {'player': 'Andre Russell', 'role': 'All-Rounder', 'rating': 93.0, 'transfer_value_cr': 16.00},
        {'player': 'Sunil Narine', 'role': 'All-Rounder', 'rating': 93.8, 'transfer_value_cr': 12.00},
        {'player': 'Shubman Gill', 'role': 'Batsman', 'rating': 90.1, 'transfer_value_cr': 16.50},
        {'player': 'Travis Head', 'role': 'Batsman', 'rating': 92.5, 'transfer_value_cr': 14.00},
        {'player': 'Arshdeep Singh', 'role': 'Bowler', 'rating': 87.2, 'transfer_value_cr': 18.00},
        {'player': 'Yuzvendra Chahal', 'role': 'Bowler', 'rating': 88.0, 'transfer_value_cr': 18.00},
        {'player': 'KL Rahul', 'role': 'Wicketkeeper', 'rating': 88.5, 'transfer_value_cr': 14.00},
        {'player': 'Rinku Singh', 'role': 'Batsman', 'rating': 86.0, 'transfer_value_cr': 13.00},
        {'player': 'Varun Chakravarthy', 'role': 'Bowler', 'rating': 85.0, 'transfer_value_cr': 12.00},
        {'player': 'Axar Patel', 'role': 'All-Rounder', 'rating': 87.5, 'transfer_value_cr': 16.50},
        {'player': 'Kuldeep Yadav', 'role': 'Bowler', 'rating': 86.4, 'transfer_value_cr': 13.25},
        {'player': 'Liam Livingstone', 'role': 'All-Rounder', 'rating': 84.5, 'transfer_value_cr': 11.50},
        {'player': 'Jos Buttler', 'role': 'Wicketkeeper', 'rating': 93.1, 'transfer_value_cr': 15.75},
        {'player': 'Trent Boult', 'role': 'Bowler', 'rating': 86.8, 'transfer_value_cr': 12.50},
        {'player': 'Marcus Stoinis', 'role': 'All-Rounder', 'rating': 83.2, 'transfer_value_cr': 11.00},
        {'player': 'Kagiso Rabada', 'role': 'Bowler', 'rating': 87.0, 'transfer_value_cr': 10.75},
        {'player': 'Mohammed Shami', 'role': 'Bowler', 'rating': 86.5, 'transfer_value_cr': 10.00},
        {'player': 'Mohammed Siraj', 'role': 'Bowler', 'rating': 84.2, 'transfer_value_cr': 12.25},
        {'player': 'Domestic Emerging Talent A', 'role': 'Batter', 'rating': 62.0, 'transfer_value_cr': 0.50},
        {'player': 'Domestic Emerging Talent B', 'role': 'Bowler', 'rating': 58.0, 'transfer_value_cr': 0.30},
        {'player': 'Domestic Bench Player C', 'role': 'All-Rounder', 'rating': 52.0, 'transfer_value_cr': 0.20}
    ]

    def __init__(self, models_dir=None):
        base_dir = Path(__file__).resolve().parent.parent
        self.models_dir = Path(models_dir) if models_dir else base_dir / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.supervised_model = None

    def calculate_7factor_valuation_index(self, row):
        """
        Computes transparent 7-Factor Market Valuation Index (0-100 scale) and monetary value in INR Crore.
        Index = (0.50*Sperf + 0.17*Sform + 0.10*Scontext + 0.08*Sconsistency + 0.05*Sexperience + 0.05*Sscarcity + 0.05*Savailability) * Mrole
        """
        # 1. Performance Score (Sperf, 50%)
        s_perf = float(row.get('current_rating', 50.0))

        # 2. Recent Form Score (Sform, 17%)
        r5 = float(row.get('runs_last_5', 0.0))
        w5 = float(row.get('wickets_last_5', 0.0))
        s_form = float(np.clip((r5 / 140.0) * 60.0 + (w5 / 6.0) * 40.0, 10.0, 100.0))

        # 3. Context Score (Scontext, 10%)
        opp_diff = float(row.get('opponent_bowler_difficulty', 50.0))
        s_context = float(np.clip(opp_diff, 15.0, 95.0))

        # 4. Consistency Score (Sconsistency, 8%)
        ff = float(row.get('form_factor', 1.0))
        s_consistency = float(np.clip(ff * 80.0, 20.0, 100.0))

        # 5. Experience Score (Sexperience, 5%)
        exp_years = float(row.get('career_span_years', 1.0))
        tot_matches = float(row.get('total_matches', 1.0))
        s_exp = float(np.clip((exp_years / 10.0) * 60.0 + (tot_matches / 80.0) * 40.0, 10.0, 100.0))

        # 6. Role Scarcity Score (Sscarcity, 5%)
        role = str(row.get('role', 'Batsman'))
        role_rank = float(row.get('role_rank', 50.0))
        base_scarcity = 90.0 if 'All' in role else (85.0 if 'Keep' in role else 75.0)
        s_scarcity = float(np.clip(base_scarcity - min(role_rank * 0.8, 40.0), 15.0, 100.0))

        # 7. Availability Score (Savailability, 5%)
        is_active = bool(row.get('is_active', True))
        s_avail = 100.0 if is_active else 45.0

        # Role Multiplier (Mrole)
        m_role = self.ROLE_MULTIPLIERS.get(role, 1.00)

        # Weighted Composite Raw Index
        raw_index = (
            (0.50 * s_perf) +
            (0.17 * s_form) +
            (0.10 * s_context) +
            (0.08 * s_consistency) +
            (0.05 * s_exp) +
            (0.05 * s_scarcity) +
            (0.05 * s_avail)
        ) * m_role

        valuation_index = float(np.clip(raw_index, 0.0, 100.0))

        # Non-Linear Monetary Conversion Curve:
        # Estimated Transfer Value (₹ Cr) = MIN_VALUE + (MAX_VALUE - MIN_VALUE) * (Index / 100) ^ 2.15
        x = valuation_index / 100.0
        transfer_val_cr = self.MIN_VALUE_CR + (self.MAX_VALUE_CR - self.MIN_VALUE_CR) * (x ** self.POWER_EXPONENT)
        transfer_val_cr = float(np.clip(round(transfer_val_cr, 2), self.MIN_VALUE_CR, self.MAX_VALUE_CR))

        return {
            'valuation_index': round(valuation_index, 1),
            'transfer_value_cr': transfer_val_cr,
            's_perf': round(s_perf, 1),
            's_form': round(s_form, 1),
            's_context': round(s_context, 1),
            's_consistency': round(s_consistency, 1),
            's_experience': round(s_exp, 1),
            's_scarcity': round(s_scarcity, 1),
            's_availability': round(s_avail, 1),
            'm_role': m_role
        }

    def train_supervised_valuation_model(self):
        """
        Trains a Supervised CatBoost Regressor on log(1 + transfer_value)
        calibrated against auction anchor points.
        """
        print("Training Supervised Market Mode Valuation Model...")
        df_bench = pd.DataFrame(self.AUCTION_BENCHMARKS)
        
        # Synthesize features matching the anchor points
        features = []
        for _, r in df_bench.iterrows():
            m_role = self.ROLE_MULTIPLIERS.get(r['role'], 1.0)
            features.append({
                'current_rating': r['rating'],
                'role_mult': m_role,
                'is_active': 1,
                'log_target': np.log1p(r['transfer_value_cr'])
            })
            
        df_feat = pd.DataFrame(features)
        X = df_feat[['current_rating', 'role_mult', 'is_active']]
        y = df_feat['log_target']

        self.supervised_model = CatBoostRegressor(
            iterations=250,
            learning_rate=0.08,
            depth=4,
            loss_function='RMSE',
            random_seed=42,
            verbose=False
        )
        self.supervised_model.fit(X, y, verbose=False)
        self.supervised_model.save_model(str(self.models_dir / "transfer_value_catboost_model.cbm"))
        print("Supervised Valuation Model trained and saved.")

    def predict_supervised_transfer_value(self, rating, role, is_active=True):
        """Generates ML predicted transfer value in INR Crore with exponential back-transformation."""
        if self.supervised_model is None:
            model_path = self.models_dir / "transfer_value_catboost_model.cbm"
            if model_path.exists():
                self.supervised_model = CatBoostRegressor()
                self.supervised_model.load_model(str(model_path))
            else:
                self.train_supervised_valuation_model()

        m_role = self.ROLE_MULTIPLIERS.get(role, 1.0)
        X = pd.DataFrame([{
            'current_rating': float(rating),
            'role_mult': float(m_role),
            'is_active': 1 if is_active else 0
        }])
        
        pred_log = float(self.supervised_model.predict(X)[0])
        pred_cr = float(np.expm1(pred_log))
        return float(np.clip(round(pred_cr, 2), self.MIN_VALUE_CR, self.MAX_VALUE_CR))
