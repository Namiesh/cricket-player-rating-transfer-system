"""
Machine Learning Rating Engine & Role-Aware Aggregation.
Implements CatBoost Batting & Bowling regressors, walk-forward chronological evaluation,
role-specific dynamic weighting, recent form factor multiplier, and active player pool filtering.
"""

import os
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class CricketRatingEngine:
    BATTING_FEATURES = [
        'career_matches_before',
        'career_innings_before',
        'career_runs_before',
        'career_batting_avg_before',
        'career_batting_sr_before',
        'career_boundary_rate_before',
        'career_dot_pct_before',
        'runs_last_3',
        'runs_last_5',
        'runs_last_10',
        'sr_last_5',
        'opponent_bowler_difficulty',
        'venue_batting_factor'
    ]

    BOWLING_FEATURES = [
        'career_matches_before',
        'career_overs_before',
        'career_wickets_before',
        'career_economy_before',
        'career_bowling_avg_before',
        'career_bowling_dot_pct_before',
        'wickets_last_3',
        'wickets_last_5',
        'wickets_last_10',
        'econ_last_5',
        'opponent_batter_difficulty',
        'venue_bowling_factor'
    ]

    ROLE_WEIGHTS = {
        'Batsman': {'bat': 0.90, 'bowl': 0.10},
        'Batter': {'bat': 0.90, 'bowl': 0.10},
        'Bowler': {'bat': 0.10, 'bowl': 0.90},
        'All-rounder': {'bat': 0.55, 'bowl': 0.45},
        'All-Rounder': {'bat': 0.55, 'bowl': 0.45},
        'Wicketkeeper': {'bat': 0.85, 'bowl': 0.15},
        'Unknown': {'bat': 0.50, 'bowl': 0.50}
    }

    def __init__(self, models_dir=None):
        base_dir = Path(__file__).resolve().parent.parent
        self.models_dir = Path(models_dir) if models_dir else base_dir / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.batting_model = None
        self.bowling_model = None
        self.metrics = {}

    def train_models(self, df_train):
        """
        Trains CatBoost regressors using chronological walk-forward split.
        First 80% chronological matches = Train, Last 20% = Validation.
        """
        print(f"Training CatBoost Rating Regressors on {len(df_train)} player-match instances...")
        
        # Sort chronologically by date
        df_sorted = df_train.sort_values(['date', 'match_id']).reset_index(drop=True)
        split_idx = int(len(df_sorted) * 0.80)
        
        train_df = df_sorted.iloc[:split_idx]
        val_df = df_sorted.iloc[split_idx:]
        
        # 1. Batting Performance Model (train only on records where player actually batted or has batting history)
        bat_train = train_df[(train_df['batting_balls'] > 0) | (train_df['career_runs_before'] > 0)]
        bat_val = val_df[(val_df['batting_balls'] > 0) | (val_df['career_runs_before'] > 0)]
        
        X_bat_tr = bat_train[self.BATTING_FEATURES].fillna(0)
        y_bat_tr = bat_train['target_batting_index']
        X_bat_val = bat_val[self.BATTING_FEATURES].fillna(0)
        y_bat_val = bat_val['target_batting_index']
        
        print(f"Training Batting Model: {len(X_bat_tr)} train rows, {len(X_bat_val)} validation rows...")
        self.batting_model = CatBoostRegressor(
            iterations=400,
            learning_rate=0.05,
            depth=6,
            loss_function='RMSE',
            eval_metric='RMSE',
            random_seed=42,
            verbose=False
        )
        self.batting_model.fit(X_bat_tr, y_bat_tr, eval_set=(X_bat_val, y_bat_val), early_stopping_rounds=40, verbose=False)
        
        y_bat_pred = self.batting_model.predict(X_bat_val)
        bat_rmse = float(np.sqrt(mean_squared_error(y_bat_val, y_bat_pred)))
        bat_mae = float(mean_absolute_error(y_bat_val, y_bat_pred))
        bat_r2 = float(r2_score(y_bat_val, y_bat_pred))
        print(f"Batting Model Evaluated - RMSE: {bat_rmse:.3f}, MAE: {bat_mae:.3f}, R2: {bat_r2:.3f}")
        
        # 2. Bowling Performance Model
        bowl_train = train_df[(train_df['bowling_legal_balls'] > 0) | (train_df['career_overs_before'] > 0)]
        bowl_val = val_df[(val_df['bowling_legal_balls'] > 0) | (val_df['career_overs_before'] > 0)]
        
        X_bowl_tr = bowl_train[self.BOWLING_FEATURES].fillna(0)
        y_bowl_tr = bowl_train['target_bowling_index']
        X_bowl_val = bowl_val[self.BOWLING_FEATURES].fillna(0)
        y_bowl_val = bowl_val['target_bowling_index']
        
        print(f"Training Bowling Model: {len(X_bowl_tr)} train rows, {len(X_bowl_val)} validation rows...")
        self.bowling_model = CatBoostRegressor(
            iterations=400,
            learning_rate=0.05,
            depth=6,
            loss_function='RMSE',
            eval_metric='RMSE',
            random_seed=42,
            verbose=False
        )
        self.bowling_model.fit(X_bowl_tr, y_bowl_tr, eval_set=(X_bowl_val, y_bowl_val), early_stopping_rounds=40, verbose=False)
        
        y_bowl_pred = self.bowling_model.predict(X_bowl_val)
        bowl_rmse = float(np.sqrt(mean_squared_error(y_bowl_val, y_bowl_pred)))
        bowl_mae = float(mean_absolute_error(y_bowl_val, y_bowl_pred))
        bowl_r2 = float(r2_score(y_bowl_val, y_bowl_pred))
        print(f"Bowling Model Evaluated - RMSE: {bowl_rmse:.3f}, MAE: {bowl_mae:.3f}, R2: {bowl_r2:.3f}")
        
        # Save models & metadata
        self.metrics = {
            'batting_model': {'rmse': bat_rmse, 'mae': bat_mae, 'r2': bat_r2},
            'bowling_model': {'rmse': bowl_rmse, 'mae': bowl_mae, 'r2': bowl_r2}
        }
        
        self.batting_model.save_model(str(self.models_dir / "batting_catboost_model.cbm"))
        self.bowling_model.save_model(str(self.models_dir / "bowling_catboost_model.cbm"))
        
        with open(self.models_dir / "model_evaluation.json", "w", encoding="utf-8") as f:
            json.dump(self.metrics, f, indent=2)
            
        return self.metrics

    def infer_player_role(self, career_runs, career_balls, career_overs, career_wkts):
        """Classifies role dynamically based on workload and historical specialization."""
        has_batting = career_runs > 40 or career_balls > 35
        has_bowling = career_overs > 8.0 or career_wkts >= 4

        if has_batting and has_bowling:
            return 'All-Rounder'
        elif has_bowling and not has_batting:
            return 'Bowler'
        elif has_batting and not has_bowling:
            return 'Batsman'
        else:
            return 'All-Rounder' if (career_overs > 3.0) else 'Batsman'

    def generate_current_player_ratings(self, df_pm, rolling_years=4.0):
        """
        Computes current ratings for all active players anchored to a rolling 4-year window.
        Incorporates predicted rating, role weights, recent form factor multiplier,
        and competitive rating band classification.
        """
        print("Generating active player ratings anchored to rolling window...")
        max_date = df_pm['date'].max()
        cutoff_date = max_date - pd.DateOffset(years=int(rolling_years))
        
        # Get latest observation state for each player
        latest_idx = df_pm.groupby('player')['date'].idxmax()
        df_latest = df_pm.loc[latest_idx].copy()

        # Compute full career summary for each player
        career_agg = df_pm.groupby('player').agg(
            total_matches=('match_id', 'nunique'),
            career_runs=('batting_runs', 'sum'),
            career_balls=('batting_balls', 'sum'),
            career_dismissals=('batting_dismissals', 'sum'),
            career_fours=('batting_fours', 'sum'),
            career_sixes=('batting_sixes', 'sum'),
            career_overs=('bowling_overs', 'sum'),
            career_runs_conc=('bowling_runs_conceded', 'sum'),
            career_wickets=('bowling_wickets', 'sum'),
            career_dot_balls=('bowling_dot_balls', 'sum'),
            first_match_date=('date', 'min'),
            latest_match_date=('date', 'max')
        ).reset_index()

        df_latest = pd.merge(df_latest.drop(columns=[c for c in career_agg.columns if c in df_latest.columns and c != 'player']), career_agg, on='player', how='left')

        # Derived career metrics
        df_latest['career_batting_avg'] = df_latest['career_runs'] / df_latest['career_dismissals'].replace(0, 1)
        df_latest['career_batting_sr'] = (df_latest['career_runs'] / df_latest['career_balls'].replace(0, 1)) * 100.0
        df_latest['career_economy'] = df_latest['career_runs_conc'] / df_latest['career_overs'].replace(0, 1)
        df_latest['career_bowling_avg'] = df_latest['career_runs_conc'] / df_latest['career_wickets'].replace(0, 1)

        # Active flag
        df_latest['is_active'] = df_latest['latest_match_date'] >= cutoff_date
        df_latest['career_span_years'] = (df_latest['latest_match_date'] - df_latest['first_match_date']).dt.days / 365.25

        # Infer Roles
        df_latest['role'] = df_latest.apply(
            lambda r: self.infer_player_role(r['career_runs'], r['career_balls'], r['career_overs'], r['career_wickets']),
            axis=1
        )

        # Predict Batting & Bowling Ratings
        X_bat = df_latest[self.BATTING_FEATURES].fillna(0)
        X_bowl = df_latest[self.BOWLING_FEATURES].fillna(0)

        df_latest['pred_batting_rating'] = np.clip(self.batting_model.predict(X_bat), 5.0, 99.0)
        df_latest['pred_bowling_rating'] = np.clip(self.bowling_model.predict(X_bowl), 5.0, 99.0)

        # Apply Dynamic Role Weights
        def aggregate_role(row):
            w = self.ROLE_WEIGHTS.get(row['role'], {'bat': 0.50, 'bowl': 0.50})
            return (w['bat'] * row['pred_batting_rating']) + (w['bowl'] * row['pred_bowling_rating'])

        raw_scores = df_latest.apply(aggregate_role, axis=1)

        # Standardize and apply Arena Credibility-weighted Normal CDF Calibration:
        # Credibility c = matches / (matches + k), with k = 8 matches
        # Prevents 1-match or 2-match anomalies from saturating top bands
        mu = raw_scores.mean()
        sigma = raw_scores.std() if raw_scores.std() > 0 else 1.0
        z_scores = (raw_scores - mu) / sigma

        k_matches = 8.0
        credibility = df_latest['total_matches'] / (df_latest['total_matches'] + k_matches)
        z_shrunk = z_scores * credibility

        from scipy.stats import norm
        calibrated_overall = norm.cdf(z_shrunk * 1.35) * 100.0

        df_latest['base_overall_rating'] = np.clip(calibrated_overall, 15.0, 99.5)

        # Recent Form Factor Multiplier:
        # Form Factor = clip((Recent 5 Matches Output + 1) / (Career Avg per 5 Matches + 1), 0.75, 1.25)
        def calc_form_factor(row):
            career_5m_bat = (row['career_runs'] / max(row['total_matches'], 1)) * 5.0
            career_5m_bowl = (row['career_wickets'] / max(row['total_matches'], 1)) * 5.0
            career_5m_output = career_5m_bat + (career_5m_bowl * 20.0)
            
            recent_output = row['runs_last_5'] + (row['wickets_last_5'] * 20.0)
            
            ratio = (recent_output + 1.0) / (career_5m_output + 1.0)
            return float(np.clip(ratio, 0.75, 1.25))

        df_latest['form_factor'] = df_latest.apply(calc_form_factor, axis=1)

        # Current Rating = clip(Predicted Rating * 0.90 + 0.10 * (Form Factor * Predicted Rating), 0, 100)
        df_latest['current_rating'] = np.clip(
            (df_latest['base_overall_rating'] * 0.90) + (df_latest['base_overall_rating'] * 0.10 * df_latest['form_factor']),
            10.0,
            99.5
        ).round(1)

        # Competitive Rating Bands:
        # Elite (90.0–100.0), World Class (80.0–89.9), International (70.0–79.9), Domestic Star (60.0–69.9), Squad Player (< 60.0)
        def assign_band(rating):
            if rating >= 90.0:
                return 'Elite'
            elif rating >= 80.0:
                return 'World Class'
            elif rating >= 70.0:
                return 'International'
            elif rating >= 60.0:
                return 'Domestic Star'
            else:
                return 'Squad Player'

        df_latest['rating_band'] = df_latest['current_rating'].apply(assign_band)
        
        # Rank within role & overall
        df_latest['overall_rank'] = df_latest['current_rating'].rank(ascending=False, method='min').astype(int)
        df_latest['role_rank'] = df_latest.groupby('role')['current_rating'].rank(ascending=False, method='min').astype(int)

        print(f"Ratings generated for {len(df_latest)} players. Active players: {df_latest['is_active'].sum()}")
        return df_latest
