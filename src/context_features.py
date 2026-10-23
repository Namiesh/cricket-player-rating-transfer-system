"""
Walk-Forward Leak-Free Feature Engine & Context Adjustments.
Computes strictly causal career features, rolling form windows, venue scoring multipliers,
opponent difficulty indices, and context-adjusted target metrics (0-100 scale).
"""

import os
import pickle
from pathlib import Path
import pandas as pd
import numpy as np


class WalkForwardFeatureEngine:
    def __init__(self, splits_cache_path=None):
        base_coding = Path(r"c:\Users\gnami\OneDrive\Desktop\coding")
        self.splits_path = Path(splits_cache_path) if splits_cache_path else base_coding / "batter_splits_cache.pkl"
        self.splits_cache = {}
        if self.splits_path.exists():
            try:
                with open(self.splits_path, 'rb') as f:
                    self.splits_cache = pickle.load(f)
                print(f"Loaded splits cache for {len(self.splits_cache)} player keys.")
            except Exception as e:
                print(f"Note: Could not load splits cache: {e}")

    def build_venue_factors(self, df_pm):
        """Computes venue run-scoring and bowling difficulty baselines."""
        venue_stats = df_pm.groupby('venue').agg(
            total_matches=('match_id', 'nunique'),
            total_runs=('batting_runs', 'sum'),
            total_balls=('batting_balls', 'sum'),
            total_wickets=('bowling_wickets', 'sum'),
            total_overs=('bowling_overs', 'sum')
        ).reset_index()

        global_run_rate = (venue_stats['total_runs'].sum() / max(venue_stats['total_overs'].sum(), 1.0))
        global_wkt_rate = (venue_stats['total_wickets'].sum() / max(venue_stats['total_overs'].sum(), 1.0))

        venue_stats['venue_run_rate'] = venue_stats['total_runs'] / venue_stats['total_overs'].replace(0, np.nan)
        venue_stats['venue_batting_factor'] = (venue_stats['venue_run_rate'] / global_run_rate).fillna(1.0).clip(0.70, 1.35)
        venue_stats['venue_bowling_factor'] = (global_run_rate / venue_stats['venue_run_rate']).fillna(1.0).clip(0.70, 1.35)

        venue_map = venue_stats.set_index('venue')[['venue_batting_factor', 'venue_bowling_factor']].to_dict(orient='index')
        return venue_map

    def compute_walk_forward_features(self, df_pm):
        """
        Calculates career cumulative and rolling features with strict shift(1).
        Ensures match N only uses data from matches 1 to N-1.
        """
        print("Calculating temporal walk-forward features for every player...")
        df = df_pm.sort_values(['date', 'match_id']).copy()

        # Step 1: Assign Venue Factors
        venue_map = self.build_venue_factors(df)
        df['venue_batting_factor'] = df['venue'].map(lambda v: venue_map.get(v, {}).get('venue_batting_factor', 1.0))
        df['venue_bowling_factor'] = df['venue'].map(lambda v: venue_map.get(v, {}).get('venue_bowling_factor', 1.0))

        # Group by player for walk-forward shifts
        grouped = df.groupby('player', group_keys=False)

        # Batting prior accumulators
        df['career_matches_before'] = grouped.cumcount()
        df['career_innings_before'] = grouped['batting_innings'].apply(lambda x: x.shift(1).fillna(0).cumsum())
        df['career_runs_before'] = grouped['batting_runs'].apply(lambda x: x.shift(1).fillna(0).cumsum())
        df['career_balls_before'] = grouped['batting_balls'].apply(lambda x: x.shift(1).fillna(0).cumsum())
        df['career_dismissals_before'] = grouped['batting_dismissals'].apply(lambda x: x.shift(1).fillna(0).cumsum())
        df['career_fours_before'] = grouped['batting_fours'].apply(lambda x: x.shift(1).fillna(0).cumsum())
        df['career_sixes_before'] = grouped['batting_sixes'].apply(lambda x: x.shift(1).fillna(0).cumsum())
        df['career_dots_before'] = grouped['batting_dots'].apply(lambda x: x.shift(1).fillna(0).cumsum())

        # Derived Batting Rates
        df['career_batting_avg_before'] = df['career_runs_before'] / df['career_dismissals_before'].replace(0, 1)
        df['career_batting_sr_before'] = (df['career_runs_before'] / df['career_balls_before'].replace(0, 1)) * 100.0
        df['career_boundary_rate_before'] = ((df['career_fours_before'] + df['career_sixes_before']) / df['career_balls_before'].replace(0, 1)) * 100.0
        df['career_dot_pct_before'] = (df['career_dots_before'] / df['career_balls_before'].replace(0, 1)) * 100.0

        # Batting Rolling Form (Strictly preceding)
        df['runs_last_3'] = grouped['batting_runs'].apply(lambda x: x.shift(1).rolling(3, min_periods=1).sum().fillna(0))
        df['runs_last_5'] = grouped['batting_runs'].apply(lambda x: x.shift(1).rolling(5, min_periods=1).sum().fillna(0))
        df['runs_last_10'] = grouped['batting_runs'].apply(lambda x: x.shift(1).rolling(10, min_periods=1).sum().fillna(0))
        
        df['balls_last_5'] = grouped['batting_balls'].apply(lambda x: x.shift(1).rolling(5, min_periods=1).sum().fillna(0))
        df['sr_last_5'] = (df['runs_last_5'] / df['balls_last_5'].replace(0, 1)) * 100.0

        # Bowling prior accumulators
        df['career_overs_before'] = grouped['bowling_overs'].apply(lambda x: x.shift(1).fillna(0).cumsum())
        df['career_wickets_before'] = grouped['bowling_wickets'].apply(lambda x: x.shift(1).fillna(0).cumsum())
        df['career_runs_conc_before'] = grouped['bowling_runs_conceded'].apply(lambda x: x.shift(1).fillna(0).cumsum())
        df['career_dot_balls_before'] = grouped['bowling_dot_balls'].apply(lambda x: x.shift(1).fillna(0).cumsum())
        df['career_legal_balls_before'] = grouped['bowling_legal_balls'].apply(lambda x: x.shift(1).fillna(0).cumsum())

        # Derived Bowling Rates
        df['career_economy_before'] = df['career_runs_conc_before'] / df['career_overs_before'].replace(0, 1)
        df['career_bowling_avg_before'] = df['career_runs_conc_before'] / df['career_wickets_before'].replace(0, 1)
        df['career_bowling_dot_pct_before'] = (df['career_dot_balls_before'] / df['career_legal_balls_before'].replace(0, 1)) * 100.0

        # Bowling Rolling Form (Strictly preceding)
        df['wickets_last_3'] = grouped['bowling_wickets'].apply(lambda x: x.shift(1).rolling(3, min_periods=1).sum().fillna(0))
        df['wickets_last_5'] = grouped['bowling_wickets'].apply(lambda x: x.shift(1).rolling(5, min_periods=1).sum().fillna(0))
        df['wickets_last_10'] = grouped['bowling_wickets'].apply(lambda x: x.shift(1).rolling(10, min_periods=1).sum().fillna(0))
        
        df['overs_last_5'] = grouped['bowling_overs'].apply(lambda x: x.shift(1).rolling(5, min_periods=1).sum().fillna(0))
        df['runs_conc_last_5'] = grouped['bowling_runs_conceded'].apply(lambda x: x.shift(1).rolling(5, min_periods=1).sum().fillna(0))
        df['econ_last_5'] = df['runs_conc_last_5'] / df['overs_last_5'].replace(0, 1)

        # Step 2: Opponent Difficulty Indices (Pre-match strength of opponent)
        team_pre_bat = df.groupby(['match_id', 'team'])['career_batting_sr_before'].mean().to_dict()
        team_pre_bowl = df.groupby(['match_id', 'team'])['career_economy_before'].mean().to_dict()

        df['opponent_bowler_difficulty'] = df.apply(
            lambda r: np.clip(100.0 - (team_pre_bowl.get((r['match_id'], r['opponent']), 8.2) * 8.0), 20.0, 95.0),
            axis=1
        )
        df['opponent_batter_difficulty'] = df.apply(
            lambda r: np.clip((team_pre_bat.get((r['match_id'], r['opponent']), 125.0) / 1.70), 25.0, 95.0),
            axis=1
        )

        # Step 3: Context-Adjusted Match Performance Indices (Target Variables 0-100)
        # Batting Performance Index: Venue-normalized runs, SR impact, Opponent Bowler difficulty
        norm_runs = (df['batting_runs'] / df['venue_batting_factor']).clip(0, 120)
        match_sr = np.where(df['batting_balls'] > 0, (df['batting_runs'] / df['batting_balls']) * 100.0, 0.0)
        sr_impact = np.clip(match_sr / 1.5, 0, 100)
        boundary_impact = np.clip((df['batting_fours'] * 4 + df['batting_sixes'] * 6) * 1.5, 0, 100)
        
        df['target_batting_index'] = (
            0.35 * np.clip(norm_runs * 1.5, 0, 100) +
            0.25 * sr_impact +
            0.20 * df['opponent_bowler_difficulty'] +
            0.20 * boundary_impact
        ).clip(0.0, 100.0)

        # Bowling Performance Index: Wickets, Venue-normalized economy, Dots, Opponent Batter difficulty
        wkt_pts = np.clip(df['bowling_wickets'] * 25.0, 0, 100)
        match_econ = np.where(df['bowling_overs'] > 0, df['bowling_runs_conceded'] / df['bowling_overs'], 8.0)
        norm_econ = match_econ * df['venue_bowling_factor']
        econ_pts = np.clip((14.0 - norm_econ) * 8.0, 0, 100)
        dot_pts = np.clip(df['bowling_dot_balls'] * 6.0, 0, 100)
        
        df['target_bowling_index'] = (
            0.40 * wkt_pts +
            0.25 * econ_pts +
            0.20 * dot_pts +
            0.15 * df['opponent_batter_difficulty']
        ).clip(0.0, 100.0)

        print("Feature engineering complete. Sample columns created:")
        print([c for c in df.columns if 'before' in c or 'last_' in c or 'target_' in c])
        return df
