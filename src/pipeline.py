"""
End-to-End Orchestration Pipeline for Cricket Player Rating & Transfer Value System.
Coordinates ingestion, walk-forward feature engineering, CatBoost model training,
dynamic role aggregation, dual-mode valuation, and Excel/CSV exporting.
"""

import os
import json
from pathlib import Path
import pandas as pd
import numpy as np

from src.ingestion import CricketDataIngestion
from src.context_features import WalkForwardFeatureEngine
from src.rating_engine import CricketRatingEngine
from src.valuation_engine import CricketTransferValuationEngine
from src.arena_v1_compatibility import calculate_arena_v1_rating


class CricketPipelineOrchestrator:
    def __init__(self, base_dir=None):
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).resolve().parent.parent
        self.data_dir = self.base_dir / "data" / "processed"
        self.models_dir = self.base_dir / "models"
        self.export_dir = self.base_dir / "exported_workbooks"
        
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.export_dir.mkdir(parents=True, exist_ok=True)

    def run(self, max_matches=1200):
        print("=" * 80)
        print("STARTING CRICKET PLAYER RATING & TRANSFER VALUE SYSTEM PIPELINE")
        print("=" * 80)

        # 1. Ingestion & Feature Extraction (with caching)
        features_csv = self.data_dir / "player_walkforward_features.csv"
        if features_csv.exists() and features_csv.stat().st_size > 1000:
            print(f"Loading cached walk-forward features from: {features_csv}")
            df_features = pd.read_csv(features_csv)
            df_features['date'] = pd.to_datetime(df_features['date'])
            df_matches = pd.DataFrame({'match_id': df_features['match_id'].unique()})
            df_pm = df_features
        else:
            ingestor = CricketDataIngestion()
            match_files = ingestor.find_all_json_matches(max_matches=max_matches)
            df_matches, df_pm = ingestor.parse_matches(match_files)

            feature_engine = WalkForwardFeatureEngine()
            df_features = feature_engine.compute_walk_forward_features(df_pm)
            df_features.to_csv(features_csv, index=False)
            print(f"Saved processed features to: {features_csv}")

        # 3. Model Training
        rating_engine = CricketRatingEngine(models_dir=self.models_dir)
        rating_metrics = rating_engine.train_models(df_features)

        # 4. Generate Current Ratings for Players
        df_players = rating_engine.generate_current_player_ratings(df_features, rolling_years=4.0)

        # 5. Dual-Mode Transfer Value Calculation
        valuation_engine = CricketTransferValuationEngine(models_dir=self.models_dir)
        valuation_engine.train_supervised_valuation_model()

        print("Computing Transfer Values and Arena v1 benchmarks for all players...")
        valuation_results = []
        arena_v1_results = []
        supervised_vals = []

        for _, row in df_players.iterrows():
            val_dict = valuation_engine.calculate_7factor_valuation_index(row)
            valuation_results.append(val_dict)

            # Supervised Transfer Value
            sup_val = valuation_engine.predict_supervised_transfer_value(
                rating=row['current_rating'],
                role=row['role'],
                is_active=row['is_active']
            )
            supervised_vals.append(sup_val)

            # Arena v1 baseline rating
            v1_dict = calculate_arena_v1_rating(
                format_type='T20',
                role=row['role'],
                matches=row['total_matches'],
                innings=row.get('career_innings', row.get('total_matches', 0)),
                runs=row['career_runs'],
                bat_avg=row['career_batting_avg'],
                bat_sr=row['career_batting_sr'],
                wickets=row['career_wickets'],
                bowl_avg=row['career_bowling_avg'] if row['career_wickets'] > 0 else None,
                economy=row['career_economy'] if row['career_overs'] > 0 else None,
                overs_bowled=row['career_overs']
            )
            arena_v1_results.append(v1_dict)

        df_vals = pd.DataFrame(valuation_results)
        df_v1 = pd.DataFrame(arena_v1_results)

        df_final = pd.concat([
            df_players.reset_index(drop=True),
            df_vals.reset_index(drop=True),
            df_v1.reset_index(drop=True)
        ], axis=1)

        df_final['supervised_transfer_value_cr'] = supervised_vals

        # Sort by Current Rating
        df_final = df_final.sort_values('current_rating', ascending=False).reset_index(drop=True)
        df_final['rank'] = range(1, len(df_final) + 1)

        # 6. Save Final Outputs
        out_csv = self.data_dir / "active_players_ratings_valuations.csv"
        df_final.to_csv(out_csv, index=False)
        print(f"Saved master player ratings & valuations CSV to: {out_csv}")

        # 7. Generate Excel Workbook
        excel_path = self.export_dir / "arena_player_rating_and_valuation_calculated.xlsx"
        self.export_rich_excel(df_final, excel_path)

        # 8. Evaluation Report
        eval_report = {
            'total_matches_processed': len(df_matches),
            'total_player_match_instances': len(df_pm),
            'unique_players_evaluated': len(df_final),
            'active_players_in_pool': int(df_final['is_active'].sum()),
            'rating_engine_metrics': rating_metrics,
            'rating_band_distribution': df_final['rating_band'].value_counts().to_dict(),
            'role_distribution': df_final['role'].value_counts().to_dict(),
            'valuation_summary': {
                'min_transfer_value_cr': float(df_final['transfer_value_cr'].min()),
                'max_transfer_value_cr': float(df_final['transfer_value_cr'].max()),
                'mean_transfer_value_cr': float(round(df_final['transfer_value_cr'].mean(), 2)),
                'median_transfer_value_cr': float(round(df_final['transfer_value_cr'].median(), 2))
            }
        }
        with open(self.models_dir / "full_pipeline_report.json", "w", encoding="utf-8") as f:
            json.dump(eval_report, f, indent=2)

        print("=" * 80)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"Elite Players: {(df_final['rating_band'] == 'Elite').sum()}")
        print(f"World Class Players: {(df_final['rating_band'] == 'World Class').sum()}")
        print(f"International Players: {(df_final['rating_band'] == 'International').sum()}")
        print(f"Top 5 Players by Transfer Value:\n{df_final[['rank', 'player', 'role', 'current_rating', 'rating_band', 'transfer_value_cr']].head(5).to_string(index=False)}")
        print("=" * 80)
        return df_final

    def export_rich_excel(self, df_final, filepath):
        """Exports styled multi-sheet Excel workbook with leaderboard, benchmarks, and comparison."""
        print(f"Generating rich Excel workbook at: {filepath}...")
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Sheet 1: Master Rating & Valuation Board
            cols_sheet1 = [
                'rank', 'player', 'role', 'current_rating', 'rating_band',
                'transfer_value_cr', 'supervised_transfer_value_cr', 'valuation_index',
                'form_factor', 'pred_batting_rating', 'pred_bowling_rating',
                'total_matches', 'career_runs', 'career_batting_avg', 'career_batting_sr',
                'career_wickets', 'career_economy', 'career_bowling_avg',
                'arena_v1_overall_rating', 'arena_v1_credibility', 'arena_v1_reliability',
                'arena_v1_status', 'is_active', 'latest_match_date'
            ]
            export_cols = [c for c in cols_sheet1 if c in df_final.columns]
            df_final[export_cols].to_excel(writer, sheet_name='Player Ratings & Valuations', index=False)

            # Sheet 2: Active Players Only
            df_active = df_final[df_final['is_active']].copy()
            df_active['active_rank'] = range(1, len(df_active) + 1)
            active_cols = ['active_rank'] + [c for c in export_cols if c != 'rank']
            df_active[active_cols].to_excel(writer, sheet_name='Active Player Pool', index=False)

            # Sheet 3: Role Breakdown & Scarcity
            role_summary = df_final.groupby('role').agg(
                player_count=('player', 'count'),
                avg_rating=('current_rating', 'mean'),
                max_rating=('current_rating', 'max'),
                avg_transfer_value_cr=('transfer_value_cr', 'mean'),
                max_transfer_value_cr=('transfer_value_cr', 'max')
            ).reset_index()
            role_summary.to_excel(writer, sheet_name='Role Market Analytics', index=False)

            # Sheet 4: Competitive Rating Bands
            band_summary = df_final.groupby('rating_band').agg(
                player_count=('player', 'count'),
                avg_rating=('current_rating', 'mean'),
                avg_transfer_value_cr=('transfer_value_cr', 'mean'),
                min_transfer_value_cr=('transfer_value_cr', 'min'),
                max_transfer_value_cr=('transfer_value_cr', 'max')
            ).reset_index()
            band_summary.to_excel(writer, sheet_name='Competitive Bands', index=False)

        print(f"Workbook successfully exported with 4 sheets to: {filepath}")
