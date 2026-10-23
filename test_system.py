"""
Unit & Integration Tests for Cricket Player Rating & Transfer Value System.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding='utf-8')

from src.arena_v1_compatibility import calculate_arena_v1_rating
from src.valuation_engine import CricketTransferValuationEngine


def test_arena_v1_calculator():
    print("Testing Arena v1 Compatibility Engine...")
    # Test high performing batter
    res_batter = calculate_arena_v1_rating(
        format_type='T20',
        role='Batter',
        matches=38,
        innings=36,
        runs=1100,
        bat_avg=45.0,
        bat_sr=155.0
    )
    assert res_batter['arena_v1_bat_rating'] is not None
    assert res_batter['arena_v1_overall_rating'] > 7.0
    assert res_batter['arena_v1_reliability'] == 'High'
    print(f"  [PASS] Arena v1 Batter Rating: {res_batter['arena_v1_overall_rating']} ({res_batter['arena_v1_reliability']})")

    # Test bowler
    res_bowler = calculate_arena_v1_rating(
        format_type='T20',
        role='Bowler',
        matches=40,
        wickets=55,
        bowl_avg=19.5,
        economy=6.8,
        overs_bowled=150.0
    )
    assert res_bowler['arena_v1_bowl_rating'] is not None
    assert res_bowler['arena_v1_overall_rating'] > 7.0
    print(f"  [PASS] Arena v1 Bowler Rating: {res_bowler['arena_v1_overall_rating']} ({res_bowler['arena_v1_reliability']})")


def test_transfer_valuation_engine():
    print("Testing 7-Factor Valuation Engine...")
    engine = CricketTransferValuationEngine()
    
    # Elite All-Rounder (like Rashid Khan / Hardik Pandya)
    sample_elite = {
        'current_rating': 94.5,
        'runs_last_5': 130.0,
        'wickets_last_5': 8.0,
        'opponent_bowler_difficulty': 78.0,
        'form_factor': 1.15,
        'career_span_years': 8.5,
        'total_matches': 110.0,
        'role': 'All-Rounder',
        'role_rank': 1,
        'is_active': True
    }
    val = engine.calculate_7factor_valuation_index(sample_elite)
    print(f"  [PASS] Elite All-Rounder Valuation: ₹{val['transfer_value_cr']} Cr (Index: {val['valuation_index']})")
    assert val['transfer_value_cr'] >= 15.0
    assert val['transfer_value_cr'] <= 30.0

    # Squad Player
    sample_squad = {
        'current_rating': 56.0,
        'runs_last_5': 20.0,
        'wickets_last_5': 1.0,
        'opponent_bowler_difficulty': 45.0,
        'form_factor': 0.85,
        'career_span_years': 2.0,
        'total_matches': 12.0,
        'role': 'Batsman',
        'role_rank': 45,
        'is_active': True
    }
    val_squad = engine.calculate_7factor_valuation_index(sample_squad)
    print(f"  [PASS] Squad Player Valuation: ₹{val_squad['transfer_value_cr']} Cr (Index: {val_squad['valuation_index']})")
    assert val_squad['transfer_value_cr'] < val['transfer_value_cr']
    assert val_squad['transfer_value_cr'] >= 0.20


if __name__ == "__main__":
    test_arena_v1_calculator()
    test_transfer_valuation_engine()
    print("\nALL UNIT TESTS PASSED SUCCESSFULLY!")
