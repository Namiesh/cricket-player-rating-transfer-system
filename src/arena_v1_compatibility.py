"""
Arena Player Rating v1 Methodology Implementation & Calibration Module.
Replicates and validates the exact benchmark z-score + normal CDF + credibility shrinkage
from arena_player_rating_workbook.xlsx and provides baseline comparison metrics.
"""

import math
from scipy.stats import norm
import numpy as np
import pandas as pd


# Default Settings from Arena Workbook (sheet 'Benchmarks and Settings')
DEFAULT_SETTINGS = {
    'MinCoverage': 0.60,
    'HighCov': 0.90,
    'HighCred': 0.75,
    'MedCred': 0.50,
    'MissingZ': -0.25,
    'MinGroupSize': 5,
    'BatSRMax': 400.0,
    'EconMax': 20.0,
    'ConsTol': 0.15
}

# Sample-size parameters by format
FORMAT_SAMPLE_PARAMS = {
    'T20': {'bat_k': 12, 'bat_min': 5, 'bowl_k': 30, 'bowl_min': 12},
    'ODI': {'bat_k': 10, 'bat_min': 5, 'bowl_k': 60, 'bowl_min': 30},
    'Test': {'bat_k': 10, 'bat_min': 6, 'bowl_k': 200, 'bowl_min': 100}
}

# Metric weights by format
FORMAT_METRIC_WEIGHTS = {
    'T20': {
        'bat': {'runs_per_inn': 0.30, 'avg': 0.25, 'sr': 0.45},
        'bowl': {'wkts_per_match': 0.30, 'avg': 0.30, 'econ': 0.40}
    },
    'ODI': {
        'bat': {'runs_per_inn': 0.35, 'avg': 0.30, 'sr': 0.35},
        'bowl': {'wkts_per_match': 0.30, 'avg': 0.35, 'econ': 0.35}
    },
    'Test': {
        'bat': {'runs_per_inn': 0.35, 'avg': 0.45, 'sr': 0.20},
        'bowl': {'wkts_per_match': 0.35, 'avg': 0.45, 'econ': 0.20}
    }
}

# Benchmarks (Mean, SD) by Format and Comparison Group
BENCHMARKS = {
    'T20|Batter': {
        'bat': {'runs_per_inn': (24.0, 7.0), 'avg': (27.0, 7.0), 'sr': (132.0, 15.0)}
    },
    'T20|All-rounder': {
        'bat': {'runs_per_inn': (15.0, 6.0), 'avg': (20.0, 6.0), 'sr': (128.0, 16.0)},
        'bowl': {'wkts_per_match': (0.8, 0.35), 'avg': (29.0, 7.0), 'econ': (8.3, 1.2)}
    },
    'T20|Bowler': {
        'bowl': {'wkts_per_match': (1.0, 0.4), 'avg': (26.0, 6.0), 'econ': (7.8, 1.1)}
    },
    'T20|Wicketkeeper': {
        'bat': {'runs_per_inn': (22.0, 7.0), 'avg': (25.0, 7.0), 'sr': (130.0, 15.0)}
    },
    'ODI|Batter': {
        'bat': {'runs_per_inn': (31.0, 9.0), 'avg': (36.0, 8.0), 'sr': (85.0, 12.0)}
    },
    'ODI|All-rounder': {
        'bat': {'runs_per_inn': (22.0, 8.0), 'avg': (27.0, 7.0), 'sr': (84.0, 14.0)},
        'bowl': {'wkts_per_match': (1.0, 0.4), 'avg': (37.0, 7.0), 'econ': (5.5, 0.8)}
    },
    'ODI|Bowler': {
        'bowl': {'wkts_per_match': (1.5, 0.5), 'avg': (33.0, 6.0), 'econ': (5.1, 0.7)}
    },
    'ODI|Wicketkeeper': {
        'bat': {'runs_per_inn': (28.0, 8.0), 'avg': (32.0, 8.0), 'sr': (86.0, 12.0)}
    }
}


def is_valid_num(val):
    return val is not None and not pd.isna(val) and isinstance(val, (int, float)) and math.isfinite(val)


def calculate_arena_v1_rating(
    format_type='T20',
    role='Batter',
    matches=0,
    innings=0,
    runs=0,
    bat_avg=None,
    bat_sr=None,
    wickets=0,
    bowl_avg=None,
    bowl_sr=None,
    economy=None,
    overs_bowled=0.0
):
    """
    Computes exact Arena v1 Rating (0-10 scale), Data Coverage, Credibility, and Reliability.
    """
    fmt = format_type if format_type in FORMAT_SAMPLE_PARAMS else 'T20'
    role_clean = 'Batter' if 'bat' in role.lower() else ('Bowler' if 'bowl' in role.lower() else ('All-rounder' if 'all' in role.lower() else ('Wicketkeeper' if 'keep' in role.lower() else 'Batter')))
    
    settings = DEFAULT_SETTINGS
    sp = FORMAT_SAMPLE_PARAMS[fmt]
    weights = FORMAT_METRIC_WEIGHTS[fmt]
    
    bench_key = f"{fmt}|{role_clean}"
    if bench_key not in BENCHMARKS:
        bench_key = f"{fmt}|Batter" if 'Batter' in bench_key else f"{fmt}|Bowler"
    bench = BENCHMARKS.get(bench_key, BENCHMARKS[f"{fmt}|Batter"])

    # --- Batting Discipline ---
    bat_applies = role_clean in ['Batter', 'All-rounder', 'Wicketkeeper']
    bat_rating = None
    bat_cov = 0.0
    c_bat = 0.0
    z_bat_shrunk = 0.0
    
    if bat_applies and 'bat' in bench:
        b_bench = bench['bat']
        b_weights = weights['bat']
        
        # Step 1: Derived rates
        runs_per_inn = runs / innings if (is_valid_num(innings) and innings > 0 and is_valid_num(runs)) else None
        
        # Step 2: Metric z-scores
        z_dict = {}
        avail_w = 0.0
        tot_w = sum(b_weights.values())
        
        if is_valid_num(runs_per_inn):
            m, s = b_bench['runs_per_inn']
            z_dict['runs_per_inn'] = (runs_per_inn - m) / s
            avail_w += b_weights['runs_per_inn']
            
        if is_valid_num(bat_avg):
            m, s = b_bench['avg']
            z_dict['avg'] = (bat_avg - m) / s
            avail_w += b_weights['avg']
            
        if is_valid_num(bat_sr):
            m, s = b_bench.get('sr', (130.0, 15.0))
            z_dict['sr'] = (bat_sr - m) / s
            avail_w += b_weights['sr']
            
        # Step 3: Composite Z
        missing_w = tot_w - avail_w
        weighted_sum = sum(b_weights[k] * z for k, z in z_dict.items()) + (missing_w * settings['MissingZ'])
        z_bat = weighted_sum / tot_w if tot_w > 0 else 0.0
        
        # Step 4: Coverage
        bat_cov = avail_w / tot_w if tot_w > 0 else 0.0
        
        # Step 5: Credibility
        n_bat = innings if is_valid_num(innings) else 0
        c_bat = n_bat / (n_bat + sp['bat_k']) if n_bat >= sp['bat_min'] else 0.0
        
        # Step 6: Discipline Rating
        if bat_cov >= settings['MinCoverage'] and c_bat > 0:
            bat_rating = float(np.round(10.0 * norm.cdf(z_bat * c_bat), 2))
            z_bat_shrunk = z_bat * c_bat

    # --- Bowling Discipline ---
    bowl_applies = role_clean in ['Bowler', 'All-rounder']
    bowl_rating = None
    bowl_cov = 0.0
    c_bowl = 0.0
    z_bowl_shrunk = 0.0
    
    if bowl_applies and ('bowl' in bench or f"{fmt}|Bowler" in BENCHMARKS):
        bw_bench = bench.get('bowl', BENCHMARKS[f"{fmt}|Bowler"]['bowl'])
        bw_weights = weights['bowl']
        
        # Step 1: Derived rates
        wkts_per_match = wickets / matches if (is_valid_num(matches) and matches > 0 and is_valid_num(wickets)) else None
        
        # Step 2: Metric z-scores (lower is better for avg & econ)
        z_dict = {}
        avail_w = 0.0
        tot_w = sum(bw_weights.values())
        
        if is_valid_num(wkts_per_match):
            m, s = bw_bench['wkts_per_match']
            z_dict['wkts_per_match'] = (wkts_per_match - m) / s
            avail_w += bw_weights['wkts_per_match']
            
        if is_valid_num(bowl_avg) and bowl_avg > 0:
            m, s = bw_bench['avg']
            z_dict['avg'] = -(bowl_avg - m) / s
            avail_w += bw_weights['avg']
            
        if is_valid_num(economy) and economy > 0:
            m, s = bw_bench['econ']
            z_dict['econ'] = -(economy - m) / s
            avail_w += bw_weights['econ']
            
        # Step 3: Composite Z
        missing_w = tot_w - avail_w
        weighted_sum = sum(bw_weights[k] * z for k, z in z_dict.items()) + (missing_w * settings['MissingZ'])
        z_bowl = weighted_sum / tot_w if tot_w > 0 else 0.0
        
        # Step 4: Coverage
        bowl_cov = avail_w / tot_w if tot_w > 0 else 0.0
        
        # Step 5: Credibility
        n_bowl = overs_bowled if is_valid_num(overs_bowled) else 0.0
        c_bowl = n_bowl / (n_bowl + sp['bowl_k']) if n_bowl >= sp['bowl_min'] else 0.0
        
        # Step 6: Discipline Rating
        if bowl_cov >= settings['MinCoverage'] and c_bowl > 0:
            bowl_rating = float(np.round(10.0 * norm.cdf(z_bowl * c_bowl), 2))
            z_bowl_shrunk = z_bowl * c_bowl

    # --- Step 7: Overall Rating ---
    if role_clean in ['Batter', 'Wicketkeeper']:
        overall_rating = bat_rating
        overall_cov = bat_cov
        c_overall = c_bat
    elif role_clean == 'Bowler':
        overall_rating = bowl_rating
        overall_cov = bowl_cov
        c_overall = c_bowl
    else:  # All-rounder
        if c_bat > 0 and c_bowl > 0:
            c_combined = 1.0 - (1.0 - c_bat) * (1.0 - c_bowl)
            z_overall = ((c_bat * z_bat_shrunk / (c_bat if c_bat>0 else 1) + c_bowl * z_bowl_shrunk / (c_bowl if c_bowl>0 else 1)) / (c_bat + c_bowl)) * c_combined
            overall_rating = float(np.round(10.0 * norm.cdf(z_overall), 2))
            c_overall = c_combined
            overall_cov = (bat_cov + bowl_cov) / 2.0
        elif c_bat > 0:
            overall_rating = bat_rating
            c_overall = c_bat
            overall_cov = bat_cov
        elif c_bowl > 0:
            overall_rating = bowl_rating
            c_overall = c_bowl
            overall_cov = bowl_cov
        else:
            overall_rating = None
            c_overall = 0.0
            overall_cov = (bat_cov + bowl_cov) / 2.0

    # --- Step 8: Reliability ---
    if c_overall >= settings['HighCred'] and overall_cov >= settings['HighCov']:
        reliability = 'High'
    elif c_overall >= settings['MedCred'] and overall_cov >= settings['MinCoverage']:
        reliability = 'Medium'
    else:
        reliability = 'Low'

    # --- Step 9: Status ---
    if overall_rating is None:
        status = 'Insufficient data'
    elif reliability == 'Low':
        status = 'Provisional'
    else:
        status = 'Rated'

    return {
        'arena_v1_bat_rating': bat_rating,
        'arena_v1_bowl_rating': bowl_rating,
        'arena_v1_overall_rating': overall_rating,
        'arena_v1_coverage': round(overall_cov * 100, 1),
        'arena_v1_credibility': round(c_overall, 3),
        'arena_v1_reliability': reliability,
        'arena_v1_status': status
    }
