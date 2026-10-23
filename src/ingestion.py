"""
Data Ingestion Engine for Cricket Player Rating & Transfer Value System.
Processes ball-by-ball JSON records (Cricsheet T20s / T20Is) and integrates
Opta / Arena metadata, computing exact ball-level aggregates without historical data leakage.
"""

import os
import json
import glob
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np


class CricketDataIngestion:
    def __init__(self, cricsheet_dir=None, venue_dir=None, arena_workbook=None):
        base_coding = Path(r"c:\Users\gnami\OneDrive\Desktop\coding")
        self.cricsheet_dir = Path(cricsheet_dir) if cricsheet_dir else base_coding / "cricket-win-predictor" / "data" / "raw" / "t20s_male_json"
        self.venue_dir = Path(venue_dir) if venue_dir else base_coding / "cricket-venue-analytics" / "data" / "raw" / "t20i"
        self.arena_workbook = Path(arena_workbook) if arena_workbook else base_coding / "arena_player_rating_workbook.xlsx"
        
    def find_all_json_matches(self, max_matches=None):
        """Discovers and deduplicates match JSON files across available sources."""
        files = []
        if self.cricsheet_dir.exists():
            files.extend(list(self.cricsheet_dir.glob("*.json")))
        if self.venue_dir.exists():
            files.extend(list(self.venue_dir.glob("*.json")))
            
        # Deduplicate by match filename stem
        unique_files = {}
        for f in files:
            stem = f.stem
            if stem not in unique_files:
                unique_files[stem] = f
                
        all_unique = list(unique_files.values())
        print(f"Found {len(all_unique)} unique match files across repositories.")
        if max_matches and len(all_unique) > max_matches:
            all_unique = all_unique[:max_matches]
            print(f"Sampling {len(all_unique)} matches for processing.")
        return all_unique

    def parse_matches(self, match_files):
        """
        Extracts exact match-level and player-match deliveries.
        Guarantees strict separation of extras, dismissal attribution, and legal balls.
        """
        match_rows = []
        batting_deliveries = []
        bowling_deliveries = []

        total = len(match_files)
        for i, filepath in enumerate(match_files):
            try:
                with open(filepath, 'r', encoding='utf-8') as fp:
                    data = json.load(fp)
            except Exception:
                continue

            info = data.get('info', {})
            dates = info.get('dates', [])
            match_date = str(dates[0]) if dates else '2000-01-01'
            venue = info.get('venue', 'Neutral Venue')
            event_name = info.get('event', {}).get('name', 'T20 Match')
            match_type = info.get('match_type', 'T20')
            teams = info.get('teams', ['Team A', 'Team B'])
            match_id = filepath.stem

            match_rows.append({
                'match_id': match_id,
                'date': match_date,
                'venue': venue,
                'event': event_name,
                'match_type': match_type,
                'team1': teams[0] if len(teams) > 0 else 'Unknown',
                'team2': teams[1] if len(teams) > 1 else 'Unknown'
            })

            innings_list = data.get('innings', [])
            for inn_idx, innings in enumerate(innings_list):
                batting_team = innings.get('team', teams[0] if inn_idx == 0 else teams[1])
                bowling_team = teams[1] if batting_team == teams[0] else teams[0]

                overs_list = innings.get('overs', [])
                for over_obj in overs_list:
                    over_num = over_obj.get('over', 0)
                    deliveries = over_obj.get('deliveries', [])
                    for deliv in deliveries:
                        batter = str(deliv.get('batter', '')).strip()
                        bowler = str(deliv.get('bowler', '')).strip()
                        non_striker = str(deliv.get('non_striker', '')).strip()

                        if not batter or not bowler:
                            continue

                        runs_info = deliv.get('runs', {})
                        r_bat = int(runs_info.get('batter', 0))
                        r_extra = int(runs_info.get('extras', 0))
                        r_total = int(runs_info.get('total', 0))

                        extras_info = deliv.get('extras', {})
                        is_wide = 'wides' in extras_info
                        is_noball = 'noballs' in extras_info
                        is_bye = 'byes' in extras_info
                        is_legbye = 'legbyes' in extras_info
                        is_legal = not is_wide and not is_noball

                        # Official batting ball: wide is NOT faced; noball is faced
                        ball_faced = 0 if is_wide else 1

                        # Dismissals: official bowler wicket excludes run-outs & retirements
                        wickets = deliv.get('wickets', [])
                        dismissed_player = None
                        is_bowler_wkt = False
                        if wickets:
                            w = wickets[0]
                            dismissed_player = str(w.get('player_out', '')).strip()
                            w_kind = str(w.get('kind', '')).lower()
                            if w_kind not in ['run out', 'retired hurt', 'retired out', 'obstructing the field', 'hit the ball twice']:
                                is_bowler_wkt = True

                        # Bowling runs conceded: excludes byes and legbyes
                        bowler_runs_conceded = (r_bat + r_extra) if not (is_bye or is_legbye) else r_bat

                        batting_deliveries.append({
                            'match_id': match_id,
                            'date': match_date,
                            'player': batter,
                            'team': batting_team,
                            'opponent': bowling_team,
                            'runs': r_bat,
                            'balls_faced': ball_faced,
                            'four': 1 if r_bat == 4 else 0,
                            'six': 1 if r_bat == 6 else 0,
                            'dot': 1 if (r_bat == 0 and not is_wide) else 0,
                            'is_dismissed': 1 if (dismissed_player == batter) else 0,
                            'innings_id': inn_idx + 1
                        })

                        bowling_deliveries.append({
                            'match_id': match_id,
                            'date': match_date,
                            'player': bowler,
                            'team': bowling_team,
                            'opponent': batting_team,
                            'over_num': over_num,
                            'legal_ball': 1 if is_legal else 0,
                            'runs_conceded': bowler_runs_conceded,
                            'wicket': 1 if is_bowler_wkt else 0,
                            'dot_ball': 1 if (bowler_runs_conceded == 0 and is_legal) else 0,
                            'boundary_conceded': 1 if r_bat >= 4 else 0
                        })

        df_matches = pd.DataFrame(match_rows).drop_duplicates(subset=['match_id']).sort_values('date').reset_index(drop=True)
        print(f"Parsed {len(df_matches)} total matches.")
        
        # Aggregate Batting at Player-Match level
        df_bat = pd.DataFrame(batting_deliveries)
        if not df_bat.empty:
            df_pm_bat = df_bat.groupby(['match_id', 'date', 'player', 'team', 'opponent']).agg(
                batting_runs=('runs', 'sum'),
                batting_balls=('balls_faced', 'sum'),
                batting_fours=('four', 'sum'),
                batting_sixes=('six', 'sum'),
                batting_dots=('dot', 'sum'),
                batting_dismissals=('is_dismissed', 'sum'),
                batting_innings=('innings_id', 'nunique')
            ).reset_index()
        else:
            df_pm_bat = pd.DataFrame()

        # Aggregate Bowling at Player-Match level
        df_bowl = pd.DataFrame(bowling_deliveries)
        if not df_bowl.empty:
            df_pm_bowl = df_bowl.groupby(['match_id', 'date', 'player', 'team', 'opponent']).agg(
                bowling_legal_balls=('legal_ball', 'sum'),
                bowling_runs_conceded=('runs_conceded', 'sum'),
                bowling_wickets=('wicket', 'sum'),
                bowling_dot_balls=('dot_ball', 'sum'),
                bowling_boundaries_conceded=('boundary_conceded', 'sum')
            ).reset_index()
            df_pm_bowl['bowling_overs'] = df_pm_bowl['bowling_legal_balls'] / 6.0
        else:
            df_pm_bowl = pd.DataFrame()

        # Full outer join of player appearances in each match
        if not df_pm_bat.empty and not df_pm_bowl.empty:
            df_pm = pd.merge(df_pm_bat, df_pm_bowl, on=['match_id', 'date', 'player', 'team', 'opponent'], how='outer')
        elif not df_pm_bat.empty:
            df_pm = df_pm_bat
        else:
            df_pm = df_pm_bowl

        df_pm = pd.merge(df_pm, df_matches[['match_id', 'venue', 'event', 'match_type']], on='match_id', how='left')

        # Fill NaNs for players who only batted or only bowled
        fill_zeros = ['batting_runs', 'batting_balls', 'batting_fours', 'batting_sixes', 'batting_dots',
                      'batting_dismissals', 'batting_innings', 'bowling_legal_balls', 'bowling_overs',
                      'bowling_runs_conceded', 'bowling_wickets', 'bowling_dot_balls', 'bowling_boundaries_conceded']
        for col in fill_zeros:
            if col in df_pm.columns:
                df_pm[col] = df_pm[col].fillna(0)

        df_pm['date'] = pd.to_datetime(df_pm['date'])
        df_pm = df_pm.sort_values(['date', 'match_id', 'player']).reset_index(drop=True)
        print(f"Generated {len(df_pm)} unified player-match records across {df_pm['player'].nunique()} unique players.")
        return df_matches, df_pm
