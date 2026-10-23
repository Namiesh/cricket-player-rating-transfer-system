"""
Cricket Player Rating & Transfer Value System - Interactive Analytics Dashboard.
Provides a modern visual interface for player performance ratings, competitive bands,
dual-mode market valuation (₹ Crore), 7-factor index decomposition, and head-to-head comparisons.
"""

import os
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Cricket Player Rating & Transfer Value System",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .badge-elite {
        background-color: #f59e0b;
        color: #000;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.82rem;
    }
    .badge-world-class {
        background-color: #8b5cf6;
        color: #fff;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.82rem;
    }
    .badge-intl {
        background-color: #0284c7;
        color: #fff;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.82rem;
    }
    .badge-domestic {
        background-color: #0d9488;
        color: #fff;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.82rem;
    }
    .badge-squad {
        background-color: #64748b;
        color: #fff;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.82rem;
    }
    .valuation-highlight {
        font-size: 2.2rem;
        font-weight: 800;
        color: #10b981;
    }
    .rating-highlight {
        font-size: 2.2rem;
        font-weight: 800;
        color: #38bdf8;
    }
</style>
""", unsafe_allow_html=True)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "active_players_ratings_valuations.csv"
EXCEL_PATH = BASE_DIR / "exported_workbooks" / "arena_player_rating_and_valuation_calculated.xlsx"


@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        return None
    df = pd.read_csv(DATA_PATH)
    # Ensure correct data types
    df['current_rating'] = df['current_rating'].astype(float)
    df['transfer_value_cr'] = df['transfer_value_cr'].astype(float)
    if 'supervised_transfer_value_cr' in df.columns:
        df['supervised_transfer_value_cr'] = df['supervised_transfer_value_cr'].astype(float)
    return df


def main():
    st.title("🏏 Cricket Player Rating & Transfer Value System")
    st.caption("Production-grade Context-Adjusted Machine Learning Ratings, Dynamic Role Weighting & Dual-Mode Market Valuation Engine")

    df = load_data()

    if df is None or df.empty:
        st.warning("⚠️ Processed dataset not found yet. The pipeline may still be executing or needs to be run.")
        if st.button("🚀 Run Pipeline Now"):
            with st.spinner("Executing pipeline..."):
                os.system(f"python {BASE_DIR}/run_pipeline.py --matches 800")
            st.success("Pipeline finished. Please refresh the page.")
            st.rerun()
        st.stop()

    # --- Sidebar Controls ---
    st.sidebar.header("🎯 Filters & Search")

    search_player = st.sidebar.text_input("🔍 Search Player", "", placeholder="e.g. Rashid Khan, Kohli...")

    active_only = st.sidebar.checkbox("Active Players Only (Rolling 4-Yr Window)", value=True)

    roles = ['All'] + sorted(list(df['role'].dropna().unique()))
    selected_role = st.sidebar.selectbox("Filter by Role", roles)

    bands = ['All', 'Elite (90+)', 'World Class (80-89.9)', 'International (70-79.9)', 'Domestic Star (60-69.9)', 'Squad Player (<60)']
    selected_band = st.sidebar.selectbox("Filter by Competitive Band", bands)

    valuation_mode = st.sidebar.radio(
        "Valuation Mode Display",
        ["7-Factor Valuation Index Mode", "Supervised ML Market Mode"]
    )

    # Filter DataFrame
    filtered_df = df.copy()
    if active_only and 'is_active' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['is_active']]

    if selected_role != 'All':
        filtered_df = filtered_df[filtered_df['role'] == selected_role]

    if selected_band != 'All':
        band_name = selected_band.split(' ')[0]
        filtered_df = filtered_df[filtered_df['rating_band'].str.startswith(band_name)]

    if search_player:
        filtered_df = filtered_df[filtered_df['player'].str.contains(search_player, case=False, na=False)]

    # --- Top KPI Metrics Bar ---
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        st.metric("Total Players", len(filtered_df))
    with kpi2:
        elite_count = (filtered_df['rating_band'] == 'Elite').sum()
        st.metric("Elite Players (90+)", elite_count)
    with kpi3:
        wc_count = (filtered_df['rating_band'] == 'World Class').sum()
        st.metric("World Class (80-89.9)", wc_count)
    with kpi4:
        avg_val = filtered_df['transfer_value_cr'].mean() if not filtered_df.empty else 0.0
        st.metric("Avg Valuation", f"₹{avg_val:.2f} Cr")
    with kpi5:
        max_val = filtered_df['transfer_value_cr'].max() if not filtered_df.empty else 0.0
        st.metric("Top Valuation", f"₹{max_val:.2f} Cr")

    st.divider()

    # --- Main Tab Layout ---
    tab_board, tab_profile, tab_h2h, tab_method = st.tabs([
        "📊 Player Leaderboard",
        "👤 Player Deep Dive & Valuation Card",
        "⚔️ Head-to-Head Comparison",
        "📐 System Methodology & Formulas"
    ])

    # ─────────────────────────────────────────────────────────────
    # TAB 1: Player Leaderboard
    # ─────────────────────────────────────────────────────────────
    with tab_board:
        st.subheader("🏆 Global Player Ratings & Transfer Valuation Leaderboard")
        
        display_cols = [
            'rank', 'player', 'role', 'current_rating', 'rating_band',
            'transfer_value_cr', 'supervised_transfer_value_cr', 'form_factor',
            'total_matches', 'career_runs', 'career_batting_sr',
            'career_wickets', 'career_economy', 'arena_v1_overall_rating'
        ]
        valid_cols = [c for c in display_cols if c in filtered_df.columns]
        
        col_rename = {
            'rank': 'Rank',
            'player': 'Player',
            'role': 'Role',
            'current_rating': 'Rating (V2)',
            'rating_band': 'Competitive Band',
            'transfer_value_cr': 'Valuation Index (₹ Cr)',
            'supervised_transfer_value_cr': 'Supervised ML (₹ Cr)',
            'form_factor': 'Form Factor',
            'total_matches': 'Matches',
            'career_runs': 'Career Runs',
            'career_batting_sr': 'Bat SR',
            'career_wickets': 'Career Wkts',
            'career_economy': 'Econ Rate',
            'arena_v1_overall_rating': 'Arena v1'
        }

        view_df = filtered_df[valid_cols].rename(columns=col_rename)

        st.dataframe(
            view_df,
            width="stretch",
            height=500,
            column_config={
                'Rating (V2)': st.column_config.ProgressColumn(
                    'Rating (V2)',
                    help='Context-Adjusted ML Rating (0-100)',
                    format="%.1f",
                    min_value=0,
                    max_value=100
                ),
                'Valuation Index (₹ Cr)': st.column_config.NumberColumn(
                    'Valuation (₹ Cr)',
                    help='Estimated Transfer Value in INR Crore',
                    format="₹%.2f Cr"
                ),
                'Supervised ML (₹ Cr)': st.column_config.NumberColumn(
                    'Supervised ML (₹ Cr)',
                    format="₹%.2f Cr"
                ),
                'Form Factor': st.column_config.NumberColumn(
                    'Form Factor',
                    format="%.2fx"
                )
            }
        )

        if EXCEL_PATH.exists():
            with open(EXCEL_PATH, 'rb') as f:
                st.download_button(
                    label="📥 Download Full Calculated Excel Workbook (.xlsx)",
                    data=f,
                    file_name="arena_player_rating_and_valuation_calculated.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    width="stretch"
                )

    # ─────────────────────────────────────────────────────────────
    # TAB 2: Player Deep Dive & Valuation Card
    # ─────────────────────────────────────────────────────────────
    with tab_profile:
        st.subheader("🔍 Individual Player Analytics & Valuation Breakdown")
        
        player_list = sorted(filtered_df['player'].dropna().unique())
        selected_p = st.selectbox("Select Player for Deep Dive", player_list, index=0 if player_list else None)

        if selected_p:
            p_data = df[df['player'] == selected_p].iloc[0]

            # Header Card
            card_col1, card_col2, card_col3 = st.columns([2, 1, 1])
            with card_col1:
                st.markdown(f"## {p_data['player']}")
                st.markdown(f"**Role:** `{p_data['role']}` | **Status:** `{'Active (4-Yr Window)' if p_data.get('is_active', True) else 'Historic'}` | **Career Span:** `{p_data.get('career_span_years', 0):.1f} Years`")
            with card_col2:
                st.markdown("### Current V2 Rating")
                st.markdown(f"<div class='rating-highlight'>{p_data['current_rating']:.1f} / 100</div>", unsafe_allow_html=True)
                band = p_data.get('rating_band', 'Squad Player')
                st.markdown(f"<span class='badge-elite'>{band}</span>", unsafe_allow_html=True)
            with card_col3:
                st.markdown("### Transfer Value")
                val_display = p_data['transfer_value_cr'] if valuation_mode.startswith('7-Factor') else p_data.get('supervised_transfer_value_cr', p_data['transfer_value_cr'])
                st.markdown(f"<div class='valuation-highlight'>₹{val_display:.2f} Cr</div>", unsafe_allow_html=True)
                st.caption(f"Role Multiplier: {p_data.get('m_role', 1.0)}x")

            st.divider()

            # Stats Matrix
            s1, s2, s3, s4, s5, s6 = st.columns(6)
            s1.metric("Matches", int(p_data.get('total_matches', 0)))
            s2.metric("Career Runs", int(p_data.get('career_runs', 0)))
            s3.metric("Batting Strike Rate", f"{p_data.get('career_batting_sr', 0):.1f}")
            s4.metric("Career Wickets", int(p_data.get('career_wickets', 0)))
            s5.metric("Economy Rate", f"{p_data.get('career_economy', 0):.2f}")
            s6.metric("Recent Form Factor", f"{p_data.get('form_factor', 1.0):.2f}x")

            st.divider()

            # 7-Factor Decomposition
            st.markdown("### 📊 7-Factor Market Valuation Decomposition")
            st.caption("Decomposition of the 7 underlying pillars defining player market value per the technical specification.")

            factor_cols = st.columns(7)
            factors = [
                ("Performance (50%)", p_data.get('s_perf', 0)),
                ("Recent Form (17%)", p_data.get('s_form', 0)),
                ("Context (10%)", p_data.get('s_context', 0)),
                ("Consistency (8%)", p_data.get('s_consistency', 0)),
                ("Experience (5%)", p_data.get('s_experience', 0)),
                ("Scarcity (5%)", p_data.get('s_scarcity', 0)),
                ("Availability (5%)", p_data.get('s_availability', 0)),
            ]
            for col, (fname, fval) in zip(factor_cols, factors):
                with col:
                    st.metric(fname, f"{fval:.1f}")
                    st.progress(float(min(max(fval / 100.0, 0.0), 1.0)))

            st.markdown(f"""
            **Market Conversion Curve:**
            $$\\text{{Index}} = {p_data.get('valuation_index', 0):.1f} / 100 \\implies \\text{{Transfer Value}} = ₹0.20 + (₹30.00 - ₹0.20) \\times \\left({p_data.get('valuation_index', 0) / 100:.3f}\\right)^{{2.15}} = \\mathbf{{₹{p_data['transfer_value_cr']:.2f}\\text{{ Cr}}}}$$
            """)

    # ─────────────────────────────────────────────────────────────
    # TAB 3: Head-to-Head Comparison
    # ─────────────────────────────────────────────────────────────
    with tab_h2h:
        st.subheader("⚔️ Head-to-Head Player Comparison")
        all_players = sorted(df['player'].dropna().unique())
        
        c1, c2 = st.columns(2)
        with c1:
            p1_name = st.selectbox("Player A", all_players, index=0)
        with c2:
            p2_name = st.selectbox("Player B", all_players, index=min(1, len(all_players)-1))

        if p1_name and p2_name:
            p1 = df[df['player'] == p1_name].iloc[0]
            p2 = df[df['player'] == p2_name].iloc[0]

            col_a, col_mid, col_b = st.columns([4, 1, 4])
            with col_a:
                st.markdown(f"### {p1['player']}")
                st.markdown(f"**Role:** `{p1['role']}` | **Band:** `{p1['rating_band']}`")
                st.metric("V2 Rating", f"{p1['current_rating']:.1f} / 100")
                st.metric("Transfer Value", f"₹{p1['transfer_value_cr']:.2f} Cr")
                st.metric("Batting Predicted", f"{p1.get('pred_batting_rating', 0):.1f}")
                st.metric("Bowling Predicted", f"{p1.get('pred_bowling_rating', 0):.1f}")
                st.metric("Career Runs", int(p1.get('career_runs', 0)))
                st.metric("Bat Strike Rate", f"{p1.get('career_batting_sr', 0):.1f}")
                st.metric("Career Wickets", int(p1.get('career_wickets', 0)))
                st.metric("Economy Rate", f"{p1.get('career_economy', 0):.2f}")

            with col_mid:
                st.markdown("<div style='text-align: center; margin-top: 100px; font-weight: bold; font-size: 1.5rem;'>VS</div>", unsafe_allow_html=True)

            with col_b:
                st.markdown(f"### {p2['player']}")
                st.markdown(f"**Role:** `{p2['role']}` | **Band:** `{p2['rating_band']}`")
                st.metric("V2 Rating", f"{p2['current_rating']:.1f} / 100", delta=round(p2['current_rating'] - p1['current_rating'], 1))
                st.metric("Transfer Value", f"₹{p2['transfer_value_cr']:.2f} Cr", delta=round(p2['transfer_value_cr'] - p1['transfer_value_cr'], 2))
                st.metric("Batting Predicted", f"{p2.get('pred_batting_rating', 0):.1f}")
                st.metric("Bowling Predicted", f"{p2.get('pred_bowling_rating', 0):.1f}")
                st.metric("Career Runs", int(p2.get('career_runs', 0)))
                st.metric("Bat Strike Rate", f"{p2.get('career_batting_sr', 0):.1f}")
                st.metric("Career Wickets", int(p2.get('career_wickets', 0)))
                st.metric("Economy Rate", f"{p2.get('career_economy', 0):.2f}")

    # ─────────────────────────────────────────────────────────────
    # TAB 4: Methodology & Formulas
    # ─────────────────────────────────────────────────────────────
    with tab_method:
        st.subheader("📐 Architecture & Mathematical Specification")
        st.markdown("""
        ### 1. Leak-Free Walk-Forward Feature Extraction
        - For match $N$, features are computed strictly using historical matches $1$ to $N-1$.
        - Eliminates data leakage and temporal bias.

        ### 2. Machine Learning Rating Models
        - **Batting Regressor:** CatBoost on decision trees evaluating venue-adjusted runs, strike rate impact, opponent bowler difficulty, and boundary frequency.
        - **Bowling Regressor:** CatBoost regressor evaluating wickets, venue-adjusted economy, dot ball pressure, and opponent batter difficulty.

        ### 3. Role-Aware Rating Aggregation
        | Player Role | Batting Weight | Bowling Weight | Rationale |
        |---|---|---|---|
        | **Batsman** | 90% | 10% | Primary batting focus; minor credit for secondary bowling |
        | **Bowler** | 10% | 90% | Primary bowling focus; rewards lower-order runs |
        | **All-Rounder** | 55% | 45% | Balanced dual-skill contribution |
        | **Wicketkeeper** | 85% | 15% | Primary batting focus |
        | **Unknown** | 50% | 50% | Equal distribution |

        ### 4. Recent Form Factor Multiplier
        $$\\text{Form Factor} = \\text{clip}\\left(\\frac{\\text{Recent 5 Matches Output} + 1}{\\text{Career Avg per 5 Matches} + 1}, 0.75, 1.25\\right)$$
        $$\\text{Current Rating} = \\text{clip}\\left(\\text{Predicted Rating} \\times 0.90 + 0.10 \\times (\\text{Form Factor} \\times \\text{Predicted Rating}), 0, 100\\right)$$

        ### 5. 7-Factor Market Transfer Value Engine
        $$\\text{Index} = \\left(0.50 S_{\\text{perf}} + 0.17 S_{\\text{form}} + 0.10 S_{\\text{context}} + 0.08 S_{\\text{consistency}} + 0.05 S_{\\text{experience}} + 0.05 S_{\\text{scarcity}} + 0.05 S_{\\text{availability}}\\right) \\times M_{\\text{role}}$$
        $$\\text{Transfer Value (₹ Cr)} = ₹0.20 + (₹30.00 - ₹0.20) \\times \\left(\\frac{\\text{Index}}{100}\\right)^{2.15}$$
        """)


if __name__ == "__main__":
    main()
