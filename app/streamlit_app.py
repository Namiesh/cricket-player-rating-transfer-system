"""
Cricket Player Rating & Transfer Value System - Interactive Analytics Dashboard.
Provides a modern visual interface for player performance ratings, competitive bands,
dual-mode market valuation with live multi-currency conversion, 7-factor index
decomposition, and head-to-head comparisons.
"""

import os
import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np

# ── Currency module (local) ────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from currency_converter import (
    SUPPORTED_CURRENCIES,
    fetch_rates_from_api,
    convert_inr_crore,
    format_currency_label,
)

# ── Page configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cricket Player Rating & Transfer Value System",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Base dark card */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.2);
    }
    /* Rating band badges */
    .badge-elite      { background:#f59e0b; color:#000; padding:4px 10px; border-radius:9999px; font-weight:700; font-size:.82rem; }
    .badge-world-class{ background:#8b5cf6; color:#fff; padding:4px 10px; border-radius:9999px; font-weight:700; font-size:.82rem; }
    .badge-intl       { background:#0284c7; color:#fff; padding:4px 10px; border-radius:9999px; font-weight:700; font-size:.82rem; }
    .badge-domestic   { background:#0d9488; color:#fff; padding:4px 10px; border-radius:9999px; font-weight:700; font-size:.82rem; }
    .badge-squad      { background:#64748b; color:#fff; padding:4px 10px; border-radius:9999px; font-weight:700; font-size:.82rem; }
    /* Value highlights */
    .valuation-highlight { font-size:2.2rem; font-weight:800; color:#10b981; }
    .rating-highlight    { font-size:2.2rem; font-weight:800; color:#38bdf8; }
    /* Currency tag */
    .currency-tag { font-size:.72rem; color:#94a3b8; margin-left:6px; }
    /* Rate bar */
    .rate-info { font-size:.75rem; color:#64748b; margin-top:4px; }
</style>
""", unsafe_allow_html=True)

# ── File paths ─────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_PATH  = BASE_DIR / "data" / "processed" / "active_players_ratings_valuations.csv"
EXCEL_PATH = BASE_DIR / "exported_workbooks" / "arena_player_rating_and_valuation_calculated.xlsx"


# ── Cached loaders ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        return None
    df = pd.read_csv(DATA_PATH)
    df["current_rating"]  = df["current_rating"].astype(float)
    df["transfer_value_cr"] = df["transfer_value_cr"].astype(float)
    if "supervised_transfer_value_cr" in df.columns:
        df["supervised_transfer_value_cr"] = df["supervised_transfer_value_cr"].astype(float)
    return df


@st.cache_data(ttl=3600)          # refresh exchange rates at most once per hour
def get_live_rates() -> dict:
    return fetch_rates_from_api()


# ── Helpers ────────────────────────────────────────────────────────────────────
def fmt(val_cr: float, currency_label: str, rates: dict) -> str:
    """Short alias: convert INR Crore → display string."""
    return format_currency_label(val_cr, currency_label, rates)


def badge_class(band: str) -> str:
    mapping = {"Elite": "elite", "World Class": "world-class",
               "International": "intl", "Domestic Star": "domestic"}
    return mapping.get(band, "squad")


# ══════════════════════════════════════════════════════════════════════════════
def main():
    st.title("🏏 Cricket Player Rating & Transfer Value System")
    st.caption(
        "Production-grade Context-Adjusted ML Ratings · Dynamic Role Weighting · "
        "Dual-Mode Market Valuation · Live Multi-Currency Conversion"
    )

    df    = load_data()
    rates = get_live_rates()

    if df is None or df.empty:
        st.warning("⚠️ Processed dataset not found. Run the pipeline first.")
        if st.button("🚀 Run Pipeline Now"):
            with st.spinner("Executing pipeline…"):
                os.system(f"python {BASE_DIR}/run_pipeline.py --matches 800")
            st.success("Pipeline finished. Please refresh the page.")
            st.rerun()
        st.stop()

    # ── Sidebar ────────────────────────────────────────────────────────────────
    st.sidebar.header("🎯 Filters & Search")

    search_player = st.sidebar.text_input(
        "🔍 Search Player", "", placeholder="e.g. Babar Azam, Rashid Khan…"
    )
    active_only = st.sidebar.checkbox("Active Players Only (Rolling 4-Yr Window)", value=True)

    roles = ["All"] + sorted(df["role"].dropna().unique().tolist())
    selected_role = st.sidebar.selectbox("Filter by Role", roles)

    bands = [
        "All", "Elite (90+)", "World Class (80-89.9)",
        "International (70-79.9)", "Domestic Star (60-69.9)", "Squad Player (<60)"
    ]
    selected_band = st.sidebar.selectbox("Filter by Competitive Band", bands)

    valuation_mode = st.sidebar.radio(
        "Valuation Mode",
        ["7-Factor Valuation Index Mode", "Supervised ML Market Mode"]
    )

    st.sidebar.divider()

    # ── Currency selector ──────────────────────────────────────────────────────
    st.sidebar.markdown("### 💱 Display Currency")
    currency_options = list(SUPPORTED_CURRENCIES.keys())
    selected_currency = st.sidebar.selectbox(
        "Select Currency",
        currency_options,
        index=0,                      # default: INR Crore
        help="All transfer values are computed in ₹ INR Crore internally and "
             "converted accurately using live exchange rates."
    )

    cur_info   = SUPPORTED_CURRENCIES[selected_currency]
    cur_code   = cur_info["code"]
    cur_symbol = cur_info["symbol"]
    cur_unit   = cur_info["unit"]

    # Display live rate info in sidebar
    rate_inr_to_cur = rates.get(cur_code, 1.0)
    if cur_code != "INR":
        rate_display = f"1 INR = {rate_inr_to_cur:.5f} {cur_code}"
        st.sidebar.caption(f"🔄 Live rate: **{rate_display}**")
    else:
        st.sidebar.caption("🔄 Base currency: INR Crore (no conversion)")

    # ── Filter DataFrame ───────────────────────────────────────────────────────
    filtered_df = df.copy()
    if active_only and "is_active" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["is_active"]]
    if selected_role != "All":
        filtered_df = filtered_df[filtered_df["role"] == selected_role]
    if selected_band != "All":
        band_name = selected_band.split(" ")[0]
        filtered_df = filtered_df[filtered_df["rating_band"].str.startswith(band_name)]
    if search_player:
        filtered_df = filtered_df[filtered_df["player"].str.contains(search_player, case=False, na=False)]

    # Helper: get the right INR Crore column per valuation mode
    def val_col(row_or_series=None):
        col = "transfer_value_cr" if valuation_mode.startswith("7-Factor") else "supervised_transfer_value_cr"
        if col not in filtered_df.columns:
            col = "transfer_value_cr"
        if row_or_series is not None:
            return row_or_series.get(col, row_or_series["transfer_value_cr"])
        return col

    # ── Top KPI Bar ────────────────────────────────────────────────────────────
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        st.metric("Total Players", len(filtered_df))
    with kpi2:
        st.metric("Elite (90+)", int((filtered_df["rating_band"] == "Elite").sum()))
    with kpi3:
        st.metric("World Class (80-89.9)", int((filtered_df["rating_band"] == "World Class").sum()))
    with kpi4:
        avg_cr = filtered_df[val_col()].mean() if not filtered_df.empty else 0.0
        st.metric("Avg Valuation", fmt(avg_cr, selected_currency, rates))
    with kpi5:
        max_cr = filtered_df[val_col()].max() if not filtered_df.empty else 0.0
        st.metric("Top Valuation", fmt(max_cr, selected_currency, rates))

    st.divider()

    # ── Tabs ───────────────────────────────────────────────────────────────────
    tab_board, tab_profile, tab_h2h, tab_method = st.tabs([
        "📊 Player Leaderboard",
        "👤 Player Deep Dive & Valuation Card",
        "⚔️ Head-to-Head Comparison",
        "📐 System Methodology & Formulas",
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — Leaderboard
    # ══════════════════════════════════════════════════════════════════════════
    with tab_board:
        val_mode_col = val_col()
        st.subheader(f"🏆 Global Player Ratings & Transfer Valuation Leaderboard  •  {cur_symbol} {cur_unit}")

        display_df = filtered_df.copy()

        # Build converted valuation column
        display_df["_val_display"] = display_df[val_mode_col].apply(
            lambda cr: convert_inr_crore(cr, selected_currency, rates)[0]
        )
        display_df["_val_inr_cr"] = display_df[val_mode_col]

        show_cols = [
            "rank", "player", "role", "current_rating", "rating_band",
            "_val_display", "form_factor",
            "total_matches", "career_runs", "career_batting_sr",
            "career_wickets", "career_economy", "arena_v1_overall_rating"
        ]
        valid_cols = [c for c in show_cols if c in display_df.columns]

        val_col_label = f"Valuation ({cur_symbol} {cur_unit})"
        col_rename = {
            "rank":                 "Rank",
            "player":               "Player",
            "role":                 "Role",
            "current_rating":       "Rating (V2)",
            "rating_band":          "Band",
            "_val_display":         val_col_label,
            "form_factor":          "Form",
            "total_matches":        "Matches",
            "career_runs":          "Career Runs",
            "career_batting_sr":    "Bat SR",
            "career_wickets":       "Career Wkts",
            "career_economy":       "Econ",
            "arena_v1_overall_rating": "Arena v1",
        }

        view_df = display_df[valid_cols].rename(columns=col_rename)

        st.dataframe(
            view_df,
            width="stretch",
            height=500,
            column_config={
                "Rating (V2)": st.column_config.ProgressColumn(
                    "Rating (V2)", help="Context-Adjusted ML Rating (0–100)",
                    format="%.1f", min_value=0, max_value=100
                ),
                val_col_label: st.column_config.NumberColumn(
                    val_col_label,
                    help=f"Estimated Transfer Value in {cur_code} (converted from ₹ INR Crore using live rates)",
                    format=f"%.2f"
                ),
                "Form": st.column_config.NumberColumn("Form Factor", format="%.2fx"),
            }
        )

        # Note that INR Cr is the base and conversion is live
        if cur_code != "INR":
            st.caption(
                f"💡 All values converted from ₹ INR Crore using live exchange rate  "
                f"**1 INR = {rate_inr_to_cur:.6f} {cur_code}**.  "
                f"Source: open.er-api.com  •  Rates refresh every 60 minutes."
            )

        if EXCEL_PATH.exists():
            with open(EXCEL_PATH, "rb") as f:
                st.download_button(
                    label="📥 Download Full Excel Workbook (.xlsx)",
                    data=f,
                    file_name="arena_player_rating_and_valuation_calculated.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    width="stretch"
                )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — Player Deep Dive
    # ══════════════════════════════════════════════════════════════════════════
    with tab_profile:
        st.subheader("🔍 Individual Player Analytics & Valuation Breakdown")

        player_list = sorted(filtered_df["player"].dropna().unique())
        if not player_list:
            st.info("No players match the current filters.")
        else:
            selected_p = st.selectbox("Select Player for Deep Dive", player_list, index=0)
            p_data = df[df["player"] == selected_p].iloc[0]

            val_cr = val_col(p_data)
            val_fmt = fmt(val_cr, selected_currency, rates)

            # ── Header row ──────────────────────────────────────────────────
            card_col1, card_col2, card_col3 = st.columns([2, 1, 1])
            with card_col1:
                st.markdown(f"## {p_data['player']}")
                active_str = "Active (4-Yr Window)" if p_data.get("is_active", True) else "Historic"
                st.markdown(
                    f"**Role:** `{p_data['role']}` | **Status:** `{active_str}` | "
                    f"**Career Span:** `{p_data.get('career_span_years', 0):.1f} Yrs`"
                )
            with card_col2:
                st.markdown("### Current V2 Rating")
                st.markdown(f"<div class='rating-highlight'>{p_data['current_rating']:.1f} / 100</div>", unsafe_allow_html=True)
                band = p_data.get("rating_band", "Squad Player")
                st.markdown(f"<span class='badge-{badge_class(band)}'>{band}</span>", unsafe_allow_html=True)
            with card_col3:
                st.markdown("### Transfer Value")
                st.markdown(f"<div class='valuation-highlight'>{val_fmt}</div>", unsafe_allow_html=True)
                # Also show INR Crore as canonical reference if non-INR selected
                if cur_code != "INR":
                    st.caption(f"≈ ₹{val_cr:.2f} Cr (INR)")
                st.caption(f"Role Multiplier: {p_data.get('m_role', 1.0):.2f}x")

            st.divider()

            # ── Currency switcher inside profile ─────────────────────────────
            st.markdown("#### 💱 View Valuation in Another Currency")
            mini_currency_opts = list(SUPPORTED_CURRENCIES.keys())
            mini_sel = st.selectbox(
                "Quick Currency Switch (Profile Only)",
                mini_currency_opts,
                index=mini_currency_opts.index(selected_currency),
                key="profile_currency"
            )
            if mini_sel != selected_currency:
                alt_fmt = fmt(val_cr, mini_sel, rates)
                st.info(f"**{p_data['player']}** transfer value in **{mini_sel.split('–')[0].strip()}** → **{alt_fmt}**")

            st.divider()

            # ── Stats grid ───────────────────────────────────────────────────
            s1, s2, s3, s4, s5, s6 = st.columns(6)
            s1.metric("Matches",           int(p_data.get("total_matches", 0)))
            s2.metric("Career Runs",       int(p_data.get("career_runs", 0)))
            s3.metric("Bat Strike Rate",   f"{p_data.get('career_batting_sr', 0):.1f}")
            s4.metric("Career Wickets",    int(p_data.get("career_wickets", 0)))
            s5.metric("Economy Rate",      f"{p_data.get('career_economy', 0):.2f}")
            s6.metric("Form Factor",       f"{p_data.get('form_factor', 1.0):.2f}x")

            st.divider()

            # ── 7-Factor Decomposition ────────────────────────────────────────
            st.markdown("### 📊 7-Factor Market Valuation Decomposition")
            st.caption("Decomposition of the 7 pillars defining player market value.")

            factor_cols = st.columns(7)
            factors = [
                ("Performance (50%)", p_data.get("s_perf", 0)),
                ("Recent Form (17%)", p_data.get("s_form", 0)),
                ("Context (10%)",     p_data.get("s_context", 0)),
                ("Consistency (8%)",  p_data.get("s_consistency", 0)),
                ("Experience (5%)",   p_data.get("s_experience", 0)),
                ("Scarcity (5%)",     p_data.get("s_scarcity", 0)),
                ("Availability (5%)", p_data.get("s_availability", 0)),
            ]
            for col, (fname, fval) in zip(factor_cols, factors):
                with col:
                    st.metric(fname, f"{fval:.1f}")
                    st.progress(float(min(max(fval / 100.0, 0.0), 1.0)))

            idx = p_data.get("valuation_index", 0)
            mini_code = SUPPORTED_CURRENCIES[mini_sel]["code"]
            mini_rate = rates.get(mini_code, 1.0)
            st.markdown("**Market Conversion Curve (base INR Crore):**")
            st.latex(
                rf"\text{{Index}} = {idx:.1f}/100 \implies"
                rf"\text{{Transfer Value}} = \text{{Rs}}0.20 + (\text{{Rs}}30.00 - \text{{Rs}}0.20)"
                rf"\times \left({idx / 100:.3f}\right)^{{2.15}}"
                rf"= \mathbf{{\text{{Rs}}{val_cr:.2f}\text{{ Cr}}}}"
            )
            st.caption(
                f"Converted to {mini_sel.split('–')[0].strip()} → "
                f"**{fmt(val_cr, mini_sel, rates)}**  "
                f"(1 INR = {mini_rate:.6f} {mini_code})"
            )

            # ── Full currency table for this player ──────────────────────────
            with st.expander("🌍 See valuation in all 16 supported currencies"):
                rows = []
                for lbl, info in SUPPORTED_CURRENCIES.items():
                    _, formatted = convert_inr_crore(val_cr, lbl, rates)
                    r = rates.get(info["code"], 1.0)
                    rows.append({
                        "Currency":     lbl,
                        "Transfer Value": formatted,
                        "Rate (1 INR)": f"{r:.6f}",
                    })
                st.dataframe(
                    pd.DataFrame(rows),
                    width="stretch",
                    hide_index=True,
                )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 — Head-to-Head
    # ══════════════════════════════════════════════════════════════════════════
    with tab_h2h:
        st.subheader(f"⚔️ Head-to-Head Player Comparison  •  {cur_symbol} {cur_unit}")
        all_players = sorted(df["player"].dropna().unique().tolist())

        c1, c2 = st.columns(2)
        with c1:
            p1_name = st.selectbox("Player A", all_players, index=0, key="h2h_p1")
        with c2:
            p2_name = st.selectbox("Player B", all_players, index=min(1, len(all_players) - 1), key="h2h_p2")

        if p1_name and p2_name:
            p1 = df[df["player"] == p1_name].iloc[0]
            p2 = df[df["player"] == p2_name].iloc[0]
            p1_val_cr = val_col(p1)
            p2_val_cr = val_col(p2)

            col_a, col_mid, col_b = st.columns([4, 1, 4])

            with col_a:
                st.markdown(f"### {p1['player']}")
                st.markdown(f"**Role:** `{p1['role']}` | **Band:** `{p1['rating_band']}`")
                st.metric("V2 Rating",         f"{p1['current_rating']:.1f} / 100")
                st.metric("Transfer Value",     fmt(p1_val_cr, selected_currency, rates))
                st.metric("Batting Predicted",  f"{p1.get('pred_batting_rating', 0):.1f}")
                st.metric("Bowling Predicted",  f"{p1.get('pred_bowling_rating', 0):.1f}")
                st.metric("Career Runs",        int(p1.get("career_runs", 0)))
                st.metric("Bat Strike Rate",    f"{p1.get('career_batting_sr', 0):.1f}")
                st.metric("Career Wickets",     int(p1.get("career_wickets", 0)))
                st.metric("Economy Rate",       f"{p1.get('career_economy', 0):.2f}")

            with col_mid:
                st.markdown(
                    "<div style='text-align:center;margin-top:110px;"
                    "font-weight:bold;font-size:1.5rem;'>VS</div>",
                    unsafe_allow_html=True
                )

            with col_b:
                p2_val_disp, _ = convert_inr_crore(p2_val_cr, selected_currency, rates)
                p1_val_disp, _ = convert_inr_crore(p1_val_cr, selected_currency, rates)
                val_delta       = round(p2_val_disp - p1_val_disp, 3)

                st.markdown(f"### {p2['player']}")
                st.markdown(f"**Role:** `{p2['role']}` | **Band:** `{p2['rating_band']}`")
                st.metric("V2 Rating",         f"{p2['current_rating']:.1f} / 100",
                          delta=round(p2["current_rating"] - p1["current_rating"], 1))
                st.metric("Transfer Value",     fmt(p2_val_cr, selected_currency, rates),
                          delta=f"{val_delta:+.3f} {cur_unit}")
                st.metric("Batting Predicted",  f"{p2.get('pred_batting_rating', 0):.1f}")
                st.metric("Bowling Predicted",  f"{p2.get('pred_bowling_rating', 0):.1f}")
                st.metric("Career Runs",        int(p2.get("career_runs", 0)))
                st.metric("Bat Strike Rate",    f"{p2.get('career_batting_sr', 0):.1f}")
                st.metric("Career Wickets",     int(p2.get("career_wickets", 0)))
                st.metric("Economy Rate",       f"{p2.get('career_economy', 0):.2f}")

            if cur_code != "INR":
                st.caption(
                    f"💡 Values shown in **{selected_currency.split('–')[0].strip()}**.  "
                    f"Live rate: 1 INR = {rate_inr_to_cur:.6f} {cur_code}"
                )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4 — Methodology
    # ══════════════════════════════════════════════════════════════════════════
    with tab_method:
        st.subheader("📐 Architecture & Mathematical Specification")
        st.markdown("""
        ### 1. Leak-Free Walk-Forward Feature Extraction
        - For match $N$, features are computed strictly using historical matches $1$ to $N-1$.
        - Eliminates data leakage and temporal bias.

        ### 2. Machine Learning Rating Models
        - **Batting Regressor:** CatBoost evaluating venue-adjusted runs, strike rate impact,
          opponent bowler difficulty, and boundary frequency.
        - **Bowling Regressor:** CatBoost evaluating wickets, venue-adjusted economy,
          dot-ball pressure, and opponent batter difficulty.

        ### 3. Role-Aware Rating Aggregation
        | Player Role | Batting Weight | Bowling Weight |
        |---|---|---|
        | **Batsman**     | 90% | 10% |
        | **Bowler**      | 10% | 90% |
        | **All-Rounder** | 55% | 45% |
        | **Wicketkeeper**| 85% | 15% |
        | **Unknown**     | 50% | 50% |

        ### 4. Recent Form Factor Multiplier
        $$\\text{Form Factor} = \\text{clip}\\left(\\frac{\\text{Recent 5 Matches Output}+1}{\\text{Career Avg per 5 Matches}+1},\\ 0.75,\\ 1.25\\right)$$

        ### 5. 7-Factor Market Transfer Value Engine (base: INR Crore)
        $$\\text{Index} = \\left(0.50 S_{\\text{perf}} + 0.17 S_{\\text{form}} + 0.10 S_{\\text{context}}
          + 0.08 S_{\\text{consistency}} + 0.05 S_{\\text{experience}}
          + 0.05 S_{\\text{scarcity}} + 0.05 S_{\\text{availability}}\\right) \\times M_{\\text{role}}$$
        $$\\text{Transfer Value} = 0.20 + (30.00 - 0.20)
          \\times \\left(\\frac{\\text{Index}}{100}\\right)^{2.15} \\quad (\\text{INR Crore})$$

        ### 6. Currency Conversion
        - The **canonical valuation** is always stored and computed in **INR Crore (Rs Cr)**.
        - Live exchange rates are fetched from [open.er-api.com](https://open.er-api.com)
          (free, no API key required) and **cached for 60 minutes** per session.
        - Conversion: Raw INR = Value x 10,000,000 -> multiply by live rate -> divide by display unit (Crore or Million).
        - If the network is unavailable, conservative hardcoded fallback rates are used.
        """)

        # Show current rate table
        st.markdown("#### Current Exchange Rates (1 INR =)")
        rate_rows = [
            {
                "Currency": lbl.split("–")[0].strip(),
                "Code": info["code"],
                "Rate (1 INR)": f"{rates.get(info['code'], 1.0):.6f}",
            }
            for lbl, info in SUPPORTED_CURRENCIES.items()
        ]
        st.dataframe(pd.DataFrame(rate_rows), hide_index=True, width="stretch")


if __name__ == "__main__":
    main()
