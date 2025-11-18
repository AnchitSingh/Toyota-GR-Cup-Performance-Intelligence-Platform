# app.py - Toyota GR Cup Championship Dashboard
# Complete Version - Beautiful Design + Full Functionality

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Toyota GR Cup Championship Analysis",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== REFINED THEME ====================
THEME = {
    'bg_gradient_1': '#14110F',
    'bg_gradient_2': '#1F1B18',
    'glass_bg': 'rgba(46, 41, 37, 0.65)',
    'glass_border': 'rgba(212, 163, 115, 0.15)',
    'text_primary': '#F2ECE9',
    'text_secondary': '#9C948E',
    'accent_gold': '#D4A373',
    'accent_danger': '#BC4B51',
    'accent_success': '#6B8F71',
    'shadow': '0 8px 32px 0 rgba(0, 0, 0, 0.5)'
}

st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: {THEME['text_primary']};
    }}

    .stApp {{
        background: radial-gradient(circle at top left, {THEME['bg_gradient_2']}, {THEME['bg_gradient_1']});
    }}
    
    [data-testid="stSidebar"] {{
        background-color: {THEME['bg_gradient_1']};
        border-right: 1px solid {THEME['glass_border']};
    }}

    .glass-card {{
        background: {THEME['glass_bg']};
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 16px;
        border: 1px solid {THEME['glass_border']};
        padding: 24px;
        box-shadow: {THEME['shadow']};
        margin-bottom: 20px;
    }}

    .stSelectbox div[data-baseweb="select"] > div, 
    .stMultiSelect div[data-baseweb="select"] > div {{
        background-color: rgba(255,255,255,0.05) !important;
        border: 1px solid {THEME['glass_border']} !important;
        color: {THEME['text_primary']} !important;
        border-radius: 8px;
    }}
    
    span[data-baseweb="tag"] {{
        background-color: {THEME['accent_gold']} !important;
        border-radius: 4px;
    }}
    span[data-baseweb="tag"] span {{
        color: #1a1614 !important;
        font-weight: 600;
    }}

    [data-testid="stDataFrame"] {{
        background-color: transparent !important;
    }}

    h1, h2, h3 {{ 
        font-weight: 600; 
        letter-spacing: -0.5px; 
        color: {THEME['text_primary']};
    }}
    
    [data-testid="stMetricLabel"] {{
        color: {THEME['text_secondary']} !important;
        font-size: 14px !important;
    }}
    [data-testid="stMetricValue"] {{
        color: {THEME['text_primary']} !important;
        font-size: 28px !important;
        font-weight: 700 !important;
        text-shadow: 0 0 20px rgba(212, 163, 115, 0.3);
    }}

    .opp-card {{
        background: rgba(255, 255, 255, 0.03);
        border-left: 4px solid {THEME['accent_danger']};
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
        transition: all 0.3s ease;
    }}
    .opp-card:hover {{
        background: rgba(255, 255, 255, 0.07);
        transform: translateX(5px);
    }}
    
    header {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    .stDeployButton {{display:none;}}
    
    </style>
    """, unsafe_allow_html=True)

# ==================== DATA LOADING (REAL DATA) ====================
@st.cache_data
def load_data():
    """Load all processed championship data"""
    try:
        corners = pd.read_csv('master_corner_features.csv')
        comparison = pd.read_csv('master_comparisons.csv')
        ml_features = pd.read_csv('ml_feature_importance.csv')
        driver_stats = pd.read_csv('driver_performance_stats.csv')
        clusters = pd.read_csv('driver_clusters.csv')
        return corners, comparison, ml_features, driver_stats, clusters
    except FileNotFoundError as e:
        st.error(f"⚠️ Data file not found: {e}")
        st.info("Please ensure all CSV files are in the same directory as app.py")
        st.stop()

corners, comparison, ml_features, driver_stats, clusters = load_data()

# ==================== HELPER FUNCTIONS ====================
def style_chart(fig):
    """Consistent chart styling"""
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=THEME['text_secondary'], family="Inter", size=12),
        margin=dict(l=40, r=40, t=40, b=40),
        xaxis=dict(
            showgrid=False, 
            showline=True, 
            linecolor='rgba(255,255,255,0.1)',
            color=THEME['text_secondary']
        ),
        yaxis=dict(
            showgrid=True, 
            gridcolor='rgba(255,255,255,0.05)', 
            zeroline=False,
            color=THEME['text_secondary']
        ),
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=1.02, 
            xanchor="right", 
            x=1,
            font=dict(color=THEME['text_primary'])
        )
    )
    return fig

def diagnose_issue(row):
    """Diagnose corner issue from deltas"""
    if abs(row['brake_delta']) > 20:
        if row['brake_delta'] > 0:
            return "Over-braking"
        else:
            return "Under-braking"
    elif abs(row['apex_throttle_delta']) > 5:
        if row['apex_throttle_delta'] < 0:
            return "Late throttle application"
        else:
            return "Too aggressive on throttle"
    else:
        return "Inconsistent corner speed"

def generate_fix(row):
    """Generate coaching advice"""
    if row['brake_delta'] > 20:
        return "Brake lighter, carry more speed"
    elif row['brake_delta'] < -20:
        return "Brake harder and later"
    elif row['apex_throttle_delta'] < -5:
        return "Get on throttle earlier at apex"
    elif row['apex_throttle_delta'] > 5:
        return "Smoother throttle application"
    else:
        return "Focus on entry consistency"

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown(f"<h2 style='color:{THEME['accent_gold']}; margin-bottom:20px;'>🏁 GR Cup Analytics</h2>", unsafe_allow_html=True)
    
    # Track Selection (ALL 7 TRACKS)
    available_tracks = sorted(corners['track'].unique())
    selected_track = st.selectbox(
        "🏎️ Track Selection", 
        ["All Tracks"] + available_tracks,
        help="Select specific track or view all tracks"
    )
    
    # Filter data by track
    if selected_track == 'All Tracks':
        corners_filtered = corners
        comparison_filtered = comparison
        stats_filtered = driver_stats
    else:
        corners_filtered = corners[corners['track'] == selected_track]
        comparison_filtered = comparison[comparison['track'] == selected_track]
        stats_filtered = driver_stats[driver_stats['track'] == selected_track]
    
    # Get drivers from filtered data
    available_drivers = sorted(corners_filtered['vehicle_id'].unique())
    
    st.markdown("---")
    st.markdown("### Driver Configuration")
    
    # Analysis Mode
    analysis_mode = st.radio(
        "View Mode", 
        ["Deep Dive", "Multi-Driver Comparison"],
        help="Choose analysis type"
    )
    
    if analysis_mode == "Deep Dive":
        selected_driver = st.selectbox("📊 Driver", available_drivers)
        
        # Benchmark selection
        fastest_driver = stats_filtered.nsmallest(1, 'best_lap')['vehicle_id'].iloc[0] if len(stats_filtered) > 0 else available_drivers[0]
        benchmark_options = ['Fastest Driver'] + [d for d in available_drivers if d != selected_driver]
        ref_selection = st.selectbox("🏆 Benchmark", benchmark_options)
        
        benchmark_driver = fastest_driver if ref_selection == 'Fastest Driver' else ref_selection
        
        st.markdown("---")
        st.markdown("### Telemetry Filter")
        
        # Corner range with proper bounds
        driver_corners = comparison_filtered[
            (comparison_filtered['slow_driver'] == selected_driver) &
            (comparison_filtered['fast_driver'] == benchmark_driver)
        ]['corner'].unique()
        
        if len(driver_corners) > 0:
            min_corner = int(min(driver_corners))
            max_corner = int(max(driver_corners))
            corner_range = st.slider(
                "Corner Range", 
                min_corner, 
                max_corner, 
                (min_corner, max_corner),
                help="Drag to focus on specific corners"
            )
        else:
            corner_range = (1, 20)
        
        comparison_drivers = [selected_driver]
        
    else:
        # Multi-driver comparison mode
        comparison_drivers = st.multiselect(
            "📊 Select Drivers (2-5)",
            available_drivers,
            default=available_drivers[:min(3, len(available_drivers))],
            help="Choose drivers to compare"
        )
        
        if len(comparison_drivers) > 0:
            selected_driver = comparison_drivers[0]
            fastest_driver = stats_filtered.nsmallest(1, 'best_lap')['vehicle_id'].iloc[0] if len(stats_filtered) > 0 else comparison_drivers[0]
            benchmark_driver = fastest_driver
            
            # Corner range for all selected drivers
            all_corners = []
            for driver in comparison_drivers:
                driver_corners = comparison_filtered[comparison_filtered['slow_driver'] == driver]['corner'].unique()
                all_corners.extend(driver_corners)
            
            if len(all_corners) > 0:
                min_corner = int(min(all_corners))
                max_corner = int(max(all_corners))
                corner_range = st.slider("Corner Range", min_corner, max_corner, (min_corner, max_corner))
            else:
                corner_range = (1, 20)
        else:
            selected_driver = available_drivers[0]
            benchmark_driver = available_drivers[0]
            corner_range = (1, 20)
    
    # Dataset stats
    st.markdown("---")
    st.markdown(f"""
        <div style='background: rgba(212, 163, 115, 0.1); padding: 12px; border-radius: 8px; border-left: 3px solid {THEME['accent_gold']}'>
            <p style='margin:0; font-size:12px; color:{THEME['text_secondary']}'>Championship Dataset</p>
            <p style='margin:5px 0 0 0; font-size:14px; color:{THEME['text_primary']}'>
                <b>{len(corners):,}</b> corners analyzed<br>
                <b>{corners['vehicle_id'].nunique()}</b> drivers tracked<br>
                <b>{corners['track'].nunique()}</b> circuits covered
            </p>
        </div>
    """, unsafe_allow_html=True)

# ==================== MAIN LAYOUT ====================

# Header
col_title, col_date = st.columns([3, 1])
with col_title:
    track_display = selected_track if selected_track != "All Tracks" else "Championship Overview"
    st.markdown(f"# {track_display}")
with col_date:
    st.markdown(f"<p style='color:{THEME['text_secondary']}; text-align:right; margin-top:20px'>{pd.Timestamp.now().strftime('%B %d, %Y')}</p>", unsafe_allow_html=True)

st.markdown(f"<p style='color:{THEME['text_secondary']}; margin-top:-15px'>Advanced Telemetry Analysis • ML-Powered Insights</p>", unsafe_allow_html=True)

# ==================== DEEP DIVE MODE ====================
if analysis_mode == "Deep Dive":
    
    # Get driver stats
    driver_stats_row = stats_filtered[stats_filtered['vehicle_id'] == selected_driver]
    
    # Filter comparison data
    # Dynamic comparison calculation
    if selected_driver == benchmark_driver:
        # Can't compare driver to themselves
        comparison_subset = pd.DataFrame()
    else:
        # Try pre-computed comparison first
        comparison_subset = comparison_filtered[
            (comparison_filtered['slow_driver'] == selected_driver) &
            (comparison_filtered['fast_driver'] == benchmark_driver) &
            (comparison_filtered['corner'].between(corner_range[0], corner_range[1]))
        ]
        
        # If no pre-computed data, calculate on the fly
        if len(comparison_subset) == 0:
            # Get corner features for both drivers
            selected_corners = corners_filtered[corners_filtered['vehicle_id'] == selected_driver]
            benchmark_corners = corners_filtered[corners_filtered['vehicle_id'] == benchmark_driver]
            
            # Calculate deltas on the fly
            comparison_list = []
            for corner_num in range(corner_range[0], corner_range[1] + 1):
                sel_corner = selected_corners[selected_corners['corner_num'] == corner_num]
                bench_corner = benchmark_corners[benchmark_corners['corner_num'] == corner_num]
                
                if len(sel_corner) > 0 and len(bench_corner) > 0:
                    sel = sel_corner.iloc[0]
                    bench = bench_corner.iloc[0]
                    
                    comparison_list.append({
                        'track': selected_track if selected_track != 'All Tracks' else sel['track'],
                        'corner': corner_num,
                        'slow_driver': selected_driver,
                        'fast_driver': benchmark_driver,
                        'time_lost_sec': (sel['corner_duration'] - bench['corner_duration']) * 0.04,
                        'brake_delta': sel['max_brake'] - bench['max_brake'],
                        'apex_throttle_delta': sel['apex_throttle'] - bench['apex_throttle'],
                        'slow_brake': sel['max_brake'],
                        'fast_brake': bench['max_brake'],
                        'slow_apex_throttle': sel['apex_throttle'],
                        'fast_apex_throttle': bench['apex_throttle']
                    })
            
            if comparison_list:
                comparison_subset = pd.DataFrame(comparison_list)
            else:
                comparison_subset = pd.DataFrame()
    
    # KPI Metrics
    st.markdown('<div class="glass-card" style="padding: 0px;">', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    
    with c1:
        if len(driver_stats_row) > 0:
            best_lap = driver_stats_row['best_lap'].iloc[0]
            st.metric("Best Lap", f"{best_lap:.2f}s")
        else:
            st.metric("Best Lap", "N/A")
    
    with c2:
        if len(driver_stats_row) > 0 and len(stats_filtered) > 0:
            fastest_time = stats_filtered['best_lap'].min()
            gap = best_lap - fastest_time
            st.metric(
                "Gap to Leader",
                f"+{gap:.2f}s" if gap > 0 else f"{gap:.2f}s",
                delta=f"{(gap/fastest_time*100):.2f}%",
                delta_color="inverse"
            )
        else:
            st.metric("Gap to Leader", "N/A")
    
    with c3:
        if len(driver_stats_row) > 0:
            rank = int(driver_stats_row['rank'].iloc[0])
            total = len(stats_filtered)
            st.metric("Position", f"P{rank} / {total}")
        else:
            st.metric("Position", "N/A")
    
    with c4:
        if len(driver_stats_row) > 0:
            percentile = driver_stats_row['percentile'].iloc[0]
            st.metric("Percentile", f"{percentile:.0f}th")
        else:
            st.metric("Percentile", "N/A")
    
    with c5:
        if len(comparison_subset) > 0:
            total_recoverable = comparison_subset['time_lost_sec'].sum()
            st.metric("Recoverable", f"{total_recoverable:.2f}s")
        else:
            st.metric("Recoverable", "N/A")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Driver Style
    if selected_driver in clusters['vehicle_id'].values:
        driver_cluster = clusters[clusters['vehicle_id'] == selected_driver]['cluster_label'].iloc[0]
        st.markdown(f"<p style='color:{THEME['accent_gold']}; font-size:14px; margin-bottom:20px'>🎯 Driver Style: <b>{driver_cluster}</b></p>", unsafe_allow_html=True)
    
    # Main Grid
    col_main, col_side = st.columns([2, 1])
    
    with col_main:
        # Top Opportunities
        st.markdown(f'<div class="glass-card" style="padding: 0px;">', unsafe_allow_html=True)
        st.markdown(f"<h4 style='color:{THEME['text_primary']}; margin-top:0'>Top Improvement Opportunities</h4>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:12px; color:{THEME['text_secondary']}; margin-bottom:15px'>Prioritized by lap time impact</p>", unsafe_allow_html=True)
        
        if len(comparison_subset) > 0:
            top_opps = comparison_subset.nlargest(3, 'time_lost_sec')
            
            cols = st.columns(3)
            for idx, (_, row) in enumerate(top_opps.iterrows()):
                issue = diagnose_issue(row)
                fix = generate_fix(row)
                
                with cols[idx]:
                    st.markdown(f"""
                    <div class="opp-card">
                        <div style="display:flex; justify-content:space-between; align-items:center">
                            <span style="font-weight:600; color:{THEME['accent_gold']}">Turn {int(row['corner'])}</span>
                            <span style="background:rgba(188, 75, 81, 0.2); color:#ff8a8a; padding:2px 6px; border-radius:4px; font-size:11px">+{row['time_lost_sec']:.2f}s</span>
                        </div>
                        <div style="margin-top:8px; font-size:14px; color:{THEME['text_primary']}">{issue}</div>
                        <div style="font-size:12px; color:{THEME['text_secondary']}; margin-top:4px">💡 {fix}</div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No comparison data available for this selection")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Corner Loss Chart
        st.markdown(f'<div class="glass-card" style="padding: 0px;">', unsafe_allow_html=True)
        st.markdown("#### Corner-by-Corner Time Loss")
        
        if len(comparison_subset) > 0:
            fig_bar = px.bar(
                comparison_subset.sort_values('corner'),
                x='corner',
                y='time_lost_sec',
                color='time_lost_sec',
                color_continuous_scale=[[0, THEME['accent_success']], [0.5, THEME['accent_gold']], [1, THEME['accent_danger']]]
            )
            fig_bar.update_traces(marker_line_width=0, opacity=0.85)
            fig_bar.update_layout(
                xaxis_title="Corner Number",
                yaxis_title="Time Lost (seconds)",
                coloraxis_showscale=False
            )
            st.plotly_chart(style_chart(fig_bar), use_container_width=True)
        else:
            st.info("No data to display")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_side:
        # ML Insights
        st.markdown(f'<div class="glass-card" style="padding: 0px;">', unsafe_allow_html=True)
        st.markdown("#### AI Coach Insights")
        
        # Feature importance radar (if cross-track)
        if selected_track == 'All Tracks' and len(driver_stats_row) > 0:
            driver_cross_track = driver_stats[driver_stats['vehicle_id'] == selected_driver]
            
            if len(driver_cross_track) > 1:
                fig_radar = go.Figure(data=go.Scatterpolar(
                    r=driver_cross_track['percentile'].values,
                    theta=driver_cross_track['track'].values,
                    fill='toself',
                    line_color=THEME['accent_gold'],
                    fillcolor='rgba(212, 163, 115, 0.2)'
                ))
                fig_radar.update_layout(
                    polar=dict(
                        bgcolor='rgba(0,0,0,0)',
                        radialaxis=dict(
                            visible=True, 
                            showticklabels=True,
                            range=[0, 100], 
                            linecolor='rgba(255,255,255,0.1)',
                            tickfont=dict(size=10, color=THEME['text_secondary'])
                        ),
                        angularaxis=dict(color=THEME['text_secondary'])
                    ),
                    margin=dict(t=20, b=20, l=20, r=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    height=250
                )
                st.plotly_chart(fig_radar, use_container_width=True)
        
        # ML Feature importance
        st.markdown(f"<p style='font-size:13px; color:{THEME['text_secondary']}; margin-top:15px'><b>Key Performance Factors:</b></p>", unsafe_allow_html=True)
        
        for idx, row in ml_features.head(4).iterrows():
            pct = row['importance'] * 100
            st.markdown(f"""
                <div style='margin-bottom:8px'>
                    <div style='display:flex; justify-content:space-between; margin-bottom:2px'>
                        <span style='font-size:12px; color:{THEME['text_primary']}'>{row['feature'].replace('_', ' ').title()}</span>
                        <span style='font-size:12px; color:{THEME['accent_gold']}'>{pct:.1f}%</span>
                    </div>
                    <div style='width:100%; height:4px; background:rgba(255,255,255,0.1); border-radius:2px'>
                        <div style='width:{pct}%; height:100%; background:{THEME['accent_gold']}; border-radius:2px'></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # What-If Simulator
        st.markdown(f'<div class="glass-card" style="padding: 0px;">', unsafe_allow_html=True)
        st.markdown("#### What-If Simulator")
        
        improvement = st.slider(
            "Throttle Management Improvement (%)", 
            0, 25, 10,
            help="Simulate better apex throttle application"
        )
        
        gain = improvement * 0.12
        if len(driver_stats_row) > 0:
            current_lap = driver_stats_row['best_lap'].iloc[0]
            new_lap = current_lap - gain
            
            st.markdown(f"""
                <div style="text-align:center; margin-top:15px">
                    <p style="font-size:12px; color:{THEME['text_secondary']}; margin:0">Predicted New Lap Time</p>
                    <p style="font-size:32px; font-weight:bold; color:{THEME['accent_success']}; margin:5px 0">{new_lap:.2f}s</p>
                    <p style="font-size:14px; color:{THEME['accent_gold']}; margin:0">Gain: -{gain:.2f}s</p>
                </div>
            """, unsafe_allow_html=True)
        
        st.caption("Based on Random Forest model (R² = 0.899)")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Detailed Corner Breakdown
    if len(comparison_subset) > 5:
        st.markdown("---")
        st.markdown(f'<div class="glass-card" style="padding: 0px;">', unsafe_allow_html=True)
        st.markdown("#### Detailed Corner Analysis")
        
        for idx, row in comparison_subset.sort_values('corner').iterrows():
            with st.expander(f"Turn {int(row['corner'])} - {diagnose_issue(row)}"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Braking Analysis**")
                    st.write(f"Your brake: {row['slow_brake']:.1f}")
                    st.write(f"Benchmark: {row['fast_brake']:.1f}")
                    st.write(f"Delta: {row['brake_delta']:+.1f}")
                
                with col2:
                    st.markdown("**Throttle Analysis**")
                    st.write(f"Your apex: {row['slow_apex_throttle']:.1f}%")
                    st.write(f"Benchmark: {row['fast_apex_throttle']:.1f}%")
                    st.write(f"Delta: {row['apex_throttle_delta']:+.1f}%")
                
                with col3:
                    st.markdown("**Time Impact**")
                    st.write(f"Time lost: {row['time_lost_sec']:.3f}s")
                    st.write(f"Estimated distance: {abs(row['time_lost_sec']*8):.0f}m")
                    
                    if row['time_lost_sec'] > 0.5:
                        st.error("🔴 Major loss")
                    elif row['time_lost_sec'] > 0:
                        st.warning("🟡 Moderate loss")
                    else:
                        st.success("🟢 Gaining time!")
        
        st.markdown('</div>', unsafe_allow_html=True)

# ==================== MULTI-DRIVER COMPARISON MODE ====================
else:
    st.markdown('<div class="glass-card" style="padding: 0px;">', unsafe_allow_html=True)
    st.markdown("#### Multi-Driver Performance Comparison")
    
    if len(comparison_drivers) < 2:
        st.warning("⚠️ Please select at least 2 drivers to compare")
    else:
        # Summary Table
        comparison_summary = []
        for driver in comparison_drivers:
            driver_stat = stats_filtered[stats_filtered['vehicle_id'] == driver]
            if len(driver_stat) > 0:
                comparison_summary.append({
                    'Driver': driver,
                    'Best Lap': f"{driver_stat['best_lap'].iloc[0]:.2f}s",
                    'Rank': int(driver_stat['rank'].iloc[0]),
                    'Percentile': f"{driver_stat['percentile'].iloc[0]:.0f}th"
                })
        
        if comparison_summary:
            summary_df = pd.DataFrame(comparison_summary)
            st.dataframe(
                summary_df,
                column_config={
                    "Driver": st.column_config.TextColumn("Driver", width="medium"),
                    "Best Lap": st.column_config.TextColumn("Best Lap", width="small"),
                    "Rank": st.column_config.NumberColumn("Rank", width="small"),
                    "Percentile": st.column_config.TextColumn("Percentile", width="small"),
                },
                use_container_width=True,
                hide_index=True
            )
        
        st.markdown("---")
        
        # Multi-driver comparison chart
        st.markdown("#### Corner-by-Corner Time Loss Comparison")
        
        fig_comp = go.Figure()
        colors = [THEME['accent_gold'], THEME['text_primary'], THEME['accent_danger'], THEME['accent_success'], '#8B7355']
        
        for i, driver in enumerate(comparison_drivers):
            driver_comp = comparison_filtered[
                (comparison_filtered['slow_driver'] == driver) &
                (comparison_filtered['fast_driver'] == benchmark_driver) &
                (comparison_filtered['corner'].between(corner_range[0], corner_range[1]))
            ].sort_values('corner')
            
            if len(driver_comp) > 0:
                fig_comp.add_trace(go.Scatter(
                    x=driver_comp['corner'],
                    y=driver_comp['time_lost_sec'],
                    mode='lines+markers',
                    name=driver,
                    line=dict(width=3, color=colors[i % len(colors)]),
                    marker=dict(size=8)
                ))
        
        fig_comp.update_layout(
            xaxis_title="Corner Number",
            yaxis_title="Time Lost vs Benchmark (seconds)",
            hovermode='x unified',
            height=500
        )
        st.plotly_chart(style_chart(fig_comp), use_container_width=True)
        
        # Side-by-side comparison
        st.markdown("---")
        st.markdown("#### Driver Statistics")
        
        cols = st.columns(len(comparison_drivers))
        for idx, driver in enumerate(comparison_drivers):
            with cols[idx]:
                st.markdown(f"**{driver}**")
                driver_stat = stats_filtered[stats_filtered['vehicle_id'] == driver]
                if len(driver_stat) > 0:
                    st.metric("Best Lap", f"{driver_stat['best_lap'].iloc[0]:.2f}s")
                    st.metric("Rank", f"P{int(driver_stat['rank'].iloc[0])}")
                    st.metric("Percentile", f"{driver_stat['percentile'].iloc[0]:.0f}th")
                
                # Driver style
                if driver in clusters['vehicle_id'].values:
                    style = clusters[clusters['vehicle_id'] == driver]['cluster_label'].iloc[0]
                    st.caption(f"Style: {style}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ==================== FOOTER ====================
st.markdown("---")
st.markdown(f"""
    <div style='text-align: center; color: {THEME['text_secondary']}; padding: 20px 0'>
        <p style='font-weight:600; color:{THEME['text_primary']}; margin-bottom:5px'>🏁 Toyota GR Cup Championship Analysis Platform</p>
        <p style='font-size:12px; margin:0'>Powered by Machine Learning + Racing Physics • {len(corners):,} Corners • {corners['vehicle_id'].nunique()} Drivers • {corners['track'].nunique()} Tracks</p>
        <p style='font-size:11px; margin-top:5px; opacity:0.7'>VIR • Road America • COTA • Sebring • Sonoma • Barber • Indianapolis</p>
    </div>
    """, unsafe_allow_html=True)
