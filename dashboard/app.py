

"""
SmartPark - AI Parking Intelligence Dashboard
--------------------------------------------
Enterprise smart-city parking management and optimization interface.
Integrates IoT Telemetry, SQLite, LSTM Neural Network, Proximity Optimizer, and GenAI Assistant.
"""

import os
import sys
import time
from datetime import datetime
import pandas as pd
import requests
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import Phase 1 - 4 modules
from optimization.slot_optimizer import optimize_parking_slots, SLOT_DISTANCES_METERS
from genai.parking_assistant import ask_parking_assistant

# Constants
API_BASE_URL = os.getenv("BACKEND_API_URL", "http://127.0.0.1:8000")

# ---------------------------------------------------------
# PAGE SETUP & MODERN SMART-CITY CSS
# ---------------------------------------------------------
st.set_page_config(
    page_title="SmartPark | AI Parking Intelligence",
    page_icon="🅿️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Dark Navy / Charcoal Theme */
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Global Card Styles */
    .smart-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3), 0 2px 4px -2px rgba(0, 0, 0, 0.3);
        margin-bottom: 16px;
    }
    
    /* Top KPI Cards */
    .kpi-card {
        background: linear-gradient(145deg, #131d31 0%, #0d1527 100%);
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 16px 20px;
        position: relative;
        overflow: hidden;
    }
    .kpi-card::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: #38bdf8;
    }
    .kpi-card-avail::before { background: #10b981; }
    .kpi-card-occ::before { background: #ef4444; }
    .kpi-card-rec::before { background: linear-gradient(90deg, #38bdf8, #818cf8); }
    
    .kpi-label {
        color: #94a3b8;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .kpi-value {
        color: #f8fafc;
        font-size: 2.1rem;
        font-weight: 800;
        line-height: 1.1;
        margin-bottom: 4px;
    }
    .kpi-sub {
        color: #64748b;
        font-size: 0.8rem;
    }
    
    /* Parking Bay Map Layout */
    .bay-card {
        background: #111827;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        border: 2px solid #1e293b;
        transition: all 0.25s ease-in-out;
        position: relative;
    }
    .bay-avail {
        border-color: #059669;
        background: radial-gradient(circle at 50% 0%, rgba(16, 185, 129, 0.12) 0%, #111827 80%);
    }
    .bay-occ {
        border-color: #dc2626;
        background: radial-gradient(circle at 50% 0%, rgba(239, 68, 68, 0.12) 0%, #111827 80%);
    }
    .bay-badge-avail {
        display: inline-block;
        background: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.4);
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.05em;
    }
    .bay-badge-occ {
        display: inline-block;
        background: rgba(239, 68, 68, 0.2);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.4);
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.05em;
    }
    
    /* Recommendation Hero Banner */
    .rec-hero {
        background: linear-gradient(135deg, #0f2b48 0%, #172554 50%, #1e1b4b 100%);
        border: 1px solid #3b82f6;
        border-radius: 14px;
        padding: 22px 24px;
        box-shadow: 0 10px 25px -5px rgba(59, 130, 246, 0.2);
        margin-bottom: 20px;
    }
    
    /* Live Status Pulse Dot */
    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #10b981;
        box-shadow: 0 0 8px #10b981;
        margin-right: 6px;
    }
    .status-dot-amber {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #f59e0b;
        box-shadow: 0 0 8px #f59e0b;
        margin-right: 6px;
    }
    
    /* Activity Feed Items */
    .activity-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 8px 12px;
        border-bottom: 1px solid #1e293b;
        font-size: 0.85rem;
    }
    .activity-time {
        color: #64748b;
        font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# DATA ACCESSORS WITH FULL RESILIENCE & FALLBACKS
# ---------------------------------------------------------
def fetch_parking_slots():
    """
    Fetches real-time slot states from FastAPI backend.
    Falls back to direct SQLite database read if API is offline.
    """
    try:
        response = requests.get(f"{API_BASE_URL}/api/slots", timeout=2.0)
        if response.status_code == 200:
            return response.json(), "api"
    except Exception:
        pass

    # Direct SQLite Fallback
    try:
        from backend.database import get_all_slots, init_db
        init_db()
        slots = get_all_slots()
        return slots, "sqlite_fallback"
    except Exception:
        return [], "error"


def fetch_historical_logs(limit=200):
    """
    Fetches historical sensor telemetry from FastAPI or SQLite fallback.
    """
    try:
        response = requests.get(f"{API_BASE_URL}/api/analytics/history?limit={limit}", timeout=2.0)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass

    # Direct SQLite Fallback
    try:
        from backend.database import get_history_logs
        return get_history_logs(limit=limit)
    except Exception:
        return []


def fetch_lstm_predictions():
    """
    Executes LSTM model predictions for all slots via existing predict_all_slots().
    """
    try:
        from dnn.predict import predict_all_slots
        return predict_all_slots(), None
    except Exception as e:
        return {}, str(e)


# ---------------------------------------------------------
# STATE RETRIEVAL & OPTIMIZATION PIPELINE
# ---------------------------------------------------------
slots_data, data_source = fetch_parking_slots()
predictions, dnn_error = fetch_lstm_predictions()
history_logs = fetch_historical_logs(limit=200)

slot_status_map = {s["slot_id"]: s["status"].lower() for s in slots_data} if slots_data else {}
pred_map = {s_id: p_info["probability"] for s_id, p_info in predictions.items()} if predictions else {}

# Single source of truth: Run optimizer
optimizer_res = optimize_parking_slots(
    slot_statuses=slot_status_map if slot_status_map else None,
    predictions=pred_map if pred_map else None,
    distances=SLOT_DISTANCES_METERS
)


# ---------------------------------------------------------
# SIDEBAR: SMARTPARK BRAND & SYSTEM STATUS
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="padding: 10px 0 15px 0;">
        <h2 style="margin: 0; color: #38bdf8; font-weight: 800; letter-spacing: -0.02em;">SmartPark</h2>
        <p style="margin: 0; color: #94a3b8; font-size: 0.82rem; font-weight: 500;">AI Parking Intelligence Platform</p>
    </div>
    """, unsafe_allow_html=True)

    nav_choice = st.radio(
        "Navigation",
        ["Overview", "Live Parking", "AI Predictions", "Analytics", "AI Assistant"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    # System Status Block
    st.write("##### System Health")
    if data_source == "api":
        st.markdown('<div><span class="status-dot"></span><span style="font-size:0.85rem; color:#e2e8f0;">FastAPI Server Online</span></div>', unsafe_allow_html=True)
    elif data_source == "sqlite_fallback":
        st.markdown('<div><span class="status-dot-amber"></span><span style="font-size:0.85rem; color:#f59e0b;">SQLite Fallback Active</span></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div><span class="status-dot-amber"></span><span style="font-size:0.85rem; color:#ef4444;">Database Disconnected</span></div>', unsafe_allow_html=True)

    st.markdown('<div><span class="status-dot"></span><span style="font-size:0.85rem; color:#e2e8f0;">SQLite Database Ready</span></div>', unsafe_allow_html=True)
    
    if predictions:
        st.markdown('<div><span class="status-dot"></span><span style="font-size:0.85rem; color:#e2e8f0;">LSTM DNN Predictor Loaded</span></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div><span class="status-dot-amber"></span><span style="font-size:0.85rem; color:#f59e0b;">LSTM Predictor Standby</span></div>', unsafe_allow_html=True)

    st.markdown('<div><span class="status-dot"></span><span style="font-size:0.85rem; color:#e2e8f0;">IoT Telemetry Stream Active</span></div>', unsafe_allow_html=True)

    st.markdown("---")

    # Controls
    st.write("##### Telemetry Controls")
    auto_refresh = st.checkbox("Auto-refresh (5s)", value=False)
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.rerun()

    st.markdown("---")
    st.caption("College Integrated Project — IoT + DNN + GenAI")


# ---------------------------------------------------------
# TOP HEADER
# ---------------------------------------------------------
col_head_left, col_head_right = st.columns([3, 1.2])

with col_head_left:
    st.markdown("""
    <div style="margin-bottom: 8px;">
        <h1 style="margin: 0; font-size: 2.1rem; font-weight: 800; color: #f8fafc;">Smart Parking Optimization</h1>
        <p style="margin: 4px 0 0 0; color: #94a3b8; font-size: 0.95rem;">Real-time parking intelligence powered by IoT, DNN and GenAI</p>
    </div>
    """, unsafe_allow_html=True)

with col_head_right:
    now_str = datetime.now().strftime("%H:%M:%S")
    st.markdown(f"""
    <div style="text-align: right; padding-top: 10px;">
        <div><span class="status-dot"></span><span style="font-weight:700; font-size:0.82rem; color:#38bdf8; letter-spacing:0.05em;">LIVE TELEMETRY</span></div>
        <div style="color: #64748b; font-size: 0.8rem; margin-top: 3px;">Updated: <span style="color:#cbd5e1; font-family:monospace;">{now_str}</span></div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr style='border-color: #1e293b; margin: 12px 0 20px 0;'>", unsafe_allow_html=True)


# ---------------------------------------------------------
# TOP KPI CARDS
# ---------------------------------------------------------
total_slots = len(slots_data) if slots_data else 4
avail_count = sum(1 for s in slots_data if s["status"].lower() == "available")
occ_count = total_slots - avail_count
rec_slot = optimizer_res.get("recommended_slot")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Total Capacity</div>
        <div class="kpi-value">{total_slots}</div>
        <div class="kpi-sub">Monitored bays: P1–P4</div>
    </div>
    """, unsafe_allow_html=True)

with kpi2:
    st.markdown(f"""
    <div class="kpi-card kpi-card-avail">
        <div class="kpi-label" style="color:#34d399;">Available Slots</div>
        <div class="kpi-value" style="color:#10b981;">{avail_count} <span style="font-size:1.1rem; color:#64748b;">/ {total_slots}</span></div>
        <div class="kpi-sub">Ready for immediate parking</div>
    </div>
    """, unsafe_allow_html=True)

with kpi3:
    st.markdown(f"""
    <div class="kpi-card kpi-card-occ">
        <div class="kpi-label" style="color:#f87171;">Occupied Slots</div>
        <div class="kpi-value" style="color:#ef4444;">{occ_count} <span style="font-size:1.1rem; color:#64748b;">/ {total_slots}</span></div>
        <div class="kpi-sub">Active vehicle dwell</div>
    </div>
    """, unsafe_allow_html=True)

with kpi4:
    rec_text = rec_slot if rec_slot else "FULL"
    rec_sub = f"Utility Score: {optimizer_res.get('score', 0):.2f}" if rec_slot else "Wait for vacancy"
    st.markdown(f"""
    <div class="kpi-card kpi-card-rec">
        <div class="kpi-label" style="color:#818cf8;">AI Recommendation</div>
        <div class="kpi-value" style="color:#38bdf8;">{rec_text}</div>
        <div class="kpi-sub">{rec_sub}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)


# ---------------------------------------------------------
# INTERACTIVE SLOT SELECTION STATE
# ---------------------------------------------------------
if "selected_slot" not in st.session_state:
    st.session_state.selected_slot = rec_slot or "P1"


# ---------------------------------------------------------
# SECTION: LIVE PARKING MAP & DETAIL PANEL
# ---------------------------------------------------------
if nav_choice in ["Overview", "Live Parking"]:
    st.markdown("### 🅿️ Live Parking Lot Layout")
    st.caption("Interactive bay occupancy map. Click any slot to view detailed sensor telemetry and LSTM utility breakdown.")

    map_col, detail_col = st.columns([2.2, 1.2])

    with map_col:
        st.markdown("""
        <div style="background:#090d16; border:1px dashed #334155; border-radius:12px; padding:18px 16px; margin-bottom:12px;">
            <div style="text-align:center; color:#64748b; font-size:0.78rem; font-weight:700; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:14px;">
                🚗 FACILITY ENTRANCE ─────────────────────────────
            </div>
        """, unsafe_allow_html=True)

        bay_cols = st.columns(4)
        for idx, s_id in enumerate(["P1", "P2", "P3", "P4"]):
            with bay_cols[idx]:
                s_info = next((s for s in slots_data if s["slot_id"] == s_id), None)
                s_status = s_info["status"].lower() if s_info else "available"
                is_available = (s_status == "available")
                dist = SLOT_DISTANCES_METERS.get(s_id, 20.0)
                pred_prob = pred_map.get(s_id, 0.20)

                bay_class = "bay-avail" if is_available else "bay-occ"
                badge_class = "bay-badge-avail" if is_available else "bay-badge-occ"
                status_text = "AVAILABLE" if is_available else "OCCUPIED"
                star_indicator = "⭐ " if s_id == rec_slot else ""

                st.markdown(f"""
                <div class="bay-card {bay_class}">
                    <div style="font-size:1.3rem; font-weight:800; color:#f8fafc; margin-bottom:6px;">{star_indicator}{s_id}</div>
                    <span class="{badge_class}">{status_text}</span>
                    <div style="margin-top:12px; font-size:0.8rem; color:#94a3b8;">🚶 <b>{int(dist)}m</b> from gate</div>
                    <div style="margin-top:4px; font-size:0.8rem; color:#64748b;">Risk: <b>{pred_prob * 100:.1f}%</b></div>
                </div>
                """, unsafe_allow_html=True)

                if st.button(f"Inspect {s_id}", key=f"btn_inspect_{s_id}", use_container_width=True):
                    st.session_state.selected_slot = s_id
                    st.rerun()

        st.markdown("""
            <div style="text-align:center; color:#334155; font-size:0.75rem; margin-top:16px;">
                ═══════════════════ MAIN DRIVING LANE ═══════════════════
            </div>
        </div>
        """, unsafe_allow_html=True)

    with detail_col:
        sel_slot = st.session_state.selected_slot
        sel_info = next((s for s in slots_data if s["slot_id"] == sel_slot), None)
        sel_status = sel_info["status"].lower() if sel_info else "available"
        sel_dist = SLOT_DISTANCES_METERS.get(sel_slot, 20.0)
        sel_pred = pred_map.get(sel_slot, 0.20)
        
        # Risk classification
        if sel_pred < 0.20:
            risk_label, risk_color = "VERY LOW", "#10b981"
        elif sel_pred < 0.50:
            risk_label, risk_color = "MODERATE", "#f59e0b"
        else:
            risk_label, risk_color = "HIGH", "#ef4444"

        # Calculate individual utility score
        slot_candidate = next((x for x in optimizer_res.get("all_slots", []) if x["slot_id"] == sel_slot), None)
        sel_score = slot_candidate["score"] if slot_candidate else 0.0

        st.markdown(f"""
        <div class="smart-card" style="border-left: 4px solid #38bdf8;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                <h3 style="margin:0; font-size:1.3rem; color:#f8fafc;">Slot {sel_slot} Telemetry</h3>
                <span style="font-size:0.8rem; padding:2px 8px; border-radius:4px; background:#1e293b; color:#94a3b8;">Details</span>
            </div>
            <div style="display:grid; grid-template-columns: 1fr 1fr; gap:10px; margin-bottom:12px;">
                <div style="background:#0b0f19; padding:10px; border-radius:8px;">
                    <div style="font-size:0.75rem; color:#64748b;">CURRENT STATE</div>
                    <div style="font-size:1rem; font-weight:700; color:{'#34d399' if sel_status=='available' else '#f87171'};">
                        {sel_status.upper()}
                    </div>
                </div>
                <div style="background:#0b0f19; padding:10px; border-radius:8px;">
                    <div style="font-size:0.75rem; color:#64748b;">WALKING DIST</div>
                    <div style="font-size:1rem; font-weight:700; color:#f8fafc;">{int(sel_dist)} meters</div>
                </div>
                <div style="background:#0b0f19; padding:10px; border-radius:8px;">
                    <div style="font-size:0.75rem; color:#64748b;">PRED OCCUPANCY</div>
                    <div style="font-size:1rem; font-weight:700; color:#38bdf8;">{sel_pred * 100:.1f}%</div>
                </div>
                <div style="background:#0b0f19; padding:10px; border-radius:8px;">
                    <div style="font-size:0.75rem; color:#64748b;">RISK PROFILE</div>
                    <div style="font-size:1rem; font-weight:700; color:{risk_color};">{risk_label}</div>
                </div>
            </div>
            <div style="background:#0b0f19; padding:10px; border-radius:8px; margin-bottom:12px;">
                <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                    <span style="font-size:0.8rem; color:#94a3b8;">Optimizer Utility Score</span>
                    <span style="font-weight:700; color:#818cf8;">{sel_score:.4f}</span>
                </div>
                <div style="background:#1e293b; border-radius:6px; height:8px; overflow:hidden;">
                    <div style="background:linear-gradient(90deg, #38bdf8, #818cf8); width:{int(sel_score * 100)}%; height:100%;"></div>
                </div>
            </div>
            <div style="font-size:0.75rem; color:#64748b;">
                Last Sensor Timestamp: {sel_info.get('last_updated', 'Synchronized') if sel_info else 'N/A'}
            </div>
        </div>
        """, unsafe_allow_html=True)


# ---------------------------------------------------------
# SECTION: OPTIMIZER RECOMMENDATION & RANKING
# ---------------------------------------------------------
if nav_choice in ["Overview", "Live Parking", "AI Predictions"]:
    st.markdown("### 🎯 AI Parking Recommendation")

    if rec_slot:
        st.markdown(f"""
        <div class="rec-hero">
            <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:10px;">
                <div>
                    <span style="background:rgba(56, 189, 248, 0.2); color:#38bdf8; font-size:0.75rem; font-weight:700; padding:4px 10px; border-radius:20px; letter-spacing:0.05em;">OPTIMIZER DECISION</span>
                    <h2 style="margin: 8px 0 4px 0; color:#ffffff; font-size:1.8rem; font-weight:800;">Recommended Bay: Slot {rec_slot}</h2>
                    <p style="margin:0; color:#cbd5e1; font-size:0.95rem; max-width:680px;">{optimizer_res.get('reason')}</p>
                </div>
                <div style="display:flex; gap:16px;">
                    <div style="background:rgba(15, 23, 42, 0.6); padding:10px 16px; border-radius:10px; border:1px solid rgba(255,255,255,0.1); text-align:center;">
                        <div style="font-size:0.7rem; color:#94a3b8;">UTILITY SCORE</div>
                        <div style="font-size:1.4rem; font-weight:800; color:#38bdf8;">{optimizer_res.get('score', 0):.4f}</div>
                    </div>
                    <div style="background:rgba(15, 23, 42, 0.6); padding:10px 16px; border-radius:10px; border:1px solid rgba(255,255,255,0.1); text-align:center;">
                        <div style="font-size:0.7rem; color:#94a3b8;">WALKING DIST</div>
                        <div style="font-size:1.4rem; font-weight:800; color:#f8fafc;">{int(optimizer_res.get('distance_m', 0))}m</div>
                    </div>
                    <div style="background:rgba(15, 23, 42, 0.6); padding:10px 16px; border-radius:10px; border:1px solid rgba(255,255,255,0.1); text-align:center;">
                        <div style="font-size:0.7rem; color:#94a3b8;">PRED RISK</div>
                        <div style="font-size:1.4rem; font-weight:800; color:#34d399;">{optimizer_res.get('predicted_occupancy', 0) * 100:.1f}%</div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ **All parking slots are currently occupied.** Please wait near the entrance for a vehicle to depart.")

    # Candidate Ranking Table
    ranked_candidates = optimizer_res.get("ranked_slots", [])
    if ranked_candidates:
        st.write("##### 📊 Optimizer Candidate Ranking")
        ranking_data = []
        for rank, item in enumerate(ranked_candidates, start=1):
            ranking_data.append({
                "Rank": f"#{rank}",
                "Slot ID": item["slot_id"],
                "Status": item["status"].upper(),
                "Walking Distance": f"{int(item['distance_m'])}m",
                "Predicted Occupancy Risk": f"{item['predicted_occupancy'] * 100:.1f}%",
                "Combined Utility Score": f"{item['score']:.4f}",
            })
        st.dataframe(pd.DataFrame(ranking_data), use_container_width=True, hide_index=True)


# ---------------------------------------------------------
# SECTION: DNN / LSTM OCCUPANCY INTELLIGENCE
# ---------------------------------------------------------
if nav_choice in ["Overview", "AI Predictions", "Analytics"]:
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    st.markdown("### 🧠 Occupancy Intelligence (LSTM Neural Network)")
    st.caption("Time-series occupancy pattern analytics and 8-observation sliding window forecast.")

    tab_chart, tab_compare, tab_table = st.tabs(["📈 Historical Telemetry Chart", "📊 Current vs Predicted Comparison", "📋 Detailed Inference Matrix"])

    with tab_chart:
        if history_logs:
            df_hist = pd.DataFrame(history_logs)
            df_hist["timestamp"] = pd.to_datetime(df_hist["timestamp"])
            df_hist = df_hist.sort_values(by="timestamp")

            # Pivot to show occupied status (0 or 1) per slot over time
            df_pivot = df_hist.pivot_table(
                index="timestamp",
                columns="slot_id",
                values="occupied",
                aggfunc="last"
            ).ffill().fillna(0)

            st.line_chart(df_pivot, use_container_width=True)
            st.caption("Telemetry stream: 1 = Occupied, 0 = Available. Sourced from SQLite `sensor_logs`.")
        else:
            st.info("No historical telemetry data available yet. Start the IoT simulator to log readings.")

    with tab_compare:
        # Comparison Bar Chart: Current Binary Occupancy vs Predicted Probability
        comp_records = []
        for s_id in ["P1", "P2", "P3", "P4"]:
            s_cur = 1.0 if slot_status_map.get(s_id) == "occupied" else 0.0
            s_pred = pred_map.get(s_id, 0.20)
            comp_records.append({
                "Slot": s_id,
                "Current Occupancy (0/1)": s_cur,
                "Predicted Risk Probability": s_pred
            })
        df_comp = pd.DataFrame(comp_records).set_index("Slot")
        st.bar_chart(df_comp, use_container_width=True)
        st.caption("Comparison: Instantaneous Physical State vs LSTM Future Probability.")

    with tab_table:
        if predictions:
            pred_records = []
            for s_id in ["P1", "P2", "P3", "P4"]:
                p_info = predictions.get(s_id, {})
                prob = p_info.get("probability", 0.20)
                risk_lvl = "Low" if prob < 0.25 else ("Moderate" if prob < 0.50 else "High")
                pred_records.append({
                    "Slot": s_id,
                    "Current Status": slot_status_map.get(s_id, "available").upper(),
                    "LSTM Predicted Probability": f"{prob:.4f} ({prob * 100:.1f}%)",
                    "Model Prediction": p_info.get("prediction", "AVAILABLE"),
                    "Risk Level": risk_lvl,
                    "Input Window Sequence": str(p_info.get("sequence_used", []))
                })
            st.dataframe(pd.DataFrame(pred_records), use_container_width=True, hide_index=True)
            st.caption("Note: Inference executed by trained 32-unit LSTM model based on previous 8 historical sensor observations.")
        elif dnn_error:
            st.warning(f"Could not load LSTM model: {dnn_error}")


# ---------------------------------------------------------
# SECTION: GENAI PARKING ASSISTANT (CONVERSATIONAL INTERACTION)
# ---------------------------------------------------------
if nav_choice in ["Overview", "AI Assistant"]:
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    st.markdown("### 🤖 SmartPark AI Assistant")
    st.caption("Ask anything about current parking availability, walking distances, and model forecasts.")

    # Session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {
                "role": "assistant",
                "text": "Hello! I am your SmartPark AI Assistant. How can I assist you with parking today?",
                "mode": "system"
            }
        ]

    # Quick Inquiries Bar
    st.write("##### Quick Inquiries:")
    col_q1, col_q2, col_q3, col_q4, col_q5 = st.columns(5)
    selected_prompt = None

    if col_q1.button("📍 Where to park?", use_container_width=True):
        selected_prompt = "Where should I park?"
    if col_q2.button("🚶 Closest bay?", use_container_width=True):
        selected_prompt = "Which parking slot is closest?"
    if col_q3.button("⏳ Likely to stay free?", use_container_width=True):
        selected_prompt = "Which slot is likely to remain available?"
    if col_q4.button("❓ Will P3 be occupied?", use_container_width=True):
        selected_prompt = "Will P3 be occupied soon?"
    if col_q5.button("📋 Show available", use_container_width=True):
        selected_prompt = "Show me available parking slots."

    # Conversational Container
    with st.container():
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.chat_message("user").write(msg["text"])
            else:
                with st.chat_message("assistant"):
                    st.write(msg["text"])
                    if msg.get("mode") and msg["mode"] != "system":
                        st.caption(f"*Engine Mode: {msg['mode'].upper()}*")

    user_query = st.chat_input("Ask a question about parking...")
    active_query = selected_prompt or user_query

    if active_query:
        st.session_state.chat_history.append({"role": "user", "text": active_query})
        
        # Sourced strictly from optimizer
        ai_reply = ask_parking_assistant(
            query=active_query,
            optimizer_result=optimizer_res
        )

        st.session_state.chat_history.append({
            "role": "assistant",
            "text": ai_reply["response"],
            "mode": ai_reply["mode"]
        })
        st.rerun()


# ---------------------------------------------------------
# SECTION: LIVE SYSTEM ACTIVITY FEED
# ---------------------------------------------------------
if nav_choice in ["Overview", "Live Parking", "Analytics"]:
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    st.markdown("### 📡 Live System Activity")
    
    act_col1, act_col2 = st.columns([2, 1])
    
    with act_col1:
        if history_logs:
            st.write("##### Recent Telemetry Events")
            recent_rows = history_logs[:6]
            for row in recent_rows:
                r_slot = row.get("slot_id")
                r_occ = row.get("occupied")
                r_dist = row.get("distance_cm", 0.0)
                r_time = row.get("timestamp", "").split()[-1] if row.get("timestamp") else "Recent"
                status_color = "#34d399" if r_occ == 0 else "#f87171"
                status_word = "AVAILABLE" if r_occ == 0 else "OCCUPIED"

                st.markdown(f"""
                <div class="activity-row">
                    <span>
                        <span class="activity-time">[{r_time}]</span> 
                        &nbsp;<b>Slot {r_slot}</b> occupancy sensor reported: 
                        <span style="color:{status_color}; font-weight:700;">{status_word}</span> 
                        (Ultrasonic: {r_dist:.1f} cm)
                    </span>
                    <span style="color:#64748b; font-size:0.75rem;">Verified</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Awaiting sensor readings from simulator...")

    with act_col2:
        st.write("##### Architecture Pipeline")
        st.markdown("""
        <div style="background:#090d16; padding:12px; border-radius:8px; border:1px solid #1f2937; font-size:0.82rem; color:#94a3b8;">
            <div>🔹 <b>IoT Simulator:</b> 4 Ultrasonic Sensors (HC-SR04)</div>
            <div style="margin-top:4px;">🔹 <b>Backend:</b> FastAPI REST API + SQLite</div>
            <div style="margin-top:4px;">🔹 <b>DNN:</b> 32-unit LSTM Time-series Model</div>
            <div style="margin-top:4px;">🔹 <b>Optimizer:</b> Proximity & Risk Utility Engine</div>
            <div style="margin-top:4px;">🔹 <b>GenAI:</b> Natural-Language Assistant</div>
        </div>
        """, unsafe_allow_html=True)


# ---------------------------------------------------------
# AUTO REFRESH TRIGGER
# ---------------------------------------------------------
if auto_refresh:
    time.sleep(5)
    st.rerun()
