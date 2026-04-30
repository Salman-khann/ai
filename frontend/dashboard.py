# dashboard.py
import streamlit as st
import requests
import os
import pandas as pd

st.set_page_config(page_title="Sentinel: AI Governance", layout="wide")

# Extreme UI Styling (Hides all platform footers, headers, and menus)
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            #stDecoration {display:none;}
            .viewerBadge_container__1QSob {display:none !important;}
            .st-emotion-cache-164366 {display:none !important;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# Cloud Deployment Config (Now hidden from the user)
# On Streamlit Cloud, add BACKEND_URL to your "Secrets"
BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:5000")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VELOCITY_PATH = os.path.join(BASE_DIR, "src", "data", "processed", "risk_velocity.csv")

st.title("🛡️ Sentinel: Proactive Educational Governance")
st.markdown("Institutional intelligence for early student intervention and strategic risk management.")

tab1, tab2 = st.tabs(["Individual Analysis", "Institutional Overview"])

with tab1:
    st.sidebar.header("Student Search")
    student_id = st.sidebar.text_input("Enter Student ID", value="STU-1042")

    if st.sidebar.button("Analyze Risk Profile"):
        with st.spinner("Analyzing data through AI models..."):
            try:
                response = requests.post(f"{BACKEND_URL}/analyze_student", json={"student_id": student_id})
                if response.status_code == 200:
                    result = response.json()
                    st.markdown(f"### Target: `{result['student_id']}`")
                    col1, col2, col3 = st.columns(3)
                    risk_score = result['risk_score'] * 100
                    col1.metric("Dropout Risk Probability", f"{risk_score:.1f}%")
                    col2.metric("Escalation Tier", f"{result['escalation_level']}")
                    
                    if "Level 3" in result['escalation_level']:
                        st.error(f"**Prescribed Action:** {result['prescribed_intervention']}")
                    elif "Level 2" in result['escalation_level'] or "Level 1" in result['escalation_level']:
                        st.warning(f"**Prescribed Action:** {result['prescribed_intervention']}")
                    else:
                        st.success(f"**Prescribed Action:** Low Risk. No intervention required.")
                    st.divider()
                    st.markdown("### 🔍 Explainable AI (XAI) Intervention Card")
                    reasons = result['intervention_card']
                    for idx, reason in enumerate(reasons):
                        metric = reason['metric'].replace('_', ' ').title()
                        impact = reason['shap_weight']
                        if impact > 0:
                            st.error(f"**{idx+1}. {metric}** (+{impact:.2f} Impact) ➔ {reason['explanation']}")
                        else:
                            st.success(f"**{idx+1}. {metric}** ({impact:.2f} Impact) ➔ {reason['explanation']}")
            except Exception as e:
                st.error(f"API Connection Failed: {e}")

with tab2:
    st.header("🏢 Institutional Risk Heatmap")
    st.markdown("Current distribution of risk across the student population.")
    
    if st.button("Refresh Institutional Data"):
        try:
            summary_res = requests.get(f"{BACKEND_URL}/risk_summary")
            if summary_res.status_code == 200:
                summary_data = summary_res.json()
                
                # Sort levels for the chart
                order = ["Level 0 (Standard)", "Level 1 (Advisor)", "Level 2 (Dean/Counselor)", "Level 3 (Academic VP)"]
                chart_data = {lvl: summary_data.get(lvl, 0) for lvl in order}
                
                st.bar_chart(chart_data)
                
                # Metrics for quick view
                m1, m2, m3 = st.columns(3)
                m1.metric("High Risk (Level 3)", chart_data["Level 3 (Academic VP)"])
                m2.metric("Medium Risk (Level 2)", chart_data["Level 2 (Dean/Counselor)"])
                m3.metric("Total Flagged", sum(chart_data.values()) - chart_data["Level 0 (Standard)"])

                # Add Velocity of Risk (Temporal Analysis)
                st.divider()
                st.subheader("📈 Velocity of Risk (Temporal Analysis)")
                st.markdown("Tracks how the risk distribution evolves across multiple semesters.")
                
                if os.path.exists(VELOCITY_PATH):
                    velocity_df = pd.read_csv(VELOCITY_PATH)
                    st.line_chart(velocity_df.set_index("Semester"))
                else:
                    st.info("Velocity data not yet simulated. Please run the simulation script.")

        except Exception as e:
            st.error(f"Failed to fetch summary: {e}")