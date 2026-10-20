import streamlit as st
import requests
import json
import pandas as pd
from pathlib import Path

API_URL = "http://127.0.0.1:8000"
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"

st.set_page_config(
    page_title="Student MindCare AI",
    page_icon="🧠",
    layout="wide"
)

# ----------------- CLEAN, POLISHED DARK CARD STYLING -----------------
st.markdown("""
<style>
    /* 1. Base Dark Slate Canvas */
    .stApp {
        background-color: #0b0f19 !important;
        color: #f1f5f9 !important;
    }

    /* 2. Distinct Dark Cards with Subtle Borders */
    div[data-testid="stVerticalBlockBorderWrapper"] > div {
        background-color: #111827 !important;
        border: 1px solid #1f293d !important;
        border-radius: 10px !important;
        padding: 1.25rem 1.4rem !important;
        margin-bottom: 0.75rem !important;
    }

    /* 3. Typography & Text Hierarchy */
    h1, h2, h3, h4 {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    p, span, label {
        color: #cbd5e1 !important;
    }

    /* 4. Fix Radio Button Text & Spacing (Without breaking the circular icons) */
    div[role="radiogroup"] {
        display: flex;
        flex-wrap: wrap;
        gap: 1.2rem !important;
        margin-top: 0.5rem;
    }

    div[role="radiogroup"] label {
        color: #e2e8f0 !important;
        font-size: 0.88rem !important;
        cursor: pointer;
    }

    /* 5. Clean Action Buttons */
    .stButton > button[kind="primary"] {
        background-color: #ef4444 !important;
        border: 1px solid #dc2626 !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        padding: 0.55rem 1.6rem !important;
        border-radius: 8px !important;
    }

    .stButton > button:not([kind="primary"]) {
        background-color: #1f2937 !important;
        border: 1px solid #374151 !important;
        color: #cbd5e1 !important;
        font-weight: 500 !important;
        border-radius: 8px !important;
        padding: 0.55rem 1.3rem !important;
    }

    /* 6. Tabs & Progress */
    .stProgress > div > div > div > div {
        background-color: #ef4444 !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 1.5rem;
        border-bottom: 1px solid #1f293d !important;
    }

    .stTabs [data-baseweb="tab"] {
        color: #94a3b8 !important;
    }

    .stTabs [aria-selected="true"] {
        color: #ffffff !important;
        border-bottom: 2px solid #ef4444 !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR STATUS -----------------
try:
    health_resp = requests.get(f"{API_URL}/health", timeout=1.5)
    if health_resp.status_code == 200:
        st.sidebar.success("FastAPI Backend: Online")
    else:
        st.sidebar.warning("FastAPI Backend: Degraded")
except Exception:
    st.sidebar.error("FastAPI Backend: Offline (Run 'uvicorn api:app --reload')")

# State initialization
if "step" not in st.session_state:
    st.session_state.step = 1

if "survey_inputs" not in st.session_state:
    st.session_state.survey_inputs = {}

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

# Title
st.title("🧠 Student MindCare AI")
st.markdown("<p style='color: #94a3b8; margin-top: -10px; font-size: 0.95rem;'>Multi-Factor Academic Stress, Lifestyle & Mental Well-Being Analysis</p>", unsafe_allow_html=True)

tab_survey, tab_analytics = st.tabs(["Assessment Survey", "Population Analytics"])

# ----------------- DESCRIPTIVE SCALES -----------------
SEVERITY_OPTIONS = [0, 1, 3, 4, 5]
SEVERITY_LABELS = {
    0: "None (Not at all)",
    1: "Mild (Manageable)",
    3: "Moderate (Noticeable)",
    4: "Severe (Disruptive)",
    5: "Extreme (Overwhelming)"
}

FREQUENCY_OPTIONS = [0, 1, 3, 5]
FREQUENCY_LABELS = {
    0: "Never (0 days/wk)",
    1: "Rarely (1–2 days/wk)",
    3: "Often (3–4 days/wk)",
    5: "Almost Daily (5–7 days/wk)"
}

WORKLOAD_OPTIONS = [0, 2, 4, 5]
WORKLOAD_LABELS = {
    0: "Light (Plenty of time)",
    2: "Balanced (Manageable)",
    4: "Demanding (Overtime)",
    5: "Crushing (Severe burnout)"
}

SUPPORT_OPTIONS = [0, 2, 4, 5]
SUPPORT_LABELS = {
    0: "Strong (Always supported)",
    2: "Moderate (A few peers)",
    4: "Limited (Rarely heard)",
    5: "Isolated (Completely alone)"
}

LIVING_OPTIONS = [0, 2, 4, 5]
LIVING_LABELS = {
    0: "Peaceful & Comfortable",
    2: "Adequate (Minor noise)",
    4: "Noisy & Disruptive",
    5: "Extremely Uncomfortable"
}

FINANCE_OPTIONS = [0, 2, 4, 5]
FINANCE_LABELS = {
    0: "Comfortable / Secure",
    2: "Tight but manageable",
    4: "Frequent strain",
    5: "Critical financial stress"
}

# ----------------- TAB 1: SURVEY -----------------
with tab_survey:
    step_col1, step_col2 = st.columns([3, 1])
    with step_col1:
        if st.session_state.step == 1:
            st.markdown("### Step 1: Psychological Health & Immediate Symptoms")
            st.markdown("<p style='color: #94a3b8; font-size: 0.88rem; margin-top: -6px;'>Reflect on your emotional and physical well-being over the past 7 days.</p>", unsafe_allow_html=True)
        else:
            st.markdown("### Step 2: Environmental, Social & Academic Pressures")
            st.markdown("<p style='color: #94a3b8; font-size: 0.88rem; margin-top: -6px;'>Answer based on your daily workload, housing environment, and support network.</p>", unsafe_allow_html=True)

    with step_col2:
        st.markdown(f"<p style='text-align: right; color: #cbd5e1; font-size: 0.88rem; font-weight: 600; margin-top: 10px;'>Step {st.session_state.step} of 2</p>", unsafe_allow_html=True)
        st.progress(0.5 if st.session_state.step == 1 else 1.0)

    st.write("")

    # --- STEP 1 ---
    if st.session_state.step == 1:
        c1, c2 = st.columns(2)

        with c1:
            with st.container(border=True):
                st.markdown("**1. Anxiety & Nervous Tension**")
                st.caption("Feeling on edge, restless, or unable to relax:")
                anxiety = st.radio(
                    "Anxiety", options=SEVERITY_OPTIONS,
                    format_func=lambda x: SEVERITY_LABELS[x],
                    index=0, horizontal=True, label_visibility="collapsed", key="anx_q"
                )

            with st.container(border=True):
                st.markdown("**2. Depression & Low Mood**")
                st.caption("Lack of motivation or persistent downhearted feelings:")
                depression = st.radio(
                    "Depression", options=SEVERITY_OPTIONS,
                    format_func=lambda x: SEVERITY_LABELS[x],
                    index=0, horizontal=True, label_visibility="collapsed", key="dep_q"
                )

            with st.container(border=True):
                st.markdown("**3. Tension Headaches**")
                st.caption("Occurrence of tension headaches or physical head pain:")
                headache = st.radio(
                    "Headache", options=FREQUENCY_OPTIONS,
                    format_func=lambda x: FREQUENCY_LABELS[x],
                    index=0, horizontal=True, label_visibility="collapsed", key="hd_q"
                )

        with c2:
            with st.container(border=True):
                st.markdown("**4. Sleep Disruptions & Insomnia**")
                st.caption("Trouble falling asleep, waking up tired, or poor sleep:")
                sleep = st.radio(
                    "Sleep", options=FREQUENCY_OPTIONS,
                    format_func=lambda x: FREQUENCY_LABELS[x],
                    index=0, horizontal=True, label_visibility="collapsed", key="slp_q"
                )

            with st.container(border=True):
                st.markdown("**5. Academic & Study Workload**")
                st.caption("Volume of coursework, assignments, and deadlines:")
                workload = st.radio(
                    "Workload", options=WORKLOAD_OPTIONS,
                    format_func=lambda x: WORKLOAD_LABELS[x],
                    index=0, horizontal=True, label_visibility="collapsed", key="wk_q"
                )

            with st.container(border=True):
                st.markdown("**6. Exam & Assessment Panic**")
                st.caption("Physical nervousness or apprehension before exams:")
                exam = st.radio(
                    "Exam", options=SEVERITY_OPTIONS,
                    format_func=lambda x: SEVERITY_LABELS[x],
                    index=0, horizontal=True, label_visibility="collapsed", key="ex_q"
                )

        st.write("")
        if st.button("Continue to Step 2 ➔", type="primary"):
            st.session_state.survey_inputs.update({
                "anxiety": anxiety,
                "depression": depression,
                "headache_frequency": headache,
                "sleep_deprivation": sleep,
                "academic_workload": workload,
                "exam_panic": exam
            })
            st.session_state.step = 2
            st.rerun()

    # --- STEP 2 ---
    elif st.session_state.step == 2:
        c1, c2 = st.columns(2)

        with c1:
            with st.container(border=True):
                st.markdown("**7. Living Environment Conditions**")
                st.caption("Noise, study privacy, and room comfort:")
                living = st.radio(
                    "Living", options=LIVING_OPTIONS,
                    format_func=lambda x: LIVING_LABELS[x],
                    index=0, horizontal=True, label_visibility="collapsed", key="liv_q"
                )

            with st.container(border=True):
                st.markdown("**8. Social Isolation & Support Network**")
                st.caption("Availability of peers or mentors to discuss difficulties:")
                support = st.radio(
                    "Support", options=SUPPORT_OPTIONS,
                    format_func=lambda x: SUPPORT_LABELS[x],
                    index=0, horizontal=True, label_visibility="collapsed", key="sup_q"
                )

            with st.container(border=True):
                st.markdown("**9. Peer Competition & Pressure**")
                st.caption("Stress caused by academic comparison and rankings:")
                peer = st.radio(
                    "Peer", options=SEVERITY_OPTIONS,
                    format_func=lambda x: SEVERITY_LABELS[x],
                    index=0, horizontal=True, label_visibility="collapsed", key="pr_q"
                )

        with c2:
            with st.container(border=True):
                st.markdown("**10. Career & Placement Anxiety**")
                st.caption("Apprehension regarding graduation, jobs, or internships:")
                career = st.radio(
                    "Career", options=SEVERITY_OPTIONS,
                    format_func=lambda x: SEVERITY_LABELS[x],
                    index=0, horizontal=True, label_visibility="collapsed", key="car_q"
                )

            with st.container(border=True):
                st.markdown("**11. Financial Strain**")
                st.caption("Difficulty managing tuition fees, living expenses, or supplies:")
                finance = st.radio(
                    "Finance", options=FINANCE_OPTIONS,
                    format_func=lambda x: FINANCE_LABELS[x],
                    index=0, horizontal=True, label_visibility="collapsed", key="fin_q"
                )

            with st.container(border=True):
                st.markdown("**12. Physical Exhaustion & Fatigue**")
                st.caption("Feeling physically depleted or drained throughout the day:")
                exhaustion = st.radio(
                    "Exhaustion", options=FREQUENCY_OPTIONS,
                    format_func=lambda x: FREQUENCY_LABELS[x],
                    index=0, horizontal=True, label_visibility="collapsed", key="exh_q"
                )

        st.write("")
        col_back, col_submit = st.columns([1, 4])

        with col_back:
            if st.button("⬅ Back to Step 1"):
                st.session_state.step = 1
                st.rerun()

        with col_submit:
            if st.button("Submit & Run Stress Analysis", type="primary"):
                st.session_state.survey_inputs.update({
                    "poor_living_condition": living,
                    "lack_of_social_support": support,
                    "peer_competition_pressure": peer,
                    "future_career_worry": career,
                    "financial_hardship": finance,
                    "physical_exhaustion": exhaustion
                })

                try:
                    res = requests.post(
                        f"{API_URL}/predict",
                        json={"inputs": st.session_state.survey_inputs}
                    )
                    if res.status_code == 200:
                        st.session_state.prediction_result = res.json()
                    else:
                        st.error(f"Inference error from server: {res.text}")
                except Exception as e:
                    st.error(f"Cannot connect to FastAPI: {e}")

    # --- RESULTS & PERSONALIZED INSIGHTS ---
    if st.session_state.prediction_result:
        res = st.session_state.prediction_result
        label = res["prediction"]
        conf = res["confidence"]

        st.write("---")
        st.subheader("🎯 Assessment Findings")

        r1, r2 = st.columns([1, 2])
        with r1:
            color = "#ef4444" if label == "High" else ("#f59e0b" if label == "Moderate" else "#10b981")
            st.markdown(f"""
                <div style='background-color: #111827; border: 1px solid #1f293d; border-radius: 10px; padding: 1.6rem; text-align: center;'>
                    <p style='color: #94a3b8; font-size: 0.85rem; text-transform: uppercase; margin: 0; font-weight: 600;'>PREDICTED STATE</p>
                    <h2 style='color: {color} !important; font-size: 2.2rem; margin: 0.4rem 0;'>{label} Stress</h2>
                    <p style='color: #cbd5e1; font-size: 0.9rem; margin: 0;'>Model Confidence: <b style='color: #ffffff;'>{conf}%</b></p>
                </div>
            """, unsafe_allow_html=True)

        with r2:
            inputs = st.session_state.survey_inputs
            psych_score = (inputs.get("anxiety", 0) + inputs.get("depression", 0) + inputs.get("exam_panic", 0)) / 3
            physical_score = (inputs.get("sleep_deprivation", 0) + inputs.get("physical_exhaustion", 0) + inputs.get("headache_frequency", 0)) / 3
            academic_score = (inputs.get("academic_workload", 0) + inputs.get("peer_competition_pressure", 0)) / 2
            env_score = (inputs.get("poor_living_condition", 0) + inputs.get("lack_of_social_support", 0) + inputs.get("financial_hardship", 0) + inputs.get("future_career_worry", 0)) / 4

            breakdown_df = pd.DataFrame({
                "Dimension": ["Psychological", "Physical / Sleep", "Academic", "Environmental"],
                "Severity (0-5)": [round(psych_score, 1), round(physical_score, 1), round(academic_score, 1), round(env_score, 1)]
            }).set_index("Dimension")
            st.bar_chart(breakdown_df, height=190)

        # Action Recommendations
        st.write("")
        st.subheader("💡 Tailored Interventions")

        if physical_score >= 2.5:
            with st.container(border=True):
                st.markdown("**🩺 Sleep Hygiene & Somatic Recovery**")
                st.caption("Identified elevated sleep disruptions or physical exhaustion.")
                st.write("- Maintain a consistent 7-8 hour sleep schedule; stop screen use 45 minutes before sleep.\n- Ensure steady daily hydration to alleviate tension headaches.")

        if academic_score >= 2.5:
            with st.container(border=True):
                st.markdown("**📚 Coursework & Time Management**")
                st.caption("Identified elevated study workload and competitive peer pressure.")
                st.write("- Use 25-minute Pomodoro focus blocks with 5-minute pauses.\n- Prioritize personal progress milestones rather than peer rankings.")

        if psych_score >= 2.5:
            with st.container(border=True):
                st.markdown("**🧠 Emotional Well-Being & De-escalation**")
                st.caption("Identified persistent nervous tension or mood friction.")
                st.write("- Schedule time with student counseling or campus wellness advisory.\n- Reserve 20 minutes daily for screen-free mental downtime.")

        if physical_score < 2.5 and academic_score < 2.5 and psych_score < 2.5:
            with st.container(border=True):
                st.markdown("**🌱 Healthy Routine Maintenance**")
                st.write("- All parameters are balanced. Keep maintaining your regular routine and daily habits.")

# ----------------- TAB 2: ANALYTICS -----------------
with tab_analytics:
    st.subheader("System Telemetry & Database Records (SQLite)")
    try:
        stats = requests.get(f"{API_URL}/stats").json()
        total = stats.get("total", 0)
        dist = stats.get("distribution", {})

        m1, m2 = st.columns(2)
        m1.metric("Evaluations Stored", total)
        m2.metric("Inference Engine", "Random Forest (Scikit-Learn)")

        if dist:
            st.write("##### Population Stress Distribution")
            chart_df = pd.DataFrame(
                list(dist.items()),
                columns=["Category", "Count"]
            ).set_index("Category")
            st.bar_chart(chart_df)
        else:
            st.info("No records in database yet.")
    except Exception:
        st.warning("Could not reach FastAPI backend.")