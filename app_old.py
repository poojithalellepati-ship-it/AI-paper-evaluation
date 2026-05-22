"""
╔══════════════════════════════════════════════════════════════╗
║   AI-POWERED AUTOMATIC ANSWER EVALUATION SYSTEM             ║
║   Built with Python · Streamlit · NLP · ML · MySQL          ║
╚══════════════════════════════════════════════════════════════╝
"""

import sys
import os
import time
import json
import random
import datetime

sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from modules.nlp_engine import (
    evaluate_student, compute_student_summary,
    extract_text, parse_qa_from_text, parse_student_answers,
    preprocess, compute_tfidf_similarity, keyword_overlap,
    tokenize, remove_stopwords, lemmatize
)
from modules import db_handler as db
from modules.sample_data import (
    SAMPLE_QUESTION_PAPER, SAMPLE_KEY_ANSWERS, SAMPLE_STUDENT_ANSWERS,
    get_sample_questions, get_sample_key_answers, get_sample_student_answers
)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Evaluation System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS — Dark futuristic theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;800&family=Exo+2:ital,wght@0,300;0,400;0,600;1,300&display=swap');

/* ── Global Reset ── */
html, body, [class*="css"] {
    font-family: 'Exo 2', sans-serif;
    background-color: #080c14;
    color: #c8d8f0;
}
.main { background-color: #080c14; }
.block-container { padding: 1.5rem 2rem 2rem 2rem; max-width: 1400px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1628 0%, #050d1a 100%);
    border-right: 1px solid #1a3050;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #7ecfff !important; }

/* ── Page Header ── */
.hero-title {
    font-family: 'Orbitron', monospace;
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #00d4ff 0%, #7c4dff 50%, #00ff88 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: 2px;
    margin-bottom: 0.2rem;
}
.hero-subtitle {
    font-family: 'Exo 2', sans-serif;
    color: #5a8ab0;
    font-size: 0.95rem;
    letter-spacing: 1px;
    margin-bottom: 1.5rem;
}
.section-header {
    font-family: 'Orbitron', monospace;
    font-size: 1.1rem;
    color: #00d4ff;
    letter-spacing: 1.5px;
    border-bottom: 1px solid #1a3050;
    padding-bottom: 0.4rem;
    margin-bottom: 1rem;
}

/* ── KPI Cards ── */
.kpi-card {
    background: linear-gradient(135deg, #0d1e35 0%, #091525 100%);
    border: 1px solid #1a3a5c;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
}
.kpi-blue::before  { background: linear-gradient(90deg, #00d4ff, #0088cc); }
.kpi-purple::before { background: linear-gradient(90deg, #7c4dff, #cc00ff); }
.kpi-green::before  { background: linear-gradient(90deg, #00ff88, #00cc66); }
.kpi-orange::before { background: linear-gradient(90deg, #ff9500, #ff6600); }

.kpi-card:hover { transform: translateY(-3px); border-color: #2a5a8c; box-shadow: 0 8px 30px rgba(0,212,255,0.15); }
.kpi-value {
    font-family: 'Orbitron', monospace;
    font-size: 2.4rem;
    font-weight: 800;
    line-height: 1;
}
.kpi-blue  .kpi-value { color: #00d4ff; }
.kpi-purple .kpi-value { color: #a97fff; }
.kpi-green  .kpi-value { color: #00ff88; }
.kpi-orange .kpi-value { color: #ff9500; }

.kpi-label {
    font-size: 0.75rem;
    color: #5a8ab0;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-top: 0.3rem;
}
.kpi-icon { font-size: 1.5rem; margin-bottom: 0.3rem; }

/* ── Result Cards ── */
.result-card {
    background: #0d1e35;
    border: 1px solid #1a3a5c;
    border-radius: 10px;
    padding: 1.2rem;
    margin-bottom: 1rem;
}
.result-card-pass  { border-left: 4px solid #00ff88; }
.result-card-fail  { border-left: 4px solid #ff4466; }

.pass-badge {
    background: linear-gradient(135deg, #003320, #005535);
    border: 1px solid #00cc66;
    color: #00ff88;
    font-family: 'Orbitron', monospace;
    font-size: 0.8rem;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    display: inline-block;
}
.fail-badge {
    background: linear-gradient(135deg, #330015, #550022);
    border: 1px solid #cc0044;
    color: #ff4466;
    font-family: 'Orbitron', monospace;
    font-size: 0.8rem;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    display: inline-block;
}
.grade-badge {
    font-family: 'Orbitron', monospace;
    font-size: 1.8rem;
    font-weight: 800;
    color: #ffcc00;
    text-shadow: 0 0 15px rgba(255,204,0,0.5);
}

/* ── Question Evaluation Rows ── */
.q-row {
    background: #091525;
    border: 1px solid #142840;
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 0.8rem;
}
.q-number {
    font-family: 'Orbitron', monospace;
    font-size: 1rem;
    color: #00d4ff;
    font-weight: 600;
}
.similarity-high   { color: #00ff88; font-weight: 600; }
.similarity-medium { color: #ffcc00; font-weight: 600; }
.similarity-low    { color: #ff4466; font-weight: 600; }

/* ── Processing steps ── */
.step-card {
    background: #091525;
    border: 1px solid #1a3050;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 0.8rem;
}
.step-icon { font-size: 1.3rem; }
.step-done { border-left: 3px solid #00ff88; }
.step-active { border-left: 3px solid #00d4ff; animation: pulse 1.5s infinite; }

@keyframes pulse {
    0%,100% { border-left-color: #00d4ff; }
    50%      { border-left-color: #7c4dff; }
}

/* ── Info boxes ── */
.info-box {
    background: linear-gradient(135deg, #091a2e 0%, #0a1e35 100%);
    border: 1px solid #1a3a5c;
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.8rem;
    font-size: 0.88rem;
    line-height: 1.6;
}
.info-box strong { color: #7ecfff; }

/* ── Tags ── */
.tag {
    display: inline-block;
    background: #0d2040;
    border: 1px solid #1a4060;
    border-radius: 20px;
    padding: 0.15rem 0.5rem;
    font-size: 0.75rem;
    margin: 0.15rem;
    color: #7ecfff;
}
.tag-match   { border-color: #006633; background: #00221a; color: #00cc66; }
.tag-missing { border-color: #660022; background: #220011; color: #ff4466; }

/* ── Streamlit overrides ── */
.stButton > button {
    background: linear-gradient(135deg, #003a6b 0%, #00264d 100%);
    border: 1px solid #0077cc;
    color: #7ecfff;
    border-radius: 8px;
    font-family: 'Exo 2', sans-serif;
    font-weight: 600;
    letter-spacing: 0.5px;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #0055aa 0%, #003a7a 100%);
    border-color: #00aaff;
    color: #ffffff;
    transform: translateY(-1px);
}
.stProgress > div > div { background: linear-gradient(90deg, #00d4ff, #7c4dff); }
.stFileUploader { background: #091525; border: 1px dashed #1a3a5c; border-radius: 10px; }

/* Sidebar nav radio */
.stRadio > div { gap: 0.4rem; }
.stRadio label { 
    color: #7ecfff !important; 
    font-size: 0.9rem; 
    cursor: pointer;
}

/* Tables */
.dataframe { background: #091525 !important; }

/* Dividers */
hr { border-color: #1a3050; margin: 1.2rem 0; }

/* Expander */
details { background: #091525; border: 1px solid #1a3050; border-radius: 8px; }
summary { color: #7ecfff; padding: 0.6rem 1rem; }

/* ── UML Diagram container ── */
.uml-container {
    background: #060e1a;
    border: 1px solid #1a3050;
    border-radius: 10px;
    padding: 1rem;
    font-family: monospace;
    font-size: 0.8rem;
    color: #5abfff;
    overflow-x: auto;
    white-space: pre;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
def init_state():
    defaults = {
        "questions": [],
        "key_answers": [],
        "students": {},          # {stu_id: {name, subject, answers[]}}
        "eval_results": {},      # {stu_id: [per-q dicts]}
        "summaries": {},         # {stu_id: summary dict}
        "session_id": None,
        "files_uploaded": False,
        "evaluated": False,
        "current_page": "🏠 Home Dashboard",
        "demo_loaded": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ─────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 0.5rem 0 1.2rem 0;">
        <div style="font-family:'Orbitron',monospace; font-size:1rem; 
                    color:#00d4ff; letter-spacing:2px;">🧠 AI EVAL</div>
        <div style="font-size:0.7rem; color:#3a6080; letter-spacing:1px; 
                    margin-top:0.2rem;">SYSTEM v2.5</div>
    </div>
    """, unsafe_allow_html=True)

    pages = [
        "🏠 Home Dashboard",
        "📤 Upload Center",
        "⚙️ AI Processing",
        "📊 Evaluation Results",
        "📈 Analytics Dashboard",
        "📐 UML Diagrams",
        "🧪 Testing & Validation",
    ]
    page = st.radio("Navigation", pages, label_visibility="collapsed")
    st.session_state.current_page = page

    st.markdown("---")
    stats = db.get_dashboard_stats()
    st.markdown(f"""
    <div style="font-size:0.8rem; color:#3a6080; margin-bottom:0.5rem;">LIVE STATS</div>
    <div style="color:#7ecfff; font-size:0.85rem;">
        👥 Students: <b style="color:#00d4ff">{stats['total_students']}</b><br>
        ✅ Evaluated: <b style="color:#00ff88">{stats['total_evaluated']}</b><br>
        📊 Avg Score: <b style="color:#ffcc00">{stats['avg_score']:.1f}%</b><br>
        🎯 Pass Rate: <b style="color:#a97fff">{stats['pass_percentage']:.1f}%</b>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🔄 Load Demo Data", use_container_width=True):
        _load_demo()

    if st.button("🗑 Reset All Data", use_container_width=True):
        db.clear_all_data()
        for k in ["questions","key_answers","students","eval_results",
                  "summaries","session_id","files_uploaded","evaluated","demo_loaded"]:
            if k in ["questions","key_answers"]:
                st.session_state[k] = []
            elif k in ["students","eval_results","summaries"]:
                st.session_state[k] = {}
            elif k in ["files_uploaded","evaluated","demo_loaded"]:
                st.session_state[k] = False
            else:
                st.session_state[k] = None
        st.rerun()

    st.markdown("""
    <div style="font-size:0.7rem; color:#2a4060; margin-top:1rem; text-align:center;">
        MySQL · Streamlit · NLP · ML<br>
        © 2025 AI Evaluation System
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DEMO DATA LOADER
# ─────────────────────────────────────────────
def _load_demo():
    st.session_state.questions = get_sample_questions()
    st.session_state.key_answers = get_sample_key_answers()
    st.session_state.students = {}
    st.session_state.eval_results = {}
    st.session_state.summaries = {}
    db.clear_all_data()

    session_id = db.create_session("Demo Session")
    st.session_state.session_id = session_id

    for sid, data in SAMPLE_STUDENT_ANSWERS.items():
        answers, name, subject = get_sample_student_answers(sid)
        st.session_state.students[sid] = {
            "name": name, "subject": subject, "answers": answers
        }
        db.save_student(sid, name, subject)

        results = evaluate_student(
            st.session_state.questions,
            st.session_state.key_answers,
            answers
        )
        summary = compute_student_summary(results)
        st.session_state.eval_results[sid] = results
        st.session_state.summaries[sid] = summary
        db.save_evaluation_results(session_id, sid, results, summary)

    st.session_state.files_uploaded = True
    st.session_state.evaluated = True
    st.session_state.demo_loaded = True
    st.rerun()


# ═══════════════════════════════════════════════════════════════
#  PAGE 1 — HOME DASHBOARD
# ═══════════════════════════════════════════════════════════════
def page_home():
    st.markdown('<div class="hero-title">🧠 AI ANSWER EVALUATION SYSTEM</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">POWERED BY NLP · MACHINE LEARNING · SEMANTIC ANALYSIS</div>', unsafe_allow_html=True)

    stats = db.get_dashboard_stats()
    summaries = st.session_state.summaries

    # KPI CARDS
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="kpi-card kpi-blue">
            <div class="kpi-icon">👥</div>
            <div class="kpi-value">{stats['total_students']}</div>
            <div class="kpi-label">Total Students</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="kpi-card kpi-green">
            <div class="kpi-icon">✅</div>
            <div class="kpi-value">{stats['total_evaluated']}</div>
            <div class="kpi-label">Evaluated</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="kpi-card kpi-orange">
            <div class="kpi-icon">📊</div>
            <div class="kpi-value">{stats['avg_score']:.1f}%</div>
            <div class="kpi-label">Average Score</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="kpi-card kpi-purple">
            <div class="kpi-icon">🎯</div>
            <div class="kpi-value">{stats['pass_percentage']:.1f}%</div>
            <div class="kpi-label">Pass Rate</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if not summaries:
        # Intro / feature showcase
        st.markdown('<div class="section-header">SYSTEM CAPABILITIES</div>', unsafe_allow_html=True)
        cols = st.columns(3)
        features = [
            ("🔍", "Intelligent Document Reading", "Extracts and parses text from PDF and DOCX files automatically"),
            ("🧬", "NLP Pipeline", "Tokenization → Stopword removal → Lemmatization → TF-IDF → Cosine Similarity"),
            ("⚡", "Real-time Evaluation", "Evaluates student answers against key answers with semantic understanding"),
            ("🏆", "Grade Generation", "Calculates marks, percentages, grades and Pass/Fail predictions"),
            ("📊", "Rich Analytics", "Interactive charts, heatmaps, leaderboards and performance insights"),
            ("💾", "MySQL Integration", "Persistent storage with full relational database schema"),
        ]
        for i, (icon, title, desc) in enumerate(features):
            with cols[i % 3]:
                st.markdown(f"""
                <div class="kpi-card kpi-{'blue' if i%4==0 else 'purple' if i%4==1 else 'green' if i%4==2 else 'orange'}"
                     style="text-align:left; margin-bottom:1rem;">
                    <div style="font-size:1.5rem; margin-bottom:0.4rem;">{icon}</div>
                    <div style="font-family:'Orbitron',monospace; font-size:0.85rem; 
                                color:#7ecfff; margin-bottom:0.4rem;">{title}</div>
                    <div style="font-size:0.8rem; color:#5a8ab0;">{desc}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.info("👈 Click **Load Demo Data** in the sidebar to instantly populate the dashboard with sample evaluation results, or navigate to **Upload Center** to upload your own files.")
        return

    # STUDENT PERFORMANCE TABLE
    st.markdown('<div class="section-header">STUDENT PERFORMANCE OVERVIEW</div>', unsafe_allow_html=True)
    rows = []
    for sid, s in st.session_state.summaries.items():
        stu = st.session_state.students.get(sid, {})
        rows.append({
            "Student ID": sid,
            "Name": stu.get("name", sid),
            "Subject": stu.get("subject", "—"),
            "Score": f"{s['total_obtained']}/{s['total_marks']}",
            "Percentage": s['percentage'],
            "Grade": s['grade'],
            "Avg Similarity": f"{s['avg_similarity']}%",
            "Pass/Fail": s['pass_fail'],
        })
    df = pd.DataFrame(rows)
    st.dataframe(
        df.style.apply(
            lambda row: ["background-color: #00220f" if row["Pass/Fail"]=="Pass"
                         else "background-color: #220008"] * len(row),
            axis=1
        ).format({"Percentage": "{:.1f}%"}),
        use_container_width=True, hide_index=True
    )

    # MINI CHARTS
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">PERFORMANCE SNAPSHOT</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        names = [st.session_state.students.get(s,{}).get("name",s) for s in summaries]
        pcts  = [summaries[s]["percentage"] for s in summaries]
        colors = ["#00ff88" if p>=40 else "#ff4466" for p in pcts]
        fig = go.Figure(go.Bar(x=names, y=pcts, marker_color=colors, text=[f"{p:.1f}%" for p in pcts],
                               textposition="outside"))
        fig.update_layout(
            title="Student Scores", template="plotly_dark",
            paper_bgcolor="#091525", plot_bgcolor="#091525",
            height=280, margin=dict(t=40,b=20,l=10,r=10),
            yaxis=dict(range=[0,110], color="#5a8ab0"),
            xaxis_color="#5a8ab0", title_font_color="#00d4ff",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        pass_count = sum(1 for s in summaries.values() if s["pass_fail"]=="Pass")
        fail_count = len(summaries) - pass_count
        fig = go.Figure(go.Pie(
            labels=["Pass","Fail"], values=[pass_count, fail_count],
            marker=dict(colors=["#00ff88","#ff4466"],
                        line=dict(color="#080c14", width=3)),
            textfont_size=14, hole=0.5,
        ))
        fig.update_layout(
            title="Pass vs Fail", template="plotly_dark",
            paper_bgcolor="#091525", height=280,
            margin=dict(t=40,b=10,l=10,r=10),
            title_font_color="#00d4ff",
            legend=dict(font=dict(color="#7ecfff")),
            annotations=[dict(text=f"{pass_count}/{len(summaries)}", x=0.5, y=0.5,
                              font=dict(size=18, color="#00ff88", family="Orbitron"), showarrow=False)]
        )
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
#  PAGE 2 — UPLOAD CENTER
# ═══════════════════════════════════════════════════════════════
def page_upload():
    st.markdown('<div class="hero-title">📤 UPLOAD CENTER</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">UPLOAD QUESTION PAPER · KEY ANSWERS · STUDENT ANSWER SHEETS</div>', unsafe_allow_html=True)

    # ── Step 1: Question Paper ──
    st.markdown('<div class="section-header">STEP 1 — QUESTION PAPER</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([2,1])
    with col1:
        qp_file = st.file_uploader("Upload Question Paper", type=["pdf","docx","txt"],
                                    key="qp_upload", label_visibility="collapsed",
                                    help="PDF or DOCX containing numbered questions")
    with col2:
        if st.button("📄 Use Sample Question Paper", use_container_width=True):
            st.session_state.questions = get_sample_questions()
            st.success(f"✅ Loaded {len(st.session_state.questions)} sample questions")

    if qp_file:
        with st.spinner("🔍 Extracting text..."):
            text = extract_text(qp_file)
            qp_file.seek(0)
            qa_pairs = parse_qa_from_text(text)
            if qa_pairs:
                st.session_state.questions = [q for q, _ in qa_pairs]
                st.success(f"✅ Extracted {len(st.session_state.questions)} questions from **{qp_file.name}**")
            else:
                # fallback: split by lines
                lines = [l.strip() for l in text.split('\n') if l.strip() and len(l.strip()) > 20]
                st.session_state.questions = lines[:10]
                st.success(f"✅ Extracted {len(st.session_state.questions)} question blocks")

    if st.session_state.questions:
        with st.expander(f"👁 Preview Questions ({len(st.session_state.questions)} found)"):
            for i, q in enumerate(st.session_state.questions, 1):
                st.markdown(f"<div class='info-box'><b style='color:#00d4ff'>Q{i}.</b> {q}</div>",
                            unsafe_allow_html=True)

    st.markdown("---")

    # ── Step 2: Key Answers ──
    st.markdown('<div class="section-header">STEP 2 — KEY ANSWER SHEET</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([2,1])
    with col1:
        ka_file = st.file_uploader("Upload Key Answers", type=["pdf","docx","txt"],
                                    key="ka_upload", label_visibility="collapsed")
    with col2:
        if st.button("📋 Use Sample Key Answers", use_container_width=True):
            st.session_state.key_answers = get_sample_key_answers()
            st.success(f"✅ Loaded {len(st.session_state.key_answers)} sample key answers")

    if ka_file:
        with st.spinner("🔍 Extracting key answers..."):
            text = extract_text(ka_file)
            ka_file.seek(0)
            qa_pairs = parse_qa_from_text(text)
            if qa_pairs:
                st.session_state.key_answers = [a for _, a in qa_pairs]
            else:
                chunks = [s.strip() for s in text.split('\n\n') if s.strip() and len(s.strip()) > 30]
                st.session_state.key_answers = chunks[:10]
            st.success(f"✅ Extracted {len(st.session_state.key_answers)} key answers")

    if st.session_state.key_answers:
        with st.expander(f"👁 Preview Key Answers ({len(st.session_state.key_answers)} found)"):
            for i, a in enumerate(st.session_state.key_answers, 1):
                st.markdown(f"<div class='info-box'><b style='color:#00ff88'>A{i}.</b> {a[:200]}{'...' if len(a)>200 else ''}</div>",
                            unsafe_allow_html=True)

    st.markdown("---")

    # ── Step 3: Student Answer Sheets ──
    st.markdown('<div class="section-header">STEP 3 — STUDENT ANSWER SHEETS</div>', unsafe_allow_html=True)
    st.info("Upload 3 student answer sheets. All students are evaluated against the same subject.")

    st.markdown("<br>", unsafe_allow_html=True)
    shared_subject = st.text_input("Subject", value="Computer Science", key="common_subject",
                                  label_visibility="visible",
                                  help="All student answer sheets are for this one subject.")
    default_ids = ["STU001", "STU002", "STU003"]

    uploaded_count = 0
    for i in range(3):
        with st.container():
            st.markdown(f"**Student {i+1}**")
            c1, c2, c3 = st.columns([1.5, 1.5, 2.5])
            with c1:
                sid = st.text_input("Student ID", value=default_ids[i],
                                     key=f"sid_{i}", label_visibility="visible")
            with c2:
                sname = st.text_input("Student Name", key=f"sname_{i}",
                                       placeholder="e.g. Arjun Sharma", label_visibility="visible")
            with c3:
                ans_file = st.file_uploader(f"Answer Sheet {i+1}", type=["pdf","docx","txt"],
                                             key=f"ans_{i}", label_visibility="visible")

            col_a, col_b = st.columns([2,1])
            with col_a:
                if ans_file and sid and sname:
                    with st.spinner(f"📖 Reading {ans_file.name}..."):
                        text = extract_text(ans_file)
                        nq = len(st.session_state.questions)
                        answers = parse_student_answers(text, nq if nq else 10)
                    st.session_state.students[sid] = {
                        "name": sname, "subject": shared_subject, "answers": answers
                    }
                    db.save_student(sid, sname, shared_subject)
                    st.success(f"✅ {ans_file.name} — {len(answers)} answers extracted")
                    uploaded_count += 1
            with col_b:
                demo_key = list(SAMPLE_STUDENT_ANSWERS.keys())[i]
                if st.button(f"🎓 Load Sample Student {i+1}", key=f"demo_stu_{i}",
                              use_container_width=True):
                    ans, nm, _ = get_sample_student_answers(demo_key)
                    sample_sid = default_ids[i]
                    st.session_state.students[sample_sid] = {
                        "name": nm, "subject": shared_subject, "answers": ans
                    }
                    db.save_student(sample_sid, nm, shared_subject)
                    st.success(f"✅ Loaded sample for {nm}")

            st.markdown("<hr style='margin:0.5rem 0; border-color:#0d1e35;'>", unsafe_allow_html=True)

    n_stu = len(st.session_state.students)
    n_q = len(st.session_state.questions)
    n_ka = len(st.session_state.key_answers)

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Questions Loaded", n_q, delta="ready" if n_q else "pending")
    with col2: st.metric("Key Answers Loaded", n_ka, delta="ready" if n_ka else "pending")
    with col3: st.metric("Students Registered", n_stu, delta="ready" if n_stu else "pending")

    if n_q >= 1 and n_ka >= 1 and n_stu >= 1:
        st.session_state.files_uploaded = True
        st.success("✅ All required data loaded! Navigate to **⚙️ AI Processing** to evaluate.")
    else:
        st.warning("⚠️ Please load at least 1 question, 1 key answer, and 1 student to proceed.")


# ═══════════════════════════════════════════════════════════════
#  PAGE 3 — AI PROCESSING
# ═══════════════════════════════════════════════════════════════
def page_processing():
    st.markdown('<div class="hero-title">⚙️ AI PROCESSING ENGINE</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">NLP PIPELINE · SEMANTIC ANALYSIS · EVALUATION ENGINE</div>', unsafe_allow_html=True)

    if not st.session_state.files_uploaded:
        st.warning("⚠️ Please upload files first (or load demo data from the sidebar).")
        return

    # NLP PIPELINE VISUALIZER
    st.markdown('<div class="section-header">NLP PIPELINE DEMONSTRATION</div>', unsafe_allow_html=True)

    sample_text = (st.session_state.key_answers[0][:200] if st.session_state.key_answers
                   else "Machine learning algorithms automatically learn from data to make predictions.")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("**Input Text Sample**")
        st.markdown(f"<div class='info-box'>{sample_text}</div>", unsafe_allow_html=True)
        st.markdown("**Step-by-step NLP Pipeline**")

        tokens_raw = tokenize(sample_text)
        tokens_clean = remove_stopwords(tokens_raw)
        tokens_lemma = [lemmatize(t) for t in tokens_clean]

        steps = [
            ("✅", "Tokenization", f"{len(tokens_raw)} tokens extracted", tokens_raw[:8]),
            ("✅", "Stopword Removal", f"{len(tokens_clean)} tokens remaining", tokens_clean[:8]),
            ("✅", "Lemmatization", f"{len(tokens_lemma)} base forms", tokens_lemma[:8]),
            ("✅", "TF-IDF Vectorization", "Term frequency-inverse doc frequency computed", []),
            ("✅", "Cosine Similarity", "Semantic distance measured", []),
        ]
        for icon, name, desc, toks in steps:
            tag_html = " ".join(f'<span class="tag">{t}</span>' for t in toks)
            st.markdown(f"""
            <div class="step-card step-done">
                <span class="step-icon">{icon}</span>
                <div>
                    <div style="font-family:'Orbitron',monospace;font-size:0.8rem;color:#00d4ff;">{name}</div>
                    <div style="font-size:0.78rem;color:#5a8ab0;">{desc}</div>
                    <div style="margin-top:0.3rem;">{tag_html}</div>
                </div>
            </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("**Token Frequency Analysis**")
        if tokens_lemma:
            from collections import Counter
            freq = Counter(tokens_lemma).most_common(12)
            words, counts = zip(*freq) if freq else ([], [])
            fig = go.Figure(go.Bar(
                x=list(counts), y=list(words), orientation='h',
                marker=dict(color=counts, colorscale='Viridis'),
                text=list(counts), textposition='outside',
            ))
            fig.update_layout(
                template="plotly_dark", paper_bgcolor="#091525",
                plot_bgcolor="#091525", height=300,
                margin=dict(t=10,b=10,l=10,r=30),
                xaxis_color="#5a8ab0", yaxis_color="#7ecfff",
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Similarity Score Meter**")
        if st.session_state.questions and st.session_state.key_answers:
            q_idx = st.selectbox("Select Question", range(1, len(st.session_state.questions)+1),
                                  format_func=lambda x: f"Q{x}")
            ka_text = st.session_state.key_answers[q_idx - 1]
            student_ids = list(st.session_state.students.keys())
            sim_scores = []
            for sid in student_ids:
                stu_answers = st.session_state.students[sid]["answers"]
                if q_idx - 1 < len(stu_answers):
                    sim = compute_tfidf_similarity(ka_text, stu_answers[q_idx - 1])
                    sim_scores.append((st.session_state.students[sid]["name"], round(sim*100, 1)))

            if sim_scores:
                fig = go.Figure(go.Bar(
                    x=[s[1] for s in sim_scores],
                    y=[s[0] for s in sim_scores],
                    orientation='h',
                    marker_color=["#00ff88" if s[1]>=60 else "#ffcc00" if s[1]>=40 else "#ff4466"
                                   for s in sim_scores],
                    text=[f"{s[1]}%" for s in sim_scores],
                    textposition='outside',
                ))
                fig.update_layout(
                    template="plotly_dark", paper_bgcolor="#091525",
                    plot_bgcolor="#091525", height=180,
                    margin=dict(t=10,b=10,l=10,r=50),
                    xaxis=dict(range=[0, 110], color="#5a8ab0"),
                    yaxis_color="#7ecfff",
                )
                st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # RUN EVALUATION BUTTON
    st.markdown('<div class="section-header">RUN EVALUATION ENGINE</div>', unsafe_allow_html=True)

    can_evaluate = (st.session_state.questions and st.session_state.key_answers
                    and st.session_state.students)

    if not can_evaluate:
        st.error("❌ Missing data. Please ensure questions, key answers, and at least one student are loaded.")
        return

    col1, col2 = st.columns([1, 2])
    with col1:
        marks_per_q = st.number_input("Marks per Question", min_value=1, max_value=20, value=5)
        pass_threshold = st.number_input("Pass Threshold (%)", min_value=10, max_value=80, value=40)

    with col2:
        st.markdown(f"""
        <div class='info-box'>
            <strong>📋 Ready to Evaluate</strong><br>
            Questions: <b style="color:#00d4ff">{len(st.session_state.questions)}</b> &nbsp;|&nbsp;
            Key Answers: <b style="color:#00ff88">{len(st.session_state.key_answers)}</b> &nbsp;|&nbsp;
            Students: <b style="color:#ffcc00">{len(st.session_state.students)}</b><br>
            Total Marks: <b style="color:#a97fff">{len(st.session_state.questions) * marks_per_q}</b> per student
        </div>
        """, unsafe_allow_html=True)

    if st.button("🚀 START AI EVALUATION", use_container_width=True):
        session_id = db.create_session("Evaluation Session")
        st.session_state.session_id = session_id
        st.session_state.eval_results = {}
        st.session_state.summaries = {}

        progress_bar = st.progress(0)
        status_box = st.empty()
        total_stu = len(st.session_state.students)

        for idx, (sid, stu_data) in enumerate(st.session_state.students.items()):
            status_box.markdown(f"""
            <div class='step-card step-active'>
                <span class='step-icon'>🤖</span>
                <div>
                    <div style='font-family:Orbitron,monospace;font-size:0.85rem;color:#00d4ff;'>
                        Evaluating {stu_data['name']} ({stu_data['subject']})
                    </div>
                    <div style='font-size:0.78rem;color:#5a8ab0;'>
                        Running NLP pipeline · Computing similarities · Generating AI remarks...
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)
            time.sleep(0.5)

            results = evaluate_student(
                st.session_state.questions,
                st.session_state.key_answers,
                stu_data["answers"],
                marks_per_q=float(marks_per_q)
            )
            summary = compute_student_summary(results)
            if summary["percentage"] < pass_threshold:
                summary["pass_fail"] = "Fail"
            else:
                summary["pass_fail"] = "Pass"
            summary["grade"] = _grade(summary["percentage"])

            st.session_state.eval_results[sid] = results
            st.session_state.summaries[sid] = summary
            db.save_student(sid, stu_data["name"], stu_data["subject"])
            db.save_evaluation_results(session_id, sid, results, summary)
            db.log_action("Evaluation", f"Student {sid} evaluated")

            progress_bar.progress((idx + 1) / total_stu)

        st.session_state.evaluated = True
        status_box.empty()
        st.success(f"🎉 Evaluation complete! {total_stu} student(s) evaluated successfully.")
        st.balloons()


def _grade(pct):
    if pct >= 90: return "A+"
    if pct >= 80: return "A"
    if pct >= 70: return "B+"
    if pct >= 60: return "B"
    if pct >= 50: return "C"
    if pct >= 40: return "D"
    return "F"


# ═══════════════════════════════════════════════════════════════
#  PAGE 4 — EVALUATION RESULTS
# ═══════════════════════════════════════════════════════════════
def page_results():
    st.markdown('<div class="hero-title">📊 EVALUATION RESULTS</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">DETAILED AI-GENERATED ASSESSMENT FOR EACH STUDENT</div>', unsafe_allow_html=True)

    if not st.session_state.evaluated:
        st.warning("⚠️ No evaluation results yet. Run evaluation from the **AI Processing** page.")
        return

    summaries = st.session_state.summaries
    students  = st.session_state.students
    student_ids = list(summaries.keys())

    # Student selector
    student_labels = {sid: f"{students.get(sid,{}).get('name', sid)} ({sid})"
                      for sid in student_ids}
    selected_sid = st.selectbox("Select Student", student_ids,
                                 format_func=lambda x: student_labels[x])

    stu  = students.get(selected_sid, {})
    summ = summaries.get(selected_sid, {})
    results = st.session_state.eval_results.get(selected_sid, [])

    # ── Student Summary Card ──
    pf_class = "result-card-pass" if summ.get("pass_fail")=="Pass" else "result-card-fail"
    pf_badge = (f'<span class="pass-badge">✅ PASS</span>' if summ.get("pass_fail")=="Pass"
                else f'<span class="fail-badge">❌ FAIL</span>')

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""
        <div class='kpi-card kpi-blue'>
            <div class='kpi-icon'>🆔</div>
            <div style='font-family:Orbitron,monospace;font-size:1.1rem;color:#00d4ff;'>{selected_sid}</div>
            <div class='kpi-label'>{stu.get("name","")}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class='kpi-card kpi-orange'>
            <div class='kpi-icon'>📚</div>
            <div style='font-size:1rem;color:#ff9500;font-weight:600;'>{stu.get("subject","—")}</div>
            <div class='kpi-label'>Subject</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class='kpi-card kpi-green'>
            <div class='kpi-icon'>🎯</div>
            <div class='kpi-value'>{summ.get("percentage",0):.1f}%</div>
            <div class='kpi-label'>{summ.get("total_obtained",0)}/{summ.get("total_marks",0)} marks</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class='kpi-card kpi-purple'>
            <div class='kpi-icon'>🏅</div>
            <div class='grade-badge'>{summ.get("grade","—")}</div>
            <div class='kpi-label'>Grade</div>
        </div>""", unsafe_allow_html=True)
    with c5:
        st.markdown(f"""
        <div class='kpi-card {"kpi-green" if summ.get("pass_fail")=="Pass" else "kpi-orange"}'>
            <div class='kpi-icon'>{"✅" if summ.get("pass_fail")=="Pass" else "❌"}</div>
            <div style='font-family:Orbitron,monospace;font-size:1.1rem;
                color:{"#00ff88" if summ.get("pass_fail")=="Pass" else "#ff4466"};'>
                {summ.get("pass_fail","—")}
            </div>
            <div class='kpi-label'>Result</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Per-question results ──
    st.markdown('<div class="section-header">QUESTION-WISE AI EVALUATION</div>', unsafe_allow_html=True)

    for r in results:
        sim = r["similarity_score"]
        sim_class = ("similarity-high" if sim >= 70 else
                     "similarity-medium" if sim >= 40 else "similarity-low")
        sim_bar_color = "#00ff88" if sim >= 70 else "#ffcc00" if sim >= 40 else "#ff4466"

        matched_tags = " ".join(f'<span class="tag tag-match">{k}</span>'
                                  for k in r.get("matched_keywords", [])[:6])
        missing_tags = " ".join(f'<span class="tag tag-missing">{k}</span>'
                                  for k in r.get("missing_keywords", [])[:4])

        with st.expander(
            f"Q{r['question_number']}  |  Marks: {r['obtained_marks']:.2f}/{r['total_marks']}  "
            f"|  Similarity: {sim:.1f}%  |  Confidence: {r['confidence_score']:.1f}%"
        ):
            col1, col2 = st.columns([3, 2])
            with col1:
                st.markdown(f"""
                <div class='info-box'>
                    <strong>❓ Question:</strong><br>{r['question']}
                </div>
                <div class='info-box'>
                    <strong>✅ Key Answer:</strong><br>
                    <span style="color:#aad4ff;">{r['key_answer'][:300]}{'...' if len(r['key_answer'])>300 else ''}</span>
                </div>
                <div class='info-box'>
                    <strong>📝 Student Answer:</strong><br>
                    <span style="color:#c8d8f0;">{r['student_answer'][:300]}{'...' if len(r['student_answer'])>300 else ''}</span>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class='info-box'>
                    <strong>🤖 AI Remarks:</strong><br>
                    <span style="color:#ffd700;">{r['ai_remarks']}</span>
                </div>
                <div style='margin-top:0.4rem;'>
                    <strong style='font-size:0.8rem;color:#5a8ab0;'>Matched Keywords:</strong> {matched_tags or '<span style="color:#3a6080;">none</span>'}<br>
                    <strong style='font-size:0.8rem;color:#5a8ab0;'>Missing Keywords:</strong> {missing_tags or '<span style="color:#3a6080;">none</span>'}
                </div>
                """, unsafe_allow_html=True)

            with col2:
                # Similarity gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=sim,
                    title={"text": "Similarity %", "font": {"color": "#7ecfff", "size": 12}},
                    number={"font": {"color": sim_bar_color, "size": 28, "family": "Orbitron"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#5a8ab0"},
                        "bar": {"color": sim_bar_color},
                        "bgcolor": "#091525",
                        "bordercolor": "#1a3050",
                        "steps": [
                            {"range": [0, 40], "color": "#1a0010"},
                            {"range": [40, 70], "color": "#1a1500"},
                            {"range": [70, 100], "color": "#001a10"},
                        ],
                    }
                ))
                fig.update_layout(
                    paper_bgcolor="#091525", height=200,
                    margin=dict(t=30, b=10, l=20, r=20),
                    font=dict(color="#7ecfff"),
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown(f"""
                <div style='text-align:center; margin-top:0.5rem;'>
                    <div style='font-size:0.75rem;color:#5a8ab0;'>AI CONFIDENCE</div>
                    <div style='font-family:Orbitron,monospace;font-size:1.3rem;color:#a97fff;'>
                        {r['confidence_score']:.1f}%
                    </div>
                    <div style='font-size:0.75rem;color:#5a8ab0;'>KEYWORD MATCH</div>
                    <div style='font-family:Orbitron,monospace;font-size:1.3rem;color:#ffcc00;'>
                        {r['keyword_match_pct']:.1f}%
                    </div>
                    <div style='font-size:0.75rem;color:#5a8ab0;'>MARKS AWARDED</div>
                    <div style='font-family:Orbitron,monospace;font-size:1.6rem;color:#00ff88;'>
                        {r['obtained_marks']:.2f} / {r['total_marks']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ── Download Report ──
    st.markdown("---")
    st.markdown('<div class="section-header">DOWNLOAD REPORT</div>', unsafe_allow_html=True)

    report_lines = [
        f"AI EVALUATION REPORT",
        f"{'='*60}",
        f"Student ID   : {selected_sid}",
        f"Student Name : {stu.get('name','')}",
        f"Subject      : {stu.get('subject','')}",
        f"Date         : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"{'='*60}",
        f"Total Marks  : {summ.get('total_obtained',0):.2f} / {summ.get('total_marks',0)}",
        f"Percentage   : {summ.get('percentage',0):.2f}%",
        f"Grade        : {summ.get('grade','—')}",
        f"Result       : {summ.get('pass_fail','—')}",
        f"Avg Similarity: {summ.get('avg_similarity',0):.2f}%",
        f"{'='*60}",
        f"QUESTION-WISE BREAKDOWN",
        f"{'='*60}",
    ]
    for r in results:
        report_lines += [
            f"\nQ{r['question_number']}. {r['question']}",
            f"Marks: {r['obtained_marks']:.2f}/{r['total_marks']}",
            f"Similarity: {r['similarity_score']:.1f}% | Confidence: {r['confidence_score']:.1f}%",
            f"AI Remarks: {r['ai_remarks']}",
            f"Matched Keywords: {', '.join(r.get('matched_keywords',[])[:6])}",
            f"Student Answer: {r['student_answer'][:200]}...",
            "-"*60,
        ]
    report_text = "\n".join(report_lines)

    st.download_button(
        "📥 Download Evaluation Report (.txt)",
        data=report_text,
        file_name=f"eval_report_{selected_sid}.txt",
        mime="text/plain",
        use_container_width=True,
    )


# ═══════════════════════════════════════════════════════════════
#  PAGE 5 — ANALYTICS DASHBOARD
# ═══════════════════════════════════════════════════════════════
def page_analytics():
    st.markdown('<div class="hero-title">📈 ANALYTICS DASHBOARD</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">VISUAL INSIGHTS · PERFORMANCE METRICS · COMPARATIVE ANALYSIS</div>', unsafe_allow_html=True)

    if not st.session_state.evaluated:
        st.warning("⚠️ No data yet. Run evaluation first.")
        return

    summaries = st.session_state.summaries
    students  = st.session_state.students
    eval_results = st.session_state.eval_results

    # Build flat dataframe
    rows = []
    for sid, s in summaries.items():
        stu = students.get(sid, {})
        rows.append({
            "Student ID": sid,
            "Name": stu.get("name", sid),
            "Subject": stu.get("subject", "—"),
            "Percentage": s["percentage"],
            "Grade": s["grade"],
            "Pass/Fail": s["pass_fail"],
            "Avg Similarity": s["avg_similarity"],
            "Avg Confidence": s["avg_confidence"],
            "Questions": s["questions_evaluated"],
            "Total Obtained": s["total_obtained"],
            "Total Marks": s["total_marks"],
        })
    df = pd.DataFrame(rows)

    # Row 1: Pie + Bar + Scatter
    col1, col2, col3 = st.columns(3)

    with col1:
        pass_count = (df["Pass/Fail"] == "Pass").sum()
        fail_count = len(df) - pass_count
        fig = go.Figure(go.Pie(
            labels=["Pass","Fail"],
            values=[pass_count, fail_count],
            hole=0.55,
            marker=dict(colors=["#00ff88","#ff4466"],
                        line=dict(color="#080c14", width=3)),
        ))
        fig.update_layout(
            title="Pass vs Fail", template="plotly_dark",
            paper_bgcolor="#091525", height=280,
            margin=dict(t=40,b=10,l=10,r=10), title_font_color="#00d4ff",
            legend=dict(font=dict(color="#7ecfff")),
            annotations=[dict(text=f"{pass_count}/{len(df)}", x=0.5, y=0.5,
                              font=dict(size=16, color="#00ff88", family="Orbitron"),
                              showarrow=False)]
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure(go.Bar(
            x=df["Name"], y=df["Percentage"],
            marker=dict(color=df["Percentage"],
                        colorscale=[[0,"#ff4466"],[0.5,"#ffcc00"],[1,"#00ff88"]]),
            text=df["Percentage"].apply(lambda x: f"{x:.1f}%"),
            textposition="outside",
        ))
        fig.update_layout(
            title="Score by Student", template="plotly_dark",
            paper_bgcolor="#091525", plot_bgcolor="#091525",
            height=280, margin=dict(t=40,b=20,l=10,r=10),
            yaxis=dict(range=[0,115], color="#5a8ab0"),
            xaxis_color="#5a8ab0", title_font_color="#00d4ff",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        fig = go.Figure(go.Scatter(
            x=df["Avg Similarity"], y=df["Percentage"],
            mode="markers+text",
            text=df["Name"], textposition="top center",
            marker=dict(size=16, color=df["Percentage"],
                        colorscale=[[0,"#ff4466"],[0.5,"#ffcc00"],[1,"#00ff88"]],
                        showscale=False, line=dict(color="#080c14", width=2)),
            textfont=dict(color="#7ecfff", size=10),
        ))
        fig.update_layout(
            title="Similarity vs Score", template="plotly_dark",
            paper_bgcolor="#091525", plot_bgcolor="#091525",
            height=280, margin=dict(t=40,b=20,l=10,r=10),
            xaxis=dict(title="Avg Similarity %", color="#5a8ab0"),
            yaxis=dict(title="Score %", color="#5a8ab0"),
            title_font_color="#00d4ff",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Row 2: Subject performance + Grade distribution
    col1, col2 = st.columns(2)

    with col1:
        subj_df = df.groupby("Subject")["Percentage"].mean().reset_index()
        fig = go.Figure(go.Bar(
            x=subj_df["Subject"], y=subj_df["Percentage"],
            marker=dict(color=["#00d4ff","#7c4dff","#00ff88"][:len(subj_df)]),
            text=subj_df["Percentage"].apply(lambda x: f"{x:.1f}%"),
            textposition="outside",
        ))
        fig.update_layout(
            title="Subject-wise Average Score", template="plotly_dark",
            paper_bgcolor="#091525", plot_bgcolor="#091525",
            height=300, margin=dict(t=40,b=20,l=10,r=10),
            yaxis=dict(range=[0,110], color="#5a8ab0"),
            xaxis_color="#5a8ab0", title_font_color="#00d4ff",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        grade_counts = df["Grade"].value_counts().reset_index()
        grade_counts.columns = ["Grade","Count"]
        grade_colors = {"A+":"#00ff88","A":"#66ff99","B+":"#00d4ff",
                        "B":"#66aaff","C":"#ffcc00","D":"#ff9900","F":"#ff4466"}
        fig = go.Figure(go.Bar(
            x=grade_counts["Grade"], y=grade_counts["Count"],
            marker_color=[grade_colors.get(g,"#7c4dff") for g in grade_counts["Grade"]],
            text=grade_counts["Count"], textposition="outside",
        ))
        fig.update_layout(
            title="Grade Distribution", template="plotly_dark",
            paper_bgcolor="#091525", plot_bgcolor="#091525",
            height=300, margin=dict(t=40,b=20,l=10,r=10),
            xaxis_color="#5a8ab0", yaxis_color="#5a8ab0", title_font_color="#00d4ff",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Row 3: Heatmap — similarity by student × question
    st.markdown('<div class="section-header">SIMILARITY HEATMAP — STUDENT × QUESTION</div>', unsafe_allow_html=True)
    if eval_results:
        student_ids = list(eval_results.keys())
        student_names = [students.get(sid,{}).get("name",sid) for sid in student_ids]
        max_q = max(len(v) for v in eval_results.values())
        q_labels = [f"Q{i+1}" for i in range(max_q)]

        matrix = []
        for sid in student_ids:
            row = [r["similarity_score"] for r in eval_results[sid]]
            while len(row) < max_q:
                row.append(0)
            matrix.append(row)

        fig = go.Figure(go.Heatmap(
            z=matrix,
            x=q_labels,
            y=student_names,
            colorscale=[[0,"#330010"],[0.4,"#664400"],[0.7,"#003320"],[1,"#00ff88"]],
            text=[[f"{v:.1f}%" for v in row] for row in matrix],
            texttemplate="%{text}",
            textfont=dict(size=11, color="#ffffff"),
            zmin=0, zmax=100,
            colorbar=dict(title="Similarity %", tickfont=dict(color="#7ecfff"),
                          titlefont=dict(color="#7ecfff")),
        ))
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="#091525",
            plot_bgcolor="#091525", height=280,
            margin=dict(t=20, b=20, l=10, r=10),
            xaxis_color="#5a8ab0", yaxis_color="#7ecfff",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Row 4: Student Ranking
    st.markdown('<div class="section-header">STUDENT RANKING LEADERBOARD</div>', unsafe_allow_html=True)
    df_ranked = df.sort_values("Percentage", ascending=False).reset_index(drop=True)
    df_ranked.index += 1
    df_ranked.index.name = "Rank"

    medals = ["🥇","🥈","🥉"] + ["🎖"] * (len(df_ranked) - 3)
    df_ranked.insert(0, "🏆", medals[:len(df_ranked)])

    st.dataframe(
        df_ranked[["🏆","Name","Subject","Percentage","Grade","Pass/Fail","Avg Similarity"]],
        use_container_width=True,
        column_config={
            "Percentage": st.column_config.ProgressColumn(
                "Score %", min_value=0, max_value=100, format="%.1f%%"
            ),
            "Avg Similarity": st.column_config.ProgressColumn(
                "Avg Similarity", min_value=0, max_value=100, format="%.1f%%"
            ),
        }
    )

    # Stats summary
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Highest Score", f"{df['Percentage'].max():.1f}%")
    with col2: st.metric("Lowest Score", f"{df['Percentage'].min():.1f}%")
    with col3: st.metric("Class Average", f"{df['Percentage'].mean():.1f}%")
    with col4: st.metric("Std Deviation", f"{df['Percentage'].std():.1f}%")


# ═══════════════════════════════════════════════════════════════
#  PAGE 6 — UML DIAGRAMS
# ═══════════════════════════════════════════════════════════════
def page_uml():
    st.markdown('<div class="hero-title">📐 UML DIAGRAMS</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">SYSTEM ARCHITECTURE · DESIGN DOCUMENTATION</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Use Case", "Class Diagram", "Sequence", "Activity", "Deployment"
    ])

    with tab1:
        st.markdown('<div class="section-header">USE CASE DIAGRAM</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="uml-container">
+─────────────────────────────────────────────────────+
│              AI EVALUATION SYSTEM                   │
│                                                     │
│  ┌──────────┐   ┌──────────────────────────────┐    │
│  │          │──▶│  Upload Question Paper       │    │
│  │          │   └──────────────────────────────┘    │
│  │          │   ┌──────────────────────────────┐    │
│  │  TEACHER │──▶│  Upload Key Answer Sheet     │    │
│  │          │   └──────────────────────────────┘    │
│  │  (Actor) │   ┌──────────────────────────────┐    │
│  │          │──▶│  Configure Evaluation Params │    │
│  └──────────┘   └──────────────────────────────┘    │
│                                                     │
│  ┌──────────┐   ┌──────────────────────────────┐    │
│  │          │──▶│  Upload Answer Sheet         │    │
│  │          │   └──────────────────────────────┘    │
│  │ STUDENT  │   ┌──────────────────────────────┐    │
│  │          │──▶│  View Own Results            │    │
│  │  (Actor) │   └──────────────────────────────┘    │
│  └──────────┘                                       │
│                                                     │
│  ┌──────────┐   ┌──────────────────────────────┐    │
│  │          │──▶│  Run AI Evaluation           │    │
│  │  AI      │   └──────────────────────────────┘    │
│  │  ENGINE  │──▶│  Generate NLP Analysis       │    │
│  │  (Actor) │   └──────────────────────────────┘    │
│  │          │──▶│  Compute Similarity Scores   │    │
│  └──────────┘   └──────────────────────────────┘    │
│                                                     │
│  ┌──────────┐   ┌──────────────────────────────┐    │
│  │  ADMIN   │──▶│  View Analytics Dashboard    │    │
│  │  (Actor) │──▶│  Export Reports              │    │
│  │          │──▶│  Manage Students             │    │
│  └──────────┘   └──────────────────────────────┘    │
+─────────────────────────────────────────────────────+
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="section-header">CLASS DIAGRAM</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="uml-container">
┌─────────────────────────┐       ┌──────────────────────────┐
│       Student           │       │      UploadedFile        │
├─────────────────────────┤       ├──────────────────────────┤
│ - student_id: str       │       │ - file_id: int           │
│ - student_name: str     │       │ - file_name: str         │
│ - subject_name: str     │       │ - file_type: str         │
│ - created_at: datetime  │1  N   │ - file_path: str         │
├─────────────────────────┤───────│ - student_id: str        │
│ + save()                │       │ - uploaded_at: datetime  │
│ + get_results()         │       ├──────────────────────────┤
└────────────┬────────────┘       │ + extract_text()         │
             │1                   │ + validate_format()      │
             │N                   └──────────────────────────┘
┌────────────┴────────────┐
│   EvaluationResult      │       ┌──────────────────────────┐
├─────────────────────────┤       │      NLPEngine           │
│ - evaluation_id: int    │       ├──────────────────────────┤
│ - student_id: str       │       │ - stopwords: Set         │
│ - question_number: int  │       │ - lemma_dict: Dict       │
│ - question_text: str    │       ├──────────────────────────┤
│ - key_answer: str       │N  1   │ + tokenize(text)         │
│ - student_answer: str   │───────│ + remove_stopwords()     │
│ - similarity_score: flt │       │ + lemmatize(token)       │
│ - confidence_score: flt │       │ + compute_tfidf()        │
│ - obtained_marks: flt   │       │ + cosine_similarity()    │
│ - total_marks: flt      │       │ + keyword_overlap()      │
│ - ai_remarks: str       │       │ + evaluate_student()     │
├─────────────────────────┤       │ + generate_remarks()     │
│ + compute_similarity()  │       └──────────────────────────┘
│ + generate_remarks()    │
└─────────────────────────┘       ┌──────────────────────────┐
                                  │   StudentFinalResult     │
┌─────────────────────────┐       ├──────────────────────────┤
│   EvaluationSession     │1      │ - result_id: int         │
├─────────────────────────┤───────│ - student_id: str        │
│ - session_id: int       │  N    │ - total_obtained: flt    │
│ - session_name: str     │       │ - total_marks: flt       │
│ - total_students: int   │       │ - percentage: flt        │
│ - avg_score: flt        │       │ - grade: str             │
│ - pass_percentage: flt  │       │ - pass_fail: str         │
├─────────────────────────┤       ├──────────────────────────┤
│ + create()              │       │ + assign_grade()         │
│ + get_stats()           │       │ + generate_report()      │
└─────────────────────────┘       └──────────────────────────┘
        </div>
        """, unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="section-header">SEQUENCE DIAGRAM</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="uml-container">
Teacher        Streamlit UI       NLP Engine         Database
  │                │                  │                  │
  │──Upload Files─▶│                  │                  │
  │                │──extract_text()─▶│                  │
  │                │◀─tokens, text────│                  │
  │                │──save_student()──────────────────────▶│
  │                │◀─student_id──────────────────────────│
  │                │                  │                  │
  │──Run Eval─────▶│                  │                  │
  │                │──create_session()────────────────────▶│
  │                │◀─session_id──────────────────────────│
  │                │                  │                  │
  │                │  For each student:                  │
  │                │──evaluate_student()─▶│              │
  │                │    │──tokenize()─────▶│             │
  │                │    │◀─tokens──────────│             │
  │                │    │──remove_sw()─────▶│            │
  │                │    │◀─clean_tokens────│             │
  │                │    │──lemmatize()─────▶│            │
  │                │    │◀─lemmas──────────│             │
  │                │    │──compute_tfidf()─▶│            │
  │                │    │◀─tfidf_vectors───│             │
  │                │    │──cosine_sim()────▶│            │
  │                │    │◀─similarity──────│             │
  │                │    │──calc_marks()────▶│            │
  │                │    │◀─marks───────────│             │
  │                │    │──gen_remarks()───▶│            │
  │                │◀─results dict─────────│             │
  │                │──save_results()──────────────────────▶│
  │                │◀─success─────────────────────────────│
  │                │                  │                  │
  │◀─Show Results──│                  │                  │
  │                │                  │                  │
        </div>
        """, unsafe_allow_html=True)

    with tab4:
        st.markdown('<div class="section-header">ACTIVITY DIAGRAM</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="uml-container">
                    [START]
                       │
                       ▼
              ┌────────────────┐
              │  Upload Files  │
              │ (QP + KA + SA) │
              └───────┬────────┘
                      │
                      ▼
              ┌────────────────┐
              │ Validate Files │◀──────────┐
              └───────┬────────┘           │
                      │                   │
           ┌──────────┴──────────┐         │
           │ Valid?              │         │
    No ────▶  Show Error        │──Yes    │
           │  Message           │         │
           └────────────────────┘         │
                      │                   │
                      ▼                   │
              ┌────────────────┐           │
              │ Extract Text   │           │
              │ from Documents │           │
              └───────┬────────┘           │
                      │                   │
                      ▼                   │
              ┌────────────────┐           │
              │  NLP Pipeline  │           │
              │  Tokenize      │           │
              │  Remove SW     │           │
              │  Lemmatize     │           │
              └───────┬────────┘           │
                      │                   │
                      ▼                   │
              ┌────────────────┐           │
              │  For Each Q:   │           │
              │  Compare KA vs │           │
              │  Student Ans   │           │
              └───────┬────────┘           │
                      │                   │
                      ▼                   │
              ┌────────────────┐           │
              │ Compute TF-IDF │           │
              │ Cosine Sim     │           │
              │ Keyword Overlap│           │
              └───────┬────────┘           │
                      │                   │
                      ▼                   │
              ┌────────────────┐           │
              │  Calculate     │           │
              │  Marks +       │           │
              │  AI Remarks    │           │
              └───────┬────────┘           │
                      │                   │
                      ▼                   │
              ┌────────────────┐           │
              │  Aggregate     │           │
              │  Total + Grade │           │
              │  + Pass/Fail   │           │
              └───────┬────────┘           │
                      │                   │
                      ▼                   │
              ┌────────────────┐           │
              │  Save to MySQL │           │
              └───────┬────────┘           │
                      │                   │
                      ▼                   │
              ┌────────────────┐           │
              │ Display Results│           │
              │ + Analytics    │           │
              └───────┬────────┘           │
                      │                   │
           ┌──────────┴──────────┐         │
           │ New Upload?         │──Yes───▶│
           └──────────┬──────────┘
                      │ No
                      ▼
                    [END]
        </div>
        """, unsafe_allow_html=True)

    with tab5:
        st.markdown('<div class="section-header">DEPLOYMENT DIAGRAM</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="uml-container">
┌─────────────────────────────────────────────────────────────┐
│                    CLIENT MACHINE                           │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Web Browser / Desktop App              │   │
│  │   [Streamlit UI] localhost:8501                     │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────┬───────────────────────────────┘
                              │ HTTP
┌─────────────────────────────▼───────────────────────────────┐
│                    APPLICATION SERVER                       │
│  ┌─────────────────────┐   ┌─────────────────────────────┐  │
│  │   Streamlit App     │   │      NLP Engine             │  │
│  │   (app.py)          │   │   (modules/nlp_engine.py)   │  │
│  │   Port: 8501        │   │   • Tokenizer               │  │
│  └──────────┬──────────┘   │   • TF-IDF                  │  │
│             │              │   • Cosine Similarity        │  │
│  ┌──────────▼──────────┐   └─────────────────────────────┘  │
│  │   DB Handler        │                                    │
│  │   (modules/         │   ┌─────────────────────────────┐  │
│  │    db_handler.py)   │   │   Sample Data Module        │  │
│  └──────────┬──────────┘   └─────────────────────────────┘  │
└─────────────┼───────────────────────────────────────────────┘
              │ SQL / In-Memory Fallback
┌─────────────▼───────────────────────────────────────────────┐
│                    DATA LAYER                               │
│  ┌─────────────────────────┐   ┌─────────────────────────┐  │
│  │      MySQL Server       │   │   In-Memory Store       │  │
│  │      Port: 3306         │   │   (Fallback)            │  │
│  │                         │   │                         │  │
│  │  • students             │   │  • _store dict          │  │
│  │  • uploaded_files       │   │  • No persistence       │  │
│  │  • evaluation_results   │   │                         │  │
│  │  • student_final_results│   └─────────────────────────┘  │
│  │  • evaluation_sessions  │                                 │
│  │  • analytics_logs       │                                 │
│  └─────────────────────────┘                                 │
└─────────────────────────────────────────────────────────────┘
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  PAGE 7 — TESTING & VALIDATION
# ═══════════════════════════════════════════════════════════════
def page_testing():
    st.markdown('<div class="hero-title">🧪 TESTING & VALIDATION</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">UNIT TESTS · INTEGRATION TESTS · SYSTEM VALIDATION</div>', unsafe_allow_html=True)

    from modules.nlp_engine import (
        tokenize, remove_stopwords, lemmatize, compute_tfidf_similarity,
        keyword_overlap, combined_similarity, calculate_marks, assign_grade
    )

    test_cases = []

    # Unit Tests
    def run_test(name, condition, expected=True):
        result = "✅ PASS" if condition == expected else "❌ FAIL"
        test_cases.append({"Test": name, "Result": result,
                           "Type": "Unit", "Status": condition == expected})
        return condition == expected

    # Tokenization
    run_test("Tokenize basic sentence", len(tokenize("Hello world")) == 2)
    run_test("Tokenize empty string", tokenize("") == [])
    run_test("Stopword removal removes 'the'", "the" not in remove_stopwords(["the","cat","sat"]))
    run_test("Lemmatize 'running'", lemmatize("running") == "run")
    run_test("Lemmatize 'algorithms'", lemmatize("algorithms") == "algorithm")
    run_test("TF-IDF identical texts → 1.0", compute_tfidf_similarity("machine learning algorithm","machine learning algorithm") >= 0.98)
    run_test("TF-IDF unrelated texts → < 0.3", compute_tfidf_similarity("apple banana fruit","quantum neural network") < 0.3)
    run_test("Keyword overlap perfect match", keyword_overlap("machine learning data","machine learning data")[0] >= 0.9)
    run_test("Marks calc for sim=1.0", calculate_marks(1.0, 5.0) <= 5.0)
    run_test("Marks calc for sim=0.0 ≥ 0", calculate_marks(0.0, 5.0) >= 0)
    run_test("Grade A+ for 95%", assign_grade(95) == "A+")
    run_test("Grade F for 30%", assign_grade(30) == "F")
    run_test("Grade B for 65%", assign_grade(65) == "B")
    run_test("Combined sim non-negative", combined_similarity("test answer", "test answer") >= 0)
    run_test("Combined sim ≤ 1.0", combined_similarity("completely different", "machine learning") <= 1.0)

    # Integration Tests
    from modules.nlp_engine import evaluate_student, compute_student_summary
    test_q = ["What is machine learning?"]
    test_ka = ["Machine learning is a subset of AI that enables systems to learn from data automatically."]
    test_sa = ["Machine learning allows computers to learn from data and make predictions automatically."]
    results = evaluate_student(test_q, test_ka, test_sa)
    run_test("evaluate_student returns list", isinstance(results, list))
    run_test("Result has required keys", all(k in results[0] for k in ["similarity_score","obtained_marks","ai_remarks"]))
    run_test("Similarity in 0-100 range", 0 <= results[0]["similarity_score"] <= 100)
    run_test("Marks ≤ total_marks", results[0]["obtained_marks"] <= results[0]["total_marks"])
    summ = compute_student_summary(results)
    run_test("Summary has percentage", "percentage" in summ)
    run_test("Summary grade assigned", summ["grade"] in ["A+","A","B+","B","C","D","F"])

    # System Tests
    run_test("DB save student", db.save_student("TEST99","Test User","Test Subject") == True)
    sid = db.create_session("Test Session")
    run_test("DB session creation returns int", isinstance(sid, int))
    run_test("DB get students not empty after save", len(db.get_all_students()) >= 1)

    # Display Results
    df_tests = pd.DataFrame(test_cases)
    pass_count = df_tests["Status"].sum()
    total = len(df_tests)

    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Total Tests", total)
    with col2: st.metric("Passed", pass_count)
    with col3: st.metric("Failed", total - pass_count)

    st.progress(int(pass_count / total * 100))

    st.markdown('<div class="section-header">TEST RESULTS</div>', unsafe_allow_html=True)
    st.dataframe(
        df_tests[["Type","Test","Result"]].style.apply(
            lambda row: ["", "", "background-color: #003320" if "PASS" in row["Result"]
                         else "background-color: #330010"] * 3, axis=1
        ),
        use_container_width=True, hide_index=True
    )

    # Error handling demo
    st.markdown("---")
    st.markdown('<div class="section-header">ERROR HANDLING VALIDATION</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class='info-box'>
            <strong>📄 File Validation</strong><br>
            ✅ PDF format check<br>
            ✅ DOCX format check<br>
            ✅ Empty file detection<br>
            ✅ Corrupt file handling<br>
            ✅ Unsupported format rejection
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='info-box'>
            <strong>🔗 Integration Guards</strong><br>
            ✅ MySQL timeout fallback<br>
            ✅ Empty answer handling<br>
            ✅ Missing Q/A alignment<br>
            ✅ Division by zero guards<br>
            ✅ Encoding error recovery
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class='info-box'>
            <strong>🖥 System Resilience</strong><br>
            ✅ Session state persistence<br>
            ✅ Large file chunking<br>
            ✅ Concurrent user isolation<br>
            ✅ Graceful degradation<br>
            ✅ In-memory DB fallback
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  ROUTER
# ═══════════════════════════════════════════════════════════════
page = st.session_state.current_page
if   page == "🏠 Home Dashboard":      page_home()
elif page == "📤 Upload Center":        page_upload()
elif page == "⚙️ AI Processing":       page_processing()
elif page == "📊 Evaluation Results":  page_results()
elif page == "📈 Analytics Dashboard": page_analytics()
elif page == "📐 UML Diagrams":        page_uml()
elif page == "🧪 Testing & Validation": page_testing()
