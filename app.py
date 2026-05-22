"""
Simple AI Answer Evaluation System
"""

import sys
from pathlib import Path
import importlib.util

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd

try:
    from modules.nlp_engine import (
        evaluate_student, compute_student_summary,
        extract_text, parse_qa_from_text, parse_student_answers
    )
    from modules import db_handler as db
    from modules.sample_data import get_sample_questions, get_sample_key_answers, get_sample_student_answers
except ModuleNotFoundError:
    # Fallback for environments where the root package path is not set correctly.
    nlp_path = ROOT_DIR / "modules" / "nlp_engine.py"
    db_path = ROOT_DIR / "modules" / "db_handler.py"
    sample_data_path = ROOT_DIR / "modules" / "sample_data.py"

    def load_module(name, path):
        spec = importlib.util.spec_from_file_location(name, str(path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    if not nlp_path.exists() or not db_path.exists() or not sample_data_path.exists():
        raise

    module_nlp = load_module("modules.nlp_engine", nlp_path)
    module_db = load_module("modules.db_handler", db_path)
    module_sample = load_module("modules.sample_data", sample_data_path)

    evaluate_student = module_nlp.evaluate_student
    compute_student_summary = module_nlp.compute_student_summary
    extract_text = module_nlp.extract_text
    parse_qa_from_text = module_nlp.parse_qa_from_text
    parse_student_answers = module_nlp.parse_student_answers

    db = module_db
    get_sample_questions = module_sample.get_sample_questions
    get_sample_key_answers = module_sample.get_sample_key_answers
    get_sample_student_answers = module_sample.get_sample_student_answers

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Evaluation System",
    page_icon="🧠",
    layout="wide",
)

# ─────────────────────────────────────────────
# SIMPLE CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
.main { background-color: #f5f5f5; }
h1 { color: #333; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "questions" not in st.session_state:
    st.session_state.questions = []
if "key_answers" not in st.session_state:
    st.session_state.key_answers = []
if "students" not in st.session_state:
    st.session_state.students = {}
if "results" not in st.session_state:
    st.session_state.results = {}
if "shared_subject" not in st.session_state:
    st.session_state.shared_subject = "Computer Science"

# ─────────────────────────────────────────────
# MAIN DASHBOARD
# ─────────────────────────────────────────────
st.title("🧠 AI Answer Evaluation System")
st.markdown("Simple upload → evaluate → results workflow")

col1, col2 = st.columns(2)

# ─────────────────────────────────────────────
# LEFT COLUMN: UPLOADS
# ─────────────────────────────────────────────
with col1:
    st.header("📋 Step 1: Upload Files")
    
    # Question Paper
    st.subheader("Question Paper")
    qp_file = st.file_uploader("Upload question paper (PDF, DOCX, or TXT)", 
                               type=["pdf", "docx", "txt"], key="qp")
    if qp_file:
        with st.spinner("Extracting questions..."):
            text = extract_text(qp_file)
            qa_pairs = parse_qa_from_text(text)
            if qa_pairs:
                st.session_state.questions = [a for _, a in qa_pairs]
            else:
                chunks = [s.strip() for s in text.split('\n\n') if s.strip() and len(s.strip()) > 30]
                st.session_state.questions = chunks[:10]
        st.success(f"✅ {len(st.session_state.questions)} questions extracted")
    
    # Answer Key
    st.subheader("Answer Key")
    ka_file = st.file_uploader("Upload answer key (PDF, DOCX, or TXT)", 
                               type=["pdf", "docx", "txt"], key="ka")
    if ka_file:
        with st.spinner("Extracting answers..."):
            text = extract_text(ka_file)
            qa_pairs = parse_qa_from_text(text)
            if qa_pairs:
                st.session_state.key_answers = [a for _, a in qa_pairs]
            else:
                chunks = [s.strip() for s in text.split('\n\n') if s.strip() and len(s.strip()) > 30]
                st.session_state.key_answers = chunks[:10]
        st.success(f"✅ {len(st.session_state.key_answers)} answers extracted")
    
    # Subject
    st.subheader("Subject")
    st.session_state.shared_subject = st.text_input(
        "Subject name", 
        value=st.session_state.shared_subject,
        key="subject_input"
    )

# ─────────────────────────────────────────────
# RIGHT COLUMN: STATUS
# ─────────────────────────────────────────────
with col2:
    st.header("📊 Status")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Questions", len(st.session_state.questions))
    with col_b:
        st.metric("Answers", len(st.session_state.key_answers))
    with col_c:
        st.metric("Students", len(st.session_state.students))

st.markdown("---")

# ─────────────────────────────────────────────
# STUDENT UPLOADS (3 Students)
# ─────────────────────────────────────────────
st.header("👥 Step 2: Upload 3 Student Answer Sheets")

uploaded_students = {}

for student_idx in range(3):
    st.subheader(f"Student {student_idx + 1}")
    
    uploaded_file = st.file_uploader(
        f"Upload answer sheet for Student {student_idx + 1}",
        type=["pdf", "docx", "txt"],
        key=f"student_file_{student_idx}"
    )
    
    col1, col2 = st.columns([2, 2])
    with col1:
        student_name = st.text_input(
            "Student Name",
            key=f"name_{student_idx}",
            placeholder="Enter student name"
        )
    with col2:
        student_id = st.text_input(
            "Student ID",
            value=f"STU{student_idx + 1:03d}",
            key=f"id_{student_idx}"
        )
    
    if uploaded_file and student_name and student_id:
        with st.spinner(f"Reading {uploaded_file.name}..."):
            text = extract_text(uploaded_file)
            nq = len(st.session_state.questions)
            answers = parse_student_answers(text, nq if nq else 10)
        
        uploaded_students[student_id] = {
            "name": student_name,
            "subject": st.session_state.shared_subject,
            "answers": answers,
            "file": uploaded_file.name
        }
        st.success(f"✅ {uploaded_file.name} - {len(answers)} answers extracted")
    
    st.markdown("<hr style='margin:1rem 0;'>", unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────────
# EVALUATE BUTTON
# ─────────────────────────────────────────────
st.header("Step 3: Evaluate")

col1, col2 = st.columns(2)

with col1:
    if st.button("🚀 EVALUATE ALL STUDENTS", use_container_width=True, type="primary"):
        if not st.session_state.questions:
            st.error("❌ Please upload question paper first")
        elif not st.session_state.key_answers:
            st.error("❌ Please upload answer key first")
        elif not uploaded_students:
            st.error("❌ Please upload student answer sheets first")
        else:
            # Store students in session
            st.session_state.students = uploaded_students
            
            # Create evaluation session
            session_id = db.create_session("Evaluation Session")
            
            # Evaluate all students
            progress_bar = st.progress(0)
            results = {}
            
            for idx, (sid, stu_data) in enumerate(uploaded_students.items()):
                with st.spinner(f"Evaluating {stu_data['name']}..."):
                    eval_result = evaluate_student(
                        st.session_state.questions,
                        st.session_state.key_answers,
                        stu_data["answers"]
                    )
                    summary = compute_student_summary(eval_result)
                    
                    results[sid] = {
                        "name": stu_data["name"],
                        "subject": stu_data["subject"],
                        "eval": eval_result,
                        "summary": summary
                    }
                    
                    # Save to database
                    db.save_student(sid, stu_data["name"], stu_data["subject"])
                    db.save_evaluation_results(session_id, sid, eval_result, summary)
                
                progress_bar.progress((idx + 1) / len(uploaded_students))
            
            st.session_state.results = results
            st.success(f"✅ Evaluation complete for {len(results)} students")

with col2:
    if st.button("📋 Load Sample Data", use_container_width=True):
        st.session_state.questions = get_sample_questions()
        st.session_state.key_answers = get_sample_key_answers()
        
        sample_students = {}
        for i, sid in enumerate(["STU001", "STU002", "STU003"]):
            ans, nm, _ = get_sample_student_answers(sid)
            sample_students[sid] = {
                "name": nm,
                "subject": st.session_state.shared_subject,
                "answers": ans,
                "file": f"sample_{sid}.pdf"
            }
        
        uploaded_students.update(sample_students)
        st.success("✅ Sample data loaded")
        st.rerun()

st.markdown("---")

# ─────────────────────────────────────────────
# RESULTS DISPLAY
# ─────────────────────────────────────────────
if st.session_state.results:
    st.header("📈 Evaluation Results")
    
    # Summary table
    st.subheader("Results Summary")
    summary_data = []
    for sid, res in st.session_state.results.items():
        summary_data.append({
            "Student ID": sid,
            "Name": res["name"],
            "Subject": res["subject"],
            "Score": f"{res['summary']['total_obtained']}/{res['summary']['total_marks']}",
            "Percentage": f"{res['summary']['percentage']:.1f}%",
            "Grade": res["summary"]["grade"],
            "Status": res["summary"]["pass_fail"]
        })
    
    df = pd.DataFrame(summary_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Detailed results
    st.subheader("Detailed Feedback")
    for sid, res in st.session_state.results.items():
        with st.expander(f"{res['name']} ({sid})"):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Percentage", f"{res['summary']['percentage']:.1f}%")
            with col2:
                st.metric("Grade", res["summary"]["grade"])
            with col3:
                st.metric("Status", res["summary"]["pass_fail"])
            with col4:
                st.metric("Avg Similarity", f"{res['summary']['avg_similarity']:.1f}%")
            
            st.markdown("**Question-wise Performance:**")
            for q_idx, q_eval in enumerate(res["eval"], 1):
                similarity = q_eval.get("similarity", 0)
                color = "🟢" if similarity >= 70 else "🟡" if similarity >= 50 else "🔴"
                st.markdown(f"{color} Q{q_idx}: {similarity:.1f}% match")
else:
    st.info("👆 Upload files and click 'EVALUATE ALL STUDENTS' to see results")
