# 🧠 AI-Powered Automatic Answer Evaluation System

A futuristic, interactive dashboard for automated student answer evaluation using
NLP, Machine Learning, and semantic similarity analysis.

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd ai_evaluation_system
pip install -r requirements.txt
```

### 2. (Optional) Set Up MySQL

```bash
mysql -u root -p < schema.sql
```

> If MySQL is not available, the system automatically uses an in-memory store.

### 3. Run the Application

```bash
streamlit run app.py
```

Open your browser at: **http://localhost:8501**

---

## 📁 Project Structure

```
ai_evaluation_system/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── schema.sql                # MySQL database schema
├── modules/
│   ├── __init__.py
│   ├── nlp_engine.py         # NLP & AI evaluation engine
│   ├── db_handler.py         # MySQL + in-memory database
│   └── sample_data.py        # Demo Q&A data
└── README.md
```

---

## 🎯 Dashboard Pages

| Page | Description |
|------|-------------|
| 🏠 Home Dashboard | KPI cards, student overview, quick charts |
| 📤 Upload Center | Upload Q paper, key answers, student sheets (PDF/DOCX) |
| ⚙️ AI Processing | NLP pipeline demo, run evaluation engine |
| 📊 Evaluation Results | Per-student, per-question detailed AI assessment |
| 📈 Analytics Dashboard | Charts, heatmaps, leaderboards |
| 📐 UML Diagrams | Use case, class, sequence, activity, deployment |
| 🧪 Testing | Unit/integration/system test runner |

---

## ⚡ Demo Mode

Click **"🔄 Load Demo Data"** in the sidebar to instantly populate the system with:
- 10 Computer Science / AI questions
- Full model answer key
- 3 student answer sheets (varying quality levels)
- Complete evaluation results and analytics

---

## 🧬 NLP Pipeline

```
Raw Text
   └─▶ Tokenization       → Split text into words
         └─▶ Stopword Removal → Remove common words (a, the, is...)
               └─▶ Lemmatization    → Reduce to base form (running → run)
                     └─▶ TF-IDF            → Compute term importance
                           └─▶ Cosine Similarity → Measure semantic closeness
                                 └─▶ Keyword Overlap  → Check concept coverage
                                       └─▶ Mark Calculation → Convert to score
                                             └─▶ AI Remarks    → Generate feedback
```

---

## 🗄️ Database Schema

- **students** — Student ID, name, subject
- **uploaded_files** — File metadata and paths
- **evaluation_sessions** — Grouping for each evaluation run
- **evaluation_results** — Per-question NLP scores and remarks
- **student_final_results** — Totals, grades, pass/fail
- **analytics_logs** — System activity log

---

## 📊 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Similarity Score | TF-IDF + Jaccard + Keyword overlap (weighted) |
| Confidence Score | AI's certainty about its evaluation |
| Keyword Match % | Proportion of key concepts present |
| Marks Awarded | Non-linear scaling of similarity to marks |

---

## 📂 Supported File Formats

- **PDF** — via pdfplumber (primary) + PyPDF2 (fallback)
- **DOCX** — via python-docx
- **TXT** — direct UTF-8 read

---

## 🧪 Answer Sheet Format

For best parsing, format student answer sheets as:

```
Answer 1. [student answer text]

Answer 2. [student answer text]

Answer 3. [student answer text]
```

Or use Q1./A1. format:

```
Q1. What is machine learning?
A1. Machine learning is...

Q2. Explain neural networks.
A2. Neural networks are...
```

---

## 💡 Tips

- Use the **Load Sample** buttons to quickly test each section
- Adjust **Marks per Question** and **Pass Threshold** before running evaluation
- Download individual student reports as `.txt` files
- The heatmap shows which questions students struggled with most

---

## 🛠️ Technology Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit (dark theme, Orbitron + Exo 2 fonts) |
| NLP | Custom TF-IDF, Cosine Similarity, Lemmatizer |
| Visualization | Plotly (interactive), Pandas |
| Database | MySQL (with in-memory fallback) |
| File Parsing | pdfplumber, PyPDF2, python-docx |

---

*Built as a futuristic AI-powered educational evaluation platform.*
