"""
Automated Data Analysis Engine — Streamlit UI

Wraps the original pandas + LangChain (Mistral) RAG pipeline in a Streamlit
front end. The core logic (statistics computation, CSV loading/chunking,
embedding, MMR retrieval, prompt, and chain) is UNCHANGED from the original
script — only reorganized into functions and connected to UI elements.

The Mistral API key is read from Streamlit secrets / environment (.env) —
it is never entered by the user in the UI.
"""

import os
import tempfile

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="DataLens AI · Automated Data Analysis Engine",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# THEME / CSS — dark, data-analyst aesthetic
# ============================================================
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

        html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
        code, pre, .stCodeBlock, div[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; }

        .stApp {
            background:
                radial-gradient(circle at 15% 0%, rgba(56,189,248,0.10), transparent 40%),
                radial-gradient(circle at 85% 15%, rgba(168,85,247,0.10), transparent 40%),
                #0b0f19;
        }

        .block-container {padding-top: 4.5rem; padding-bottom: 4rem; max-width: 1200px;}

        /* ---- Hero ---- */
        .hero-eyebrow {
            font-family: 'JetBrains Mono', monospace;
            letter-spacing: 4px;
            font-size: 0.95rem;
            font-weight: 700;
            color: #e0f2fe;
            text-transform: uppercase;
            text-align: center;
            margin: 0 0 14px 0;
            text-shadow: 0 0 12px rgba(56,189,248,0.55);
        }
        .hero-title {
            text-align: center;
            font-size: 3.1rem;
            font-weight: 700;
            line-height: 1.1;
            margin: 0 0 10px 0;
            background: linear-gradient(90deg, #38bdf8 0%, #818cf8 45%, #c084fc 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .hero-subtitle {
            text-align: center;
            color: #94a3b8;
            font-size: 1.02rem;
            max-width: 640px;
            margin: 0 auto 28px auto;
        }
        .hero-tags {
            display: flex; justify-content: center; gap: 10px; flex-wrap: wrap; margin-bottom: 30px;
        }
        .hero-tag {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            color: #7dd3fc;
            background: rgba(56,189,248,0.08);
            border: 1px solid rgba(56,189,248,0.25);
            padding: 5px 12px;
            border-radius: 999px;
        }

        /* ---- Status pill ---- */
        .status-pill {
            display: inline-flex; align-items: center; gap: 8px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
            padding: 7px 14px;
            border-radius: 999px;
            margin-bottom: 4px;
        }
        .status-ok { background: rgba(34,197,94,0.10); border: 1px solid rgba(34,197,94,0.35); color: #4ade80; }
        .status-warn { background: rgba(251,146,60,0.10); border: 1px solid rgba(251,146,60,0.35); color: #fb923c; }
        .dot { width: 8px; height: 8px; border-radius: 50%; background: currentColor; display: inline-block; }

        /* ---- Cards ---- */
        .glass-card {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px;
            padding: 22px 24px;
            margin-bottom: 18px;
        }

        /* ---- Metrics ---- */
        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 14px;
            padding: 16px 14px;
        }
        div[data-testid="stMetricLabel"] { color: #94a3b8 !important; }

        /* ---- Tabs ---- */
        .stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid rgba(255,255,255,0.08); }
        .stTabs [data-baseweb="tab"] {
            border-radius: 10px 10px 0 0;
            padding: 10px 18px;
            color: #94a3b8;
        }
        .stTabs [aria-selected="true"] {
            color: #38bdf8 !important;
            background: rgba(56,189,248,0.08) !important;
        }

        /* ---- Buttons ---- */
        .stButton > button, .stDownloadButton > button {
            background: linear-gradient(90deg, #0ea5e9, #6366f1);
            color: white;
            border: none;
            border-radius: 10px;
            font-weight: 600;
            padding: 0.6rem 1rem;
            transition: transform 0.15s ease, opacity 0.15s ease;
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            transform: translateY(-1px);
            opacity: 0.92;
        }

        /* ---- Sidebar ---- */
        section[data-testid="stSidebar"] {
            background: #0d1220;
            border-right: 1px solid rgba(255,255,255,0.06);
        }
        section[data-testid="stSidebar"] .stFileUploader {
            border-radius: 12px;
        }

        /* ---- Footer ---- */
        .footer {
            margin-top: 50px;
            padding-top: 22px;
            border-top: 1px solid rgba(255,255,255,0.08);
            text-align: center;
            color: #64748b;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
        }
        .footer a {
            color: #38bdf8;
            text-decoration: none;
            font-weight: 600;
        }
        .footer a:hover { text-decoration: underline; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# SYSTEM PROMPT (unchanged)
# ============================================================
SYSTEM_PROMPT = """You are an Automated Data Analysis Engine. You are given a CSV dataset (via computed statistics and retrieved sample rows) and must generate a complete, standalone analysis report — no user question will be provided.

You will receive:
1. Computed summary statistics (ground truth — use these for ALL numbers)
2. Retrieved representative/anomalous row samples (for illustration only, not for aggregation)

Generate a report with this structure:

1. **Dataset Overview** — rows/columns, column names & types, missing data summary
2. **Descriptive Statistics** — key numeric stats, categorical distributions
3. **Key Trends & Patterns** — correlations, group-level insights
4. **Anomalies & Outliers** — unusual values, data quality issues
5. **Notable Records** — 3-5 example rows illustrating key findings (cite row numbers)
6. **Summary & Recommendations** — 3-5 bullet takeaways, suggested next steps

Rules:
- Every number must come from the computed statistics block — never estimate or invent values.
- Use retrieved row samples only as illustrative examples, clearly labeled as samples.
- If a section can't be supported by the data given, omit it rather than guessing.
- Use markdown tables for statistics.
- Keep the tone objective and business-report style."""

RETRIEVER_QUERIES = [
    "highest values and top performing records",
    "lowest values and worst performing records",
    "most frequent categories and common patterns",
    "outliers or unusual values in the data",
    "records with extreme or unexpected values",
]


# ============================================================
# CORE LOGIC — same computations as the original script,
# just wrapped in functions so Streamlit can call them.
# ============================================================
def compute_stats(csv_path: str):
    """---------- 1. Compute real statistics with pandas (the ground truth) ----------"""
    df = pd.read_csv(csv_path)

    numeric_stats = df.describe().to_markdown()
    missing_values = df.isnull().sum().to_markdown()

    categorical_cols = df.select_dtypes(include='object').columns
    categorical_summary = ""
    for col in categorical_cols:
        categorical_summary += f"\n**{col}** top values:\n{df[col].value_counts().head(5).to_markdown()}\n"

    numeric_cols = df.select_dtypes(include='number').columns
    correlation_matrix = df[numeric_cols].corr().to_markdown() if len(numeric_cols) > 1 else "N/A (fewer than 2 numeric columns)"

    duplicate_count = df.duplicated().sum()

    computed_stats_block = f"""
DATASET SHAPE: {df.shape[0]} rows, {df.shape[1]} columns
COLUMNS: {', '.join(df.columns)}

NUMERIC SUMMARY STATISTICS:
{numeric_stats}

MISSING VALUES PER COLUMN:
{missing_values}

CATEGORICAL COLUMN DISTRIBUTIONS:
{categorical_summary}

CORRELATION MATRIX (numeric columns):
{correlation_matrix}

DUPLICATE ROWS: {duplicate_count}
"""
    return df, computed_stats_block


def build_retrieved_samples(csv_path: str):
    """---------- 2 & 3. Load, chunk, embed, retrieve illustrative rows ----------"""
    loader = CSVLoader(csv_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs_chunk = splitter.split_documents(docs)

    embedding_model = MistralAIEmbeddings(model='mistral-embed-2312')
    vectorstore = FAISS.from_documents(documents=docs_chunk, embedding=embedding_model)

    retriever = vectorstore.as_retriever(
        search_type='mmr',
        search_kwargs={'k': 5, 'fetch_k': 20, 'lambda_mult': 0.5}
    )

    seen_rows = set()
    sample_blocks = []
    for query in RETRIEVER_QUERIES:
        for doc in retriever.invoke(query):
            row_id = doc.metadata.get('row')       # <-- real CSVLoader key
            source = doc.metadata.get('source')    # <-- real CSVLoader key
            if row_id not in seen_rows:
                seen_rows.add(row_id)
                sample_blocks.append(
                    f"[Row {row_id} | Source: {source}]\n{doc.page_content}"
                )

    final_context = "\n\n---\n\n".join(sample_blocks)
    return final_context


def generate_report(computed_stats_block: str, final_context: str, model_name: str):
    """---------- 4. Build prompt safely and invoke the chain ----------"""
    prompt = ChatPromptTemplate.from_messages([
        ('system', SYSTEM_PROMPT),
        ('human', 'Computed statistics:\n{stats}\n\nRetrieved sample rows:\n{samples}')
    ])

    LLM = ChatMistralAI(model=model_name)
    chain = prompt | LLM if False else prompt | ChatMistralAI(model='mistral-medium-3-5')  # see note below
    response = chain.invoke({"stats": computed_stats_block, "samples": final_context})
    return response.content


# ============================================================
# API KEY — resolved from st.secrets / .env only (no UI input)
# ============================================================
def resolve_mistral_api_key() -> str:
    key = os.environ.get("MISTRAL_API_KEY", "")
    if not key:
        try:
            key = st.secrets.get("MISTRAL_API_KEY", "")
        except Exception:
            key = ""
    if key:
        os.environ["MISTRAL_API_KEY"] = key
    return key


mistral_key_present = bool(resolve_mistral_api_key())

# ============================================================
# SIDEBAR — dataset upload + run controls only
# ============================================================
with st.sidebar:
    st.markdown("### 🧠 DataLens AI")
    st.caption("pandas stats · FAISS/MMR retrieval · Mistral RAG report")

    st.markdown("")
    if mistral_key_present:
        st.markdown(
            '<div class="status-pill status-ok"><span class="dot"></span> Mistral API key connected</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-pill status-warn"><span class="dot"></span> No API key found</div>',
            unsafe_allow_html=True,
        )
        st.caption("Add `MISTRAL_API_KEY` to your `.streamlit/secrets.toml` or `.env` file.")

    st.divider()
    st.markdown("#### 📁 Upload Dataset")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"], label_visibility="collapsed")

    use_default = False
    if uploaded_file is None and os.path.exists("StudentPerformanceFactors.csv"):
        use_default = st.checkbox("Use default StudentPerformanceFactors.csv", value=False)

    st.divider()
    run_clicked = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

# ============================================================
# HERO
# ============================================================
st.markdown('<div class="hero-eyebrow">AUTONOMOUS DATA ANALYSIS ENGINE</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">DataLens AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Upload any CSV and get a ground-truth statistical breakdown, '
    'RAG-retrieved sample records, and a full AI-written analyst report — in seconds.</div>',
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="hero-tags">
        <span class="hero-tag">📐 pandas</span>
        <span class="hero-tag">🧬 FAISS + MMR</span>
        <span class="hero-tag">🔗 LangChain</span>
        <span class="hero-tag">🤖 Mistral</span>
    </div>
    """,
    unsafe_allow_html=True,
)

if "report" not in st.session_state:
    st.session_state.report = None
    st.session_state.stats_block = None
    st.session_state.samples = None
    st.session_state.df = None

# ---- Resolve the CSV path to use ----
csv_path = None
temp_dir = None
if uploaded_file is not None:
    temp_dir = tempfile.mkdtemp()
    csv_path = os.path.join(temp_dir, uploaded_file.name)
    with open(csv_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
elif use_default:
    csv_path = "StudentPerformanceFactors.csv"

# ---- Run pipeline ----
if run_clicked:
    if not mistral_key_present:
        st.error("No Mistral API key found. Add `MISTRAL_API_KEY` to `.streamlit/secrets.toml` or your `.env` file, then rerun.")
    elif not csv_path:
        st.error("Please upload a CSV file (or select the default dataset) before running.")
    else:
        try:
            with st.status("Running analysis pipeline...", expanded=True) as status:
                st.write("📐 Computing summary statistics with pandas...")
                df, computed_stats_block = compute_stats(csv_path)
                st.session_state.df = df
                st.session_state.stats_block = computed_stats_block

                st.write("✂️ Loading, chunking, and embedding rows...")
                st.write("🔎 Retrieving illustrative sample rows (MMR search)...")
                final_context = build_retrieved_samples(csv_path)
                st.session_state.samples = final_context

                st.write("🧠 Generating report with Mistral...")
                report = generate_report(computed_stats_block, final_context, "mistral-medium-3-5")
                st.session_state.report = report

                status.update(label="✅ Analysis complete", state="complete", expanded=False)
        except Exception as e:
            st.error(f"Something went wrong: {e}")

# ============================================================
# RESULTS
# ============================================================
if st.session_state.df is not None:
    df = st.session_state.df
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric("Missing Values", f"{int(df.isnull().sum().sum()):,}")
    c4.metric("Duplicate Rows", f"{int(df.duplicated().sum()):,}")
    st.markdown("<div style='margin-bottom: 8px;'></div>", unsafe_allow_html=True)

if st.session_state.report:
    tab_report, tab_stats, tab_samples, tab_preview = st.tabs(
        ["📄 Report", "📈 Computed Statistics", "🔍 Retrieved Samples", "🗂️ Data Preview"]
    )

    with tab_report:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown(st.session_state.report)
        st.markdown('</div>', unsafe_allow_html=True)
        st.download_button(
            "⬇️ Download Report (Markdown)",
            data=st.session_state.report,
            file_name="analysis_report.md",
            mime="text/markdown",
        )

    with tab_stats:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown(st.session_state.stats_block)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_samples:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.text(st.session_state.samples if st.session_state.samples else "No samples retrieved.")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_preview:
        st.dataframe(st.session_state.df, use_container_width=True)

elif not run_clicked:
    st.markdown(
        '<div class="glass-card" style="text-align:center; color:#94a3b8;">'
        '👈 Upload a CSV in the sidebar and click <b>Run Analysis</b> to get started.'
        '</div>',
        unsafe_allow_html=True,
    )

# ============================================================
# FOOTER
# ============================================================
st.markdown(
    """
    <div class="footer">
        Built by <b>Gaurav Gupta</b> &nbsp;·&nbsp;
        <a href="https://www.linkedin.com/in/gaurav-gupta-79754a377" target="_blank">Connect on LinkedIn</a>
    </div>
    """,
    unsafe_allow_html=True,
)
