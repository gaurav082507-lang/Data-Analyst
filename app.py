"""
Automated Data Analysis Engine — Streamlit UI

Wraps the original pandas + LangChain (Mistral) RAG pipeline in a Streamlit
front end. The core logic (statistics computation, CSV loading/chunking,
embedding, MMR retrieval, prompt, and chain) is UNCHANGED from the original
script — only reorganized into functions and connected to UI elements
(file uploader, API key input, buttons, tabs).
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
    page_title="Automated Data Analysis Engine",
    page_icon="📊",
    layout="wide",
)

# ---- light styling polish ----
st.markdown(
    """
    <style>
        .block-container {padding-top: 2rem; padding-bottom: 3rem;}
        div[data-testid="stMetric"] {
            background-color: rgba(120,120,120,0.08);
            border-radius: 10px;
            padding: 12px 10px;
        }
        .stTabs [data-baseweb="tab-list"] {gap: 4px;}
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px 8px 0 0;
            padding: 8px 16px;
        }
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
# SIDEBAR — API key + file upload + run controls
# ============================================================
with st.sidebar:
    st.title("📊 Data Analysis Engine")
    st.caption("pandas stats + FAISS/MMR retrieval + Mistral RAG report")

    st.divider()
    st.subheader("🔑 Mistral API Key")
    env_key = os.environ.get("MISTRAL_API_KEY", "")
    api_key_input = st.text_input(
        "MISTRAL_API_KEY",
        value=env_key,
        type="password",
        help="Loaded from .env if present. You can override it here.",
    )
    if api_key_input:
        os.environ["MISTRAL_API_KEY"] = api_key_input

    st.divider()
    st.subheader("📁 Upload Dataset")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    use_default = False
    if uploaded_file is None and os.path.exists("StudentPerformanceFactors.csv"):
        use_default = st.checkbox("Use default StudentPerformanceFactors.csv", value=False)

    st.divider()
    run_clicked = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

# ============================================================
# MAIN AREA
# ============================================================
st.title("Automated Data Analysis Engine")
st.write("Upload a CSV in the sidebar, then click **Run Analysis** to generate a full statistical + AI-written report.")

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
    if not os.environ.get("MISTRAL_API_KEY"):
        st.error("Please provide a Mistral API key in the sidebar before running.")
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

    st.divider()

if st.session_state.report:
    tab_report, tab_stats, tab_samples, tab_preview = st.tabs(
        ["📄 Report", "📈 Computed Statistics", "🔍 Retrieved Samples", "🗂️ Data Preview"]
    )

    with tab_report:
        st.markdown(st.session_state.report)
        st.download_button(
            "⬇️ Download Report (Markdown)",
            data=st.session_state.report,
            file_name="analysis_report.md",
            mime="text/markdown",
        )

    with tab_stats:
        st.markdown(st.session_state.stats_block)

    with tab_samples:
        st.text(st.session_state.samples if st.session_state.samples else "No samples retrieved.")

    with tab_preview:
        st.dataframe(st.session_state.df, use_container_width=True)

elif not run_clicked:
    st.info("👈 Upload a CSV and click **Run Analysis** in the sidebar to get started.")
