# 🧠 DataLens AI — Automated Data Analysis Engine

DataLens AI turns any CSV file into a complete, AI-written analyst report in seconds. It combines **ground-truth statistics computed with pandas** with **RAG-retrieved illustrative sample rows** (FAISS + MMR search over Mistral embeddings), then hands both to a Mistral LLM to generate a structured, business-report-style analysis.

Built by **Gaurav Gupta** · [Connect on LinkedIn](https://www.linkedin.com/in/gaurav-gupta-79754a377)

---

## ✨ Features

- 📁 **Upload any CSV** — drag-and-drop from the sidebar, no code changes needed
- 📐 **Ground-truth stats** — pandas computes shape, describe(), missing values, categorical distributions, correlation matrix, and duplicates. The LLM is never allowed to invent numbers.
- 🧬 **RAG sample retrieval** — rows are chunked, embedded with `mistral-embed-2312`, indexed in FAISS, and retrieved via MMR search across multiple analytical queries (top performers, outliers, common patterns, etc.)
- 🤖 **AI-written report** — a Mistral chat model (`mistral-medium-3-5`) synthesizes the stats + samples into a 6-section markdown report: Overview, Descriptive Statistics, Trends & Patterns, Anomalies, Notable Records, and Recommendations
- 🎨 **Polished dark UI** — gradient hero header, glass-panel cards, live status pill for API key connectivity, tabbed results (Report / Statistics / Samples / Data Preview)
- ⬇️ **Download** the generated report as a `.md` file

---

## 🗂️ Project Structure

```
.
├── app.py                          # Streamlit app (UI + pipeline)
├── requirements.txt                # Python dependencies
└── .streamlit/
    └── secrets.toml.example        # Template for your Mistral API key
```

---

## ⚙️ Setup

### 1. Clone / download the project files

Make sure `app.py` and `requirements.txt` are in the same folder.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your Mistral API key

The key is **never entered in the UI** — it's read from Streamlit secrets or a `.env` file.

**Option A — Streamlit secrets (recommended):**

```bash
mkdir -p .streamlit
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Then edit `.streamlit/secrets.toml`:

```toml
MISTRAL_API_KEY = "your-mistral-api-key-here"
```

**Option B — `.env` file:**

Create a `.env` file in the project root:

```
MISTRAL_API_KEY=your-mistral-api-key-here
```

> Get an API key from the [Mistral AI Console](https://console.mistral.ai/).

### 4. Run the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## 🚀 Usage

1. Open the app — the sidebar shows a green **"Mistral API key connected"** pill if your key was found.
2. Upload a CSV file using the **Upload Dataset** panel in the sidebar.
3. Click **🚀 Run Analysis**.
4. Watch the pipeline run through three stages: statistics computation → embedding/retrieval → report generation.
5. Explore the results across five tabs:
   - **📊 Dashboard** — interactive BI-style charts (KPI tiles, correlation heatmap, missing-value chart, distribution histograms, outlier box plots, category breakdowns, and a relationship/scatter explorer). Computed directly from pandas + Plotly — instant, no LLM call.
   - **📄 Report** — the full AI-written analysis (downloadable as markdown)
   - **📈 Computed Statistics** — the raw pandas ground-truth numbers
   - **🔍 Retrieved Samples** — the illustrative rows pulled via MMR search
   - **🗂️ Data Preview** — the full uploaded dataset as a table

---

## 🧩 How It Works

```
CSV Upload
   │
   ├─► pandas ──► describe(), isnull(), value_counts(), corr(), duplicated()
   │                        │
   │                        ▼
   │              computed_stats_block (ground truth)
   │
   └─► CSVLoader ──► RecursiveCharacterTextSplitter ──► MistralAIEmbeddings
                                                              │
                                                              ▼
                                                         FAISS (MMR search)
                                                              │
                                                              ▼
                                                   retrieved sample rows
                        │                                    │
                        └──────────────┬─────────────────────┘
                                       ▼
                         ChatPromptTemplate + ChatMistralAI
                                       │
                                       ▼
                            Final AI-generated report
```

- **Statistics are the source of truth** — the system prompt explicitly instructs the model to never estimate or invent numbers; every figure must come from the computed stats block.
- **Retrieved rows are illustrative only** — used to cite concrete example records (e.g., "Row 42 shows...") but never used for aggregation.

---

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| UI | Streamlit |
| Data processing | pandas |
| Document loading & chunking | LangChain (`CSVLoader`, `RecursiveCharacterTextSplitter`) |
| Embeddings | `MistralAIEmbeddings` (`mistral-embed-2312`) |
| Vector store | FAISS (MMR search) |
| LLM | `ChatMistralAI` (`mistral-medium-3-5`) |

---

## 📄 License

This project is provided as-is for personal and educational use.

---

<p align="center">
Built with ❤️ by <b>Gaurav Gupta</b> · <a href="https://www.linkedin.com/in/gaurav-gupta-79754a377">LinkedIn</a>
</p>
