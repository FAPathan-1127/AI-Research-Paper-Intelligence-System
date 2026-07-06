# 📚 AI Research Paper Intelligence System + Resume Analyzer

A two-in-one AI-powered platform built on top of 50,000 ArXiv Machine Learning abstracts:

1. **Semantic Paper Search** — find research papers by *meaning*, not keywords
2. **AI Resume Analyzer** — upload your resume and get personalised paper recommendations, skill extraction, and a skill-gap report

Built as part of the Coding Blocks Internship program.

---

## 🎯 What it does

### Feature 1 — Semantic Paper Search
Type a natural-language query like:
> "deep learning for medical image analysis"

The system:
- Converts your query into a **384-dimensional semantic vector**
- Retrieves the top-k most similar papers using **cosine similarity over a FAISS index**
- **Summarizes** each abstract using a BART-based transformer
- **Extracts key topics** using KeyBERT

### Feature 2 — AI Resume Analyzer *(new)*
Upload your resume (PDF) and get:
- **Detected ML/AI skills** — what the AI finds in your resume
- **Top keyphrases** — the most semantically significant terms in your profile
- **Personalised paper recommendations** — papers matched to *your specific background*
- **Skill gap analysis** — category-wise coverage of ML/AI skills with what to learn next

---

## 🧠 System Architecture

```
                     ┌─────────────────────────┐
                     │   ArXiv ML Papers (HF)   │
                     │  ~50,000 title+abstract  │
                     └────────────┬─────────────┘
                                  │  clean & merge
                                  ▼
                     ┌─────────────────────────┐
                     │   Sentence-Transformer   │
                     │    (all-MiniLM-L6-v2)    │
                     │   text --> 384-dim vec   │
                     └────────────┬─────────────┘
                                  │
                                  ▼
                     ┌─────────────────────────┐
                     │      FAISS Index         │
                     │  (Inner Product / cosine)│
                     └────────────┬─────────────┘
                                  │
          user query ──encode────┘         PDF Resume ──extract──┐
                     │                                            │
                     ▼                                            ▼
          ┌──────────────────┐                    ┌──────────────────────┐
          │  Top-K Papers    │                    │  Resume Text + Skills│
          └────────┬─────────┘                    └──────────┬───────────┘
                   │                                         │
       ┌───────────┴──────────┐               ┌─────────────┼──────────────┐
       ▼                      ▼               ▼             ▼              ▼
  ┌─────────┐          ┌──────────┐    ┌───────────┐ ┌──────────┐ ┌────────────┐
  │  BART   │          │ KeyBERT  │    │  KeyBERT  │ │  FAISS   │ │ Skill Gap  │
  │Summarize│          │ Keywords │    │  Phrases  │ │  Papers  │ │  Analysis  │
  └─────────┘          └──────────┘    └───────────┘ └──────────┘ └────────────┘
                                               │
                                               ▼
                                  ┌─────────────────────────┐
                                  │      Streamlit UI        │
                                  │  Tab 1: Paper Search     │
                                  │  Tab 2: Resume Analyzer  │
                                  └─────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| Dataset | [CShorten/ML-ArXiv-Papers](https://huggingface.co/datasets/CShorten/ML-ArXiv-Papers) |
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` (384-dim) |
| Vector Search | `faiss-cpu` — `IndexFlatIP` (cosine similarity) |
| Summarization | `transformers` — `sshleifer/distilbart-cnn-12-6` |
| Keyword Extraction | `keybert` |
| PDF Parsing | `pypdf` |
| Data handling | `pandas`, `numpy` |
| Web App | `streamlit` |

---

## 📂 Project Structure

```
AI-Research-Paper-Intelligence-System/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md                         # explains generated data files
├── notebooks/
│   ├── 01_EDA_and_Embeddings.ipynb       # data exploration + embedding walkthrough
│   └── 02_Search_Engine.ipynb            # FAISS + summarization + keyword walkthrough
└── src/
    ├── data_prep.py                       # load & clean raw dataset
    ├── build_index.py                     # generate embeddings + build FAISS index
    ├── search_engine.py                   # PaperSearchEngine class
    ├── resume_analyzer.py                 # ResumeAnalyzer class (new feature)
    └── app.py                             # Streamlit web app (both tabs)
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/AI-Research-Paper-Intelligence-System.git
cd AI-Research-Paper-Intelligence-System
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Build the search index (one-time setup)

Downloads the dataset, generates embeddings for ~50,000 papers, and builds the FAISS index.
This step is compute-heavy (~20–30 min on CPU) but only needs to run once — results are cached in `data/`.

```bash
python src/data_prep.py
python src/build_index.py
```

### 4. Launch the app

```bash
streamlit run src/app.py
```

Open the local URL Streamlit prints (usually `http://localhost:8501`).

---

## 🔍 Feature 1 — Paper Search (How to use)

1. Go to the **🔍 Paper Search** tab
2. Type any natural-language query (e.g. "attention mechanism in transformers")
3. Adjust the number of results and toggle summary/keywords in the sidebar
4. Click **Search**

---

## 📄 Feature 2 — AI Resume Analyzer (How to use)

1. Go to the **📄 AI Resume Analyzer** tab
2. Upload your resume as a **PDF file**
3. Adjust number of paper recommendations using the slider
4. Click **🚀 Analyze Resume**

You will see:
- ✅ **Detected ML/AI Skills** — skills found in your resume
- 🔑 **Top Keyphrases** — KeyBERT's view of your core expertise
- 📚 **Paper Recommendations** — ArXiv papers matched to your profile
- 📊 **Skill Gap Analysis** — category-wise breakdown (Core ML, NLP, CV, MLOps, etc.)
- 🎯 **Skills to Learn Next** — top missing skills with tips

> 💡 After the analysis, click **🔍 Paper Search** tab and search those missing skills to find research papers that can help you learn them!

---

## 💡 Key Design Decisions

- **Why FAISS `IndexFlatIP`?** For 50k papers, an exact flat index guarantees perfect recall with no accuracy trade-off from approximate search — fast enough at this scale.
- **Why normalize embeddings before indexing?** Cosine similarity depends only on vector direction. Normalizing to unit length lets FAISS's Inner Product search give mathematically identical results to cosine similarity.
- **Why `all-MiniLM-L6-v2`?** Strong balance of speed and semantic quality — 384 dimensions is compact enough for instant querying while capturing rich sentence-level meaning.
- **Why embed the full resume for paper recommendations?** Instead of extracting a few keywords and querying with those, embedding the entire resume text gives a richer, more personalised match — the FAISS search sees the candidate's full semantic profile.
- **Why a skill taxonomy for gap analysis?** KeyBERT finds what's in the resume, but doesn't know what's *missing*. A curated ML/AI taxonomy (40+ skills across 6 categories) lets us show concrete, actionable gaps.
- **Why cache embeddings/index to disk?** Re-encoding 50,000 papers takes ~20–30 minutes; caching means the app starts in seconds on every subsequent run.

---

## 🔮 Possible Extensions

- Swap `IndexFlatIP` for `IndexIVFFlat` / `IndexHNSW` to scale to millions of papers
- Add year/category filters alongside semantic search
- Deploy publicly on Streamlit Community Cloud or Hugging Face Spaces
- Add ATS (Applicant Tracking System) score to the Resume Analyzer
- Add a "papers similar to this one" recommendation feature
- Support DOCX resume uploads in addition to PDF

---

## 🙋 Author

Built by **Foziya** — Coding Blocks Internship, Project: AI Research Paper Intelligence System.
