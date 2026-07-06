# 📚 AI Research Paper Intelligence System + Resume Analyzer

A two-in-one AI-powered platform built on top of 50,000 ArXiv Machine Learning abstracts:

1. **Semantic Paper Search** — find research papers by *meaning*, not keywords
2. **AI Resume Analyzer** — upload your resume and get personalised paper recommendations, skill extraction, job matching and a skill-gap report

Built as part of the Coding Blocks Internship program.

---

## 🎯 Features

### 🔍 Paper Search
- Semantic search over **50,000 ArXiv ML papers**
- Dense vector retrieval using Sentence-Transformers + FAISS
- KeyBERT keyword extraction
- **Named Entity Recognition (NER)** using spaCy — extracts Organizations, Models, Technical terms

### 📄 AI Resume Analyzer
- Upload PDF or DOCX resume
- Skill detection from resume text
- **Personalised paper recommendations** matched to your profile
- **Skill Gap Analysis** — 6 category breakdown
- **Job Role Matching** — suggests relevant ML/AI roles
- Top keyphrases extraction using KeyBERT

---

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| Dataset | CShorten/ML-ArXiv-Papers (Hugging Face) |
| Embeddings | sentence-transformers — all-MiniLM-L6-v2 |
| Vector Search | faiss-cpu — IndexFlatIP |
| NER | spaCy — en_core_web_sm |
| Keyword Extraction | KeyBERT |
| PDF Parsing | pypdf |
| DOCX Parsing | docx2txt |
| Web App | Streamlit |

---

## 🚀 Getting Started

### 1. Install dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Build the search index
```bash
python src/data_prep.py
python src/build_index.py
```

### 3. Launch the app
```bash
streamlit run src/app.py
```

---

## 🙋 Author
Built by **Foziya** — Coding Blocks Internship
