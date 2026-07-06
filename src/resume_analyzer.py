"""
resume_analyzer.py
-------------------
AI Resume Analyzer feature for the AI Research Paper Intelligence System.

Given an uploaded resume (PDF or plain text), this module:
  1. Extracts raw text from the PDF
  2. Identifies key ML/AI skills and experience using KeyBERT
  3. Finds the most relevant research papers from the FAISS index
     that match the candidate's background
  4. Generates an AI-powered insight report:
       - Detected skills summary
       - Paper recommendations aligned to their expertise
       - Skill gap suggestions (what to learn next in ML/AI)

Usage (standalone):
    from resume_analyzer import ResumeAnalyzer
    analyzer = ResumeAnalyzer(engine)          # pass your PaperSearchEngine
    report   = analyzer.analyze(pdf_bytes)
"""

import io
from typing import Optional

# PDF text extraction — PyPDF2 is already a lightweight transitive dep
try:
    from pypdf import PdfReader          # pypdf (newer, preferred)
except ImportError:
    from PyPDF2 import PdfReader         # PyPDF2 (older fallback)


# ------------------------------------------------------------------ #
# ML / AI skill taxonomy used for gap analysis
# ------------------------------------------------------------------ #
ML_AI_SKILLS = [
    # Core ML
    "machine learning", "deep learning", "neural networks", "supervised learning",
    "unsupervised learning", "reinforcement learning", "transfer learning",
    "feature engineering", "model evaluation", "cross-validation", "overfitting",

    # NLP / LLM
    "natural language processing", "nlp", "transformers", "bert", "gpt",
    "large language models", "llm", "text classification", "named entity recognition",
    "sentiment analysis", "text generation", "embeddings", "tokenization",

    # Computer Vision
    "computer vision", "image classification", "object detection", "cnn",
    "convolutional neural network", "image segmentation", "yolo", "resnet",

    # Frameworks
    "pytorch", "tensorflow", "keras", "scikit-learn", "hugging face",
    "langchain", "faiss", "xgboost", "lightgbm",

    # Data / MLOps
    "data preprocessing", "data pipeline", "mlops", "model deployment",
    "docker", "kubernetes", "aws", "gcp", "azure", "api", "fastapi",
    "pandas", "numpy", "matplotlib", "sql", "spark", "hadoop",

    # Research methods
    "research paper", "experimentation", "ablation study", "benchmarking",
    "hyperparameter tuning", "gradient descent", "backpropagation",
]

# Skill categories for gap report
SKILL_CATEGORIES = {
    "Core ML Algorithms":       ["machine learning", "supervised learning", "unsupervised learning",
                                  "reinforcement learning", "neural networks", "deep learning"],
    "NLP & Large Language Models": ["natural language processing", "transformers", "bert", "gpt",
                                     "llm", "embeddings", "text classification"],
    "Computer Vision":          ["computer vision", "cnn", "object detection", "image classification",
                                  "image segmentation"],
    "ML Frameworks":            ["pytorch", "tensorflow", "keras", "scikit-learn", "hugging face"],
    "MLOps & Deployment":       ["mlops", "docker", "fastapi", "model deployment", "aws", "gcp"],
    "Data Engineering":         ["pandas", "numpy", "sql", "spark", "data pipeline"],
}


# ------------------------------------------------------------------ #
# PDF → plain text
# ------------------------------------------------------------------ #
def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Read a PDF from raw bytes and return all page text as a single string.
    Works with both pypdf and PyPDF2 (same API).
    """
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    full_text = "\n".join(pages).strip()
    return full_text


# ------------------------------------------------------------------ #
# Core analyzer class
# ------------------------------------------------------------------ #
class ResumeAnalyzer:
    """
    Wraps the existing PaperSearchEngine to add resume-specific analysis.
    Designed to be instantiated once and reused across multiple resumes.
    """

    def __init__(self, engine):
        """
        Parameters
        ----------
        engine : PaperSearchEngine
            Already-initialised search engine (models + FAISS index loaded).
        """
        self.engine = engine

    # ------------------------------------------------------------------ #
    # Step 1 — Skill extraction
    # ------------------------------------------------------------------ #
    def extract_skills(self, resume_text: str, top_n: int = 15) -> list[tuple[str, float]]:
        """
        Use KeyBERT (backed by the same MiniLM model as the search engine)
        to pull the most representative ML/AI phrases from the resume.
        Returns a list of (phrase, score) tuples sorted by relevance.
        """
        keywords = self.engine.kw_model.extract_keywords(
            resume_text,
            keyphrase_ngram_range=(1, 3),
            stop_words="english",
            top_n=top_n,
        )
        return keywords

    # ------------------------------------------------------------------ #
    # Step 2 — Paper recommendations
    # ------------------------------------------------------------------ #
    def recommend_papers(self, resume_text: str, k: int = 5) -> list[dict]:
        """
        Embed the entire resume text and run a FAISS similarity search —
        the same way a query works, but the 'query' is the full resume.
        This surfaces papers that are semantically closest to the candidate's
        background, so results are personalised to their actual experience.
        """
        # Truncate to avoid overwhelming the embedding model
        truncated = resume_text[:3000]
        results = self.engine.search(truncated, k=k)

        # Enrich each result with a summary
        for r in results:
            if self.engine.summarizer is not None:
                r["summary"] = self.engine.summarize(r["abstract"])
        return results

    # ------------------------------------------------------------------ #
    # Step 3 — Skill gap analysis
    # ------------------------------------------------------------------ #
    def skill_gap_analysis(self, resume_text: str) -> dict:
        """
        Compare the resume text against the ML_AI_SKILLS taxonomy to find:
          - present_skills  : skills detected in the resume
          - missing_skills  : skills not found (potential gaps)
          - category_scores : coverage % per skill category
        """
        text_lower = resume_text.lower()

        present  = [s for s in ML_AI_SKILLS if s in text_lower]
        missing  = [s for s in ML_AI_SKILLS if s not in text_lower]

        category_scores = {}
        for category, skills in SKILL_CATEGORIES.items():
            found = [s for s in skills if s in text_lower]
            category_scores[category] = {
                "found": found,
                "missing": [s for s in skills if s not in text_lower],
                "score": round(len(found) / len(skills) * 100, 1),
            }

        return {
            "present_skills":  present,
            "missing_skills":  missing[:10],   # top 10 gaps to surface
            "category_scores": category_scores,
            "overall_coverage": round(len(present) / len(ML_AI_SKILLS) * 100, 1),
        }

    # ------------------------------------------------------------------ #
    # Step 4 — Full report (convenience method)
    # ------------------------------------------------------------------ #
    def analyze(self, pdf_bytes: bytes, k: int = 5) -> dict:
        """
        End-to-end pipeline: PDF bytes → full analysis report dict.

        Returns
        -------
        dict with keys:
            raw_text       : extracted resume text
            keywords       : KeyBERT top phrases
            papers         : recommended research papers
            gap_analysis   : skill gap breakdown
        """
        raw_text = extract_text_from_pdf(pdf_bytes)

        if not raw_text.strip():
            raise ValueError(
                "Could not extract text from this PDF. "
                "Make sure it is not a scanned image-only document."
            )

        keywords     = self.extract_skills(raw_text)
        papers       = self.recommend_papers(raw_text, k=k)
        gap_analysis = self.skill_gap_analysis(raw_text)

        return {
            "raw_text":     raw_text,
            "keywords":     keywords,
            "papers":       papers,
            "gap_analysis": gap_analysis,
        }
