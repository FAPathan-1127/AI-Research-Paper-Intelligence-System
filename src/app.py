"""
app.py  (updated)
-----------------
Streamlit front-end for the AI Research Paper Intelligence System.
Now includes two features accessible via tabs:

  Tab 1 — 🔍 Paper Search    (original semantic search feature)
  Tab 2 — 📄 Resume Analyzer (new AI-powered resume analysis feature)

Run with:
    streamlit run src/app.py
"""

import streamlit as st
from search_engine import PaperSearchEngine
from resume_analyzer import ResumeAnalyzer

# ------------------------------------------------------------------ #
# Page config
# ------------------------------------------------------------------ #
st.set_page_config(
    page_title="AI Research Paper Intelligence System",
    page_icon="📚",
    layout="wide",
)

# ------------------------------------------------------------------ #
# Shared CSS — minimal, clean, consistent across both tabs
# ------------------------------------------------------------------ #
st.markdown("""
<style>
    /* Card containers */
    .result-card {
        background: #f8f9fc;
        border-left: 4px solid #4f8ef7;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
    }
    /* Skill pill tags */
    .skill-pill {
        display: inline-block;
        background: #e8f0fe;
        color: #1a56db;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.8rem;
        margin: 3px;
    }
    /* Gap pill tags */
    .gap-pill {
        display: inline-block;
        background: #fff3cd;
        color: #856404;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.8rem;
        margin: 3px;
    }
    /* Category score bar label */
    .cat-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #374151;
    }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ #
# Load models once — shared between both tabs
# ------------------------------------------------------------------ #
@st.cache_resource(show_spinner="Loading AI models and index (first run only)...")
def get_engine() -> PaperSearchEngine:
    return PaperSearchEngine()


@st.cache_resource(show_spinner="Setting up Resume Analyzer...")
def get_analyzer(_engine: PaperSearchEngine) -> ResumeAnalyzer:
    return ResumeAnalyzer(_engine)


# ------------------------------------------------------------------ #
# App Header
# ------------------------------------------------------------------ #
st.title("📚 AI Research Paper Intelligence System")
st.caption(
    "Semantic search over 50,000 ArXiv ML papers — "
    "powered by Sentence-Transformers · FAISS · BART · KeyBERT"
)

# ------------------------------------------------------------------ #
# Tabs
# ------------------------------------------------------------------ #
tab_search, tab_resume = st.tabs(["🔍 Paper Search", "📄 AI Resume Analyzer"])


# ====================================================================== #
# TAB 1 — PAPER SEARCH  (original feature, unchanged)
# ====================================================================== #
with tab_search:

    with st.sidebar:
        st.header("⚙️ Search Settings")
        top_k        = st.slider("Number of results", 1, 10, 5)
        show_summary  = st.checkbox("Generate AI summary",  value=False)
        show_keywords = st.checkbox("Extract keywords",       value=False)
        st.markdown("---")
        st.markdown(
            "**How it works**\n\n"
            "1. Your query is embedded with `all-MiniLM-L6-v2`\n"
            "2. FAISS finds the closest papers by cosine similarity\n"
            "3. BART summarises each abstract\n"
            "4. KeyBERT extracts the key phrases"
        )

    query = st.text_input(
        "🔍 Search research papers",
        placeholder="e.g. deep learning for medical image analysis",
    )
    search_clicked = st.button("Search", type="primary")

    if search_clicked and query.strip():
        engine = get_engine()
        with st.spinner("Searching and analysing papers..."):
            results = engine.search(query, k=top_k)
            for r in results:
                if show_summary:
                    r["summary"] = engine.summarize(r["abstract"])
                if show_keywords:
                    r["keywords"] = engine.extract_keywords(r["abstract"])

        st.success(f"Found {len(results)} relevant papers")

        for r in results:
            with st.container(border=True):
                col1, col2 = st.columns([5, 1])
                with col1:
                    st.subheader(f"{r['rank']}. {r['title']}")
                with col2:
                    st.metric("Similarity", f"{r['score']:.2f}")

                if show_summary and "summary" in r:
                    st.markdown(f"**🧠 AI Summary:** {r['summary']}")

                with st.expander("View full abstract"):
                    st.write(r["abstract"])

                if show_keywords and "keywords" in r:
                    kw_tags = "  ".join(
                        f"`{kw}` ({score:.2f})" for kw, score in r["keywords"]
                    )
                    st.markdown(f"**🏷️ Keywords:** {kw_tags}")

    elif search_clicked:
        st.warning("Please enter a search query.")
    else:
        st.info("Enter a query above and click **Search** to explore the paper database.")


# ====================================================================== #
# TAB 2 — AI RESUME ANALYZER  (new feature)
# ====================================================================== #
with tab_resume:

    st.header("📄 AI Resume Analyzer")
    st.write(
        "Upload your resume and get an instant AI-powered breakdown: "
        "your top ML/AI skills, the most relevant research papers for your background, "
        "and a personalised skill-gap report to guide your learning."
    )

    # Settings sidebar (visible only inside this tab context)
    num_papers   = st.slider("Number of paper recommendations", 1, 10, 5, key="resume_k")
    top_keywords = st.slider("Keywords to extract", 5, 20, 12, key="resume_kw")

    # ------------------------------------------------------------------ #
    # Upload widget
    # ------------------------------------------------------------------ #
    uploaded_file = st.file_uploader(
        "Upload your resume (PDF only)",
        type=["pdf"],
        help="Text-based PDFs work best. Scanned/image PDFs may not extract correctly.",
    )

    analyze_clicked = st.button("🚀 Analyze Resume", type="primary", disabled=uploaded_file is None)

    if analyze_clicked and uploaded_file is not None:
        engine   = get_engine()
        analyzer = get_analyzer(engine)

        with st.spinner("Reading your resume and running analysis..."):
            try:
                pdf_bytes = uploaded_file.read()
                report    = analyzer.analyze(pdf_bytes, k=num_papers)
            except ValueError as e:
                st.error(str(e))
                st.stop()

        gap      = report["gap_analysis"]
        keywords = report["keywords"][:top_keywords]
        papers   = report["papers"]

        # ------------------------------------------------------------------ #
        # Section A — Detected Skills
        # ------------------------------------------------------------------ #
        st.markdown("---")
        st.subheader("🧠 Detected ML/AI Skills from Your Resume")

        present_skills = gap["present_skills"]
        if present_skills:
            pills = " ".join(
                f'<span class="skill-pill">✅ {s}</span>'
                for s in present_skills
            )
            st.markdown(pills, unsafe_allow_html=True)
        else:
            st.info("No common ML/AI skill keywords detected. Make sure your resume mentions tools and techniques explicitly.")

        st.caption(
            f"Overall ML/AI skill coverage: **{gap['overall_coverage']}%** "
            f"({len(present_skills)} of {len(present_skills) + len(gap['missing_skills'][:10])} checked keywords found)"
        )

        # ------------------------------------------------------------------ #
        # Section B — KeyBERT top phrases
        # ------------------------------------------------------------------ #
        st.markdown("---")
        st.subheader("🔑 Top Keyphrases in Your Resume")
        st.caption("These are the most semantically significant phrases KeyBERT identified — what the AI 'sees' as your expertise areas.")

        if keywords:
            cols = st.columns(3)
            for i, (phrase, score) in enumerate(keywords):
                with cols[i % 3]:
                    st.metric(label=phrase.title(), value=f"{score:.2f}", delta="relevance score")
        else:
            st.info("No keyphrases could be extracted.")

        # ------------------------------------------------------------------ #
        # Section C — Paper Recommendations
        # ------------------------------------------------------------------ #
        st.markdown("---")
        st.subheader("📚 Research Papers Matched to Your Profile")
        st.caption(
            "Your entire resume was embedded as a single semantic vector and searched "
            "against 50,000 ArXiv ML papers — results are papers closest in meaning to your background."
        )

        for r in papers:
            with st.container(border=True):
                col1, col2 = st.columns([5, 1])
                with col1:
                    st.subheader(f"{r['rank']}. {r['title']}")
                with col2:
                    st.metric("Match", f"{r['score']:.2f}")

                if "summary" in r:
                    st.markdown(f"**🧠 Why relevant:** {r['summary']}")

                with st.expander("Read full abstract"):
                    st.write(r["abstract"])

        # ------------------------------------------------------------------ #
        # Section D — Skill Gap Analysis
        # ------------------------------------------------------------------ #
        st.markdown("---")
        st.subheader("📊 Skill Gap Analysis by Category")
        st.caption("Coverage of core ML/AI skill areas detected in your resume:")

        for category, data in gap["category_scores"].items():
            score = data["score"]
            found = data["found"]
            missing_items = data["missing"]

            with st.expander(f"{category}  —  {score:.0f}% coverage"):
                col_l, col_r = st.columns(2)

                with col_l:
                    st.markdown("**✅ Found in your resume:**")
                    if found:
                        for s in found:
                            st.markdown(f"- {s}")
                    else:
                        st.write("None detected")

                with col_r:
                    st.markdown("**💡 Consider adding / learning:**")
                    if missing_items:
                        for s in missing_items:
                            st.markdown(f"- {s}")
                    else:
                        st.success("Great coverage in this area!")

                st.progress(int(score))

        # ------------------------------------------------------------------ #
        # Section E — Skill Gaps (top 10 across all categories)
        # ------------------------------------------------------------------ #
        st.markdown("---")
        st.subheader("🎯 Top Skills to Learn Next")
        st.caption(
            "These common ML/AI skills were not detected in your resume. "
            "Adding them (or the experience behind them) can strengthen your profile."
        )

        missing = gap["missing_skills"]
        if missing:
            pills = " ".join(
                f'<span class="gap-pill">📌 {s}</span>'
                for s in missing
            )
            st.markdown(pills, unsafe_allow_html=True)

            st.markdown(
                "\n💡 **Tip:** Search these topics in the **🔍 Paper Search** tab "
                "to find research papers that can help you get up to speed!"
            )
        else:
            st.success("Excellent! You have strong coverage across all major ML/AI skill areas.")

        # ------------------------------------------------------------------ #
        # Section F — Raw text preview (optional debug)
        # ------------------------------------------------------------------ #
        with st.expander("🔎 View extracted resume text (for debugging)"):
            st.text_area(
                "Extracted text",
                report["raw_text"][:3000] + ("..." if len(report["raw_text"]) > 3000 else ""),
                height=250,
                disabled=True,
            )

    elif not uploaded_file:
        st.info("👆 Upload a PDF resume above to get started.")
