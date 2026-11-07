"""Citebench Streamlit app.  Run with:  streamlit run app.py"""
from __future__ import annotations

import html
import os
import re
import tempfile

import streamlit as st

from citebench.compare import ClaimStatus
from citebench.ingest import Collection, chunk_text, fetch_arxiv, load_pdf_text, load_starter
from citebench.models import Family, Paper, StudyType
from citebench.nli import Support
from citebench.pipeline import Pipeline, PipelineResult

st.set_page_config(page_title="Citebench", page_icon="📑", layout="wide")

EXAMPLES = [
    "What approaches have researchers used to reduce hallucinations in language models?",
    "How can hallucinations be detected without access to the model's internals?",
    "Does retrieval augmentation improve factuality?",
    "How accurate are LLM citations?",
    "Can fine-tuning make hallucination worse?",
]

SUPPORT_STYLE = {
    Support.SUPPORTED: ("#d9f2e3", "#1f7a4d", "Supported by the cited passage"),
    Support.PARTIAL: ("#fbf0cf", "#8a5d00", "Partly supported"),
    Support.UNSUPPORTED: ("#f9dcda", "#a8322b", "Not supported by the cited passage"),
    Support.CONTRADICTED: ("#f3c9c6", "#7d1d17", "Contradicted by the cited passage"),
    Support.UNCITED: ("transparent", "inherit", "No citation"),
}
CLAIM_STYLE = {
    ClaimStatus.BACKED: ("#d9f2e3", "#1f7a4d", "Backed by the papers"),
    ClaimStatus.PARTIAL: ("#fbf0cf", "#8a5d00", "Partly backed"),
    ClaimStatus.NOT_IN_PAPERS: ("#e6e8ee", "#4a5163", "Not in these papers (can't verify here)"),
    ClaimStatus.CONTRADICTED: ("#f3c9c6", "#7d1d17", "Contradicted by a paper"),
}
STRENGTH_STYLE = {"strong": ("#d9f2e3", "#1f7a4d"), "moderate": ("#fbf0cf", "#8a5d00"), "limited": ("#f9dcda", "#a8322b")}

st.markdown("""
<style>
.cb-chip{font:500 11px ui-monospace,Menlo,monospace;background:#fbe7a1;color:#3b2f00;border-radius:3px;
  padding:1px 4px;margin:0 1px;cursor:help;white-space:nowrap}
.cb-chip.bad{background:#f9dcda;color:#a8322b;text-decoration:line-through}
.cb-sent{border-radius:3px;padding:1px 2px;cursor:help}
.cb-lead{font-family:Georgia,'Times New Roman',serif;font-size:1.12rem;line-height:1.65}
.cb-badge{font:600 11px system-ui;text-transform:uppercase;letter-spacing:.05em;border-radius:4px;padding:2px 7px}
.cb-meta{color:#6b7280;font-size:.85rem}
.cb-legend span{margin-right:10px;font-size:.8rem}
.cb-psg{border:1px solid rgba(128,128,128,.3);border-radius:8px;padding:10px 12px;margin-bottom:10px}
.cb-psg p{font-family:Georgia,serif;margin:.4rem 0}
</style>""", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------
def get_pipeline() -> Pipeline:
    if "pipeline" not in st.session_state:
        with st.spinner("Loading papers and building the search index…"):
            st.session_state.pipeline = Pipeline(load_starter())
    return st.session_state.pipeline


def esc(s: object) -> str:
    return html.escape(str(s), quote=True)


def chip(label: str, result: PipelineResult, bad: bool = False) -> str:
    r = next((x for x in result.retrieved if x.label == label), None)
    tip = f"{r.paper.citation_label()}: {r.passage.text}" if r else "Not among the retrieved passages"
    return f'<span class="cb-chip{" bad" if bad or not r else ""}" title="{esc(tip)}">{esc(label)}</span>'


def render_highlighted(sentences, result: PipelineResult) -> str:
    parts = []
    for s in sentences:
        bg, fg, why = SUPPORT_STYLE[s.support]
        tip = why
        if s.best:
            tip += f' · {s.support.value} ({s.entailment:.0%}) · best evidence [{s.best.label}]: "{s.best.sentence}"'
        chips = "".join(chip(c, result) for c in s.citations)
        style = f"background:{bg};color:{fg}" if s.support is not Support.UNCITED else ""
        parts.append(f'<span class="cb-sent" style="{style}" title="{esc(tip)}">{esc(s.text)}</span> {chips}')
    return " ".join(parts)


def render_plain_with_chips(text: str, result: PipelineResult) -> str:
    return re.sub(r"\[(P\d+)\]", lambda m: chip(m.group(1), result), esc(text))


def legend() -> str:
    items = [s for s in Support if s is not Support.UNCITED]
    return '<div class="cb-legend">' + "".join(
        f'<span><span class="cb-sent" style="background:{SUPPORT_STYLE[s][0]};color:{SUPPORT_STYLE[s][1]}">'
        f'{s.value}</span></span>' for s in items) + "</div>"


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    key_in = st.text_input("Anthropic API key (optional)", type="password",
                           help="Kept in this browser session only. To use a free local model instead, set "
                                "CITEBENCH_PROVIDER=ollama in .env. With no LLM the app runs offline "
                                "(extractive answers, no memory-only comparison).")
    if key_in and key_in != st.session_state.get("api_key"):
        from citebench.llm import ClaudeLLM
        try:
            get_pipeline().llm = ClaudeLLM(api_key=key_in)
            st.session_state.api_key = key_in
            st.success("Claude connected for this session.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Couldn't set up Claude: {exc}")
    top_k = st.slider("Passages to retrieve", 4, 20, 12)
    with_review = st.toggle("Reviewer 2 pass", value=True)
    with_compare = st.toggle("With vs. without docs", value=True, help="Needs Claude.")
    pipe = get_pipeline()
    st.divider()
    st.caption(
        f"**LLM:** {getattr(pipe.llm, 'model', 'configured') if pipe.llm else 'offline (extractive)'}  \n"
        f"**Search:** {pipe.retriever.mode}  \n"
        f"**Papers:** {len(pipe.collection)} · {len(pipe.collection.passages)} passages"
    )

# --------------------------------------------------------------------------
# Header + query
# --------------------------------------------------------------------------
st.title("Citebench")
st.caption("Ask a research question. Answers come from retrieved passages across many papers, every citation is "
           "checked, and individual findings are kept separate from what the overall evidence supports.")

if "question" not in st.session_state:
    st.session_state.question = EXAMPLES[0]
cols = st.columns(len(EXAMPLES))
for c, ex in zip(cols, EXAMPLES):
    if c.button(ex, width="stretch"):
        st.session_state.question = ex
        st.session_state.pop("result", None)

with st.form("ask"):
    question = st.text_input("Research question", key="question")
    submitted = st.form_submit_button("Search papers", type="primary")

if submitted or "result" not in st.session_state:
    with st.spinner("Retrieving, answering, checking citations…"):
        try:
            st.session_state.result = pipe.run(question, with_review=with_review,
                                               with_comparison=with_compare, top_k=top_k)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Something went wrong while answering: {exc}")
            st.stop()

result: PipelineResult = st.session_state.result
tab_answer, tab_compare, tab_review, tab_lib = st.tabs(
    ["Answer", "With vs. without docs", "Reviewer 2", "Library"])

# --------------------------------------------------------------------------
# Answer tab
# --------------------------------------------------------------------------
with tab_answer:
    left, right = st.columns([1.45, 1], gap="large")
    with left:
        mode = getattr(pipe.llm, "model", "LLM") if result.answer.mode == "llm" else "Extractive (offline)"
        st.subheader("Answer")
        st.caption(f"{mode} · hover a sentence to see its best evidence, hover a [P#] to read the passage")
        if result.highlights:
            st.markdown(f'<div class="cb-lead">{render_highlighted(result.highlights.summary, result)}</div>',
                        unsafe_allow_html=True)
            st.markdown(legend(), unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="cb-lead">{render_plain_with_chips(result.answer.summary, result)}</div>',
                        unsafe_allow_html=True)

        st.subheader("What individual papers found")
        for f in result.answer.findings:
            paper = pipe.collection.papers.get(f.paper_key)
            if not paper:
                continue
            body = (render_highlighted(result.highlights.findings.get(f.paper_key, []), result)
                    if result.highlights else esc(f.finding) + "".join(chip(c, result) for c in f.citations))
            st.markdown(
                f"**{esc(paper.title)}**  \n<span class='cb-meta'>{esc(paper.authors)} · {esc(paper.year)} · "
                f"{esc((f.study_type or paper.study_type).value)} · {esc(paper.family.value)}</span><br>{body}",
                unsafe_allow_html=True)

        st.subheader("What the overall evidence supports")
        for e in result.answer.evidence:
            bg, fg = STRENGTH_STYLE[e.strength.value]
            note = " · lowered by the citation check" if e.downgraded else ""
            st.markdown(
                f'<span class="cb-badge" style="background:{bg};color:{fg}">{e.strength.value}</span> '
                f'{esc(e.claim)} {"".join(chip(c, result) for c in e.citations)}<br>'
                f'<span class="cb-meta">{esc(e.why)}{note}</span>', unsafe_allow_html=True)
        if result.answer.tensions:
            st.markdown("**Tensions and caveats**")
            for t in result.answer.tensions:
                st.markdown(f"- {esc(t.point)} {''.join(chip(c, result) for c in t.citations)}",
                            unsafe_allow_html=True)
        if result.answer.gaps:
            st.markdown("**Not established by these passages**")
            for g in result.answer.gaps:
                st.markdown(f"- {g}")

        st.subheader("Citation check")
        rep = result.draft_report or result.report
        m = st.columns(4)
        m[0].metric("Citations checked", rep.total)
        m[1].metric("Resolve to a passage", rep.resolved)
        m[2].metric("Weak support", rep.weak)
        m[3].metric("Invented / misattributed", rep.invented + rep.misattributed)
        if rep.issues:
            with st.expander(f"{len(rep.issues)} issue(s) found in the draft", expanded=not rep.ok):
                for i in rep.issues:
                    icon = "❌" if i.kind.is_error else "⚠️"
                    st.markdown(f"{icon} **{i.kind.value}** · `{i.location}` · {i.message}")
            st.caption("Invalid citations are removed and inflated ratings lowered before the answer is shown.")
        else:
            st.success("Every citation points to a retrieved passage from the paper it is attributed to.")

    with right:
        st.subheader("Retrieved passages")
        st.caption(f"{len(result.retrieved)} passages from {len({r.paper.key for r in result.retrieved})} papers · "
                   f"{result.retrieval_mode} search")
        for r in result.retrieved:
            link = f'<a href="{esc(r.paper.url)}" target="_blank">{esc(r.paper.title)} ↗</a>' if r.paper.url \
                else esc(r.paper.title)
            st.markdown(
                f'<div class="cb-psg"><span class="cb-chip">{r.label}</span> <b>{esc(r.paper.authors)} '
                f'{esc(r.paper.year)}</b> <span class="cb-meta">score {r.score:.3f}</span>'
                f'<p>{esc(r.passage.text)}</p><span class="cb-meta">{link}</span></div>',
                unsafe_allow_html=True)

# --------------------------------------------------------------------------
# With vs without docs
# --------------------------------------------------------------------------
with tab_compare:
    cmp = result.comparison
    if not result.llm_used:
        st.info("This view needs an LLM: set CITEBENCH_PROVIDER=ollama in .env for a free local model, or add an "
                "Anthropic API key in the sidebar. It asks the same question "
                "with no documents, then checks every claim of that memory-only answer against the papers.")
    elif cmp is None:
        st.info("Turn on “With vs. without docs” in the sidebar and search again.")
    else:
        s = cmp.stats()
        m = st.columns(4)
        m[0].metric("Memory-only claims", s["claims"])
        m[1].metric("Backed by papers", f"{cmp.backed_rate:.0%}")
        m[2].metric("Not in these papers", s["not_in_papers"])
        m[3].metric("Contradicted", s["contradicted"])
        a, b = st.columns(2, gap="large")
        with a:
            st.subheader("Without documents")
            st.caption("Claude answering from memory. Each claim is checked against the whole collection.")
            for c in cmp.claims:
                bg, fg, why = CLAIM_STYLE[c.status]
                ev = f' · [{c.evidence.label}] "{c.evidence.sentence}"' if c.evidence and \
                    c.status is not ClaimStatus.NOT_IN_PAPERS else ""
                st.markdown(f'<div style="margin-bottom:6px"><span class="cb-badge" style="background:{bg};'
                            f'color:{fg}">{c.status.value.replace("_", " ")}</span> '
                            f'<span title="{esc(why + ev)}">{esc(c.claim)}</span></div>', unsafe_allow_html=True)
            st.caption("“Not in these papers” means the claim can't be checked against this collection, "
                       "not that it is false.")
        with b:
            st.subheader("With documents")
            st.caption("The grounded, cited answer.")
            st.markdown(f'<div class="cb-lead">{render_plain_with_chips(result.answer.summary, result)}</div>',
                        unsafe_allow_html=True)
            if cmp.papers_unused_by_memory:
                names = ", ".join(pipe.collection.papers[k].citation_label() for k in cmp.papers_unused_by_memory
                                  if k in pipe.collection.papers)
                st.markdown(f"**Papers the memory-only answer never drew on:** {esc(names)}")

# --------------------------------------------------------------------------
# Reviewer 2
# --------------------------------------------------------------------------
with tab_review:
    rv = result.review
    if rv is None:
        st.info("Reviewer 2 is off. Turn it on in the sidebar and search again.")
    else:
        st.markdown(f"**Verdict:** {rv.verdict}" + ("" if rv.by_llm else " · offline: notes come from the citation check"))
        if not rv.notes:
            st.success("No problems raised.")
        for n in rv.notes:
            st.markdown(f"- **{n.kind.value.replace('_', ' ')}** · `{n.target}` · {esc(n.comment)}",
                        unsafe_allow_html=True)
        if rv.revised is not None and result.draft is not None:
            a, b = st.columns(2)
            a.markdown("**Draft summary**")
            a.markdown(render_plain_with_chips(result.draft.summary, result), unsafe_allow_html=True)
            b.markdown("**Revised summary**")
            b.markdown(render_plain_with_chips(result.answer.summary, result), unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Library
# --------------------------------------------------------------------------
with tab_lib:
    col: Collection = pipe.collection
    st.subheader(f"{len(col)} papers in the collection")
    st.dataframe(
        [{"Title": p.title, "Authors": p.authors, "Year": p.year, "Approach": p.family.value,
          "Type": p.study_type.value, "Passages": sum(x.paper_key == p.key for x in col.passages), "Link": p.url}
         for p in col.papers.values()],
        width="stretch", hide_index=True,
        column_config={"Link": st.column_config.LinkColumn("Link")})

    families = [f.value for f in Family]
    types = [t.value for t in StudyType]
    up, ax = st.columns(2, gap="large")
    with up:
        st.markdown("**Add a PDF**")
        with st.form("pdf", clear_on_submit=True):
            pdf = st.file_uploader("PDF file", type=["pdf"])
            title = st.text_input("Title")
            authors = st.text_input("Authors", placeholder="Smith et al.")
            year = st.text_input("Year", placeholder="2024")
            url = st.text_input("Link (optional)")
            fam = st.selectbox("Approach", families, key="pdf_fam")
            typ = st.selectbox("Study type", types, key="pdf_type")
            if st.form_submit_button("Add PDF"):
                if not pdf or not title:
                    st.error("Choose a PDF and give it a title.")
                else:
                    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                        tmp.write(pdf.getvalue())
                    try:
                        text = load_pdf_text(tmp.name)
                    finally:
                        os.unlink(tmp.name)
                    key = re.sub(r"\W+", "", (authors.split()[0] if authors else "paper").lower()) + (year or "") \
                        + f"_{len(col)}"
                    try:
                        col.add(Paper(key=key, title=title, authors=authors or "Unknown", year=year or "n.d.",
                                      url=url, family=Family(fam), study_type=StudyType(typ), text=text))
                        pipe.rebuild_index()
                        st.success(f"Added “{title}” as {len(chunk_text(text))} passages.")
                    except ValueError as exc:
                        st.error(str(exc))
    with ax:
        st.markdown("**Add from arXiv**")
        with st.form("arxiv", clear_on_submit=True):
            aid = st.text_input("arXiv ID or URL", placeholder="2305.14251")
            full = st.checkbox("Use the full PDF text (slower), not just the abstract")
            fam2 = st.selectbox("Approach", families, key="ax_fam")
            typ2 = st.selectbox("Study type", types, key="ax_type")
            if st.form_submit_button("Fetch and add"):
                try:
                    with st.spinner("Fetching from arXiv…"):
                        paper = fetch_arxiv(aid, full_text=full, family=Family(fam2), study_type=StudyType(typ2))
                    col.add(paper)
                    pipe.rebuild_index()
                    st.success(f"Added “{paper.title}”.")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Couldn't add that paper: {exc}")
    st.caption("Added papers live in this session. Use Collection.save() to keep a collection on disk.")
