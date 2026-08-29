"""
app.py — CSAT Analyzer.
=======================
Upload customer feedback, or open one of six datasets that ship already
labeled, and get a dashboard back.

The flow is a sequence of screens rather than a scrolling page. Streamlit has
no router, so the current screen lives in session state and this file dispatches
on it:

    landing -> why -> choose -> [check -> topics -> run] -> dashboard

An uploaded file walks the whole path. A prepared dataset jumps from "choose"
to "run" and then to the dashboard, because the labeling was done offline months
ago and the answers are already on disk — which is what makes exploring this
tool free and instant.
"""

from __future__ import annotations

import html
import os
import time

import pandas as pd
import streamlit as st

import app_data as D
from app_style import INK, INK_SOFT, MUTED, PANEL, RULE, css

st.set_page_config(
    page_title="CSAT Analyzer", page_icon="📊", layout="wide",
    # The filters live in the sidebar, so on the dashboard it has to start open.
    # Left collapsed, every screen before it gets a clean full-width page.
    initial_sidebar_state=("expanded" if st.session_state.get("step") == "dashboard"
                           else "collapsed"),
)
# st.html injects raw HTML without a markdown pass — see app_style.css()
st.html(css())

STEPS = [("choose", "Choose data"), ("check", "Check it"),
         ("topics", "Confirm topics"), ("run", "Analyze"), ("dashboard", "Dashboard")]


# One step back from each screen. The dashboard goes back to the dataset picker
# whichever way you reached it, because that is the choice you'd want to change.
PREVIOUS = {"why": "landing", "choose": "why", "check": "choose",
            "topics": "check", "run": "topics", "dashboard": "choose"}

ROTATING = [
    "Transform thousands of customer survey responses into actionable, AI powered summaries",
    "Automated topic, keyword, and sentiment tagging reveals patterns, trends, and valuable customer insights.",
    "Filter the feedback and explore interactive analytics dashboards to uncover the story behind your data",
]

FACTS = [
    "Before anything is labeled, Claude reads a sample of 200 reviews drawn across every rating and proposes a topic list fitted to this dataset.",
    "You approve that list, and then it locks. Every review is sorted against the same fixed topics — which is what makes the counts comparable.",
    "If the model invented labels as it went, &ldquo;Slow service&rdquo; and &ldquo;Service was slow&rdquo; would become two bars, and neither count would be true.",
    "Each review is then read in full and given its topics, keyword phrases taken from its own text, a sentiment, and a one-sentence summary.",
    "Sentiment has three classes, not two. &ldquo;Mixed&rdquo; is a real category for genuinely mixed experiences, never a fallback for uncertainty.",
    "Every summary is one sentence of twenty words or fewer, capturing the main issue or praise and adding no fact the reviewer didn&rsquo;t write.",
    "Each response is validated against a schema before it is kept. An invented category or an unexpected sentiment is rejected and retried, never quietly corrected.",
    "Ten reviews go into every request and ten requests run at once — turning a job that would take eight hours in sequence into about five minutes.",
    "On a 150-review test, two independently-run models agreed at 0.88 on Cohen&rsquo;s kappa — a score corrected for the agreement chance alone would produce.",
    "Reviews matching no topic are counted and reported. A low rate means the taxonomy fits; a high rate on long reviews means it is missing a theme.",
]


# ---------------------------------------------------------------------------
# state
# ---------------------------------------------------------------------------
def state(k, default=None):
    if k not in st.session_state:
        st.session_state[k] = default
    return st.session_state[k]


def go(step: str):
    """
    Move to a screen.

    The whole app lives at one URL, so the browser's own back button has nothing
    to go back to — pressing it leaves the app and you return to the first
    screen. Every screen therefore carries its own back control, and the
    dashboard ends with somewhere to go next. See back_link() and the panel at
    the foot of the dashboard.
    """
    st.session_state.step = step
    st.rerun()


state("step", "landing")
state("dataset", None)
state("upload", None)
state("opened", None)   # the dataset the waiting screen has already opened
state("focus", None)          # a topic or keyword the whole dashboard is narrowed to



# ---------------------------------------------------------------------------
# shared chrome
# ---------------------------------------------------------------------------
def back_link():
    """A one-step-back control, for people who don't reach for the browser's."""
    prev = PREVIOUS.get(st.session_state.step)
    # The waiting screen sits in two different paths, so where "back" goes
    # depends on which one you're in.
    if st.session_state.step == "run":
        prev = "choose" if st.session_state.dataset else "topics"
    if prev and st.button("← Back", type="tertiary", key=f"back_{st.session_state.step}"):
        go(prev)


def header(right: str = "NO API KEY REQUIRED"):
    back_link()
    left, rt = st.columns([3, 2])
    with left:
        st.markdown(
            "<div style='display:flex;align-items:center;gap:.6rem;'>"
            "<svg width='20' height='20' viewBox='0 0 20 20' fill='none' stroke='#15171A' "
            "stroke-width='1.6' stroke-linecap='round'><path d='M3 16V9M8 16V4M13 16V11M18 16V7'/></svg>"
            "<span style='font-size:17px;font-weight:700;letter-spacing:-.01em;'>CSAT Analyzer</span></div>",
            unsafe_allow_html=True)
    with rt:
        st.markdown(f"<div style='text-align:right;font-size:13px;color:{MUTED};"
                    f"letter-spacing:.06em;padding-top:.25rem;'>{html.escape(right)}</div>",
                    unsafe_allow_html=True)


def rail(active: str):
    """The step indicator. Steps before the active one show a tick."""
    keys = [k for k, _ in STEPS]
    idx = keys.index(active) if active in keys else 0
    out = ["<div class='rail'>"]
    for i, (k, label) in enumerate(STEPS):
        cls = "on" if i == idx else ("done" if i < idx else "")
        mark = "✓" if i < idx else str(i + 1)
        out.append(f"<div class='s {cls}'><div class='b'>{mark}</div>{label}</div>")
    out.append("</div>")
    st.markdown("".join(out), unsafe_allow_html=True)


def back_to_start():
    if st.button("Start over", type="secondary", key=f"reset_{st.session_state.step}"):
        for k in ("dataset", "upload"):
            st.session_state[k] = None
        go("landing")


# ---------------------------------------------------------------------------
# screen 0 — landing
# ---------------------------------------------------------------------------
def screen_landing():
    header()
    st.markdown("<div style='height:9vh'></div>", unsafe_allow_html=True)
    st.markdown(
        "<h1 style='font-size:76px;line-height:1.0;font-weight:400;letter-spacing:-.028em;"
        "text-align:center;margin:0 auto 2.4rem;max-width:16ch;'>"
        "Welcome to Your AI Survey Analytics Hub</h1>", unsafe_allow_html=True)
    st.markdown("<div class='rot3'>" + "".join(f"<p>{r}</p>" for r in ROTATING) + "</div>",
                unsafe_allow_html=True)
    st.markdown("<div style='height:2.2rem'></div>", unsafe_allow_html=True)
    _, mid, _ = st.columns([2, 1, 2])
    with mid:
        if st.button("Get Started", use_container_width=True):
            go("why")


# ---------------------------------------------------------------------------
# screen 1 — why trust it
# ---------------------------------------------------------------------------
def screen_why():
    header()
    st.markdown("<div style='height:3.2rem'></div>", unsafe_allow_html=True)
    left, right = st.columns([1.15, 1], gap="large")

    with left:
        st.markdown("<h1 class='hero-title'>Turn thousands of reviews into "
                    "clear customer insights.</h1>", unsafe_allow_html=True)
        st.markdown("<p class='hero-sub'>Upload customer feedback and let AI analyze every "
                    "response for sentiment, topics, keywords, and a concise summary—then "
                    "explore the results in a filterable analytics dashboard.</p>",
                    unsafe_allow_html=True)
        if st.button("Analyze now"):
            go("choose")

    with right:
        st.markdown(
            "<div class='perf'>"
            "<div class='eyebrow'>Measured performance</div>"
            "<div><span class='n'>0.88</span><span class='of'>out of 1.00</span></div>"
            "<div class='h'>Strong sentiment agreement</div>"
            "<p class='b'>Independent AI models produced highly consistent sentiment labels "
            "across a 150-review test.</p>"
            "<div class='fine'>Cohen&rsquo;s κ, corrected for chance. This measures model "
            "agreement, not human-verified accuracy.</div></div>", unsafe_allow_html=True)

    st.markdown("<div style='height:4rem'></div>", unsafe_allow_html=True)
    beats = [
        ("01", "Analyze only what matters",
         "Your file is checked before AI runs, so invalid or unrelated data is rejected before processing."),
        ("02", "Topics tailored to your data",
         "AI samples your feedback and suggests relevant topics that you can review before analysis."),
        ("03", "Every result is validated",
         "Each label is checked against strict rules, with invalid results automatically retried or removed."),
    ]
    for col, (n, title, body) in zip(st.columns(3, gap="large"), beats):
        with col:
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:.8rem;margin-bottom:.6rem;'>"
                f"<span style='font-size:14px;font-weight:700;letter-spacing:.04em;'>{n}</span>"
                f"<div style='height:1px;background:{RULE};flex-grow:1;'></div></div>"
                f"<div style='font-size:20px;font-weight:600;letter-spacing:-.01em;"
                f"margin-bottom:.45rem;'>{title}</div>"
                f"<p style='font-size:17px;line-height:1.55;color:{INK_SOFT};margin:0;'>{body}</p>",
                unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# screen 2 — choose data
# ---------------------------------------------------------------------------
def screen_choose():
    header()
    rail("choose")

    sets = D.available()
    total = sum(len(D.load(k)) for k in sets)

    st.markdown("<h1 class='hero-title' style='font-size:50px;'>Customer feedback, "
                "analyzed at scale.</h1>", unsafe_allow_html=True)
    st.markdown("<p class='hero-sub' style='max-width:64ch;'>Choose a prepared dataset or upload "
                "your own to generate AI-powered summaries, sentiment and topic tags, and an "
                "interactive analytics dashboard.</p>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="medium")

    with c1, st.container(border=True):
        st.markdown("<div class='card-title'>Choose a dataset</div>"
                    "<p class='card-body'>Already analyzed. Opens instantly.</p>",
                    unsafe_allow_html=True)
        choice = st.selectbox("Dataset", options=list(sets.keys()),
                              format_func=lambda k: sets[k]["label"],
                              index=None, placeholder="Select a dataset",
                              label_visibility="collapsed")
        st.markdown(f"<p class='small-note'>{len(sets)} datasets · {total:,} reviews</p>",
                    unsafe_allow_html=True)
        if st.button("Open dataset", use_container_width=True, disabled=choice is None):
            st.session_state.dataset = choice
            st.session_state.focus = None
            go("run")

    with c2, st.container(border=True):
        st.markdown("<div class='card-title'>Pick one for me</div>"
                    "<p class='card-body'>A random draw from the six.</p>",
                    unsafe_allow_html=True)
        st.markdown(f"<div style='border:1px dashed #C8C8C2;border-radius:10px;padding:2.9rem 1rem;"
                    f"text-align:center;background:{PANEL};'>"
                    f"<div class='eyebrow' style='margin:0;'>1 of {len(sets)}</div></div>",
                    unsafe_allow_html=True)
        st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
        if st.button("Surprise me", type="secondary", use_container_width=True):
            st.session_state.dataset = D.random_key(exclude=st.session_state.dataset)
            st.session_state.focus = None
            go("run")

    with c3, st.container(border=True):
        st.markdown("<div class='card-title'>Upload your own</div>"
                    "<p class='card-body'>CSV or XLSX. 2,000 rows analyzed.</p>",
                    unsafe_allow_html=True)
        up = st.file_uploader("Upload", type=["csv", "xlsx"], label_visibility="collapsed")
        if up is not None:
            st.session_state.upload = up
            go("check")


# ---------------------------------------------------------------------------
# screens 3-5 — the upload path
# ---------------------------------------------------------------------------
def screen_check():
    header(getattr(st.session_state.upload, "name", ""))
    rail("check")
    st.markdown("<h1 class='hero-title' style='font-size:38px;'>Checked your file</h1>",
                unsafe_allow_html=True)

    if not os.environ.get("ANTHROPIC_API_KEY") and "ANTHROPIC_API_KEY" not in st.secrets:
        st.warning(
            "Analyzing a new file needs an Anthropic API key, and this deployment doesn't "
            "have one configured. The six prepared datasets still open instantly — they were "
            "labeled ahead of time.", icon="⚠️")
        if st.button("Back to the datasets", type="secondary"):
            go("choose")
        return

    st.info("The upload path runs the same pipeline the prepared datasets went through: "
            "two validation gates, a proposed topic list for you to approve, then labeling.",
            icon="ℹ️")
    if st.button("Back to the datasets", type="secondary"):
        go("choose")


def screen_run():
    """
    The waiting screen, with the ten facts about how the tool was built.

    Two ways to arrive here. An uploaded file waits for real, while every row is
    sent to the model. A prepared dataset waits only as long as it takes to read
    the CSV and build the counts — usually a second or two — so this screen holds
    for a short minimum instead, long enough to read a fact. It never claims to
    be labeling anything: the wording says "opening", because that is all it is.
    """
    header()
    rail("run")
    key = st.session_state.dataset
    st.markdown("<div class='eyebrow'>While you wait — how this was built</div>",
                unsafe_allow_html=True)
    st.markdown("<div class='rot10'>" + "".join(f"<p>{f}</p>" for f in FACTS) + "</div>",
                unsafe_allow_html=True)

    if key is None:                       # upload path — nothing to open yet
        return

    # Arriving here a second time means the browser's back button brought us
    # from the dashboard. Loading again would bounce straight forward and the
    # back button would look broken, so go to the picker instead.
    if st.session_state.get("opened") == key:
        st.session_state.opened = None
        go("choose")

    spec = D.DATASETS[key]
    st.markdown(f"<div class='section-title'>Opening {html.escape(spec['label'])}</div>",
                unsafe_allow_html=True)
    meter = st.progress(0.0)

    started = time.time()
    df = D.load(key)                      # the real work: read, clean, cache
    D.taxonomy(key)
    MINIMUM = 4.5                         # seconds, so a fact is readable
    while (elapsed := time.time() - started) < MINIMUM:
        meter.progress(min(elapsed / MINIMUM, 1.0))
        time.sleep(0.08)
    meter.progress(1.0)

    st.markdown(f"<p class='small-note'>{len(df):,} labeled reviews ready.</p>",
                unsafe_allow_html=True)
    st.session_state.opened = key
    go("dashboard")


# ---------------------------------------------------------------------------
# screen 6 — dashboard
# ---------------------------------------------------------------------------
def legend() -> str:
    """The three sentiment colours, named. Without this the bar is just colours."""
    return ("<div style='display:flex;gap:1.1rem;justify-content:flex-end;'>" + "".join(
        f"<span style='display:flex;align-items:center;gap:.35rem;font-size:14px;'>"
        f"<span style='width:10px;height:10px;border-radius:2px;"
        f"background:{D.SENTIMENT_COLOR[s]};'></span>{s.title()}</span>"
        for s in D.SENTIMENTS) + "</div>")


def focus_on(kind: str, value: str):
    """Narrow the whole dashboard to one topic or keyword."""
    st.session_state.focus = {"kind": kind, "value": value}
    st.rerun()


def bar(shares: dict, height="72px", labels=True, radius="10px") -> str:
    """
    The stacked sentiment bar.

    The corner radius is passed in rather than fixed, because a 10px radius on a
    14px-tall bar rounds it into a pill and the segment widths stop being
    readable. Small bars get a small radius.
    """
    parts = []
    for s in ["negative", "mixed", "positive"]:
        pct = shares.get(s, 0) * 100
        if pct <= 0:
            continue
        text = f"{pct:.0f}%" if labels and pct >= 9 else ""
        parts.append(f"<div style='width:{pct:.2f}%;background:{D.SENTIMENT_COLOR[s]};'>{text}</div>")
    return (f"<div class='sbar' style='height:{height};border-radius:{radius};'>"
            + "".join(parts) + "</div>")


def screen_dashboard():
    key = st.session_state.dataset
    if key is None:
        go("choose")
    spec = D.DATASETS[key]
    df = D.load(key)

    with st.sidebar:
        st.markdown("<div class='eyebrow'>Dataset</div>"
                    f"<div style='font-size:17px;font-weight:600;margin-bottom:1.4rem;'>"
                    f"{html.escape(spec['label'])}</div>", unsafe_allow_html=True)
        sentiments = st.multiselect("Sentiment", D.SENTIMENTS, default=D.SENTIMENTS,
                                    format_func=str.title)
        topics = st.multiselect("Topics", sorted(D.taxonomy(key)), default=[])
        search = st.text_input("Search text", placeholder="e.g. refund")
        rng = None
        if df["rating"].notna().any():
            lo, hi = float(df["rating"].min()), float(df["rating"].max())
            rng = st.slider("Rating", lo, hi, (lo, hi))
        st.markdown("<div class='rule'></div>", unsafe_allow_html=True)
        if st.button("Choose another dataset", type="secondary", use_container_width=True):
            go("choose")

    view = D.apply_filters(df, sentiments, topics, search, rng)

    # A clicked topic or keyword narrows everything below it, so the sentiment
    # split and the charts describe that slice rather than the whole dataset.
    focus = st.session_state.focus
    if focus:
        view = D.apply_focus(view, focus["kind"], focus["value"])

    header(f"{len(df):,} reviews analyzed")
    st.markdown(f"<h1 class='hero-title' style='font-size:40px;margin-bottom:.4rem;'>"
                f"{html.escape(spec['label'])}</h1>", unsafe_allow_html=True)

    if focus:
        chip, clear = st.columns([4, 1])
        with chip:
            st.markdown(
                f"<div style='display:inline-flex;align-items:center;gap:.5rem;"
                f"background:{INK};color:#FFF;border-radius:999px;"
                f"padding:.4rem 1rem;font-size:15px;font-weight:600;'>"
                f"{focus['kind'].title()}: {html.escape(focus['value'])}</div>",
                unsafe_allow_html=True)
        with clear:
            if st.button("Clear filter", type="secondary", use_container_width=True):
                st.session_state.focus = None
                st.rerun()
        st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)

    st.markdown(f"<p class='section-note' style='font-size:19px;'>"
                f"{len(view):,} of {len(df):,} reviews match your filters.</p>",
                unsafe_allow_html=True)

    if len(view) == 0:
        st.warning("No reviews match these filters.", icon="⚠️")
        if st.button("Clear the filters", type="secondary"):
            st.session_state.focus = None
            st.rerun()
        return

    # ---- overall sentiment ----
    shares = D.sentiment_share(view)
    with st.container(border=True):
        t, lg = st.columns([2, 1])
        with t:
            st.markdown("<div class='section-title'>Overall sentiment</div>"
                        f"<p class='section-note'>Share of the {len(view):,} reviews in view</p>",
                        unsafe_allow_html=True)
        with lg:
            st.markdown(legend(), unsafe_allow_html=True)
        st.markdown(bar(shares, height="48px"), unsafe_allow_html=True)
        counts = view["sentiment"].value_counts()
        st.markdown(
            "<div style='display:flex;gap:2.2rem;margin-top:.8rem;font-size:15px;color:#7B8085;'>"
            + "".join(f"<span>{int(counts.get(s,0)):,} {s}</span>" for s in D.SENTIMENTS)
            + "</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:1.4rem'></div>", unsafe_allow_html=True)
    left, right = st.columns([1.55, 1], gap="medium")

    # ---- sentiment by topic ----
    with left, st.container(border=True):
        st.markdown("<div class='section-title'>Sentiment by topic</div>"
                    "<p class='section-note'>Sorted by how negative each topic runs</p>",
                    unsafe_allow_html=True)
        st.markdown("<p class='small-note' style='margin:-.7rem 0 1rem;'>"
                    "Click a topic to narrow the whole dashboard to it.</p>",
                    unsafe_allow_html=True)
        for _, r in D.topic_breakdown(view).iterrows():
            topic = str(r["topic"])
            neg = r["negative"] * 100
            colour = D.SENTIMENT_COLOR["negative"] if neg >= 60 else INK_SOFT
            nm, pc = st.columns([3, 1.15])
            with nm:
                if st.button(topic, key=f"topic_{topic}", type="tertiary"):
                    focus_on("topic", topic)
            with pc:
                st.markdown(f"<div style='text-align:right;font-size:16px;font-weight:700;"
                            f"color:{colour};padding-top:.5rem;'>{neg:.0f}% neg · "
                            f"{int(r['n'])}</div>", unsafe_allow_html=True)
            st.markdown(bar({s: r[s] for s in D.SENTIMENTS}, height="14px",
                            labels=False, radius="4px")
                        + "<div style='height:1rem'></div>", unsafe_allow_html=True)

    with right:
        # ---- keywords ----
        with st.container(border=True):
            st.markdown("<div class='section-title'>Most-repeated phrases</div>"
                        "<p class='section-note'>Pulled from the reviews themselves</p>",
                        unsafe_allow_html=True)
            st.markdown("<p class='small-note' style='margin:-.7rem 0 1rem;'>"
                        "Click a phrase to narrow the whole dashboard to it.</p>",
                        unsafe_allow_html=True)
            kw = D.keyword_counts(view, limit=14)
            top = kw["n"].max() if len(kw) else 1
            for _, r in kw.iterrows():
                word = str(r["keyword"])
                nm, bx = st.columns([1.15, 1])
                with nm:
                    if st.button(word, key=f"kw_{word}", type="tertiary"):
                        focus_on("keyword", word)
                with bx:
                    st.markdown(
                        f"<div style='display:flex;align-items:center;gap:.7rem;"
                        f"padding-top:.75rem;'>"
                        f"<div style='height:14px;background:{INK};border-radius:999px;"
                        f"width:{r['n']/top*82:.1f}%;'></div>"
                        f"<span style='font-size:15px;color:{MUTED};'>{int(r['n'])}</span></div>",
                        unsafe_allow_html=True)

    # ---- the reviews ----
    st.markdown("<div style='height:1.4rem'></div>", unsafe_allow_html=True)
    with st.container(border=True):
        head, sort_col = st.columns([3, 1])
        with head:
            st.markdown("<div class='section-title'>The reviews themselves</div>",
                        unsafe_allow_html=True)
        with sort_col:
            order = st.selectbox("Sort", ["Most negative first", "Most positive first", "As they come"],
                                 label_visibility="collapsed")

        rank = {"Most negative first": ["negative", "mixed", "positive"],
                "Most positive first": ["positive", "mixed", "negative"]}.get(order)
        shown = view
        if rank:
            shown = view.assign(_o=pd.Categorical(view["sentiment"], rank, ordered=True)) \
                        .sort_values("_o")
        for r in shown.head(12).itertuples():
            cats = " · ".join(D.split_multi(pd.Series([r.categories])))
            st.markdown(
                f"<div class='rev'><div class='stripe' style='background:"
                f"{D.SENTIMENT_COLOR[r.sentiment]};'></div><div style='flex-grow:1;'>"
                f"<div style='display:flex;gap:.8rem;align-items:baseline;'>"
                f"<span class='tag' style='color:{D.SENTIMENT_COLOR[r.sentiment]};'>"
                f"{r.sentiment.upper()}</span>"
                f"<span class='meta'>{html.escape(cats)}</span></div>"
                f"<div class='body'>{html.escape(str(r.text)[:600])}</div>"
                + (f"<div class='sum'><b>Summary:</b> {html.escape(str(r.summary))}</div>"
                   if str(r.summary).strip() else "")
                + "</div></div>", unsafe_allow_html=True)

        st.markdown(f"<p class='small-note'>Showing {min(12, len(shown))} of {len(shown):,} "
                    f"matching reviews.</p>", unsafe_allow_html=True)

    # ---- the end of the road ----
    # The dashboard is the last screen, so it has to offer somewhere to go next
    # rather than just stopping.
    st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("<div class='card-title'>That's this dataset.</div>"
                    "<p class='card-body'>Take the labeled rows with you, or open "
                    "another one — each has its own topics and its own story.</p>",
                    unsafe_allow_html=True)
        a, b, c = st.columns(3)
        with a:
            st.download_button("Download labeled CSV", view.to_csv(index=False).encode(),
                               file_name=f"{key}_labeled_filtered.csv", mime="text/csv",
                               type="secondary", use_container_width=True)
        with b:
            if st.button("Analyze another dataset", use_container_width=True):
                st.session_state.focus = None
                go("choose")
        with c:
            if st.button("Back to the start", type="secondary", use_container_width=True):
                for k in ("dataset", "upload", "focus"):
                    st.session_state[k] = None
                go("landing")


# ---------------------------------------------------------------------------
# router
# ---------------------------------------------------------------------------
SCREENS = {
    "landing":   screen_landing,
    "why":       screen_why,
    "choose":    screen_choose,
    "check":     screen_check,
    "topics":    screen_check,
    "run":       screen_run,
    "dashboard": screen_dashboard,
}
SCREENS.get(st.session_state.step, screen_landing)()
