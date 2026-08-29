"""
app_style.py — the whole visual identity, injected as one CSS block.
====================================================================
Streamlit ships its own look. Everything here overrides it: Public Sans
throughout, a white ground, rounded corners, pill buttons, and the two
rotating-text animations.

Two rules the design rests on:

  1. ONE TYPEFACE. Public Sans does headings, body and the all-caps labels.
     Numbers use tabular figures so columns line up without a monospace face.

  2. COLOUR MEANS SENTIMENT. The interface is black, white and grey. The only
     hues on screen are the three sentiment colours, so every coloured pixel
     carries information. Those three were validated against colour-vision
     deficiency simulation — an intuitive red/green pair fails deutan, which is
     why the green is more saturated than instinct suggests.
"""

import re

INK          = "#15171A"
INK_SOFT     = "#474B4F"
MUTED        = "#7B8085"
RULE         = "#E3E3DF"
RULE_STRONG  = "#C8C8C2"
PANEL        = "#F7F7F5"
NEGATIVE     = "#A63125"
MIXED        = "#C08A1E"
POSITIVE     = "#0F7D45"


def css() -> str:
    """
    Return the stylesheet, stripped of anything a markdown parser would mangle.

    Two hazards, both discovered the hard way, and both silent:

      COMMENTS   markdown reads the asterisks in /* like this */ as emphasis and
                 removes them, leaving an unterminated CSS comment that swallows
                 the rest of the stylesheet. Every rule after it dies.

      INDENTS    a line indented four or more spaces becomes a code block, which
                 closes the <style> tag early and prints the remaining CSS onto
                 the page as visible text.

    The app injects this with st.html(), which does not run a markdown pass at
    all — but the cleaning stays, so the stylesheet is safe either way.
    """
    css_text = re.sub(r"/\*.*?\*/", "", _CSS, flags=re.S)
    return "\n".join(line.strip() for line in css_text.splitlines() if line.strip())


_CSS = f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet"
      href="https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;500;600;700&display=swap">
<style>

/* ---------- ground ---------- */
.stApp {{ background: #FFFFFF; }}
html, body, [class*="css"], .stMarkdown, .stButton button, input, textarea, select {{
  font-family: 'Public Sans', -apple-system, 'Helvetica Neue', Arial, sans-serif !important;
  font-variant-numeric: tabular-nums;
}}
.block-container {{ max-width: 1240px; padding-top: 1.6rem; padding-bottom: 4rem; }}
#MainMenu, footer, header[data-testid="stHeader"] {{ visibility: hidden; }}

h1, h2, h3, h4 {{ color: {INK}; letter-spacing: -0.024em; }}
p, li, label, span, div {{ color: {INK}; }}

/* ---------- buttons: black pills ---------- */
.stButton > button {{
  background: {INK}; color: #FFFFFF; border: none; border-radius: 999px;
  padding: 1.05rem 2.4rem; font-size: 1.12rem; font-weight: 600;
  transition: opacity .15s ease;
}}
.stButton > button:hover {{ background: {INK}; color: #FFFFFF; opacity: .86; }}
.stButton > button:focus {{ box-shadow: 0 0 0 3px rgba(21,23,26,.18) !important; color: #FFFFFF; }}
.stButton > button[kind="secondary"] {{
  background: #FFFFFF; color: {INK}; border: 1px solid {RULE_STRONG};
}}
.stButton > button[kind="secondary"]:hover {{ background: {PANEL}; color: {INK}; }}

/* Tertiary buttons are the clickable topic and keyword names on the dashboard.
   They have to read as part of the chart, not as buttons sitting on top of it —
   so they lose the pill entirely and behave like a heading you can press. */
.stButton > button[kind="tertiary"] {{
  background: transparent; color: {INK}; border: none; padding: 0.3rem 0;
  font-size: 17px; font-weight: 500; text-align: left; justify-content: flex-start;
}}
.stButton > button[kind="tertiary"]:hover {{
  background: transparent; color: {INK}; opacity: 1; text-decoration: underline;
  text-underline-offset: 3px;
}}
.stButton > button[kind="tertiary"] p {{ font-size: 17px; font-weight: 500; }}

/* sidebar filters, sized to match the rest */
[data-testid="stSidebar"] label p {{ font-size: 15px !important; font-weight: 600; }}

/* The selected filters are black pills, and each pill wraps its label in a
   second span. The blanket "span is dark ink" rule further up catches that
   inner span and paints black text on the black pill, so the filters look
   empty. These two rules put the label back. */
span[data-tag] {{ border-radius: 999px !important; }}
span[data-tag] span, span[data-tag] button, span[data-tag] svg {{
  color: #FFFFFF !important; font-size: 14px !important;
}}

/* ---------- inputs ---------- */
div[data-baseweb="select"] > div, .stTextInput input, .stNumberInput input {{
  border-radius: 10px !important; border-color: {RULE_STRONG} !important;
  font-size: 1rem !important;
}}
[data-testid="stFileUploaderDropzone"] {{
  border-radius: 12px; border: 1px dashed {RULE_STRONG}; background: {PANEL};
  padding: 2.4rem 1rem;
}}

/* ---------- bordered containers become cards ---------- */
div[data-testid="stVerticalBlockBorderWrapper"] {{
  background: #FFFFFF; border-color: {RULE}; border-radius: 14px; padding: 0.4rem;
}}

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] {{ background: {PANEL}; border-right: 1px solid {RULE}; }}
[data-testid="stSidebar"] .block-container {{ padding-top: 2rem; }}

/* ---------- our own bits ---------- */
.eyebrow {{
  font-size: 14px; font-weight: 600; color: {MUTED};
  letter-spacing: 0.13em; text-transform: uppercase; margin-bottom: 0.4rem;
}}
.hero-title {{ font-size: 54px; line-height: 1.06; font-weight: 700; letter-spacing: -0.028em; margin: 0 0 1.1rem; }}
.hero-sub   {{ font-size: 21.5px; line-height: 1.6; color: {INK_SOFT}; margin: 0 0 2rem; max-width: 54ch; }}
.section-title {{ font-size: 24px; font-weight: 600; letter-spacing: -0.014em; margin: 0 0 0.15rem; }}
.section-note  {{ font-size: 17px; color: {MUTED}; margin: 0 0 1.2rem; }}
.card-title {{ font-size: 27px; font-weight: 600; letter-spacing: -0.012em; margin: 0.5rem 0 0.4rem; }}
.card-body  {{ font-size: 20px; line-height: 1.5; color: {INK_SOFT}; margin: 0 0 0.9rem; }}
.small-note {{ font-size: 16.5px; color: {MUTED}; }}
.rule {{ height: 1px; background: {RULE}; margin: 2.2rem 0; }}

/* metric card */
.perf {{ border: 1px solid {RULE}; border-radius: 14px; padding: 2rem 2.1rem; background: {PANEL}; }}
.perf .n  {{ font-size: 60px; font-weight: 700; letter-spacing: -0.035em; line-height: 1; }}
.perf .of {{ font-size: 19px; color: {MUTED}; margin-left: 0.6rem; }}
.perf .h  {{ font-size: 24px; font-weight: 600; letter-spacing: -0.014em; margin: 0.85rem 0 0.5rem; }}
.perf .b  {{ font-size: 19px; line-height: 1.55; color: {INK_SOFT}; margin: 0; }}
.perf .fine {{ font-size: 16.5px; line-height: 1.5; color: {MUTED};
               margin-top: 1.3rem; padding-top: 1.2rem; border-top: 1px solid {RULE}; }}

/* step rail */
.rail {{ display: flex; gap: 2rem; align-items: center; padding: 0.9rem 0 1rem;
         border-bottom: 1px solid {RULE}; margin-bottom: 2.2rem; flex-wrap: wrap; }}
.rail .s {{ display: flex; align-items: center; gap: 0.55rem; font-size: 17px; color: {MUTED}; }}
.rail .s.on {{ color: {INK}; font-weight: 600; }}
.rail .b {{ width: 23px; height: 23px; border-radius: 50%; display: flex;
            align-items: center; justify-content: center; font-size: 13px; font-weight: 700;
            border: 1px solid {RULE_STRONG}; color: {MUTED}; }}
.rail .s.on   .b {{ background: {INK}; border-color: {INK}; color: #FFFFFF; }}
.rail .s.done .b {{ background: {RULE_STRONG}; border-color: {RULE_STRONG}; color: #FFFFFF; }}

/* stacked sentiment bar */
.sbar {{ display: flex; gap: 3px; height: 72px; border-radius: 10px; overflow: hidden; }}
.sbar div {{ display: flex; align-items: center; padding-left: 1rem;
             font-size: 19px; font-weight: 700; color: #FFFFFF; }}
.tbar {{ display: flex; gap: 2px; height: 24px; border-radius: 6px; overflow: hidden; }}

/* review card */
.rev {{ border: 1px solid {RULE}; border-radius: 14px; padding: 1.2rem 1.4rem;
        display: flex; gap: 1.1rem; margin-bottom: 0.7rem; }}
.rev .stripe {{ width: 4px; border-radius: 2px; flex-shrink: 0; }}
.rev .tag  {{ font-size: 14.5px; font-weight: 700; letter-spacing: 0.07em; }}
.rev .meta {{ font-size: 16px; color: {MUTED}; }}
.rev .body {{ font-size: 19px; line-height: 1.55; margin: 0.5rem 0 0.6rem; }}
.rev .sum  {{ font-size: 17.5px; color: {INK_SOFT}; padding-left: 0.85rem;
              border-left: 2px solid {RULE}; }}

/* ---------- rotating text ---------- */
@keyframes rot3 {{
  0%     {{ opacity: 0; transform: translateY(20px);  }}
  4.2%   {{ opacity: 1; transform: translateY(0);     }}
  29.2%  {{ opacity: 1; transform: translateY(0);     }}
  33.3%  {{ opacity: 0; transform: translateY(-12px); }}
  100%   {{ opacity: 0; transform: translateY(-12px); }}
}}
.rot3 {{ position: relative; height: 108px; }}
.rot3 p {{
  position: absolute; inset: 0; margin: 0; opacity: 0; text-align: center;
  font-size: 31px; line-height: 1.24; font-style: italic; color: #5D5D5D;
  animation: rot3 10.8s ease-out infinite;
}}
.rot3 p:nth-child(1) {{ animation-delay: 0s;   }}
.rot3 p:nth-child(2) {{ animation-delay: 3.6s; }}
.rot3 p:nth-child(3) {{ animation-delay: 7.2s; }}

@keyframes rot10 {{
  0%    {{ opacity: 0; transform: translateY(22px); }}
  0.7%  {{ opacity: 1; transform: translateY(0);    }}
  9.3%  {{ opacity: 1; transform: translateY(0);    }}
  10%   {{ opacity: 0; transform: translateY(-14px);}}
  100%  {{ opacity: 0; transform: translateY(-14px);}}
}}
.rot10 {{ position: relative; height: 180px; max-width: 1080px; }}
.rot10 p {{
  position: absolute; inset: 0; margin: 0; opacity: 0;
  font-size: 30px; line-height: 1.36; font-weight: 500;
  letter-spacing: -0.018em; color: {INK};
  animation: rot10 65s linear infinite;
}}
{"".join(f".rot10 p:nth-child({i+1}) {{ animation-delay: {i*6.5:g}s; }}" for i in range(10))}

/* Motion that cannot be paused is a real problem for some people. */
@media (prefers-reduced-motion: reduce) {{
  .rot3, .rot10 {{ height: auto; }}
  .rot3 p, .rot10 p {{ position: static; opacity: 1; animation: none;
                       margin-bottom: 0.8rem; font-size: 19px; }}
}}
</style>
"""
