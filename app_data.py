"""
app_data.py — where the app gets its data.
==========================================
Six datasets ship already labeled. Opening one reads a CSV off disk and makes
no API calls at all, which is what lets anyone explore the tool for free.

Each file has its own column names — Starbucks calls the text "Review",
McDonald's calls it "review_details" — so the registry below maps each one onto
a common shape. Everything downstream works with that shape and never has to
know which dataset it came from.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
CLEAN = ROOT / "datasets" / "clean"

# ---------------------------------------------------------------------------
# The six prepared datasets.
#
#   text     the review itself
#   summary  the one-line summary our pipeline wrote
#   rating   the customer's own score, where the dataset has one. This is the
#            only column not produced by us, which is what makes it useful as
#            an independent check on the model's reading.
#   segment  a column worth slicing by on the dashboard (airline, store, model)
# ---------------------------------------------------------------------------
DATASETS: dict[str, dict] = {
    "airline": {
        "label": "Airline reviews",
        "file": ROOT / "reviews_final.csv",
        "taxonomy": None,                      # labeled before the per-dataset taxonomy existed
        "text": "Reviews",
        "summary": "summary",
        "rating": None,
        "segment": "Airline",
        "segment_label": "Airline",
        "blurb": "8,100 reviews across ten carriers.",
    },
    "amazon_music": {
        "label": "Amazon Music",
        "file": CLEAN / "amazon_music_labeled.csv",
        "taxonomy": CLEAN / "amazon_music_taxonomy.json",
        "text": "reviewText",
        "summary": "summary_y",                # summary_x is Amazon's own headline
        "rating": "overall",
        "segment": None,
        "segment_label": None,
        "blurb": "Musical instrument and audio gear reviews.",
    },
    "amazon_alexa": {
        "label": "Amazon Alexa",
        "file": CLEAN / "amazon_alexa_labeled.csv",
        "taxonomy": CLEAN / "amazon_alexa_taxonomy.json",
        "text": "verified_reviews",
        "summary": "summary",
        "rating": "rating",
        "segment": "variation",
        "segment_label": "Model",
        "blurb": "Echo and Fire device reviews.",
    },
    "starbucks": {
        "label": "Starbucks",
        "file": CLEAN / "starbucks_labeled.csv",
        "taxonomy": CLEAN / "starbucks_taxonomy.json",
        "text": "Review",
        "summary": "summary",
        "rating": "Rating",
        "segment": None,
        "segment_label": None,
        "blurb": "Store-level customer complaints and praise.",
    },
    "mcdonalds": {
        "label": "McDonald's",
        "file": CLEAN / "mcdonalds_labeled.csv",
        "taxonomy": CLEAN / "mcdonalds_taxonomy.json",
        "text": "review_details",
        "summary": "summary",
        "rating": None,
        "segment": None,
        "segment_label": None,
        "blurb": "Australian restaurant reviews.",
    },
    "iphone14": {
        "label": "iPhone 14",
        "file": CLEAN / "iphone14_labeled.csv",
        "taxonomy": CLEAN / "iphone14_taxonomy.json",
        "text": "review",
        "summary": "summary",
        "rating": "rating",
        "segment": None,
        "segment_label": None,
        "blurb": "Retail product reviews.",
    },
}

SENTIMENTS = ["negative", "mixed", "positive"]

# The interface is black, white and grey; colour is reserved for sentiment, so
# every coloured pixel on screen carries meaning. These three were checked
# against colour-vision deficiency simulation — an intuitive red/green pair
# fails, which is why the green is this saturated.
SENTIMENT_COLOR = {
    "negative": "#A63125",
    "mixed":    "#C08A1E",
    "positive": "#0F7D45",
}


def available() -> dict[str, dict]:
    """Only the datasets whose file is actually present."""
    return {k: v for k, v in DATASETS.items() if v["file"].exists()}


@st.cache_data(show_spinner=False)
def load(key: str) -> pd.DataFrame:
    """
    Read one prepared dataset and normalise it to the common shape.

    Cached, because Streamlit re-runs this whole script on every click and
    re-reading a 1.6MB CSV each time would make the filters feel sluggish.
    """
    spec = DATASETS[key]
    df = pd.read_csv(spec["file"], low_memory=False)

    out = pd.DataFrame({
        "text":      df[spec["text"]].fillna("").astype(str).str.strip(),
        "summary":   df.get(spec["summary"], pd.Series([""] * len(df))).fillna("").astype(str),
        "sentiment": df["sentiment"].fillna("").astype(str).str.lower().str.strip(),
        "categories": df["categories"].fillna("").astype(str),
        "keywords":  df["keywords"].fillna("").astype(str),
    })
    out["rating"]  = pd.to_numeric(df[spec["rating"]], errors="coerce") if spec["rating"] else pd.NA
    out["segment"] = df[spec["segment"]].fillna("—").astype(str) if spec["segment"] else "—"

    # Rows whose sentiment isn't one of the three never made it through
    # validation, so they are not analysis — drop them rather than charting them.
    out = out[out["sentiment"].isin(SENTIMENTS)]
    return out.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def taxonomy(key: str) -> list[str]:
    """The fixed topic list this dataset was labeled against."""
    spec = DATASETS[key]
    if spec["taxonomy"] and spec["taxonomy"].exists():
        return json.load(open(spec["taxonomy"]))
    # The airline set predates per-dataset taxonomies — recover it from the data.
    return sorted(set(split_multi(load(key)["categories"])))


def split_multi(series: pd.Series) -> list[str]:
    """Pipe-joined cells back into a flat list: 'Delays | Staff' -> ['Delays','Staff']."""
    out: list[str] = []
    for cell in series:
        out.extend(p.strip() for p in str(cell).split("|") if p.strip())
    return out


def random_key() -> str:
    return random.choice(list(available().keys()))


# ---------------------------------------------------------------------------
# Filtering and aggregation — everything the dashboard draws comes from here.
# ---------------------------------------------------------------------------
def apply_filters(df, sentiments, topics, search, rating_range) -> pd.DataFrame:
    out = df[df["sentiment"].isin(sentiments or [])]
    if topics:
        pattern = "|".join(pd.Series(topics).str.replace(r"([.^$*+?()\[\]{}|\\])", r"\\\1", regex=True))
        out = out[out["categories"].str.contains(pattern, case=False, na=False, regex=True)]
    if search:
        out = out[out["text"].str.contains(search, case=False, na=False, regex=False)]
    if rating_range is not None and out["rating"].notna().any():
        lo, hi = rating_range
        out = out[out["rating"].between(lo, hi) | out["rating"].isna()]
    return out


def sentiment_share(df) -> dict[str, float]:
    if len(df) == 0:
        return {s: 0.0 for s in SENTIMENTS}
    counts = df["sentiment"].value_counts(normalize=True)
    return {s: float(counts.get(s, 0.0)) for s in SENTIMENTS}


def topic_breakdown(df, limit=12) -> pd.DataFrame:
    """One row per topic: how many reviews mention it, and its sentiment split."""
    rows = []
    for topic in set(split_multi(df["categories"])):
        hit = df[df["categories"].str.contains(topic, case=False, na=False, regex=False)]
        if len(hit) == 0:
            continue
        share = sentiment_share(hit)
        rows.append({"topic": topic, "n": len(hit), **share})
    if not rows:
        return pd.DataFrame(columns=["topic", "n", *SENTIMENTS])
    return (pd.DataFrame(rows)
            .sort_values("negative", ascending=False)
            .head(limit)
            .reset_index(drop=True))


def keyword_counts(df, limit=10) -> pd.DataFrame:
    from collections import Counter
    common = Counter(k.lower() for k in split_multi(df["keywords"]))
    return pd.DataFrame(common.most_common(limit), columns=["keyword", "n"])


def rating_effect(df, limit=8) -> pd.DataFrame:
    """
    Average rating when a topic is mentioned, against the overall average.

    The ratings were set by customers, not by us — so this is the one view on
    the dashboard that checks the model's reading of the text against something
    independent of it. Only possible where the dataset carries a rating.
    """
    if df["rating"].isna().all():
        return pd.DataFrame(columns=["topic", "delta", "n"])
    baseline = df["rating"].mean()
    rows = []
    for topic in set(split_multi(df["categories"])):
        hit = df[df["categories"].str.contains(topic, case=False, na=False, regex=False)]
        if len(hit) < 10 or hit["rating"].isna().all():
            continue
        rows.append({"topic": topic, "delta": hit["rating"].mean() - baseline, "n": len(hit)})
    if not rows:
        return pd.DataFrame(columns=["topic", "delta", "n"])
    out = pd.DataFrame(rows).reindex(
        pd.DataFrame(rows)["delta"].abs().sort_values(ascending=False).index)
    return out.head(limit).sort_values("delta").reset_index(drop=True)
