"""
taxonomy.py — fit the controlled category list to the dataset being analyzed.
=============================================================================
The pipeline's original 14 categories were written for airline reviews. They
work beautifully there and are meaningless anywhere else — "Legroom" and
"Lounge" never fire on restaurant or software feedback, so an uploaded dataset
produces an empty dashboard.

This module keeps the design decision that made the fixed taxonomy right in the
first place (topic counts are only comparable if the labels can't drift) while
dropping the assumption that every dataset is about flying. One extra API call
reads a sample of the actual reviews and proposes a category list for THEM. The
user reviews it, edits it, and it then locks for the whole run.

Fixed taxonomy, fitted per dataset.

Pipeline position:
    sample rows -> propose_taxonomy() -> user confirms -> label_batch()
                                                       -> coverage_report()

The coverage report is the safety net. Sampling 200 reviews finds any theme
appearing in ~2% or more of the data, but a rarer one can still slip through,
so after labeling we count how many rows came back with no category at all.
A high rate means the taxonomy has a hole and should be revised.
"""

from __future__ import annotations

import json

from prompts import build_taxonomy_prompt

# ---------------------------------------------------------------------------
# Shape of an acceptable taxonomy. These are deliberately strict: a bad category
# list poisons every label that follows it, and it is far cheaper to reject and
# retry here than to discover the problem after labeling 10,000 rows.
# ---------------------------------------------------------------------------
MIN_CATEGORIES = 8
MAX_CATEGORIES = 16
MAX_WORDS_PER_CATEGORY = 4
MAX_CHARS_PER_CATEGORY = 40

# How many reviews to show the model. 200 gives near-certain discovery of any
# theme present in 2%+ of reviews (a 50-row sample misses those 36% of the time)
# and costs about three cents. Going much past 300 buys little: a model reading
# a very long list attends unevenly to its middle.
DEFAULT_SAMPLE_SIZE = 200

# If more than this fraction of labeled rows get zero categories, the taxonomy
# is probably missing a real theme.
COVERAGE_WARN_THRESHOLD = 0.05


class TaxonomyError(Exception):
    """Raised when a proposed taxonomy does not meet the rules above."""


# ---------------------------------------------------------------------------
# Choosing which reviews the model gets to see.
# ---------------------------------------------------------------------------
def sample_for_taxonomy(df, text_col, rating_col=None, n=DEFAULT_SAMPLE_SIZE, seed=42):
    """
    Pick the reviews to show the proposer.

    Random sampling matters more than it sounds. Datasets are frequently sorted
    by date or by rating, so the first N rows are not a fair picture — take the
    first 200 rows of a file that opens with delivery complaints and you get a
    delivery-heavy taxonomy that misses everything else.

    Where a rating column exists we spread the draw across rating values instead
    of sampling purely at random. Complaints and praise use different vocabulary
    and surface different topics, and in a dataset that is 70% five-star a purely
    random sample is mostly praise.
    """
    usable = df[df[text_col].notna()]
    usable = usable[usable[text_col].astype(str).str.strip().str.len() > 20]

    if len(usable) <= n:
        return usable[text_col].astype(str).tolist()

    if rating_col and rating_col in usable.columns:
        groups = [g for _, g in usable.groupby(rating_col) if len(g) > 0]
        if len(groups) > 1:
            per_group = max(1, n // len(groups))
            picked = [
                g.sample(min(per_group, len(g)), random_state=seed) for g in groups
            ]
            import pandas as pd
            out = pd.concat(picked)
            # Top up from the remainder if rounding left us short.
            if len(out) < n:
                rest = usable.drop(out.index)
                if len(rest) > 0:
                    out = pd.concat(
                        [out, rest.sample(min(n - len(out), len(rest)), random_state=seed)]
                    )
            return out[text_col].astype(str).tolist()

    return usable.sample(n, random_state=seed)[text_col].astype(str).tolist()


# ---------------------------------------------------------------------------
# Validating what comes back. Same principle as validate_label(): reject rather
# than quietly repair, so a bad proposal is visible instead of silently wrong.
# ---------------------------------------------------------------------------
def validate_taxonomy(obj):
    """
    Check a proposed category list and return it normalized.

    Raises TaxonomyError with a specific reason if anything is wrong, so the
    caller can retry with that reason fed back to the model.
    """
    if not isinstance(obj, list):
        raise TaxonomyError("response is not a JSON array")

    if not all(isinstance(c, str) for c in obj):
        raise TaxonomyError("every category must be a string")

    cats = [c.strip() for c in obj if c.strip()]

    if not (MIN_CATEGORIES <= len(cats) <= MAX_CATEGORIES):
        raise TaxonomyError(
            f"got {len(cats)} categories, need between "
            f"{MIN_CATEGORIES} and {MAX_CATEGORIES}"
        )

    for c in cats:
        if len(c) > MAX_CHARS_PER_CATEGORY:
            raise TaxonomyError(f"category too long: {c!r}")
        if len(c.split()) > MAX_WORDS_PER_CATEGORY:
            raise TaxonomyError(f"category is a phrase, not a label: {c!r}")
        if c.endswith("."):
            raise TaxonomyError(f"category looks like a sentence: {c!r}")

    # Exact duplicates, ignoring case.
    lowered = [c.lower() for c in cats]
    if len(set(lowered)) != len(lowered):
        dupes = {c for c in lowered if lowered.count(c) > 1}
        raise TaxonomyError(f"duplicate categories: {sorted(dupes)}")

    # Near-duplicates: one label wholly contained in another ("Delivery" and
    # "Delivery Speed"). These are worse than exact duplicates because the model
    # will split related reviews arbitrarily between them and both counts lie.
    for i, a in enumerate(lowered):
        for j, b in enumerate(lowered):
            if i != j and a in b:
                raise TaxonomyError(f"overlapping categories: {cats[i]!r} and {cats[j]!r}")

    return cats


# ---------------------------------------------------------------------------
# The proposal call.
# ---------------------------------------------------------------------------
def propose_taxonomy(client, sample_texts, model, max_retries=3, max_tokens=600):
    """
    Ask the model for a category list fitted to these reviews.

    Returns a list of category names. Raises TaxonomyError if the model cannot
    produce a valid list within max_retries — the caller should then fall back
    to DEFAULT_CATEGORIES or ask the user to supply one.

    Costs exactly one API call when it works: roughly 30,000 input tokens for a
    200-review sample, about three cents on Haiku.
    """
    if not sample_texts:
        raise TaxonomyError("no sample reviews supplied")

    last_err = None
    for attempt in range(max_retries):
        # On a retry, tell the model what was wrong with its previous attempt.
        prompt = build_taxonomy_prompt(sample_texts, previous_error=last_err)
        try:
            msg = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = msg.content[0].text.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1].removeprefix("json").strip()
            return validate_taxonomy(json.loads(raw))
        except json.JSONDecodeError as e:
            last_err = f"response was not valid JSON: {e}"
        except TaxonomyError as e:
            last_err = str(e)

    raise TaxonomyError(f"could not produce a valid taxonomy: {last_err}")


# ---------------------------------------------------------------------------
# The safety net.
# ---------------------------------------------------------------------------
def coverage_report(labels):
    """
    After labeling, how many rows got no category at all?

    A high rate means the reviews are discussing something the taxonomy has no
    label for. This is the check that catches a theme too rare to appear in the
    200-review sample, and it is free — the number falls out of a run you were
    doing anyway.

    `labels` is the list of validated label dicts from label_batch().
    Returns a dict; `needs_revision` is True when the threshold is crossed.
    """
    scored = [l for l in labels if isinstance(l, dict) and "error" not in l]
    if not scored:
        return {"n": 0, "uncategorized": 0, "rate": 0.0, "needs_revision": False}

    uncategorized = sum(1 for l in scored if not l.get("categories"))
    rate = uncategorized / len(scored)

    return {
        "n": len(scored),
        "uncategorized": uncategorized,
        "rate": rate,
        "needs_revision": rate > COVERAGE_WARN_THRESHOLD,
        "message": (
            f"{uncategorized} of {len(scored)} reviews ({rate:.1%}) matched no category. "
            + (
                "The taxonomy is probably missing a theme — consider revising it."
                if rate > COVERAGE_WARN_THRESHOLD
                else "Coverage looks healthy."
            )
        ),
    }
