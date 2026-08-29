"""
prompts.py — the prompts used by the labeling pipeline.
=======================================================
Kept in its own file so the wording is easy to inspect and iterate on without
touching pipeline logic.

Two prompts live here:
  build_system_prompt()   — instructs the model to label ONE review
  build_taxonomy_prompt() — asks the model to propose a category list for a
                            dataset it has never seen

The category list is injected rather than written inline, so the prompt and the
validator in schemas.py can never drift apart.
"""

from schemas import DEFAULT_CATEGORIES, VALID_SENTIMENTS


def build_system_prompt(categories=None):
    """
    Return the system prompt that instructs Claude how to label one review.

    `categories` is the controlled taxonomy for this run. Omit it to use the
    airline defaults from schemas.py.
    """
    cats = categories if categories is not None else DEFAULT_CATEGORIES
    category_lines = "\n".join(f"  - {c}" for c in cats)
    sentiments = ", ".join(VALID_SENTIMENTS)

    return f"""You label individual customer reviews for a data pipeline.

Return ONLY a JSON object (no markdown, no backticks, no commentary) with exactly these keys:
  "sentiment": one of {sentiments}
  "categories": array of tags chosen ONLY from the controlled list below (may be empty)
  "keywords": array of 3-6 short free-form phrases taken from the review
  "summary": one sentence, 20 words or fewer

Controlled category list (use these labels EXACTLY, choose only what the review supports):
{category_lines}

SENTIMENT DEFINITIONS:
  - "positive": the review is predominantly praise or satisfaction.
  - "negative": the review is predominantly dissatisfaction or complaint.
  - "mixed": the review contains MEANINGFUL positive AND negative feedback.
  "mixed" describes genuinely mixed experiences. It is NOT a fallback for
  uncertainty — if you are unsure, still choose the sentiment the text best supports.

RULES:
  - Classify ONLY from the supplied review text. Do not invent context or facts.
  - Select multiple categories only when the review genuinely covers them.
  - Do not force a category that the review does not support.
  - Keywords must be concise, grounded in the review, and not redundant with each other.
  - The summary must faithfully capture the primary issue or praise, add no new facts,
    and contain NO recommendations or advice.
  - Use only information contained in the review."""


def build_taxonomy_prompt(sample_texts, previous_error=None):
    """
    Ask the model to propose a controlled category list for this dataset.

    `sample_texts` is a list of raw review strings — see
    taxonomy.sample_for_taxonomy() for how they should be drawn.

    `previous_error` is the reason a prior attempt was rejected. Feeding it back
    turns a retry into a correction rather than a re-roll.
    """
    numbered = "\n".join(
        f"{i + 1}. {' '.join(str(t).split())[:400]}"
        for i, t in enumerate(sample_texts)
    )

    retry_note = ""
    if previous_error:
        retry_note = (
            f"\nYour previous attempt was rejected for this reason: {previous_error}\n"
            "Correct that specific problem in this attempt.\n"
        )

    return f"""Below is a sample of customer reviews from a single dataset.

Propose a controlled list of topic categories for tagging every review in this
dataset. The list will be fixed for the whole analysis, so it must cover what
these customers actually talk about.

Return ONLY a JSON array of strings (no markdown, no backticks, no commentary).
Example shape: ["Delivery Speed", "Packaging", "Customer Support"]

RULES:
  - Between 10 and 14 categories.
  - Each is a short noun phrase of 1-3 words. Not a sentence.
  - Categories describe WHAT the review is about (aspects, topics, parts of the
    experience) — never how the customer felt. "Battery Life" is a category.
    "Positive Experience", "Complaint" and "Satisfied Customer" are not.
  - Categories must be clearly distinct. Never propose one label that contains
    another ("Delivery" alongside "Delivery Speed") — reviews would be split
    arbitrarily between them and both counts would be meaningless.
  - Cover the recurring themes. A theme mentioned by many customers needs a
    category; a one-off does not.
  - Together they should account for most of what these reviews discuss.
  - Use the vocabulary of this dataset's domain, not generic business language.
{retry_note}
REVIEWS:
{numbered}"""
