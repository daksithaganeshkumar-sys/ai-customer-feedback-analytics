"""
schemas.py — controlled vocabulary and structured-output validation.
====================================================================
The labeling pipeline asks Claude for a JSON object per review. This module
defines what a *valid* object looks like and validates every model response
against it, so malformed output is caught and retried/flagged instead of
silently entering the analytics as if it were correct.

The category list is no longer a fixed constant. It is passed in, so the same
validator serves the airline demo (using DEFAULT_CATEGORIES below) and any
uploaded dataset (using a list fitted to it by taxonomy.py). The guarantee that
made a fixed taxonomy the right call is unchanged: within a single run the list
cannot drift, so topic counts stay comparable. It is simply no longer assumed
that every dataset is about flying.
"""

# ---------------------------------------------------------------------------
# Controlled vocabulary
# ---------------------------------------------------------------------------
# Sentiment is a closed set and applies to every dataset. "mixed" is a real
# class (meaningful positive AND negative), never an error fallback. Invalid
# model output is rejected, not coerced into one of these.
VALID_SENTIMENTS = ("positive", "negative", "mixed")

# The airline taxonomy, used for the bundled demo dataset and as the fallback
# when no dataset-specific list has been proposed.
#
# "Premium Economy" was removed in Aug 2026: it fired on 1 of 8,100 reviews
# (0.01%), never produced a countable bar, and cost prompt tokens on every call.
DEFAULT_CATEGORIES = (
    "Seat Comfort",
    "Legroom",
    "Staff Service",
    "Food & Beverages",
    "Inflight Entertainment",
    "Cleanliness",
    "Boarding & Check-in",
    "Delays & Punctuality",
    "Baggage",
    "Value for Money",
    "Booking & Customer Service",
    "Lounge",
    "Cabin Condition",
)

# Kept so older imports don't break.
CATEGORIES = DEFAULT_CATEGORIES


class ValidationError(Exception):
    """Raised when a model response does not conform to the expected schema."""


def validate_label(obj, categories=None):
    """
    Validate a single parsed label object against the schema.

    `categories` is the controlled taxonomy for this run — pass the list that
    taxonomy.py proposed for an uploaded dataset, or leave it out to use
    DEFAULT_CATEGORIES for the airline demo.

    Returns a normalized dict with keys: sentiment, categories, keywords, summary.
    Raises ValidationError with a specific message if anything is wrong, so the
    caller can decide to retry or flag the row — never silently "fix" it.
    """
    allowed = set(categories if categories is not None else DEFAULT_CATEGORIES)

    if not isinstance(obj, dict):
        raise ValidationError("response is not a JSON object")

    # sentiment: must be one of the closed set
    sentiment = obj.get("sentiment")
    if sentiment not in VALID_SENTIMENTS:
        raise ValidationError(f"invalid sentiment: {sentiment!r}")

    # categories: list of strings, each from the controlled taxonomy; may be empty
    cats = obj.get("categories")
    if not isinstance(cats, list):
        raise ValidationError("categories is not a list")
    unknown = [c for c in cats if c not in allowed]
    if unknown:
        raise ValidationError(f"unknown categorie(s): {unknown}")

    # keywords: list of non-empty strings
    keywords = obj.get("keywords")
    if not isinstance(keywords, list) or not all(isinstance(k, str) for k in keywords):
        raise ValidationError("keywords must be a list of strings")

    # summary: non-empty string
    summary = obj.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        raise ValidationError("summary must be a non-empty string")

    return {
        "sentiment": sentiment,
        "categories": list(cats),
        "keywords": [k.strip() for k in keywords if k.strip()],
        "summary": summary.strip(),
    }
