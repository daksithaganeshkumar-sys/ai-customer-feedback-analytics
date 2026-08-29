"""
batch_labeler.py — the labeling engine, batched and concurrent.
===============================================================
label_reviews.py sends one review per request, one request at a time. That is
fine for a few hundred rows and impossible for fourteen thousand: at roughly two
seconds a call it is about eight hours.

This module does the same work two ways faster:

  BATCHING     ten reviews per request instead of one, so the system prompt
               (the taxonomy and the instructions) is sent 1,400 times rather
               than 14,000. Saves roughly 30% of the bill.

  CONCURRENCY  ten requests in flight at once. Haiku allows 1,000 requests and
               2M input tokens per minute, so ten parallel batches is well
               inside the limits and turns eight hours into about fifteen
               minutes.

Everything that made the original engine trustworthy is kept:

  - each item is numbered and the model must echo the number back, so answers
    are matched to rows rather than trusting the order they come back in
  - every label is validated before it is kept; an invented category or an
    unexpected sentiment is rejected, never quietly corrected
  - a failed batch is retried once as a batch, then row by row, and whatever
    still fails is flagged rather than dropped
  - results are written as they complete, so a crash loses nothing

Used by label_datasets.py (the command-line runner) and, later, by the
Streamlit app — so the concurrency lives in one place.
"""

from __future__ import annotations

import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from schemas import DEFAULT_CATEGORIES, VALID_SENTIMENTS, ValidationError, validate_label

# ---------------------------------------------------------------------------
# Tuning. The defaults are chosen against Haiku's published limits
# (1,000 requests/min, 2M input tokens/min) with plenty of headroom.
# ---------------------------------------------------------------------------
BATCH_SIZE = 10
CONCURRENCY = 10
MAX_TOKENS_PER_BATCH = 2200      # ~200 output tokens per review, plus slack
MAX_REVIEW_CHARS = 4000          # truncate monsters; a 20k-char review helps nobody

# Haiku 4.5 list prices, for the running cost estimate.
PRICE_IN_PER_MTOK = 1.00
PRICE_OUT_PER_MTOK = 5.00


def build_batch_prompt(items, categories=None):
    """
    Build the system prompt for a batch.

    `items` is a list of (id, text). Each is numbered, and the model is told to
    return the id with each answer — that is what lets us match answers to rows
    instead of assuming the array comes back in order.
    """
    cats = categories if categories is not None else DEFAULT_CATEGORIES
    category_lines = "\n".join(f"  - {c}" for c in cats)
    sentiments = ", ".join(VALID_SENTIMENTS)

    return f"""You label customer reviews for a data pipeline.

You will be given {len(items)} numbered reviews. Return ONLY a JSON array with one
object per review (no markdown, no backticks, no commentary). Each object must have:
  "id": the number of the review this answers, exactly as given
  "sentiment": one of {sentiments}
  "categories": array of tags chosen ONLY from the controlled list below (may be empty)
  "keywords": array of 3-6 short free-form phrases taken from that review
  "summary": one sentence, 20 words or fewer

Return an object for EVERY id you were given, in any order.

Controlled category list (use these labels EXACTLY, choose only what the review supports):
{category_lines}

SENTIMENT DEFINITIONS:
  - "positive": predominantly praise or satisfaction.
  - "negative": predominantly dissatisfaction or complaint.
  - "mixed": contains MEANINGFUL positive AND negative feedback.
  "mixed" describes genuinely mixed experiences. It is NOT a fallback for
  uncertainty — if unsure, still choose the sentiment the text best supports.

RULES:
  - Judge each review only on its own text. Do not let one review influence another.
  - Do not invent context or facts.
  - Select multiple categories only when a review genuinely covers them.
  - Keywords must be concise, grounded in that review, and not redundant.
  - The summary must capture the primary issue or praise, add no new facts,
    and contain NO recommendations or advice."""


def build_batch_input(items):
    """Format the reviews themselves as the user message."""
    parts = []
    for rid, text in items:
        clean = " ".join(str(text).split())[:MAX_REVIEW_CHARS]
        parts.append(f"[{rid}]\n{clean}")
    return "\n\n".join(parts)


def extract_json_array(raw):
    """
    Pull a JSON array out of the model's reply.

    Strips markdown fences if present, and otherwise falls back to finding the
    outermost [...] — which rescues a reply that opened with a stray sentence.
    """
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        if len(parts) > 1:
            raw = parts[1].removeprefix("json").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _label_one_batch(client, items, categories, model):
    """
    One API call for one batch. Returns (results_by_id, usage).

    Rows the model skipped or got wrong simply do not appear in the returned
    dict — the caller decides what to do about them.
    """
    msg = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS_PER_BATCH,
        system=build_batch_prompt(items, categories),
        messages=[{"role": "user", "content": build_batch_input(items)}],
    )
    raw = msg.content[0].text
    parsed = extract_json_array(raw)

    if not isinstance(parsed, list):
        raise ValidationError("response was not a JSON array")

    wanted = {rid for rid, _ in items}
    out = {}
    for obj in parsed:
        if not isinstance(obj, dict):
            continue
        rid = obj.get("id")
        # Models occasionally return the id as a string.
        try:
            rid = int(rid)
        except (TypeError, ValueError):
            continue
        if rid not in wanted or rid in out:
            continue
        try:
            out[rid] = validate_label(obj, categories)
        except ValidationError as e:
            # Leave it out; it will be retried individually.
            out.pop(rid, None)
            _ = e

    usage = getattr(msg, "usage", None)
    tokens = (
        getattr(usage, "input_tokens", 0) or 0,
        getattr(usage, "output_tokens", 0) or 0,
    )
    return out, tokens


def _label_with_retries(client, items, categories, model, max_attempts=2):
    """
    Label one batch, retrying the batch once, then falling back to labeling the
    stragglers one at a time.

    Splitting to singles matters: one malformed row in a batch of ten should not
    cost you the other nine.
    """
    results, in_tok, out_tok = {}, 0, 0
    remaining = list(items)

    for attempt in range(max_attempts):
        if not remaining:
            break
        try:
            got, (i, o) = _label_one_batch(client, remaining, categories, model)
            in_tok += i
            out_tok += o
            results.update(got)
            remaining = [(r, t) for r, t in remaining if r not in results]
        except Exception as e:  # transient API problem, bad JSON, bad shape
            if attempt == max_attempts - 1:
                break
            time.sleep(2 ** attempt)

    # Anything still missing gets one solo attempt each.
    for rid, text in list(remaining):
        try:
            got, (i, o) = _label_one_batch(client, [(rid, text)], categories, model)
            in_tok += i
            out_tok += o
            results.update(got)
        except Exception:
            pass

    for rid, _ in items:
        if rid not in results:
            results[rid] = {"error": "failed_after_retries"}

    return results, in_tok, out_tok


def label_all(client, rows, categories=None, model="claude-haiku-4-5-20251001",
              batch_size=BATCH_SIZE, concurrency=CONCURRENCY,
              on_result=None, on_progress=None):
    """
    Label every row, batched and concurrent.

    rows        list of (id, text)
    categories  the controlled taxonomy for this run (None = airline defaults)
    on_result   called with (id, label_dict) as each row completes — use it to
                append to disk so a crash loses nothing
    on_progress called with a dict of running totals after each batch

    Returns {id: label_dict}. A failed row's dict is {"error": ...}, exactly as
    label_reviews.py does, so downstream code needs no special case.
    """
    batches = [rows[i:i + batch_size] for i in range(0, len(rows), batch_size)]

    results = {}
    totals = {"done": 0, "failed": 0, "input_tokens": 0, "output_tokens": 0,
              "total": len(rows), "cost": 0.0}
    lock = threading.Lock()

    def run(batch):
        return _label_with_retries(client, batch, categories, model)

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(run, b): b for b in batches}
        for fut in as_completed(futures):
            try:
                got, in_tok, out_tok = fut.result()
            except Exception:
                got = {rid: {"error": "batch_crashed"} for rid, _ in futures[fut]}
                in_tok = out_tok = 0

            with lock:
                results.update(got)
                totals["input_tokens"] += in_tok
                totals["output_tokens"] += out_tok
                totals["done"] += len(got)
                totals["failed"] += sum(1 for v in got.values() if "error" in v)
                totals["cost"] = (
                    totals["input_tokens"] / 1_000_000 * PRICE_IN_PER_MTOK
                    + totals["output_tokens"] / 1_000_000 * PRICE_OUT_PER_MTOK
                )
                if on_result:
                    for rid, label in got.items():
                        on_result(rid, label)
                if on_progress:
                    on_progress(dict(totals))

    return results
