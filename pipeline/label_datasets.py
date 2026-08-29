"""
label_datasets.py — label every dataset in a folder, once, offline.
===================================================================
This is the script that fills the app's free demo path. Point it at a folder of
CSVs and walk away; when it finishes, every prepared dataset has been read by
Claude and the results are on disk. The app then serves those files, so a
visitor clicking "open dataset" costs nothing and waits for nothing.

For each file it:
  1. finds the text column and any rating column (the Gate 1 checks)
  2. proposes a topic taxonomy from 200 rows, stratified across ratings
  3. shows you the taxonomy and waits for your yes
  4. labels every row — 10 per request, 10 requests at once
  5. writes <name>_labeled.csv beside the source
  6. reports rows labeled, rows flagged, uncategorized rate, and cost

Resumable. Progress goes to <name>_labels.jsonl as each row finishes, and
re-running skips anything already done — so a crash, a rate limit or a closed
laptop costs you nothing but the time already spent.

----------------------------------------------------------------------
BEFORE YOU RUN
----------------------------------------------------------------------
1. pip install -r pipeline/requirements.txt
2. export ANTHROPIC_API_KEY="sk-ant-..."
3. Raise the spend limit in the Anthropic console to $20. The full run is about
   $11.50 and a $10 cap would stop it partway.
4. Try one small file first:
       python pipeline/label_datasets.py datasets/ --only mcdonalds
   That is 400 rows, about 30 cents, and proves the whole path works.

Then the real thing:
       python pipeline/label_datasets.py datasets/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

try:
    import anthropic
except ImportError:
    sys.exit("The anthropic package is required:  pip install anthropic")

from batch_labeler import label_all
from taxonomy import (
    DEFAULT_SAMPLE_SIZE,
    TaxonomyError,
    coverage_report,
    propose_taxonomy,
    sample_for_taxonomy,
)

MODEL = "claude-haiku-4-5-20251001"

# Column-name hints, same as vet_datasets.py. Keep them in step.
FEEDBACK_HINTS = ("review", "comment", "feedback", "response", "text", "content",
                  "opinion", "description", "message", "body", "remark")
RATING_HINTS = ("rating", "score", "star", "csat", "nps", "rate", "overall")

MIN_AVG_CHARS = 20


def read_any(path: Path):
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path)
    sep = "\t" if path.suffix.lower() in (".tsv", ".tab") else ","
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return pd.read_csv(path, sep=sep, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise ValueError("could not decode the file")


def find_columns(df):
    """Pick the feedback column and any rating column. Returns (text_col, rating_col)."""
    text_candidates = []
    for c in df.columns:
        vals = df[c].dropna().astype(str).str.strip()
        vals = vals[vals != ""]
        if len(vals) == 0:
            continue
        avg = vals.str.len().mean()
        if avg >= MIN_AVG_CHARS and vals.nunique() / len(vals) > 0.3:
            text_candidates.append((c, avg))
    if not text_candidates:
        return None, None

    named = [t for t in text_candidates if any(h in str(t[0]).lower() for h in FEEDBACK_HINTS)]
    text_col = max(named or text_candidates, key=lambda t: t[1])[0]

    rating_col = None
    for c in df.columns:
        if not any(h in str(c).lower() for h in RATING_HINTS):
            continue
        n_unique = df[c].dropna().nunique()
        if 1 < n_unique <= 11:
            rating_col = c
            break

    return text_col, rating_col


def load_done(jsonl_path: Path):
    """Read the progress file. Only successful rows count as done, so a re-run retries failures."""
    done = {}
    if not jsonl_path.exists():
        return done
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" not in rec and "row_id" in rec:
                done[rec["row_id"]] = rec
    return done


def process(path: Path, client, args):
    print(f"\n{'=' * 70}\n{path.name}\n{'=' * 70}")

    try:
        df = read_any(path)
    except Exception as e:
        print(f"  skipped — could not read: {e}")
        return None

    df = df.reset_index(drop=True)
    df["row_id"] = df.index

    text_col, rating_col = find_columns(df)
    if text_col is None:
        print("  skipped — no column of real sentences. Run vet_datasets.py to see why.")
        return None

    usable = df[df[text_col].notna()].copy()
    usable = usable[usable[text_col].astype(str).str.strip().str.len() > MIN_AVG_CHARS]

    # Duplicate rows are common in scraped datasets — iPhone 14 is 50% repeats,
    # Alexa 25%. Labeling the same review twice costs money and makes the
    # dashboard show it twice. Keep the first of each.
    n_before = len(usable)
    if not args.keep_duplicates:
        usable = usable.drop_duplicates(subset=[text_col], keep="first")
    n_dropped = n_before - len(usable)

    # Row cap. A per-dataset cap (--cap amazon_music=2000) beats a global one:
    # the big file gets sampled, the small ones run in full, and the intent is
    # written down rather than being true by accident.
    cap = args.caps.get(path.stem.lower(), args.limit)
    if cap and len(usable) > cap:
        # Sample rather than take the first N — files are often sorted by date
        # or rating, so head() would give a skewed slice.
        usable = usable.sample(cap, random_state=42).sort_index()
        print(f"  capped at     : {cap:,} rows (random sample of {n_before:,})")

    print(f"  text column   : {text_col}")
    print(f"  rating column : {rating_col or '— none —'}")
    if n_dropped:
        print(f"  duplicates    : {n_dropped:,} dropped ({n_dropped / n_before:.0%})")
    print(f"  rows to label : {len(usable):,}")

    jsonl_path = path.with_name(f"{path.stem}_labels.jsonl")
    out_path = path.with_name(f"{path.stem}_labeled.csv")
    tax_path = path.with_name(f"{path.stem}_taxonomy.json")

    # ---- taxonomy: reuse the saved one, or propose a new one ----
    if tax_path.exists():
        categories = json.loads(tax_path.read_text())
        print(f"  taxonomy      : reusing {tax_path.name} ({len(categories)} topics)")
    else:
        print("  taxonomy      : reading 200 rows to propose one...")
        sample = sample_for_taxonomy(usable, text_col, rating_col, n=DEFAULT_SAMPLE_SIZE)
        try:
            categories = propose_taxonomy(client, sample, MODEL)
        except TaxonomyError as e:
            print(f"  skipped — could not propose a taxonomy: {e}")
            return None

        print()
        for i, c in enumerate(categories, 1):
            print(f"      {i:2d}. {c}")
        print()
        if not args.yes:
            answer = input("  Use these topics? [Y/n/skip] ").strip().lower()
            if answer.startswith("s"):
                print("  skipped.")
                return None
            if answer.startswith("n"):
                print("  Edit them in", tax_path.name, "then re-run. Writing them out now.")
                tax_path.write_text(json.dumps(categories, indent=2))
                return None
        tax_path.write_text(json.dumps(categories, indent=2))

    # ---- label ----
    done = load_done(jsonl_path)
    todo = [(int(r.row_id), str(getattr(r, text_col)))
            for r in usable.itertuples() if int(r.row_id) not in done]

    print(f"  already done  : {len(done):,}")
    print(f"  to do now     : {len(todo):,}")

    if todo:
        est = len(todo) / 10 * (1900 / 1_000_000 * 1.0 + 1200 / 1_000_000 * 5.0)
        print(f"  rough cost    : ${est:.2f}\n")

        fh = open(jsonl_path, "a", encoding="utf-8")
        lock_note = {"n": 0}

        def on_result(rid, label):
            rec = dict(label)
            rec["row_id"] = rid
            fh.write(json.dumps(rec) + "\n")
            fh.flush()

        def on_progress(t):
            lock_note["n"] += 1
            pct = t["done"] / max(t["total"], 1)
            print(f"\r  {t['done']:,}/{t['total']:,} ({pct:.0%})  "
                  f"failed {t['failed']}  ${t['cost']:.2f}   ", end="", flush=True)

        label_all(client, todo, categories, MODEL,
                  concurrency=args.concurrency,
                  on_result=on_result, on_progress=on_progress)
        fh.close()
        print()

    # ---- join and write ----
    done = load_done(jsonl_path)
    labels = pd.DataFrame(done.values()) if done else pd.DataFrame(columns=["row_id"])
    for col in ("categories", "keywords"):
        if col in labels:
            labels[col] = labels[col].apply(
                lambda v: " | ".join(v) if isinstance(v, list) else "")

    merged = df.merge(labels, on="row_id", how="left")
    merged.to_csv(out_path, index=False)

    cov = coverage_report(list(done.values()))
    print(f"\n  wrote {out_path.name}")
    print(f"  labeled       : {len(done):,} of {len(usable):,}")
    print(f"  topics        : {len(categories)}")
    print(f"  {cov['message']}")
    if cov["needs_revision"]:
        print(f"  -> consider editing {tax_path.name} and re-running to re-label.")

    return {"file": path.name, "labeled": len(done), "coverage": cov["rate"]}


def main():
    ap = argparse.ArgumentParser(description="Label every dataset in a folder with Claude.")
    ap.add_argument("folder", nargs="?", default="datasets",
                    help="folder of CSV/TSV/XLSX files (default: datasets)")
    ap.add_argument("--only", help="substring — process only files whose name contains it")
    ap.add_argument("--limit", type=int, help="label at most this many rows per file (for a test run)")
    ap.add_argument("--cap", action="append", default=[], metavar="NAME=N",
                    help="cap one dataset, e.g. --cap amazon_music=2000. Repeatable. "
                         "Overrides --limit for that file.")
    ap.add_argument("--concurrency", type=int, default=10, help="parallel requests (default 10)")
    ap.add_argument("--yes", action="store_true", help="accept proposed taxonomies without asking")
    ap.add_argument("--keep-duplicates", action="store_true",
                    help="label repeated reviews too (off by default — duplicates cost money twice)")
    args = ap.parse_args()

    # --cap amazon_music=2000  ->  {"amazon_music": 2000}
    args.caps = {}
    for pair in args.cap:
        if "=" not in pair:
            sys.exit(f"--cap needs NAME=N, got: {pair}")
        name, _, n = pair.partition("=")
        if not n.isdigit():
            sys.exit(f"--cap needs a number, got: {pair}")
        args.caps[name.strip().lower()] = int(n)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit('ANTHROPIC_API_KEY is not set.\n  export ANTHROPIC_API_KEY="sk-ant-..."')

    folder = Path(args.folder)
    if not folder.is_dir():
        sys.exit(f"Not a folder: {folder}")

    files = sorted(p for ext in ("*.csv", "*.tsv", "*.xlsx", "*.xls")
                   for p in folder.glob(ext)
                   if not p.stem.endswith("_labeled"))
    if args.only:
        files = [p for p in files if args.only.lower() in p.name.lower()]
    if not files:
        sys.exit(f"No datasets found in {folder}")

    client = anthropic.Anthropic()

    print(f"{len(files)} dataset(s) to process, {args.concurrency} requests in flight.")
    summaries = [s for s in (process(p, client, args) for p in files) if s]

    print(f"\n{'=' * 70}\nDONE\n{'=' * 70}")
    for s in summaries:
        print(f"  {s['file']:<34} {s['labeled']:>7,} rows   "
              f"{s['coverage']:.1%} uncategorized")


if __name__ == "__main__":
    main()
