# clean_datasets.R — prepare the raw Kaggle files for labeling
# =============================================================
# Scraped review datasets arrive dirty in predictable ways: blank rows, the same
# review pasted in many times, and scraper artifacts left in the text. Labeling
# a duplicate costs money twice and, worse, inflates that review's topic count —
# so the dashboard quietly reports something untrue.
#
# This script does the cleaning as a visible, reproducible step and writes a
# clean copy of each file. Run it before label_datasets.py.
#
#   Rscript clean_datasets.R              # reads ./datasets, writes ./datasets/clean
#
# What it removes, in order:
#   1. rows where the review text is missing or blank
#   2. scraper artifacts in the text ("READ MORE", repeated whitespace)
#   3. rows shorter than 20 characters — not a real review
#   4. exact duplicate reviews, keeping the first
#
# It prints a before/after table so you can see exactly what went and defend it.

library(dplyr)
library(readr)
library(stringr)

in_dir  <- "datasets"
out_dir <- file.path(in_dir, "clean")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

MIN_CHARS <- 20

# The text column in each file. Verified against the actual data — these names
# are not guesses.
files <- tribble(
  ~file,                ~text_col,           ~delim,
  "amazon_music.csv",   "reviewText",        ",",
  "amazon_alexa.tsv",   "verified_reviews",  "\t",
  "starbucks.csv",      "Review",            ",",
  "iphone14.csv",       "review",            ",",
  "mcdonalds.csv",      "review_details",    ","
)

report <- list()

for (i in seq_len(nrow(files))) {
  f    <- files$file[i]
  txt  <- files$text_col[i]
  path <- file.path(in_dir, f)

  if (!file.exists(path)) {
    message("skipping (not found): ", f)
    next
  }

  df <- read_delim(path, delim = files$delim[i],
                   show_col_types = FALSE, progress = FALSE)
  n_raw <- nrow(df)

  df <- df %>%
    # 1. drop missing text
    filter(!is.na(.data[[txt]])) %>%
    # 2. strip scraper artifacts and normalise whitespace
    mutate(
      !!txt := .data[[txt]] %>%
        str_remove_all("READ MORE\\s*$") %>%   # Flipkart scraper leaves this on iPhone reviews
        str_replace_all("\\s+", " ") %>%       # collapse newlines and runs of spaces
        str_trim()
    ) %>%
    # 3. drop anything too short to be a real review
    filter(str_length(.data[[txt]]) >= MIN_CHARS)

  n_after_empty <- nrow(df)

  # 4. drop exact duplicates, keeping the first occurrence
  df <- df %>% distinct(.data[[txt]], .keep_all = TRUE)
  n_clean <- nrow(df)

  write_csv(df, file.path(out_dir, str_replace(f, "\\.(csv|tsv)$", ".csv")))

  report[[f]] <- tibble(
    dataset    = f,
    raw        = n_raw,
    blank      = n_raw - n_after_empty,
    duplicates = n_after_empty - n_clean,
    clean      = n_clean,
    kept       = sprintf("%.0f%%", 100 * n_clean / n_raw)
  )
}

cat("\n")
print(bind_rows(report), n = Inf)

totals <- bind_rows(report) %>%
  summarise(raw = sum(raw), blank = sum(blank),
            duplicates = sum(duplicates), clean = sum(clean))

cat("\n")
cat(sprintf("Removed %s blank and %s duplicate rows.\n",
            format(totals$blank, big.mark = ","),
            format(totals$duplicates, big.mark = ",")))
cat(sprintf("%s rows ready to label (from %s).\n",
            format(totals$clean, big.mark = ","),
            format(totals$raw,   big.mark = ",")))
cat(sprintf("Roughly $%.2f to label at 10 reviews per request.\n",
            totals$clean / 10 * (1900 / 1e6 * 1.0 + 1200 / 1e6 * 5.0)))
cat(sprintf("\nClean files written to %s/\n", out_dir))

# ---------------------------------------------------------------------------
# One thing this script deliberately does NOT fix
# ---------------------------------------------------------------------------
# In mcdonalds.csv the review title is concatenated onto the front of the body
# with no separator — "Horrible mealHad meal at kelso nsw it was disgusting".
# Splitting that reliably needs a rule for where the title ends, and every rule
# I could write breaks on some rows. Claude reads it correctly regardless, so it
# is left alone rather than half-fixed. The clean title is in `review_title` if
# you want it separately.
