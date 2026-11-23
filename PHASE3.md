# Phase 3 — Clean and analyse (Ramayana: Part 1)

Run: `pip install pandas numpy matplotlib vaderSentiment py7zr openpyxl` then
`python phase3_clean_analyse.py --imdb50k "path/to/IMDB Dataset.csv"` (Kaggle IMDb 50K, not committed).
`raw/data.7z` is extracted to `data/reddit/` automatically (gitignored).

| Task | Where |
|---|---|
| Film-name matching (`film_key()`: *Ramayana: Part 1*, *RAMAYANA*, *Rise of a Legend* → `ramayana_part1_2026`; Adipurush / 1987 serial / anime / Part 2 kept separate) | script §1, `clean/reddit_comments_clean.csv` |
| ₹ text → crore, text dates → datetime, text counts → int | `clean/box_office_clean.csv`, `clean/imdb_rating_distribution.csv` |
| Missing values: deleted comments (dropped), missing day-wise collections (kept NaN, not interpolated), reviews with no score (kept, not imputed) | `clean/daily_collections_with_gaps.csv`, cleaning log |
| Usernames, bots, duplicates removed | `clean/reddit_comments_clean.csv` |
| Spreadsheet vs pandas (same ₹ → crore step) | `clean/spreadsheet_vs_pandas.xlsx` / `.csv` |
| VADER validated on IMDb 50K | `clean/vader_imdb50k_validation.csv`, `clean/vader_imdb50k_confusion.csv` |
| Sentiment per film per source | `clean/sentiment_by_film_source.csv` |
| Normalisation (min-max 0–1 + z-score) | `clean/normalised_voices.csv`, `clean/figures/` |
| **Cleaning log** (one row per decision) | `clean/cleaning_log.csv` |

## Spreadsheet vs pandas
The spreadsheet version needs four helper columns (strip ₹ and commas → find first digit → read number → scale by unit) because LibreOffice/Excel xlsx has no portable regex function. pandas does the same thing with one `to_crore()` function reused across every source. Both agree on 16/16 rows (checked by recalculating the xlsx in LibreOffice). The spreadsheet is easier to audit cell by cell. pandas is reproducible and doesn't break when someone edits a cell.

## VADER on IMDb 50K
Accuracy 69.8% (threshold compound ≥ 0). It is biased towards positive: recall is 86% on positive reviews but only 53% on negative ones. Read Ramayana sentiment as *relative* (between sources and over days), not as an absolute level.
