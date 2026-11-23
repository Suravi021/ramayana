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

---

# Phase 3b — PySpark job and core analysis

Run order: `phase3_clean_analyse.py` → `spark_text_job.py` → `core_analysis.py`.
PySpark 4.x needs Java 17+. On Java 11, use `pip install pyspark==3.5.3`.

## PySpark text job (`spark_text_job.py` → `clean/spark/`)
**82,300 text rows.** That is 32.3k Reddit comments plus critic texts, plus the IMDb 50K corpus, which was added because we had fewer than 50k rows. Steps: clean (lowercase, strip URLs, HTML and punctuation), measure length (characters and words), count terms (English, Hinglish and filler stopwords removed), aggregate by film × source. Outputs: `length_by_film_source.csv`, `top_terms_by_film_source.csv`, `term_counts_part1.csv`.

## Core analysis (`core_analysis.py` → `clean/analysis/`)
**Agreement gap** (`agreement_gap_by_film.csv`). Only *Ramayana: Part 1* has critic data. The other works (Adipurush, the 1987 serial, the anime, Part 2) appear only in Reddit talk, so their gap is undefined and they are kept as Reddit-only comparisons. Part 1, all on a 0–1 scale:

| gap | value |
|---|---|
| critic stars − Reddit sentiment | +0.09 |
| critic text sentiment − Reddit sentiment (same instrument) | +0.23 |
| IMDb users − Reddit | +0.27 |
| critic stars − IMDb users | −0.18 |

Reddit is the coolest voice and IMDb users are the warmest.

**Correlations** (`source_pair_correlations.csv`). With one scored film, a correlation *across films* is impossible (n = 1). We correlate across **10 film aspects** instead, plus across days for the subreddits. Critics vs Reddit across aspects: r = 0.38 (ρ = 0.37). They broadly agree on which parts are better or worse, but the link is weak. r/IndianCinema vs r/RamayanaTheFilm: r = 0.82 on aspects and r = 0.42 across days. n = 10 aspects, so none of these are significant. Read them as direction only.

**Splits** (`aspect_gap_critics_vs_reddit.csv`, `split_terms_critics_vs_reddit.csv`, `complaint_terms_by_side.csv`, chart `figures/aspect_gap_critics_vs_reddit.png`):
- Critics are warmer on emotion/warmth (+0.11) and Ravana/Yash (+0.08). Reddit is warmer on faithfulness to Valmiki (−0.05) and music (−0.02).
- Critic vocabulary is about *restraint, calculated, television, liberal, offence, criticism, rightwing, excess*: they judge the film's politics and craft. Reddit vocabulary is about *story, vfx, feel, character, music, history*, and its complaints mention *evil, fight, shown, jatayu, cut*: specific scenes and how characters were portrayed.
- Caveat: critic counts per aspect are small (7–43 sentences against 774–3,990 Reddit sentences). The length & pacing gap (+0.16) rests on only 7 critic sentences.

**Box office link** (`box_office_trend.csv`, `box_office_vs_sources.csv`, `reddit_week1_vs_week2.csv`). Wikipedia gives no day-wise data for week 2. The proxy: days 5–10 averaged ₹140.6 cr/day worldwide against ₹125 cr/day over the opening weekend, a ratio of 1.12. That ratio is an upper bound, because the weekend figure is "over ₹500 cr". Collections **held** into week 2. Reddit mood was flat (mean compound 0.086 → 0.088; 40.7% → 41.1% positive). The voice that matched the hold was **IMDb users (0.82)**, not critics (0.64, middling) or Reddit's level (0.54). This is one film and partial week-2 data: suggestive, not proof.

**SQL** (`sql/queries.sql`, results in `clean/analysis/sql_*.csv`, database `clean/radar_phase3.db`). Six tables: films, sources, critic_reviews, reddit_comments, box_office, aspect_mentions. Seven queries, covering: average score by source and by language (JOIN + GROUP BY), agreement gap per film (LEFT JOIN of aggregates), daily volume and mood, week 1 vs week 2 (CASE), aspect gap (JOIN + GROUP BY + HAVING), gross per screen (self-JOIN), and above/below the critic average (subquery). Sentiment uses `/ 2.0`, because SQLite stores a compound of exactly 0 as an integer and `/ 2` would do integer division.
