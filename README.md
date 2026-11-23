# Release Day Radar: *Ramayana: Part 1* (2026)

Case Study A. **Do critics, IMDb users and Reddit users agree about the same film? Where do they split, and why?**

## Headline findings
- **The voices disagree on level, not on direction.** On a common 0–1 scale: IMDb users 0.82, critic blurb sentiment 0.78, critic stars 0.64 (3.18/5 from 17 reviews), Reddit 0.54 (31,754 comments). Reddit is the coolest voice and IMDb users the warmest.
- **Agreement gap (Part 1):** critic stars − Reddit = +0.09. Critic text − Reddit text (same VADER instrument) = +0.23. IMDb − Reddit = +0.27.
- **Where they split:** critics are warmer on *emotion & warmth* (+0.11) and *Ravana/Yash* (+0.08). Reddit is warmer on *faithfulness to Valmiki* and *music*. Critics write about politics and craft (*restraint, calculated, liberal, offence, rightwing*). Reddit writes about specific scenes and characters (*story, vfx, evil, fight, jatayu, cut*).
- **Correlation across 10 aspects:** critics vs Reddit r = 0.38 (weak). r/IndianCinema vs r/RamayanaTheFilm r = 0.82. n = 10, so these are directional only.
- **Box office:** ₹121.6 cr worldwide on day 1, over ₹500 cr on the opening weekend, ₹1,343.5 cr worldwide by 14 Nov. Days 5–10 earned about 1.12× the weekend's per-day rate: the film **held**. Reddit mood was flat from week 1 to week 2. The voice that matched the hold was **IMDb users**, not critics or Reddit.
- **Sentiment method check:** VADER scored **69.8%** on the labelled IMDb 50K corpus and is biased positive (86% recall on positive reviews, 53% on negative). Read sentiment as relative between sources, not as an absolute level.

## Data
| Source | File | Rows |
|---|---|---|
| Reddit dumps (r/RamayanaTheFilm, r/bollywood, r/IndianCinema), JSONL | `raw/data.7z` | 47,616 raw comments → 32,757 clean |
| Wikipedia box office (₹ text, dates, screens) | `raw/ramayana_part1_wikipedia_boxoffice_raw.csv` | 17 |
| Critic blurbs with ratings on /4, /5 and /10 scales | `raw/ramayana_reviews_normalized.csv` | 17 |
| IMDb vote distribution + full critic texts | `imdb.txt` | 10 star bins (692 votes), 3 texts |
| IMDb 50K labelled corpus (Kaggle, not committed) | pass with `--imdb50k` | 50,000 |

## Run it
```bash
pip install pandas numpy matplotlib vaderSentiment py7zr openpyxl pyspark==3.5.3   # pyspark 4.x needs Java 17+
python phase3_clean_analyse.py --imdb50k "path/to/IMDB Dataset.csv"   # cleaning, VADER check, normalisation
python spark_text_job.py       --imdb50k "path/to/IMDB Dataset.csv"   # PySpark job, 82,300 text rows
python core_analysis.py                                               # gaps, correlations, splits, box office, SQL
```
`raw/data.7z` is extracted to `data/reddit/` automatically (gitignored).

## Repo map
| Path | What |
|---|---|
| `phase3_clean_analyse.py` | Film-name matching, type casting, missing values, usernames/bots/duplicates removed, spreadsheet vs pandas, VADER on IMDb 50K, normalisation |
| `spark_text_job.py` | PySpark: clean → length → term counts → aggregate by film × source |
| `core_analysis.py` | Agreement gaps, source correlations, term splits, box-office link, SQLite DB + 7 queries |
| `clean/cleaning_log.csv` | **Cleaning log**: one row per decision (problem, rows affected, action, why) |
| `clean/` | Cleaned tables, `spreadsheet_vs_pandas.xlsx`, `radar_phase3.db` |
| `clean/analysis/` | Gap, correlation, aspect, term and box-office tables, plus SQL results |
| `clean/spark/` | Spark outputs |
| `clean/figures/` | Charts: normalised voices, aspect gap, Reddit volume vs mood (z-scores) |
| `sql/queries.sql` | 7 queries (JOIN, GROUP BY, HAVING, CASE, self-JOIN, subquery) |
| `data.ipynb` | Earlier team notebook (Reddit + Wikipedia API + PySpark) |
| `PHASE3.md` | Detailed Phase 3 notes |

## Limitations
- Only Part 1 has critic and IMDb data. The other Ramayana works (Adipurush, the 1987 TV serial, the 1993 anime, Part 2) appear only in Reddit talk, so cross-film gaps and correlations aren't possible.
- No day-by-day box office for week 2. The hold ratio is a proxy and an upper bound.
- There are no IMDb user review texts, only the vote distribution. Only 2 full critic reviews are used: ThePrint's text is a press roundup, so it was excluded.
- Critic sentences per aspect are few (7–43), so aspect gaps with small critic n are not reliable.
