"""Core analysis: agreement gaps, source correlations, term splits, box-office link, SQL.

Run after phase3_clean_analyse.py and spark_text_job.py:
    python core_analysis.py
"""
import re, sqlite3
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

BASE = Path(__file__).resolve().parent
CLEAN, OUT, FIG = BASE / 'clean', BASE / 'clean' / 'analysis', BASE / 'clean' / 'figures'
SQL_DIR = BASE / 'sql'
FILM = 'ramayana_part1_2026'
OUT.mkdir(parents=True, exist_ok=True); SQL_DIR.mkdir(exist_ok=True)
sia = SentimentIntensityAnalyzer()
mm = lambda c: (c + 1) / 2          # VADER compound -1..+1 -> 0..1

comments = pd.read_csv(CLEAN / 'reddit_comments_clean.csv', parse_dates=['created_ist'])
critics = pd.read_csv(CLEAN / 'critic_reviews_clean.csv')
fulltx = pd.read_csv(CLEAN / 'critic_full_texts_clean.csv')
fulltx = fulltx[fulltx.is_review]          # ThePrint press round-up excluded (see cleaning log)
imdb = pd.read_csv(CLEAN / 'imdb_rating_distribution.csv')
bo = pd.read_csv(CLEAN / 'box_office_clean.csv')
voices = pd.read_csv(CLEAN / 'normalised_voices.csv')
sent = pd.read_csv(CLEAN / 'sentiment_by_film_source.csv')

# ============================================================ 1. agreement gap per film
films = sorted(comments.film_key.unique())
critic_01 = critics.groupby('film_key').stars_5.mean() / 5
imdb_01 = pd.Series({FILM: ((imdb.stars * imdb.votes).sum() / imdb.votes.sum() - 1) / 9})
reddit_01 = mm(comments.groupby('film_key').compound.mean())
critic_sent_01 = mm(critics.groupby('film_key').compound.mean())
gap = pd.DataFrame({'critic_stars_0_1': critic_01, 'critic_text_sent_0_1': critic_sent_01, 'imdb_users_0_1': imdb_01,
                    'reddit_sent_0_1': reddit_01, 'n_reddit': comments.film_key.value_counts()}).reindex(films)
gap['gap_critic_minus_reddit'] = gap.critic_stars_0_1 - gap.reddit_sent_0_1
gap['gap_critic_minus_imdb'] = gap.critic_stars_0_1 - gap.imdb_users_0_1
gap['gap_imdb_minus_reddit'] = gap.imdb_users_0_1 - gap.reddit_sent_0_1
# Sentiment-vs-sentiment gap is the fairest (same instrument, same scale): critic text VADER minus Reddit VADER
gap['gap_critic_text_minus_reddit_text'] = gap.critic_text_sent_0_1 - gap.reddit_sent_0_1
gap['note'] = np.where(gap.critic_stars_0_1.isna(), 'No critic score in our data (only Reddit mentions); gap undefined', '')
gap.index.name = 'film_key'
gap.round(3).to_csv(OUT / 'agreement_gap_by_film.csv')
print('\n== Agreement gap per film\n', gap.round(3).to_string())

# ============================================================ 2. aspect-level scores (unit for correlations)
ASPECTS = {
    'VFX & visuals': r'\bvfx\b|\bcgi\b|visual|graphic|spectacle|grand|\b3d\b|imax',
    'Story & screenplay': r'story|screenplay|script|writing|narrative|plot',
    'Length & pacing': r'runtime|length|lengthy|\bpace|pacing|\bslow\b|boring|\bdrag|three hours|3 hours|long',
    'Music & songs': r'music|song|score|\bbgm\b|zimmer|rahman',
    'Rama / Ranbir': r'ranbir|\brama?\b',
    'Ravana / Yash': r'\byash\b|ravana?n?\b',
    'Sita / Sai Pallavi': r'sai pallavi|\bsita\b',
    'Faithfulness / Valmiki': r'valmiki|faithful|accura|source material|scripture|adaptation',
    'Emotion & warmth': r'emotion|warmth|heart|soul|feel|moving|tears',
    'Direction / Tiwari': r'tiwari|direct',
}
def sentences(text):
    return [s for s in re.split(r'(?<=[.!?])\s+|\n+', str(text)) if len(s.split()) >= 3]

rows = []
p1 = comments[comments.film_key.eq(FILM)]
for src, texts in [('critic_blurbs', critics.review_text), ('critic_full_texts', fulltx.text)] + \
                  [(f'reddit_r/{s}', g.body_clean) for s, g in p1.groupby('subreddit')]:
    for t in texts:
        for s in sentences(t):
            low = s.lower()
            hits = [a for a, pat in ASPECTS.items() if re.search(pat, low)]
            if hits:
                c = sia.polarity_scores(s)['compound']
                rows += [dict(source=src, aspect=a, compound=c, sentence=s[:300]) for a in hits]
asp = pd.DataFrame(rows)
asp['side'] = np.where(asp.source.str.startswith('critic'), 'Critics', 'Reddit')
asp_tab = asp.groupby(['aspect', 'source']).compound.agg(['mean', 'size']).unstack('source')
asp_tab.round(3).to_csv(OUT / 'aspect_sentiment_by_source.csv')
side = asp.groupby(['aspect', 'side']).compound.agg(['mean', 'size']).unstack('side')
side.columns = [f'{a}_{b}' for a, b in side.columns]
side['gap_critics_minus_reddit_0_1'] = (mm(side.mean_Critics) - mm(side.mean_Reddit))
side = side.sort_values('gap_critics_minus_reddit_0_1')
side.round(3).to_csv(OUT / 'aspect_gap_critics_vs_reddit.csv')
asp.to_csv(OUT / 'aspect_sentences.csv', index=False)
print('\n== Aspect gap (critics - Reddit, 0-1 scale)\n', side.round(3).to_string())

# ============================================================ 3. correlation between each pair of sources
# Only one film has critic scores, so correlation ACROSS FILMS is impossible (n=1).
# Two within-film units instead: (a) aspects (10 points), (b) days since release (Reddit subs + pageview-free).
MIN_N = 3
m = asp.groupby(['aspect', 'source']).compound.agg(['mean', 'size'])
mean_wide = m['mean'].unstack(); n_wide = m['size'].unstack()
mean_wide = mean_wide.where(n_wide >= MIN_N)
mean_wide['critics_all'] = asp[asp.side.eq('Critics')].groupby('aspect').compound.mean()
mean_wide['reddit_all'] = asp[asp.side.eq('Reddit')].groupby('aspect').compound.mean()
pairs = []
for a, b in combinations(mean_wide.columns, 2):
    d = mean_wide[[a, b]].dropna()
    if len(d) >= 4:
        pairs.append(dict(unit='aspect', source_a=a, source_b=b, n=len(d),
                          pearson_r=d[a].corr(d[b]), spearman_rho=d[a].corr(d[b], method='spearman')))
daily = (p1[p1.day_since_release.between(0, 14)].groupby(['day_since_release', 'subreddit']).compound
         .agg(['mean', 'size']))
dmean = daily['mean'].unstack().where(daily['size'].unstack() >= 10)
for a, b in combinations(dmean.columns, 2):
    d = dmean[[a, b]].dropna()
    if len(d) >= 4:
        pairs.append(dict(unit='day', source_a=f'reddit_r/{a}', source_b=f'reddit_r/{b}', n=len(d),
                          pearson_r=d[a].corr(d[b]), spearman_rho=d[a].corr(d[b], method='spearman')))
corr = pd.DataFrame(pairs)
corr.round(3).to_csv(OUT / 'source_pair_correlations.csv', index=False)
print('\n== Source-pair correlations\n', corr.round(3).to_string(index=False))

# ============================================================ 4. explain the splits: distinctive terms per side
STOP = set('''the a an and or but of to in on for with is are was were be been it its this that as at by from
 not no so than then too very can could would should will just about into over more most some such only own same
 have has had do does did i you he she they we me him her them my your his their our what which who whom whose
 there here when where why how all any both each few other ramayana film movie part one also like even'''.split())
NAMES = set('''ram rama ravan ravana sita sai pallavi ranbir kapoor yash tiwari nitesh namit malhotra hanuman lakshman laxman
 valmiki ramayan ramayana surpanakha indra kaikeyi dasharath bharat lanka ayodhya hai nhi bhai'''.split())
GENERIC = set('''people because much see out now make time show scene scenes good better best great bad many really movie'''.split())
def term_freq(texts):
    toks = pd.Series(' '.join(map(str, texts)).lower().replace("'", '').split())
    toks = toks.str.replace(r'[^a-z]', '', regex=True)
    toks = toks[(toks.str.len() >= 3) & ~toks.isin(STOP | NAMES | GENERIC)]
    return toks.value_counts()

def log_odds(f1, f2, prior=0.5):
    """Log-odds ratio with an informative Dirichlet prior (Monroe et al. 2008), z-scored."""
    vocab = f1.index.union(f2.index)
    a, b = f1.reindex(vocab, fill_value=0), f2.reindex(vocab, fill_value=0)
    pri = pd.Series(prior, index=vocab)    # flat prior keeps rare terms in check
    n1, n2, p0 = a.sum(), b.sum(), pri.sum()
    l1 = np.log((a + pri) / (n1 + p0 - a - pri)); l2 = np.log((b + pri) / (n2 + p0 - b - pri))
    z = (l1 - l2) / np.sqrt(1 / (a + pri) + 1 / (b + pri))
    return pd.DataFrame({'count_a': a, 'count_b': b, 'z': z}).sort_values('z')

# Compare only sentences about the film (aspect-tagged): keeps ThePrint's global-media/politics roundup from dominating
lo = log_odds(term_freq(asp[asp.side.eq('Critics')].sentence.drop_duplicates()), term_freq(asp[asp.side.eq('Reddit')].sentence.drop_duplicates()))
lo = lo[(lo.count_a >= 2) & ((lo.count_a + lo.count_b) >= 5)]
split_terms = pd.concat([lo.tail(20).iloc[::-1].assign(side='critics_over_reddit'),
                         lo.head(20).assign(side='reddit_over_critics')]).rename(columns={'count_a': 'critic_count', 'count_b': 'reddit_count'})
split_terms.round(2).to_csv(OUT / 'split_terms_critics_vs_reddit.csv')
# terms inside NEGATIVE sentences on each side: what each side complains about
neg = asp[asp.compound <= -0.05]
neg_terms = pd.DataFrame({'critics_negative': term_freq(neg[neg.side.eq('Critics')].sentence.drop_duplicates()).head(15).index,
                          'reddit_negative': term_freq(neg[neg.side.eq('Reddit')].sentence.drop_duplicates()).head(15).index})
neg_terms.to_csv(OUT / 'complaint_terms_by_side.csv', index=False)
print('\n== Terms critics use more:', ', '.join(split_terms[split_terms.side.eq('critics_over_reddit')].index[:15]))
print('== Terms Reddit uses more :', ', '.join(split_terms[split_terms.side.eq('reddit_over_critics')].index[:15]))
print('== Complaint terms\n', neg_terms.to_string(index=False))

# ============================================================ 5. link back to box office: week-2 trend
g = bo.set_index(['metric', 'territory']).value_crore
wk1_open = g[('opening_weekend_gross', 'worldwide')]             # days 1-4 (5-8 Nov), lower bound
cum_d10 = g[('cumulative_gross', 'worldwide')]                    # to 14 Nov = day 10
opening_day = g[('opening_day_gross', 'worldwide')]
bo_trend = pd.DataFrame([
    dict(period='Opening day (day 1, 5 Nov)', days=1, gross_cr=opening_day, per_day_cr=opening_day),
    dict(period='Opening weekend (days 1-4, 5-8 Nov)', days=4, gross_cr=wk1_open, per_day_cr=wk1_open / 4),
    dict(period='Days 5-10 (9-14 Nov, incl. start of week 2)', days=6, gross_cr=cum_d10 - wk1_open, per_day_cr=(cum_d10 - wk1_open) / 6),
])
bo_trend['note'] = ['', 'Reported as "over ₹500 crore" -> lower bound', 'Upper bound (cumulative minus a lower-bound weekend)']
bo_trend.round(2).to_csv(OUT / 'box_office_trend.csv', index=False)
hold = bo_trend.per_day_cr.iloc[2] / bo_trend.per_day_cr.iloc[1]

r = p1.assign(week=np.select([p1.day_since_release.between(0, 6), p1.day_since_release.between(7, 13)], ['week1', 'week2'], 'other'))
wk = r[r.week.ne('other')].groupby('week').compound.agg(['mean', 'size'])
wk['pct_positive'] = r[r.week.ne('other')].groupby('week').compound.apply(lambda c: (c >= 0.05).mean())
wk.round(3).to_csv(OUT / 'reddit_week1_vs_week2.csv')

verdict = pd.DataFrame([
    dict(source='Critics (stars)', signal_0_1=gap.loc[FILM, 'critic_stars_0_1'],
         predicted_week2='Middling (0.64): expect a normal Bollywood fall-off after the opening weekend'),
    dict(source='IMDb users', signal_0_1=gap.loc[FILM, 'imdb_users_0_1'],
         predicted_week2='Strong (0.82): expect word of mouth to hold collections up'),
    dict(source='Reddit (level)', signal_0_1=gap.loc[FILM, 'reddit_sent_0_1'],
         predicted_week2='Lukewarm (0.54): expect a drop'),
    dict(source='Reddit (week 1 -> week 2 change)', signal_0_1=mm(wk.loc['week2', 'mean']) - mm(wk.loc['week1', 'mean']),
         predicted_week2='Mood ' + ('rose' if wk.loc['week2', 'mean'] > wk.loc['week1', 'mean'] else 'fell') + ' in week 2'),
])
verdict['actual'] = f'Per-day gross on days 5-10 was {hold:.2f}x the opening-weekend per-day average (≤ because weekend is a lower bound)'
verdict.round(3).to_csv(OUT / 'box_office_vs_sources.csv', index=False)
print('\n== Box-office trend\n', bo_trend.round(2).to_string(index=False))
print('\n== Reddit week1 vs week2\n', wk.round(3).to_string())
print(f'\nhold ratio (days 5-10 per day / weekend per day) = {hold:.2f}')

# ---- charts
fig, ax = plt.subplots(figsize=(9, 4.6))
s = side.dropna(subset=['mean_Critics', 'mean_Reddit'])
y = np.arange(len(s))
ax.hlines(y, mm(s.mean_Critics), mm(s.mean_Reddit), color='#bbb', lw=2)
ax.scatter(mm(s.mean_Critics), y, color='#3b6ea5', s=60, label='Critics', zorder=3)
ax.scatter(mm(s.mean_Reddit), y, color='#d9822b', s=60, label='Reddit', zorder=3)
ax.set_yticks(y); ax.set_yticklabels([f'{a} (n={int(nc)}/{int(nr)})' for a, nc, nr in zip(s.index, s.size_Critics, s.size_Reddit)], fontsize=8)
ax.axvline(0.5, color='grey', lw=0.8, ls=':'); ax.set_xlabel('Mean sentence sentiment, 0–1 (0.5 = neutral)')
big = s[s.size_Critics >= 15].gap_critics_minus_reddit_0_1.abs().idxmax()   # ignore aspects with <15 critic sentences
ax.set_title(f'Critics are warmer than Reddit on "{big}"; Reddit is warmer on music and faithfulness\n'
             f'(aspects with <15 critic sentences, e.g. length & pacing, are too thin to call)', loc='left', fontsize=10)
ax.legend(frameon=False, loc='lower right')
for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig(FIG / 'aspect_gap_critics_vs_reddit.png', dpi=150); plt.close(fig)

# ============================================================ 6. SQL (SQLite file; DDL is MySQL-compatible)
db = CLEAN / 'radar_phase3.db'
db.unlink(missing_ok=True)
con = sqlite3.connect(db)
con.executescript('''
CREATE TABLE films (film_key VARCHAR(40) PRIMARY KEY, title VARCHAR(120), language VARCHAR(20), release_year INT, has_critic_data INT);
CREATE TABLE sources (source_id VARCHAR(40) PRIMARY KEY, voice VARCHAR(20), platform VARCHAR(40));
CREATE TABLE critic_reviews (review_id INT PRIMARY KEY, film_key VARCHAR(40) REFERENCES films(film_key), source_id VARCHAR(40) REFERENCES sources(source_id),
                             stars_5 DECIMAL(3,2), rating_scale DECIMAL(4,1), compound DECIMAL(5,4), review_text TEXT);
CREATE TABLE reddit_comments (comment_id VARCHAR(20) PRIMARY KEY, film_key VARCHAR(40) REFERENCES films(film_key), source_id VARCHAR(40) REFERENCES sources(source_id),
                              created_ist DATETIME, day_since_release INT, score INT, compound DECIMAL(5,4));
CREATE TABLE box_office (row_id INT PRIMARY KEY, film_key VARCHAR(40) REFERENCES films(film_key), metric VARCHAR(40), territory VARCHAR(20),
                         day_number INT, value_crore DECIMAL(10,2), value_usd_million DECIMAL(10,2), is_lower_bound INT);
CREATE TABLE aspect_mentions (mention_id INT PRIMARY KEY, source_id VARCHAR(40) REFERENCES sources(source_id), aspect VARCHAR(40), compound DECIMAL(5,4));
''')
titles = {FILM: ('Ramayana: Part 1', 'Hindi', 2026), 'ramayana_part2_2027': ('Ramayana: Part 2', 'Hindi', 2027),
          'adipurush_2023': ('Adipurush', 'Hindi/Telugu', 2023), 'ramayan_tv_1987': ('Ramayan (TV serial)', 'Hindi', 1987),
          'ramayana_anime_1993': ('Ramayana: The Legend of Prince Rama', 'Japanese/English', 1993)}
pd.DataFrame([dict(film_key=k, title=t, language=l, release_year=y, has_critic_data=int(k == FILM)) for k, (t, l, y) in titles.items()]
             ).to_sql('films', con, if_exists='append', index=False)
srcs = [('critic_blurbs', 'Critics', 'Rotten Tomatoes blurbs'), ('critic_full_texts', 'Critics', 'Full reviews (imdb.txt)')] + \
       [(f'reddit_r/{s}', 'Reddit', f'r/{s}') for s in comments.subreddit.unique()] + [('wikipedia', 'Box office', 'Wikipedia')]
pd.DataFrame(srcs, columns=['source_id', 'voice', 'platform']).to_sql('sources', con, if_exists='append', index=False)
critics.assign(source_id='critic_blurbs')[['review_id', 'film_key', 'source_id', 'stars_5', 'rating_scale', 'compound', 'review_text']] \
    .to_sql('critic_reviews', con, if_exists='append', index=False)
comments.assign(source_id='reddit_r/' + comments.subreddit, created_ist=comments.created_ist.astype(str)) \
    .rename(columns={'id': 'comment_id'})[['comment_id', 'film_key', 'source_id', 'created_ist', 'day_since_release', 'score', 'compound']] \
    .to_sql('reddit_comments', con, if_exists='append', index=False)
bo.reset_index().rename(columns={'index': 'row_id'}).assign(is_lower_bound=lambda d: d.is_lower_bound.astype(int)) \
    [['row_id', 'film_key', 'metric', 'territory', 'day_number', 'value_crore', 'value_usd_million', 'is_lower_bound']] \
    .to_sql('box_office', con, if_exists='append', index=False)
asp.reset_index().rename(columns={'index': 'mention_id', 'source': 'source_id'})[['mention_id', 'source_id', 'aspect', 'compound']] \
    .to_sql('aspect_mentions', con, if_exists='append', index=False)

QUERIES = {
'q1_avg_score_by_source_and_language': '''
-- JOIN + GROUP BY: average sentiment (0-1) by source and by film language
SELECT s.voice, s.source_id, f.language, COUNT(*) AS n, ROUND(AVG((c.compound + 1) / 2.0), 3) AS avg_sentiment_0_1
FROM reddit_comments c JOIN films f ON f.film_key = c.film_key JOIN sources s ON s.source_id = c.source_id
GROUP BY s.voice, s.source_id, f.language
UNION ALL
SELECT s.voice, s.source_id, f.language, COUNT(*), ROUND(AVG((r.compound + 1) / 2.0), 3)
FROM critic_reviews r JOIN films f ON f.film_key = r.film_key JOIN sources s ON s.source_id = r.source_id
GROUP BY s.voice, s.source_id, f.language
ORDER BY avg_sentiment_0_1 DESC;''',
'q2_agreement_gap_per_film': '''
-- JOIN of two aggregates: critic stars (0-1) minus Reddit sentiment (0-1) per film (LEFT JOIN keeps films without critics)
SELECT f.film_key, f.title, rd.n_comments, ROUND(rd.reddit_0_1, 3) AS reddit_0_1, ROUND(cr.critic_0_1, 3) AS critic_0_1,
       ROUND(cr.critic_0_1 - rd.reddit_0_1, 3) AS gap_critic_minus_reddit
FROM films f
LEFT JOIN (SELECT film_key, COUNT(*) AS n_comments, AVG((compound + 1) / 2.0) AS reddit_0_1 FROM reddit_comments GROUP BY film_key) rd ON rd.film_key = f.film_key
LEFT JOIN (SELECT film_key, AVG(stars_5) / 5.0 AS critic_0_1 FROM critic_reviews GROUP BY film_key) cr ON cr.film_key = f.film_key
ORDER BY rd.n_comments DESC;''',
'q3_reddit_daily_volume_and_mood': '''
-- GROUP BY day: buzz velocity and mood for Part 1
SELECT day_since_release, COUNT(*) AS comments, ROUND(AVG(compound), 3) AS mean_compound,
       ROUND(100.0 * SUM(CASE WHEN compound >= 0.05 THEN 1 ELSE 0 END) / COUNT(*), 1) AS pct_positive
FROM reddit_comments WHERE film_key = 'ramayana_part1_2026' AND day_since_release BETWEEN 0 AND 14
GROUP BY day_since_release ORDER BY day_since_release;''',
'q4_week1_vs_week2': '''
-- CASE + GROUP BY: does Reddit mood change from week 1 to week 2?
SELECT CASE WHEN day_since_release BETWEEN 0 AND 6 THEN 'week1' ELSE 'week2' END AS week,
       COUNT(*) AS comments, ROUND(AVG(compound), 3) AS mean_compound
FROM reddit_comments WHERE film_key = 'ramayana_part1_2026' AND day_since_release BETWEEN 0 AND 13
GROUP BY week;''',
'q5_aspect_gap_critics_vs_reddit': '''
-- JOIN + GROUP BY + HAVING: aspect sentiment by voice, only aspects both voices discuss
SELECT a.aspect,
       ROUND(AVG(CASE WHEN s.voice = 'Critics' THEN (a.compound + 1) / 2.0 END), 3) AS critics_0_1,
       ROUND(AVG(CASE WHEN s.voice = 'Reddit'  THEN (a.compound + 1) / 2.0 END), 3) AS reddit_0_1,
       SUM(s.voice = 'Critics') AS n_critic, SUM(s.voice = 'Reddit') AS n_reddit
FROM aspect_mentions a JOIN sources s ON s.source_id = a.source_id
GROUP BY a.aspect HAVING n_critic >= 3 AND n_reddit >= 3
ORDER BY ABS(critics_0_1 - reddit_0_1) DESC;''',
'q7_top_critic_and_bottom_critic': '''
-- subquery: reviews above / below the critic average
SELECT review_id, stars_5, ROUND(compound, 2) AS compound, SUBSTR(review_text, 1, 90) AS excerpt,
       CASE WHEN stars_5 >= (SELECT AVG(stars_5) FROM critic_reviews) THEN 'above_avg' ELSE 'below_avg' END AS vs_avg
FROM critic_reviews ORDER BY stars_5 DESC;''',
}
# screens are integers stored in value_int in the csv; keep them in the DB for q6
scr = bo[bo.metric.eq('screens')]
con.executemany('UPDATE box_office SET value_crore = ? WHERE row_id = ?', [(float(v), int(i)) for i, v in scr.value_int.items()])
con.commit()
QUERIES['q6_box_office_per_screen'] = '''
-- self-JOIN: opening-day gross per screen (₹ lakh per screen) by territory
SELECT g.territory, g.metric, g.value_crore AS opening_day_cr, CAST(s.value_crore AS INT) AS screens,
       ROUND(g.value_crore * 100.0 / s.value_crore, 2) AS lakh_per_screen
FROM box_office g JOIN box_office s ON s.film_key = g.film_key AND s.territory = g.territory AND s.metric = 'screens'
WHERE g.metric LIKE 'opening_day%';'''

with open(SQL_DIR / 'queries.sql', 'w') as f:
    for name, q in QUERIES.items():
        f.write(f'-- {name}{q}\n\n')
        res = pd.read_sql(q, con)
        res.to_csv(OUT / f'sql_{name}.csv', index=False)
        print(f'\n== SQL {name}\n', res.head(16).to_string(index=False))
con.close()
