"""Phase 3 - Clean and analyse: Ramayana: Part 1 (2026).

Reproducible from raw data to final chart:
    pip install pandas numpy matplotlib vaderSentiment py7zr openpyxl
    python phase3_clean_analyse.py --imdb50k "path/to/IMDB Dataset.csv"

Inputs  : raw/data.7z (Reddit JSONL dumps), raw/ramayana_part1_wikipedia_boxoffice_raw.csv,
          raw/ramayana_reviews_normalized.csv, imdb.txt, IMDb 50K labelled corpus (Kaggle, not committed).
Outputs : clean/*.csv (cleaned tables + cleaning_log.csv), clean/figures/*.png,
          clean/spreadsheet_vs_pandas.xlsx
"""
import argparse, html, json, re, subprocess, shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

BASE = Path(__file__).resolve().parent
RAW, CLEAN = BASE / 'raw', BASE / 'clean'
REDDIT_DIR = BASE / 'data' / 'reddit'          # extracted from raw/data.7z (gitignored)
FIG = CLEAN / 'figures'
FILM_KEY = 'ramayana_part1_2026'
RELEASE = pd.Timestamp('2026-11-05')
TZ = 'Asia/Kolkata'

CLEANING_LOG = []
def log(source, problem, rows, action, why):
    CLEANING_LOG.append(dict(step=len(CLEANING_LOG) + 1, source=source, problem=problem,
                             rows_affected=int(rows), action=action, why=why))
    print(f'[{source}] {problem}: {rows} rows -> {action}')


# ---------------------------------------------------------------- 1. film-name matching
OTHER_WORKS = [   # checked first: other Ramayana works Reddit talks about must NOT join to Part 1
    ('ramayana_part2_2027', r'ramay\w*\s*[:\-–]?\s*part\s*(2|ii|two)\b'),
    ('adipurush_2023', r'adipurush'),
    ('ramayan_tv_1987', r'ramanand|\bsagar\b.*ramay|ramay\w*.*\b1987\b'),
    ('ramayana_anime_1993', r'legend of prince rama|yugo sako|ramay\w*.*\banime\b|\b1993\b.*ramay'),
]
PART1 = re.compile(r'\br[aā]m[aā]?y[aā]n(a|am)?\b|रामायण|rise of a legend', re.I)

def film_key(text):
    """'Ramayana: Part 1', 'RAMAYANA - Official Trailer', 'Ramayana (2026)', 'Ramayana: Rise of a Legend' -> ramayana_part1_2026."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return None
    t = str(text).lower().replace('–', '-').replace('—', '-')
    for key, pat in OTHER_WORKS:
        if re.search(pat, t):
            return key
    return FILM_KEY if PART1.search(t) else None


# ---------------------------------------------------------------- 2. type-casting helpers
def to_crore(s):
    """'₹12.5 Cr' -> 12.5; '₹1,343.49 crore' -> 1343.49; '1,25,00,000' -> 1.25; '50 lakh' -> 0.5; USD -> NaN."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return np.nan
    t = str(s).lower()
    if '$' in t and not re.search(r'₹|crore|\bcr\b|lakh', t):
        return np.nan
    t = re.sub(r'\(.*?\)', '', t)          # drop "(US$430 million)" conversions
    t = t.replace('₹', '').replace('rs.', '').replace('rs', '').replace('inr', '')
    m = re.search(r'([\d,]+(?:\.\d+)?)\s*(crore|cr\b|lakh|lac)?', t)
    if not m:
        return np.nan
    num = float(m.group(1).replace(',', ''))
    unit = m.group(2)
    if unit in ('lakh', 'lac'):
        return num / 100
    if unit is None and num >= 1e5:       # bare rupee amount in Indian grouping
        return num / 1e7
    return num

def to_usd_million(s):
    t = str(s).lower()
    m = re.search(r'\$\s*([\d,.]+)\s*(million|billion)?', t)
    if not m:
        return np.nan
    v = float(m.group(1).replace(',', ''))
    return v * 1000 if m.group(2) == 'billion' else v

def to_int(s):
    m = re.search(r'[\d,]+', str(s))
    return int(m.group(0).replace(',', '')) if m else pd.NA


# ---------------------------------------------------------------- 3. box office
def clean_box_office():
    bo = pd.read_csv(RAW / 'ramayana_part1_wikipedia_boxoffice_raw.csv', dtype=str)
    n0 = len(bo)
    bo['film_key'] = bo.film_key.fillna(FILM_KEY)
    for c in ['as_of_date', 'access_date']:
        bo[c] = pd.to_datetime(bo[c], errors='coerce')
    date_rows = bo.metric.str.endswith('_date')
    bo.loc[date_rows, 'value_date'] = pd.to_datetime(bo.loc[date_rows, 'value_raw'], format='%d %B %Y', errors='coerce')
    log('box_office', 'Dates stored as text ("5 November 2026")', date_rows.sum() + bo.as_of_date.notna().sum(),
        'Parsed to datetime (format %d %B %Y; ISO for as_of_date)', 'Needed for day-since-release arithmetic and joins by date')

    money = bo.metric.str.contains('gross|net|budget|box_office') & ~date_rows
    bo['value_crore'] = np.where(money, bo.value_raw.map(to_crore), np.nan)
    bo['value_usd_million'] = np.where(money, bo.value_raw.map(to_usd_million), np.nan)
    log('box_office', 'Money stored as text ("₹62.47 crore net", "₹1,343.49 crore")', bo.value_crore.notna().sum(),
        'Converted to float crore with to_crore()', 'Arithmetic and comparison across rows')
    usd = bo.value_usd_million.notna() & bo.value_crore.isna()
    log('box_office', 'Collections reported in US$ only', usd.sum(),
        'Kept in value_usd_million, crore left NaN (not converted)', 'Exchange rate on the day is not in the source; converting would invent precision')
    lower = bo.value_raw.str.contains(r'\bover\b|\bmore than\b|\+', case=False)
    bo['is_lower_bound'] = lower
    log('box_office', 'Approximate values ("over ₹500 crore")', lower.sum(),
        'Kept the number, flagged is_lower_bound=True', 'It is a floor, not a measurement; must not be averaged as exact')
    bo['value_int'] = np.where(bo.metric.eq('screens'), bo.value_raw.map(to_int), pd.NA)
    log('box_office', 'Counts stored as text ("59,000" screens)', bo.metric.eq('screens').sum(),
        'Converted to integer', 'Counts must be numeric to compute per-screen figures')
    bo['is_combined_duology'] = bo.metric.str.contains('combined|duology') | bo.value_raw.str.contains('combined', case=False)
    log('box_office', 'Budget rows that mix Part 1 with the two-film total (₹4,000 cr)', bo.is_combined_duology.sum(),
        'Flagged is_combined_duology; Part 1 budget = ₹2,000 cr row', 'Double counting would halve any ROI estimate')
    dup = bo.metric.isin(['cumulative_gross', 'box_office_infobox']) & bo.territory.eq('worldwide')
    log('box_office', 'Two different worldwide totals (₹1,343.49 cr dated 14 Nov vs ₹1,384.50 cr undated infobox)', dup.sum(),
        'Kept both; analysis uses the dated Bollywood Hungama figure', 'A dated, cited figure is traceable; the infobox one is not')

    # missing day-wise collections
    daily = bo[bo.day_number.notna() & bo.metric.str.contains('gross|net')]
    days_have = sorted(daily.day_number.astype(int).unique())
    last_day = (bo.access_date.max() - RELEASE).days + 1
    missing = [d for d in range(1, last_day + 1) if d not in days_have]
    grid = pd.DataFrame({'film_key': FILM_KEY, 'day_number': range(1, last_day + 1)})
    grid['date'] = RELEASE + pd.to_timedelta(grid.day_number - 1, unit='D')
    india_net = bo[bo.metric.eq('opening_day_net') & bo.territory.eq('india')]
    grid = grid.merge(india_net[['day_number', 'value_crore']].assign(day_number=lambda d: d.day_number.astype(int))
                      .rename(columns={'value_crore': 'india_net_crore'}), on='day_number', how='left')
    grid['is_missing'] = grid.india_net_crore.isna()
    log('box_office', f'Missing day-wise collections (only day(s) {days_have} reported, of {last_day} days up to access date)', len(missing),
        'Kept as NaN with is_missing=True; NOT interpolated', 'Bollywood day-wise curves drop sharply after weekends; any fill would fabricate the decay we want to study')
    log('box_office', 'Raw rows in', n0, 'Kept all rows (flags added, none dropped)', 'Every row is a distinct cited figure')
    bo['day_number'] = pd.to_numeric(bo.day_number, errors='coerce').astype('Int64')
    return bo, grid


# ---------------------------------------------------------------- 4. critics + IMDb text file
def clean_critics():
    rv = pd.read_csv(RAW / 'ramayana_reviews_normalized.csv')
    rv['film_key'] = FILM_KEY
    for c in ['original_rating', 'rating_scale', 'normalized_rating']:
        rv[c] = pd.to_numeric(rv[c], errors='coerce')
    odd = ~rv.rating_scale.isin([5, 10])
    log('critics', 'Ratings on mixed scales (out of 4, 5 and 10)', rv.rating_scale.ne(5).sum(),
        'Converted every rating to stars out of 5 = original/scale*5', 'Critic scores must share one scale before averaging')
    if odd.any():
        log('critics', 'Uncommon rating scale (e.g. out of 4)', odd.sum(), 'Rescaled linearly like the others', 'Linear rescaling is the standard convention for star ratings')
    rv['stars_5'] = rv.original_rating / rv.rating_scale * 5
    chk = (rv.stars_5 * 2 - rv.normalized_rating).abs().max()
    log('critics', f'Pre-computed normalized_rating (/10) checked against our rescaling (max diff {chk:.2f})', len(rv),
        'Kept ours; dropped nothing', 'Verifies the team\'s earlier normalisation')
    nos = rv.original_rating.isna()
    log('critics', 'Critic reviews with no score', nos.sum(), 'Kept text for sentiment, score left NaN (not imputed)',
        'Missing critic score is not zero; imputing a mean would shrink disagreement')
    d = rv.duplicated('review_text')
    log('critics', 'Duplicate critic blurbs', d.sum(), 'Dropped', 'Same review counted twice biases the mean')
    rv = rv[~d]

    raw = (BASE / 'imdb.txt').read_text(encoding='utf-8').strip()
    first, _, rest = raw.partition('\n')
    dist = pd.DataFrame(re.findall(r'(\d+)\s*:\s*(\d+)', first), columns=['stars', 'votes']).astype(int)
    dist['film_key'] = FILM_KEY
    log('imdb', 'IMDb vote distribution stored as text ("1: 46, 2: 9, 3:10")', len(dist),
        'Parsed to (stars int, votes int) with regex tolerant of missing spaces', 'Counts stored as text must be integers')
    blocks = re.findall(r'(?ms)^([^\n:"“]{2,60}):\s*["“](.*?)(?=^[^\n:"“]{2,60}:\s*["“]|\Z)', rest)
    full = pd.DataFrame(blocks, columns=['publication', 'text'])
    full['text'] = full.text.str.strip().str.rstrip('"”').str.replace(r'^Also Read:.*$', '', regex=True, flags=re.M).str.strip()
    full['film_key'] = FILM_KEY   # every block is a review OF Part 1; mentions of Adipurush etc. are comparisons
    full['n_words'] = full.text.str.split().str.len()
    log('imdb', 'Full-text critic reviews in imdb.txt with no numeric score', len(full),
        'Kept for sentiment only; flagged has_score=False', 'Film-level critic score comes from the rated reviews; text still carries tone')
    log('imdb', 'Full reviews mention other works (Adipurush, 1987 serial) in passing', full.text.str.contains('adipurush|ramanand', case=False).sum(),
        'Kept film_key = ramayana_part1_2026 for whole reviews', 'A review of Part 1 that compares it to Adipurush is still a review of Part 1')
    log('imdb', '"Also Read:" cross-promo lines inside review texts', len(re.findall(r'(?m)^Also Read:', rest)),
        'Removed from text', 'Unrelated headlines would distort sentiment')
    return rv, dist, full


# ---------------------------------------------------------------- 5. Reddit
BOTS = {'AutoModerator', 'RemindMeBot', 'sneakpeekbot', 'SaveVideo', 'savevideobot', 'B0tRank', 'WikiSummarizerBot'}
def is_bot(author, body):
    a = str(author)
    return a in BOTS or a.lower().endswith('bot') or bool(re.search(r'i am a bot|^\*?beep boop', str(body), re.I))

def clean_text(s):
    s = html.unescape(str(s))
    s = re.sub(r'!\[[^\]]*\]\([^)]*\)', ' ', s)
    s = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', s)
    s = re.sub(r'https?://\S+|www\.\S+', ' ', s)
    s = re.sub(r'(?<![\w/])/?u/[A-Za-z0-9_-]{2,}', '[user]', s)    # usernames mentioned in text
    s = re.sub(r'^\s*&gt;.*$|^\s*>.*$', ' ', s, flags=re.M)         # quoted text from another comment
    return re.sub(r'\s+', ' ', s).strip()

def load_reddit():
    if not REDDIT_DIR.exists() or not any(REDDIT_DIR.glob('*.jsonl')):
        import py7zr
        REDDIT_DIR.mkdir(parents=True, exist_ok=True)
        with py7zr.SevenZipFile(RAW / 'data.7z') as z:
            z.extractall(REDDIT_DIR)
    rows, bad = [], 0
    for f in sorted(REDDIT_DIR.glob('r_*_comments.jsonl')):
        sub = f.name[2:-len('_comments.jsonl')]
        for line in f.open(encoding='utf-8'):
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            rows.append(dict(id=o.get('id'), subreddit=sub, author=o.get('author'), body=o.get('body'),
                             created_utc=o.get('created_utc'), score=o.get('score'), link_title=o.get('link_title'),
                             link_id=o.get('link_id')))
    c = pd.DataFrame(rows)
    log('reddit', 'Lines that are not valid JSON', bad, 'Skipped', 'Truncated last line of a dump still being written')
    log('reddit', 'Raw comments loaded (all subreddits)', len(c), 'Loaded', 'Starting count')

    # film matching: dedicated sub -> Part 1 unless the comment/thread names another work; general subs -> must name it
    c['film_key'] = c.body.map(film_key)
    thread_key = c.link_title.map(film_key)
    c['film_key'] = c.film_key.fillna(thread_key)
    ded = c.subreddit.eq('RamayanaTheFilm')
    c.loc[ded & c.film_key.isna(), 'film_key'] = FILM_KEY
    no_film = c.film_key.isna()
    log('reddit', 'General-sub comments (r/bollywood, r/IndianCinema) not about any Ramayana work', no_film.sum(),
        'Dropped', 'They discuss other films; keeping them would mix sentiment across films')
    c = c[~no_film].copy()
    other = c.film_key.ne(FILM_KEY)
    log('reddit', 'Comments about other Ramayana works (Adipurush, 1987 TV serial, anime, Part 2)', other.sum(),
        'Kept with their own film_key', 'Same name family, different film; needed for the cross-film comparison')

    deleted = c.body.isin(['[deleted]', '[removed]', '']) | c.body.isna()
    log('reddit', 'Deleted / removed comments ("[deleted]", "[removed]")', deleted.sum(),
        'Dropped from text analysis', 'No text to score; replies to them are kept because they stand on their own')
    c = c[~deleted].copy()
    bots = c.apply(lambda r: is_bot(r.author, r.body), axis=1)
    log('reddit', 'Bot comments (AutoModerator, *bot accounts, "I am a bot" signatures)', bots.sum(),
        'Dropped', 'Automated text is not audience opinion')
    c = c[~bots].copy()
    d_id = c.duplicated('id')
    log('reddit', 'Same comment id appearing twice (re-crawled)', d_id.sum(), 'Dropped exact id duplicates (kept first)',
        'Re-crawl artefact, not a second opinion')
    c = c[~d_id].copy()

    c['body_clean'] = c.body.map(clean_text)
    c['norm'] = c.body_clean.str.lower().str.replace(r'[^a-z0-9 ]', '', regex=True).str.strip()
    dup_txt = c.duplicated(['author', 'norm']) & c.norm.str.len().ge(20)
    log('reddit', 'Duplicate comments (same account, same text)', dup_txt.sum(), 'Dropped, kept first',
        'Double-posts would count one opinion twice')
    c = c[~dup_txt].copy()
    empty = c.body_clean.str.len().eq(0)
    log('reddit', 'Comments empty after removing links/quotes/images', empty.sum(), 'Dropped', 'Nothing left to score')
    c = c[~empty].copy()

    # privacy: no usernames in any output
    log('reddit', 'Usernames (author field + u/ mentions in text)', len(c),
        'Author column dropped; u/name in text replaced by [user]', 'Personal data is not needed for film-level analysis')
    c = c.drop(columns=['author', 'norm', 'body'])

    c['created_ist'] = pd.to_datetime(pd.to_numeric(c.created_utc), unit='s', utc=True).dt.tz_convert(TZ).dt.tz_localize(None)
    c['day_since_release'] = (c.created_ist.dt.normalize() - RELEASE).dt.days
    c['score'] = pd.to_numeric(c.score, errors='coerce').astype('Int64')
    log('reddit', 'Epoch timestamps and text scores', len(c), 'created_utc -> IST datetime; score -> Int64',
        'Daily analysis is in Indian time; score must be numeric')
    return c.drop(columns=['created_utc'])


# ---------------------------------------------------------------- 6. spreadsheet vs pandas
def spreadsheet_exercise(bo):
    money = bo.metric.str.contains('gross|net|budget|box_office')
    ex = pd.concat([bo.loc[money, ['value_raw']].rename(columns={'value_raw': 'raw'}),
                    pd.DataFrame({'raw': ['₹12.5 Cr', '1,25,00,000', '50 lakh', '57.30cr']})], ignore_index=True)
    ex['pandas_crore'] = ex.raw.map(to_crore)
    out = CLEAN / 'spreadsheet_vs_pandas.xlsx'
    from openpyxl import Workbook
    wb = Workbook(); ws = wb.active; ws.title = 'clean_step'
    # Spreadsheet version of to_crore with helper columns (LibreOffice/Excel, no regex functions needed)
    ws.append(['raw', 'B_stripped', 'C_first_digit_pos', 'D_number', 'spreadsheet_crore', 'pandas_crore', 'match'])
    for i, r in ex.iterrows():
        n = i + 2
        ws.append([r.raw,
                   f'=SUBSTITUTE(SUBSTITUTE(SUBSTITUTE(LOWER(A{n}),"₹"," "),",",""),"cr"," cr")',
                   f'=MIN(FIND({{0,1,2,3,4,5,6,7,8,9}},B{n}&"0123456789"))',
                   f'=VALUE(LEFT(MID(B{n},C{n},50),FIND(" ",MID(B{n},C{n},50)&" ")-1))',
                   f'=IF(AND(ISNUMBER(SEARCH("$",A{n})),ISERROR(SEARCH("cr",A{n}))),"",IF(ISNUMBER(SEARCH("lakh",A{n})),D{n}/100,'
                   f'IF(AND(D{n}>=100000,ISERROR(SEARCH("cr",A{n}))),D{n}/10000000,D{n})))',
                   None if pd.isna(r.pandas_crore) else r.pandas_crore,
                   f'=IF(AND(E{n}="",F{n}=""),TRUE,IFERROR(ABS(E{n}-F{n})<0.0001,FALSE))'])
    wb.save(out)
    note = 'formulas written; LibreOffice not found, so spreadsheet values were not recomputed'
    if shutil.which('soffice'):
        tmp = CLEAN / '_recalc'
        subprocess.run(['soffice', '--headless', '--convert-to', 'csv:Text - txt - csv (StarCalc):44,34,76', '--outdir', str(tmp), str(out)],
                       capture_output=True, timeout=120)
        csvs = list(tmp.glob('*.csv'))
        if csvs:
            ss = pd.read_csv(csvs[0])
            ex['spreadsheet_crore'] = pd.to_numeric(ss.spreadsheet_crore, errors='coerce')
            ex['match'] = (ex.spreadsheet_crore.fillna(-1) - ex.pandas_crore.fillna(-1)).abs() < 1e-4
            note = f'{int(ex.match.sum())}/{len(ex)} rows agree between LibreOffice formulas and pandas'
        shutil.rmtree(tmp, ignore_errors=True)
    ex.to_csv(CLEAN / 'spreadsheet_vs_pandas.csv', index=False, encoding='utf-8-sig')
    log('box_office', 'Spreadsheet vs pandas: same ₹-text -> crore step done both ways', len(ex), note,
        'Module 1 comparison: a spreadsheet formula is visible per cell but brittle; pandas is one tested function reused everywhere')
    return ex


# ---------------------------------------------------------------- 7. sentiment
def vader_on_imdb50k(sia, path):
    df = pd.read_csv(path)
    df['text'] = df.review.str.replace(r'<br\s*/?>', ' ', regex=True)
    df['compound'] = df.text.map(lambda s: sia.polarity_scores(s)['compound'])
    df['pred'] = np.where(df.compound >= 0, 'positive', 'negative')
    acc = (df.pred == df.sentiment).mean()
    cm = pd.crosstab(df.sentiment, df.pred)
    tp, fp, fn = cm.loc['positive', 'positive'], cm.loc['negative', 'positive'], cm.loc['positive', 'negative']
    tn = cm.loc['negative', 'negative']
    res = dict(n=len(df), accuracy=acc, precision_pos=tp / (tp + fp), recall_pos=tp / (tp + fn), recall_neg=tn / (tn + fp),
               threshold='compound >= 0 -> positive')
    pd.DataFrame([res]).to_csv(CLEAN / 'vader_imdb50k_validation.csv', index=False)
    cm.to_csv(CLEAN / 'vader_imdb50k_confusion.csv')
    print(f'VADER on IMDb 50K: accuracy {acc:.3f}\n{cm}')
    return res

def sentence_vader(sia, text):
    """Long reviews: average sentence compounds (document compound saturates at ±1 on long text)."""
    sents = [s for s in re.split(r'(?<=[.!?])\s+', str(text)) if len(s.split()) >= 3]
    return float(np.mean([sia.polarity_scores(s)['compound'] for s in sents])) if sents else np.nan


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--imdb50k', default=str(BASE / 'data' / 'IMDB Dataset.csv'))
    args = ap.parse_args()
    CLEAN.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)

    bo, daily = clean_box_office()
    critics, imdb_dist, imdb_texts = clean_critics()
    comments = load_reddit()
    spreadsheet_exercise(bo)

    sia = SentimentIntensityAnalyzer()
    val = vader_on_imdb50k(sia, args.imdb50k) if Path(args.imdb50k).exists() else None
    if val is None:
        print('IMDb 50K corpus not found; pass --imdb50k. Skipping validation.')

    comments['compound'] = comments.body_clean.map(lambda s: sia.polarity_scores(s)['compound'])
    critics['compound'] = critics.review_text.map(lambda s: sia.polarity_scores(s)['compound'])
    imdb_texts['compound'] = imdb_texts.text.map(lambda s: sentence_vader(sia, s))

    sent = pd.concat([
        comments.groupby(['film_key', 'subreddit']).compound.agg(['mean', 'size']).reset_index()
                .assign(source=lambda d: 'reddit_r/' + d.subreddit).drop(columns='subreddit'),
        comments.groupby('film_key').compound.agg(['mean', 'size']).reset_index().assign(source='reddit_all'),
        critics.groupby('film_key').compound.agg(['mean', 'size']).reset_index().assign(source='critic_blurbs'),
        imdb_texts.groupby('film_key').compound.agg(['mean', 'size']).reset_index().assign(source='imdb_txt_review_texts'),
    ]).rename(columns={'mean': 'mean_compound', 'size': 'n'})[['film_key', 'source', 'n', 'mean_compound']]
    sent.to_csv(CLEAN / 'sentiment_by_film_source.csv', index=False)
    print(sent.to_string(index=False))

    # ------------- normalisation to 0-1 (min-max over each scale's theoretical range) + z-score across voices
    imdb_mean = (imdb_dist.stars * imdb_dist.votes).sum() / imdb_dist.votes.sum()
    p1 = sent[sent.film_key.eq(FILM_KEY)].set_index('source').mean_compound
    voices = pd.DataFrame([
        ('Critics: star rating', critics.stars_5.mean(), 0, 5, 'stars out of 5', len(critics)),
        ('IMDb users: mean rating', imdb_mean, 1, 10, 'IMDb out of 10', int(imdb_dist.votes.sum())),
        ('Critics: blurb sentiment', p1['critic_blurbs'], -1, 1, 'VADER −1..+1', len(critics)),
        ('Critics: full-text sentiment', p1['imdb_txt_review_texts'], -1, 1, 'VADER −1..+1', len(imdb_texts)),
        ('Reddit: comment sentiment', p1['reddit_all'], -1, 1, 'VADER −1..+1', int(sent.query('film_key==@FILM_KEY and source=="reddit_all"').n.iloc[0])),
    ], columns=['voice', 'raw_value', 'scale_min', 'scale_max', 'scale', 'n'])
    voices['minmax_0_1'] = (voices.raw_value - voices.scale_min) / (voices.scale_max - voices.scale_min)
    voices['z_score'] = (voices.minmax_0_1 - voices.minmax_0_1.mean()) / voices.minmax_0_1.std(ddof=0)
    voices['film_key'] = FILM_KEY
    voices.to_csv(CLEAN / 'normalised_voices.csv', index=False)
    log('all', 'Voices on different scales (stars /5, IMDb /10, sentiment −1..+1)', len(voices),
        'Min-max to 0–1 using each scale\'s full range; z-score across voices', 'Puts every voice on one footing for comparison')
    print(voices.round(3).to_string(index=False))

    # ------------- chart 1: normalised voices
    v = voices.sort_values('minmax_0_1')
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(v.voice, v.minmax_0_1, color='#3b6ea5')
    for y, (x, raw, sc) in enumerate(zip(v.minmax_0_1, v.raw_value, v.scale)):
        ax.text(x + 0.01, y, f'{x:.2f}  (raw {raw:.2f}, {sc})', va='center', fontsize=8)
    ax.set_xlim(0, 1.15); ax.set_xlabel('Normalised score (0 = worst possible, 1 = best possible)')
    gap = v.set_index('voice').minmax_0_1
    ax.set_title(f'IMDb users rate Ramayana: Part 1 higher than critics do\n'
                 f'({gap["IMDb users: mean rating"]:.2f} vs {gap["Critics: star rating"]:.2f} on a common 0–1 scale)', fontsize=10, loc='left')
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / 'normalised_voices.png', dpi=150); plt.close(fig)

    # ------------- chart 2: Reddit sentiment by day (z-scored volume vs mood)
    p = comments[comments.film_key.eq(FILM_KEY) & comments.day_since_release.between(0, 14)]
    d = p.groupby('day_since_release').agg(comments=('id', 'size'), mood=('compound', 'mean'))
    d['z_volume'] = (d.comments - d.comments.mean()) / d.comments.std()
    d['z_mood'] = (d.mood - d.mood.mean()) / d.mood.std()
    d.to_csv(CLEAN / 'reddit_daily_z.csv')
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(d.index, d.z_volume, marker='o', label='Comment volume (z)', color='#3b6ea5')
    ax.plot(d.index, d.z_mood, marker='o', label='Mean VADER sentiment (z)', color='#d9822b')
    ax.axhline(0, color='grey', lw=0.8); ax.set_xlabel('Days since release (5 Nov 2026 = 0)'); ax.legend(frameon=False)
    peak = int(d.comments.idxmax())
    ax.set_title(f'Reddit buzz peaks on day {peak}; mood moves independently of volume', fontsize=10, loc='left')
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / 'reddit_volume_vs_mood_z.png', dpi=150); plt.close(fig)

    # ------------- save
    bo.to_csv(CLEAN / 'box_office_clean.csv', index=False)
    daily.to_csv(CLEAN / 'daily_collections_with_gaps.csv', index=False)
    critics.to_csv(CLEAN / 'critic_reviews_clean.csv', index=False)
    imdb_dist.to_csv(CLEAN / 'imdb_rating_distribution.csv', index=False)
    imdb_texts.to_csv(CLEAN / 'critic_full_texts_clean.csv', index=False)
    comments.to_csv(CLEAN / 'reddit_comments_clean.csv', index=False)
    if val:
        log('sentiment', f'VADER validated on IMDb 50K labelled corpus: accuracy {val["accuracy"]:.1%}', val['n'],
            'Threshold compound >= 0 = positive; method accepted with known bias', 'Check the method on labelled data before trusting it on unlabelled Ramayana text')
    pd.DataFrame(CLEANING_LOG).to_csv(CLEAN / 'cleaning_log.csv', index=False, encoding='utf-8-sig')
    print(f'\ncleaning log: {len(CLEANING_LOG)} decisions -> clean/cleaning_log.csv')


if __name__ == '__main__':
    main()
