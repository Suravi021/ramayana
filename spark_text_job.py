"""PySpark text job: clean -> measure length -> count terms -> aggregate by film and source.

Needs Java 8/11/17 + pyspark 3.5 (pyspark 4.x requires Java 17+).
    python spark_text_job.py --imdb50k "path/to/IMDB Dataset.csv"
Run phase3_clean_analyse.py first (it writes the clean/*.csv inputs).

Row count: ~32.8k Reddit comments + 17 critic blurbs + 3 full critic reviews < 50,000, so the
IMDb 50K labelled corpus is added as its own "film" (imdb50k_corpus) -> ~82.8k text rows.
"""
import argparse, os, sys
from pathlib import Path

os.environ.setdefault('PYSPARK_PYTHON', sys.executable)
from pyspark.sql import SparkSession, functions as F, Window
from pyspark.ml.feature import StopWordsRemover

BASE = Path(__file__).resolve().parent
CLEAN = BASE / 'clean'
OUT = CLEAN / 'spark'

HINGLISH = ['hai', 'hain', 'ka', 'ki', 'ke', 'ko', 'se', 'ne', 'bhi', 'toh', 'to', 'na', 'nahi', 'ye', 'yeh', 'woh', 'wo',
            'aur', 'kya', 'tha', 'thi', 'the', 'ho', 'hi', 'mein', 'main', 'par', 'koi', 'ek', 'kar', 'raha', 'rahi']
EXTRA = ['film', 'movie', 'movies', 'films', 's', 't', 'one', 'like', 'just', 'also', 'even', 'would', 'get', 'really',
         'see', 'much', 'people', 'dont', 'im', 'think', 'know', 'gt', 'amp', 'will', 'user', 'br', 'u', 've', 'll',
         'got', 'go', 'going', 'make', 'made', 'well', 'still', 'thing', 'things', 'way', 'say', 'said', 'didnt', 'doesnt']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--imdb50k', default=str(BASE / 'data' / 'IMDB Dataset.csv'))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    spark = (SparkSession.builder.master('local[*]').appName('RamayanaTextJob')
             .config('spark.driver.memory', '3g').config('spark.sql.shuffle.partitions', '8')
             .config('spark.ui.showConsoleProgress', 'false').getOrCreate())
    spark.sparkContext.setLogLevel('ERROR')
    rd = dict(header=True, multiLine=True, escape='"')

    reddit = (spark.read.csv(str(CLEAN / 'reddit_comments_clean.csv'), **rd)
              .select('film_key', F.concat(F.lit('reddit_r/'), 'subreddit').alias('source'), F.col('body_clean').alias('text')))
    blurbs = (spark.read.csv(str(CLEAN / 'critic_reviews_clean.csv'), **rd)
              .select('film_key', F.lit('critic_blurbs').alias('source'), F.col('review_text').alias('text')))
    fulltx = (spark.read.csv(str(CLEAN / 'critic_full_texts_clean.csv'), **rd)
              .where(F.col('is_review') == 'True')           # ThePrint press round-up excluded
              .select('film_key', F.lit('critic_full_texts').alias('source'), 'text'))
    imdb = (spark.read.csv(args.imdb50k, **rd)
            .select(F.lit('imdb50k_corpus').alias('film_key'), F.concat(F.lit('imdb50k_'), 'sentiment').alias('source'),
                    F.col('review').alias('text')))
    docs = reddit.unionByName(blurbs).unionByName(fulltx).unionByName(imdb).where(F.col('text').isNotNull())

    # ---- clean
    docs = (docs.withColumn('text', F.regexp_replace('text', r'<br\s*/?>', ' '))
                .withColumn('clean', F.lower('text'))
                .withColumn('clean', F.regexp_replace('clean', r'https?://\S+', ' '))
                .withColumn('clean', F.regexp_replace('clean', r"[^a-z\s']", ' '))
                .withColumn('clean', F.regexp_replace('clean', r"'", ''))
                .withColumn('clean', F.trim(F.regexp_replace('clean', r'\s+', ' ')))
                .where(F.length('clean') > 0))
    # ---- measure length
    docs = (docs.withColumn('tokens', F.split('clean', ' '))
                .withColumn('n_chars', F.length('text'))
                .withColumn('n_words', F.size('tokens')))
    docs.cache()
    n_rows = docs.count()
    print(f'text rows processed by Spark: {n_rows:,}')

    remover = StopWordsRemover(inputCol='tokens', outputCol='terms',
                               stopWords=StopWordsRemover.loadDefaultStopWords('english') + HINGLISH + EXTRA)
    docs = remover.transform(docs)

    # ---- aggregate by film and source
    agg = (docs.groupBy('film_key', 'source')
               .agg(F.count('*').alias('n_docs'), F.round(F.avg('n_words'), 1).alias('avg_words'),
                    F.expr('percentile_approx(n_words, 0.5)').alias('median_words'),
                    F.round(F.avg('n_chars'), 1).alias('avg_chars'), F.sum('n_words').alias('total_words'))
               .orderBy('film_key', 'source'))
    agg.toPandas().to_csv(OUT / 'length_by_film_source.csv', index=False)
    agg.show(30, truncate=False)

    # ---- count terms (per film x source), top 25 each
    terms = (docs.select('film_key', 'source', F.explode('terms').alias('term'))
                 .where(F.length('term') >= 3)
                 .groupBy('film_key', 'source', 'term').count())
    w = Window.partitionBy('film_key', 'source').orderBy(F.desc('count'), 'term')
    top = terms.withColumn('rank', F.row_number().over(w)).where('rank <= 25')
    top.toPandas().to_csv(OUT / 'top_terms_by_film_source.csv', index=False)
    # full counts for Part 1 (used by the split analysis)
    (terms.where(F.col('film_key') == 'ramayana_part1_2026').toPandas()
          .to_csv(OUT / 'term_counts_part1.csv', index=False))
    (top.groupBy('film_key', 'source').agg(F.concat_ws(', ', F.collect_list('term')).alias('top_terms'))
        .show(30, truncate=110))

    with open(OUT / 'run_summary.txt', 'w') as f:
        f.write(f'Spark {spark.version}; text rows processed: {n_rows}\n')
    spark.stop()


if __name__ == '__main__':
    main()
