-- q1_avg_score_by_source_and_language
-- JOIN + GROUP BY: average sentiment (0-1) by source and by film language
SELECT s.voice, s.source_id, f.language, COUNT(*) AS n, ROUND(AVG((c.compound + 1) / 2.0), 3) AS avg_sentiment_0_1
FROM reddit_comments c JOIN films f ON f.film_key = c.film_key JOIN sources s ON s.source_id = c.source_id
GROUP BY s.voice, s.source_id, f.language
UNION ALL
SELECT s.voice, s.source_id, f.language, COUNT(*), ROUND(AVG((r.compound + 1) / 2.0), 3)
FROM critic_reviews r JOIN films f ON f.film_key = r.film_key JOIN sources s ON s.source_id = r.source_id
GROUP BY s.voice, s.source_id, f.language
ORDER BY avg_sentiment_0_1 DESC;

-- q2_agreement_gap_per_film
-- JOIN of two aggregates: critic stars (0-1) minus Reddit sentiment (0-1) per film (LEFT JOIN keeps films without critics)
SELECT f.film_key, f.title, rd.n_comments, ROUND(rd.reddit_0_1, 3) AS reddit_0_1, ROUND(cr.critic_0_1, 3) AS critic_0_1,
       ROUND(cr.critic_0_1 - rd.reddit_0_1, 3) AS gap_critic_minus_reddit
FROM films f
LEFT JOIN (SELECT film_key, COUNT(*) AS n_comments, AVG((compound + 1) / 2.0) AS reddit_0_1 FROM reddit_comments GROUP BY film_key) rd ON rd.film_key = f.film_key
LEFT JOIN (SELECT film_key, AVG(stars_5) / 5.0 AS critic_0_1 FROM critic_reviews GROUP BY film_key) cr ON cr.film_key = f.film_key
ORDER BY rd.n_comments DESC;

-- q3_reddit_daily_volume_and_mood
-- GROUP BY day: buzz velocity and mood for Part 1
SELECT day_since_release, COUNT(*) AS comments, ROUND(AVG(compound), 3) AS mean_compound,
       ROUND(100.0 * SUM(CASE WHEN compound >= 0.05 THEN 1 ELSE 0 END) / COUNT(*), 1) AS pct_positive
FROM reddit_comments WHERE film_key = 'ramayana_part1_2026' AND day_since_release BETWEEN 0 AND 14
GROUP BY day_since_release ORDER BY day_since_release;

-- q4_week1_vs_week2
-- CASE + GROUP BY: does Reddit mood change from week 1 to week 2?
SELECT CASE WHEN day_since_release BETWEEN 0 AND 6 THEN 'week1' ELSE 'week2' END AS week,
       COUNT(*) AS comments, ROUND(AVG(compound), 3) AS mean_compound
FROM reddit_comments WHERE film_key = 'ramayana_part1_2026' AND day_since_release BETWEEN 0 AND 13
GROUP BY week;

-- q5_aspect_gap_critics_vs_reddit
-- JOIN + GROUP BY + HAVING: aspect sentiment by voice, only aspects both voices discuss
SELECT a.aspect,
       ROUND(AVG(CASE WHEN s.voice = 'Critics' THEN (a.compound + 1) / 2.0 END), 3) AS critics_0_1,
       ROUND(AVG(CASE WHEN s.voice = 'Reddit'  THEN (a.compound + 1) / 2.0 END), 3) AS reddit_0_1,
       SUM(s.voice = 'Critics') AS n_critic, SUM(s.voice = 'Reddit') AS n_reddit
FROM aspect_mentions a JOIN sources s ON s.source_id = a.source_id
GROUP BY a.aspect HAVING n_critic >= 3 AND n_reddit >= 3
ORDER BY ABS(critics_0_1 - reddit_0_1) DESC;

-- q7_top_critic_and_bottom_critic
-- subquery: reviews above / below the critic average
SELECT review_id, stars_5, ROUND(compound, 2) AS compound, SUBSTR(review_text, 1, 90) AS excerpt,
       CASE WHEN stars_5 >= (SELECT AVG(stars_5) FROM critic_reviews) THEN 'above_avg' ELSE 'below_avg' END AS vs_avg
FROM critic_reviews ORDER BY stars_5 DESC;

-- q6_box_office_per_screen
-- self-JOIN: opening-day gross per screen (₹ lakh per screen) by territory
SELECT g.territory, g.metric, g.value_crore AS opening_day_cr, CAST(s.value_crore AS INT) AS screens,
       ROUND(g.value_crore * 100.0 / s.value_crore, 2) AS lakh_per_screen
FROM box_office g JOIN box_office s ON s.film_key = g.film_key AND s.territory = g.territory AND s.metric = 'screens'
WHERE g.metric LIKE 'opening_day%';

