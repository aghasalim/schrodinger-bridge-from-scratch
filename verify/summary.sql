-- Recompute the two published tables in README.md straight from
-- results/transport.csv, in SQL, with no Python in the path.
--
-- Every cell of the results table is a median over three seeds and every cell
-- of the wall clock table is a median over nine runs, and all of them were
-- produced by one pandas free but equally single implementation in
-- scripts/check_numbers.py. A group by written from scratch is the cheapest
-- independent opinion available on whether those medians are the medians.
--
-- Output is label,value. verify/verify.sh requires every value to appear in
-- README.md. Values are rounded here to the number of digits the README quotes.
.mode csv
.headers off
-- The header row names the columns, so every reference below is by name.
.import --csv results/transport.csv t

-- The two non finite DSB runs are stored as the text nan, so a numeric filter
-- has to reject them explicitly rather than trust the cast.
CREATE VIEW ok AS
  SELECT pair, CAST(seed AS INTEGER) AS seed, method,
         CAST(stage AS INTEGER) AS stage, CAST(w2 AS REAL) AS w2,
         CAST(wall_s AS REAL) AS wall_s
  FROM t WHERE lower(w2) NOT LIKE '%nan%' AND lower(w2) NOT LIKE '%inf%';

-- Median of an odd or even count: average the middle one or two rows.
CREATE VIEW ranked AS
  SELECT pair, method, stage, w2, wall_s,
         ROW_NUMBER() OVER (PARTITION BY pair, method, stage ORDER BY w2) AS rw,
         ROW_NUMBER() OVER (PARTITION BY pair, method, stage ORDER BY wall_s) AS rt,
         COUNT(*) OVER (PARTITION BY pair, method, stage) AS n
  FROM ok;

-- results table: baseline, Sinkhorn, DSB best single run, bridge matching
SELECT pair || ' baseline', printf('%.3f', AVG(w2)) FROM ranked
 WHERE method = 'untransported' AND rw IN ((n+1)/2, (n+2)/2) GROUP BY pair;
SELECT pair || ' sinkhorn', printf('%.3f', AVG(w2)) FROM ranked
 WHERE method = 'sinkhorn-eps0.03' AND rw IN ((n+1)/2, (n+2)/2) GROUP BY pair;
SELECT pair || ' bm-ode', printf('%.3f', AVG(w2)) FROM ranked
 WHERE method = 'bridge-matching-ode' AND stage = 100 AND rw IN ((n+1)/2, (n+2)/2) GROUP BY pair;
SELECT pair || ' bm-sde', printf('%.3f', AVG(w2)) FROM ranked
 WHERE method = 'bridge-matching-sde' AND stage = 100 AND rw IN ((n+1)/2, (n+2)/2) GROUP BY pair;
-- The README quotes DSB's best of its thirty runs, not a median.
SELECT pair || ' dsb-best', printf('%.3f', MIN(w2)) FROM ok
 WHERE method = 'dsb' GROUP BY pair;

-- per IPF iteration medians, the table in notes/METHODS.md
SELECT pair || ' dsb-ipf' || stage, printf('%.3f', AVG(w2)) FROM ranked
 WHERE method = 'dsb' AND rw IN ((n+1)/2, (n+2)/2) GROUP BY pair, stage;

-- wall clock table. Sinkhorn is one solve at the tightest eps, bridge matching
-- is the single training run, DSB is the elapsed time at its last IPF iteration.
SELECT 'wall sinkhorn', printf('%.1f', AVG(wall_s)) FROM ranked
 WHERE method = 'sinkhorn-eps0.03' AND rt IN ((n+1)/2, (n+2)/2);
SELECT 'wall bridge-matching', printf('%.1f', AVG(wall_s)) FROM ranked
 WHERE method = 'bridge-matching-ode' AND stage = 100 AND rt IN ((n+1)/2, (n+2)/2);
SELECT 'wall dsb', printf('%.0f', AVG(wall_s)) FROM (
  SELECT wall_s, ROW_NUMBER() OVER (ORDER BY wall_s) AS r, COUNT(*) OVER () AS n
  FROM (SELECT pair, seed, MAX(wall_s) AS wall_s FROM ok WHERE method = 'dsb'
        GROUP BY pair, seed))
 WHERE r IN ((n+1)/2, (n+2)/2);
