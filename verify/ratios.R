# The claims the README makes in words, and one test it does not make at all.
#
# scripts/check_numbers.py says outright that it checks quoted figures and not
# claims written in words. The multiples in the results section are written in
# words: "1.84x, 1.29x and 2.71x", "four times cheaper". Those are exactly the
# sentences that survive a change to results/transport.csv, because nothing was
# looking at them.
#
# The second half of this file is inference rather than arithmetic. "Closer to
# the target on every pair" is a claim about a comparison, and with three pairs
# and three seeds there is a nine run paired comparison behind it that nobody
# had run. The exact one sided sign test is the right size of tool for nine
# paired observations and needs no distributional assumption.

args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args) > 0) args[1] else "."

d <- read.csv(file.path(root, "results", "transport.csv"),
              stringsAsFactors = FALSE)
d$w2 <- suppressWarnings(as.numeric(d$w2))
readme <- paste(readLines(file.path(root, "README.md"), warn = FALSE),
                collapse = "\n")
# Bolding a number must not read as the number being gone.
readme <- gsub("[*`]", "", readme)

fin <- d[is.finite(d$w2), ]
pairs_in_row_order <- c("circle->moons", "gaussian->8gaussians", "moons->spiral")
bad <- 0

med <- function(pair, method, stage) {
  v <- fin$w2[fin$pair == pair & fin$method == method & fin$stage == stage]
  if (length(v) == 0) stop(sprintf("no rows for %s %s stage %s", pair, method, stage))
  median(v)
}
dsb_best <- function(pair) min(fin$w2[fin$pair == pair & fin$method == "dsb"])

check_string <- function(what, s) {
  if (grepl(s, readme, fixed = TRUE)) {
    cat(sprintf("  %-34s %-9s in README\n", what, s))
  } else {
    cat(sprintf("  %-34s %-9s NOT IN README\n", what, s))
    bad <<- bad + 1
  }
}

for (method in c("bridge-matching-ode", "bridge-matching-sde")) {
  for (p in pairs_in_row_order) {
    r <- dsb_best(p) / med(p, method, 100)
    check_string(paste(p, sub("bridge-matching-", "", method)),
                 sprintf("%.2fx", r))
  }
}

# "four times cheaper": DSB elapsed at its last IPF iteration against the single
# bridge matching training run, medians over all nine runs.
dsb_wall <- tapply(d$wall_s[d$method == "dsb"],
                   paste(d$pair, d$seed)[d$method == "dsb"], max)
bm_wall <- d$wall_s[d$method == "bridge-matching-ode" & d$stage == 100]
cheaper <- median(dsb_wall) / median(bm_wall)
cat(sprintf("  %-34s %.2f\n", "dsb wall over bridge matching wall", cheaper))
if (round(cheaper) != 4) {
  cat("  the README says four times cheaper and the ratio no longer rounds to 4\n")
  bad <- bad + 1
}

# Paired comparison, one run at a time: the bridge matching SDE sample against
# the best DSB iteration of the same pair and seed.
wins <- 0
n <- 0
for (p in pairs_in_row_order) {
  for (s in sort(unique(d$seed))) {
    sde <- fin$w2[fin$pair == p & fin$seed == s &
                  fin$method == "bridge-matching-sde" & fin$stage == 100]
    dsb <- fin$w2[fin$pair == p & fin$seed == s & fin$method == "dsb"]
    if (length(sde) != 1 || length(dsb) == 0) {
      cat(sprintf("  missing rows for %s seed %d\n", p, s))
      bad <- bad + 1
      next
    }
    n <- n + 1
    if (sde < min(dsb)) wins <- wins + 1
  }
}
pval <- binom.test(wins, n, p = 0.5, alternative = "greater")$p.value
cat(sprintf("  bridge matching SDE beats the best DSB run in %d of %d paired runs, sign test p = %.4f\n",
            wins, n, pval))
if (wins != n) {
  cat("  the README claims bridge matching is closer on every pair\n")
  bad <- bad + 1
}
if (pval >= 0.01) {
  cat("  that margin is no longer significant at the 1 percent level\n")
  bad <- bad + 1
}

if (bad > 0) {
  cat(sprintf("  %d claims in words disagree with results/transport.csv\n", bad))
  quit(status = 1)
}
cat("  every multiple and comparison written in words matches the data\n")
quit(status = 0)
