#!/usr/bin/env bash
# Recompute what the repo publishes, in languages that share nothing with it.
#
# Everything in README.md came out of one implementation. The tables come from
# results/transport.csv, which comes from bench/experiment.py, and the only
# thing that had ever checked them was scripts/check_numbers.py, which reads the
# same file with the same language and the same assumptions. If the median in
# there were wrong, or the Sinkhorn iteration had a sign error, nothing in the
# repo would have noticed, because everything in the repo agrees with itself by
# construction.
#
# These do not. Each one recomputes a published number from the rawest file that
# still contains it, and a mistake would have to be repeated identically in
# several languages to get through.
#
# Anything whose toolchain is missing is skipped with a message rather than
# passed silently, so this is useful on a laptop with only some of them.
set -uo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
export PATH="$HOME/.cargo/bin:$PATH"
work="${TMPDIR:-/tmp}"

pass=0 fail=0 skip=0

run () {
    local name="$1" tool="$2"; shift 2
    printf '\n=== %s ===\n' "$name"
    if ! command -v "$tool" >/dev/null 2>&1; then
        printf '  skipped: %s is not installed\n' "$tool"
        skip=$((skip + 1)); return
    fi
    if "$@"; then pass=$((pass + 1)); else fail=$((fail + 1)); fi
}

# SQL has no way to assert, so the comparison happens here: every value the
# query recomputes has to appear in the documents that publish it.
# sqlite3 reads stdin, which inside a script is the rest of this script, so the
# redirect from /dev/null is load bearing. Its CSV output is CRLF, so the \r has
# to go before anything is matched.
check_sql () {
    local out doc miss=0 n=0 label value
    out=$(sqlite3 -init verify/summary.sql :memory: "" < /dev/null 2>/dev/null | tr -d '\r')
    if [ -z "$out" ]; then
        echo "  the query returned nothing"
        return 1
    fi
    doc=$(cat README.md notes/METHODS.md)
    while IFS= read -r line; do
        [ -n "$line" ] || continue
        label=${line%,*}; value=${line##*,}
        label=${label#\"}; label=${label%\"}
        n=$((n + 1))
        if ! printf '%s' "$doc" | grep -qF -- "$value"; then
            printf '  %s recomputes to %s, which is not in README.md or notes/METHODS.md\n' \
                   "$label" "$value"
            miss=$((miss + 1))
        fi
    done <<< "$out"
    if [ "$miss" -gt 0 ]; then
        printf '  %d of %d recomputed values are missing from the documents\n' "$miss" "$n"
        return 1
    fi
    printf '  SQL recomputes all %d published medians and every one is still in the documents\n' "$n"
    return 0
}

check_c () {
    cc -std=c99 -O2 -Wall -Wextra -Wpedantic -Werror \
       -o "$work/sb_sinkhorn" verify/sinkhorn.c -lm || return 1
    "$work/sb_sinkhorn" "$root"
}

check_go () { ( cd verify/gocheck && go run . -root "$root" ); }

check_rust () { ( cd verify/slicedw2 && cargo run --release --quiet -- "$root" ); }

run "Python, quoted figures"        python3 python3 scripts/check_numbers.py
run "SQL, published medians"        sqlite3 check_sql
run "C, the Sinkhorn kernel"        cc      check_c
run "Go, file structure"            go      check_go
run "R, claims written in words"    Rscript Rscript verify/ratios.R "$root"
run "Rust, Monte Carlo error bar"   cargo   check_rust
run "Ruby, counts in the prose"     ruby    ruby verify/counts.rb "$root"
run "JavaScript, table cells"       node    node verify/tables.mjs "$root"
run "Java, the coupling itself"     java    java verify/java/PlanCheck.java "$root"

printf '\n%s\n' "----------------------------------------"
printf '%d passed, %d failed, %d skipped\n' "$pass" "$fail" "$skip"
[ "$fail" -eq 0 ] || exit 1
[ "$pass" -gt 0 ] || { echo "nothing ran"; exit 1; }
