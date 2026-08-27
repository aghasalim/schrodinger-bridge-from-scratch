"""Fail if a number in README.md no longer matches results/transport.csv."""
from __future__ import annotations

import csv
import math
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    rows = list(csv.DictReader((ROOT / "results" / "transport.csv").open()))
    body = (ROOT / "README.md").read_text()
    # Detail moved out of the README lives in notes/METHODS.md. A figure quoted
    # there is still a quoted figure and still has to match its source.
    _methods = ROOT / "notes" / "METHODS.md"
    if _methods.exists():
        body += "\n" + _methods.read_text()
    claims, failures = [], []

    def finite(vals):
        return [v for v in vals if math.isfinite(v)]

    for pair in sorted({r["pair"] for r in rows}):
        sub = [r for r in rows if r["pair"] == pair]

        def med(method, stage=None, sub=sub):
            # sub bound as a default: closing over the loop variable would make
            # every med() call use the last pair's rows.
            v = finite([float(r["w2"]) for r in sub if r["method"] == method
                        and (stage is None or r["stage"] == str(stage))])
            return statistics.median(v) if v else None

        claims.append((f"{pair} baseline", med("untransported")))
        claims.append((f"{pair} sinkhorn", med("sinkhorn-eps0.03")))
        claims.append((f"{pair} bm-ode", med("bridge-matching-ode", 100)))
        claims.append((f"{pair} bm-sde", med("bridge-matching-sde", 100)))

        by = {}
        for r in sub:
            if r["method"] == "dsb":
                v = float(r["w2"])
                if math.isfinite(v):
                    by.setdefault(int(r["stage"]), []).append(v)
        for s in sorted(by):
            claims.append((f"{pair} dsb ipf{s}", statistics.median(by[s])))
        if by:
            claims.append((f"{pair} dsb best", min(statistics.median(v) for v in by.values())))

    for label, value in claims:
        if value is None:
            continue
        text = f"{value:.3f}"
        if not re.search(r"(?<![\d.])" + re.escape(text) + r"(?!\d)", body):
            failures.append(f"{label} should read {text}, not found")

    print(f"checked {len(claims)} quoted figures against results/transport.csv")
    if failures:
        print("\nDRIFT DETECTED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("no drift")
    return 0


if __name__ == "__main__":
    sys.exit(main())
