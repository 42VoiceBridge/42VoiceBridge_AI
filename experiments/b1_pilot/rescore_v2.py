#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Re-score saved pilot runs under the frozen rule (score-v2). No GPU: reads per_segment.csv.

Why this exists: the runs in results/ were scored with score-v1, whose "no digit" set depended on
which systems the invocation happened to include. A --with-large run therefore compared n30 and
nall on different segment sets. score-v2 fixes membership to the reference plus the paired
systems. See b0_run.number_bearing and HO §5.2.

    python3 rescore_v2.py            # prints the table and writes rescore_v2.json
"""
import csv
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from b0_run import PAIRED_SYSTEMS, SCORING_VERSION, cer, count_digits, number_bearing  # noqa: E402

FOREIGN = re.compile(r"[A-Za-z぀-ヿ一-鿿]")


def pooled(rows, key, jamo):
    E = N = 0
    for r in rows:
        c, n = cer(r["ref"], r[key], jamo=jamo)
        if n:
            E += c * n
            N += n
    return (round(E / N, 4) if N else None), N


def main():
    out = {"scoring_version": SCORING_VERSION, "paired_systems": list(PAIRED_SYSTEMS), "runs": {}}
    runs = sorted(d for d in os.listdir(os.path.join(HERE, "results"))
                  if os.path.isfile(os.path.join(HERE, "results", d, "per_segment.csv")))
    for run in runs:
        rows = [r for r in csv.DictReader(
            open(os.path.join(HERE, "results", run, "per_segment.csv"), encoding="utf-8"))
            if r["split"] == "test"]
        systems = [c for c in rows[0] if c in ("small_b0", "small_b1", "large_b0")]
        paired = [s for s in PAIRED_SYSTEMS if s in systems]
        v2 = [r for r in rows if not number_bearing(r["ref"], [r[s] for s in paired])]
        v1 = [r for r in rows if all(count_digits(r[s]) == 0 for s in systems)]
        rec = {"n_test": len(rows), "systems": systems,
               "n_no_number_v2": len(v2), "n_no_digit_v1": len(v1),
               "foreign_script_segments": sum(
                   1 for r in rows if any(FOREIGN.search(r[s]) for s in systems)),
               "variants": {}}
        for name, rr in (("test", rows), ("test_no_number", v2), ("test_no_digit__legacy_v1", v1)):
            rec["variants"][name] = {
                s: dict(zip(("cer_jamo", "n_jamo"), pooled(rr, s, True))) for s in systems}
            rec["variants"][name]["n_seg"] = len(rr)
        out["runs"][run] = rec

    print("%-13s %-26s %5s %8s %8s %8s" % ("run", "variant", "n_seg", "b0", "b1", "rel"))
    for run, rec in out["runs"].items():
        for name in ("test", "test_no_number", "test_no_digit__legacy_v1"):
            v = rec["variants"][name]
            a, b = v["small_b0"]["cer_jamo"], v["small_b1"]["cer_jamo"]
            print("%-13s %-26s %5d %8.4f %8.4f %7.1f%%" % (
                run, name, v["n_seg"], a, b, 100 * (b - a) / a))
        print("%-13s   foreign-script segments: %d" % ("", rec["foreign_script_segments"]))
    json.dump(out, open(os.path.join(HERE, "rescore_v2.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\n-> rescore_v2.json")


if __name__ == "__main__":
    main()
