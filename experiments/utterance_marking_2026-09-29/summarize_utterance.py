#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summary rule for the short-utterance eval, FROZEN 2026-09-29 before any result exists.

Same discipline as summarize_h100.py: the rule is written as code first so the reporting cannot be
chosen after seeing which cell looks good.

  - per speaker x decoder: b0, every training seed's pooled jamo CER, paired (b1 - b0), median,
    range, and the count improving / worsening
  - pooled CER weights by reference length; per-item rates are never averaged
  - speakers are never pooled; the best seed is never the headline
  - crops are 10 per speaker, so a sign count is descriptive, not a significance claim
  - the long-form matrix used a different decode path; the two are never compared cell-for-cell

    python3 summarize_utterance.py [results_utterance]
    python3 summarize_utterance.py --selftest
"""
import glob
import json
import os
import statistics as st
import sys

DECS = ("default", "nr5")


def rows(d):
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "*.json"))):
        if p.endswith("_spec.json"):
            continue
        r = json.load(open(p, encoding="utf-8"))
        if "results" not in r or "speaker" not in r:
            continue
        out[r["speaker"]] = r
    return out


def summarise(d):
    data = rows(d)
    if not data:
        return "No results in %s" % d
    L = ["# Short-utterance summary (rule fixed 2026-09-29)", "",
         "Pooled jamo CER in percentage points, weighted by reference length. "
         "diff = seed - b0 (negative = better).", ""]
    for spk in sorted(data):
        r = data[spk]
        res = r["results"]
        n = res["b0"][DECS[0]]["n_items"] if "b0" in res else "?"
        L += ["## %s (%s crops, %s)" % (spk, n, r.get("device", "?")), "",
              "| decoder | b0 | per-seed diff | median | range | improve / worsen |",
              "|---|---:|---|---:|---|---|"]
        for dec in DECS:
            if "b0" not in res or dec not in res["b0"]:
                continue
            b0 = res["b0"][dec]["pooled_cer_jamo"]
            seeds = [(k, v[dec]["pooled_cer_jamo"]) for k, v in sorted(res.items())
                     if k != "b0" and dec in v and v[dec]["pooled_cer_jamo"] is not None]
            if b0 is None or not seeds:
                continue
            diffs = [round((v - b0) * 100, 6) for _, v in seeds]
            med = st.median(diffs) + 0.0            # float noise otherwise prints a "-0.00" median
            med = 0.0 if abs(med) < 5e-3 else med
            L.append("| %s | %.2f | %s | %+.2f | %+.2f..%+.2f | %d / %d |" % (
                dec, b0 * 100, " ".join("%+.1f" % x for x in diffs), med,
                min(diffs), max(diffs), sum(x < 0 for x in diffs), sum(x >= 0 for x in diffs)))
        L.append("")
    L += ["## Scope", "",
          "Single-shot greedy decoding, i.e. the serving path for a short utterance. These numbers "
          "are **not** comparable cell-for-cell with the 112-cell long-form matrix, which used "
          "sequential long-form decoding with a sampling fallback on 40-minute files.", "",
          "10 crops per speaker, 2 speakers. Sign counts describe these crops; they are not a "
          "population claim. Windows were chosen by the timeline alone and boundaries marked by a "
          "listener, so the set is not biased toward utterances the recogniser already handled."]
    return "\n".join(L)


def selftest():
    r = dict(speaker="X", device="cuda", results=dict(
        b0={"default": dict(pooled_cer_jamo=0.40, n_items=10)},
        X_nall_s0={"default": dict(pooled_cer_jamo=0.30, n_items=10)},
        X_nall_s1={"default": dict(pooled_cer_jamo=0.50, n_items=10)}))
    import tempfile
    d = tempfile.mkdtemp()
    json.dump(r, open(os.path.join(d, "X.json"), "w"))
    out = summarise(d)
    assert "| default | 40.00 | -10.0 +10.0 | +0.00 | -10.00..+10.00 | 1 / 1 |" in out, out
    assert "not** comparable" in out
    print("selftest ok")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        selftest()
        d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "results_utterance")
        s = summarise(d)
        print(s)
        if os.path.isdir(d):
            open(os.path.join(d, "summary_utterance.md"), "w", encoding="utf-8").write(s)
