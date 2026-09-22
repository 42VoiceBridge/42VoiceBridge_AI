#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summarise the weekend batch exactly as fixed on 2026-09-22 (HO §7, "weekend batch v2 frozen").

Rule, written before any H100 result existed:
  - per speaker x decoder x VAD: every H100 training seed's CER, paired b1 - b0, median, range,
    count improving / worsening / fallback (fallback = no dev-selected adapter -> B1 := B0, diff 0)
  - eval seeds are summarised WITHIN a training seed (mean, plus the spread), never as extra n
  - speakers are never pooled; the decoder interaction is reported, not the best cell
  - archived Colab adapters are listed separately (different platform)
  - missing / failed cells are listed and the matrix is flagged incomplete

    python3 summarize_h100.py [results_h100]      # writes <dir>/summary_h100.md
    python3 summarize_h100.py --selftest
"""
import glob
import json
import os
import statistics
import sys

SPK = ("KEJ", "DTH")
DECS = ("default", "nr5")
VADS = ("off", "silero")
SEEDS = (0, 1, 2, 3, 4)
ES = (0, 1)
UNITS = ("cer_jamo", "cer_syl")


def load(d):
    cells = {}
    for p in glob.glob(os.path.join(d, "longform", "*.json")):
        r = json.load(open(p, encoding="utf-8"))
        if "cer_jamo" not in r or "eval_seed" not in r:
            continue
        spk = next(s for s in SPK if "-%s-" % s in r["id"])
        cells[(spk, r["system"], r["decoder"], r["vad"], r["eval_seed"])] = r
    fallback = set()
    lp = os.path.join(d, "ledger.jsonl")
    failed = []
    if os.path.exists(lp):
        for line in open(lp, encoding="utf-8"):
            e = json.loads(line)
            if e.get("status") == "fallback":
                fallback.add(e["run"])
            if e.get("status") == "failed":
                failed.append(e)
    return cells, fallback, failed


def summarise(d):
    cells, fallback, failed = load(d)
    out, missing = [], []
    out.append("# Weekend batch summary (rule fixed 2026-09-22)\n")
    out.append("Values are CER in percentage points; diff = b1 − b0 (negative = better). "
               "Per training seed the value is the mean over eval seeds; `es-spread` is the "
               "max−min across eval seeds. Exploratory follow-up on already-examined files.\n")
    for spk in SPK:
        out.append("\n## %s\n" % spk)
        for unit in UNITS:
            for vad in VADS:
                out.append("\n### %s, VAD %s\n" % (unit, vad))
                out.append("| decoder | b0 | per-seed b1 − b0 (s0..s4) | median | range | "
                           "improve / worsen / fallback | max es-spread |")
                out.append("|---|---:|---|---:|---|---|---:|")
                meds = {}
                for dec in DECS:
                    b0s = [cells[k][unit] for k in [(spk, "b0", dec, vad, e) for e in ES] if k in cells]
                    if len(b0s) < len(ES):
                        missing.append((spk, "b0", dec, vad))
                    if not b0s:
                        out.append("| %s | — | missing | | | | |" % dec)
                        continue
                    b0 = 100 * statistics.mean(b0s)
                    diffs, spreads, imp, wor, fb = [], [], 0, 0, 0
                    for s in SEEDS:
                        rid = "%s_nall_s%d" % (spk, s)
                        if rid in fallback:
                            diffs.append(0.0)
                            fb += 1
                            continue
                        v = [cells[k][unit] for k in [(spk, rid, dec, vad, e) for e in ES] if k in cells]
                        if len(v) < len(ES):
                            missing.append((spk, rid, dec, vad))
                        if not v:
                            diffs.append(None)
                            continue
                        dd = 100 * statistics.mean(v) - b0
                        diffs.append(dd)
                        spreads.append(100 * (max(v) - min(v)))
                        imp += dd < 0
                        wor += dd > 0
                    ok = [x for x in diffs if x is not None]
                    med = statistics.median(ok) if ok else None
                    meds[dec] = med
                    out.append("| %s | %.2f | %s | %s | %s | %d / %d / %d | %s |" % (
                        dec, b0, " ".join("—" if x is None else "%+.1f" % x for x in diffs),
                        "%+.2f" % med if med is not None else "—",
                        "%+.1f..%+.1f" % (min(ok), max(ok)) if ok else "—", imp, wor, fb,
                        "%.2f" % max(spreads) if spreads else "—"))
                if all(meds.get(x) is not None for x in DECS):
                    out.append("\nInteraction (median diff nr5 − default): %+.2f pp" %
                               (meds["nr5"] - meds["default"]))
        arch = sorted({k[1] for k in cells if k[0] == spk and k[1].startswith(spk + "_colab")})
        if arch:
            out.append("\nArchived Colab adapters (eval on H100, VAD off, es0), jamo %:")
            for a in arch:
                out.append("- %s: %s" % (a, ", ".join(
                    "%s %.2f" % (dec, 100 * cells[(spk, a, dec, "off", 0)]["cer_jamo"])
                    for dec in DECS if (spk, a, dec, "off", 0) in cells)))
    out.append("\n## Completeness\n")
    out.append("- fallback runs (B1 := B0): %s" % (sorted(fallback) or "none"))
    out.append("- failed ledger cells: %d" % len(failed))
    for e in failed:
        out.append("  - %s" % json.dumps(e, ensure_ascii=False))
    miss = sorted(set(missing))
    out.append("- missing cells: %d%s" % (len(miss), " -> **matrix INCOMPLETE**" if miss else ""))
    for m in miss:
        out.append("  - %s" % (m,))
    text = "\n".join(out) + "\n"
    open(os.path.join(d, "summary_h100.md"), "w", encoding="utf-8").write(text)
    return text


def selftest():
    import tempfile
    d = tempfile.mkdtemp()
    os.makedirs(os.path.join(d, "longform"))
    ids = {"KEJ": "ID-01-13-N-KEJ-02-04-F-36-KK", "DTH": "ID-01-13-N-DTH-02-04-M-85-KK"}
    def put(spk, system, dec, vad, es, j):
        json.dump(dict(id=ids[spk], system=system, decoder=dec, vad=vad, eval_seed=es,
                       cer_jamo=j, cer_syl=j), open(os.path.join(
                           d, "longform", "%s_%s_%s_%s_%d.json" % (spk, system, dec, vad, es)), "w"))
    for es in ES:
        put("KEJ", "b0", "default", "off", es, 0.70)
        put("KEJ", "KEJ_nall_s0", "default", "off", es, 0.60 + 0.02 * es)   # mean 0.61 -> -9.0
        put("KEJ", "KEJ_nall_s1", "default", "off", es, 0.80)               # +10.0
    open(os.path.join(d, "ledger.jsonl"), "w").write(
        json.dumps(dict(status="fallback", run="KEJ_nall_s2")) + "\n")
    t = summarise(d)
    row = [l for l in t.split("\n") if l.startswith("| default | 70.00")][0]
    assert "-9.0 +10.0 +0.0 — —" in row, row
    assert "1 / 1 / 1" in row and "| +0.00 |" in row, row      # median of -9, +10, 0 is 0
    assert "matrix INCOMPLETE" in t
    print("selftest ok")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        print(summarise(sys.argv[1] if len(sys.argv) > 1 else
                        os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_h100")))
