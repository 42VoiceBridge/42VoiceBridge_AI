#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build the B1 pilot set: sentence segments + enrollment/dev/test split per speaker.

Pilot scope (2026-09-18): two speakers whose 06-01 dialogue-reading files already have
large-v3 word timestamps (B0b run) - CYU and KJW. This is a PIPELINE test. Both speakers have
low baseline CER (large-v3 jamo 0.052 / 0.056), so these are not the users the product is for.

Split rule (fixed before any training, independent of model performance):
  - sentences are grouped in blocks of BLOCK consecutive sentences (dialogue pairs stay together)
  - blocks are shuffled with SEED and assigned test / dev / enroll by fraction
  - any normalized sentence text that occurs in test is removed from dev and enroll
  - any normalized text in dev is removed from enroll
Outputs pilot_manifest.json and segments/*.wav next to this script (or --out).
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from segment_by_asr import SEG_VERSION, cut, segment  # noqa: E402

BLOCK = 10
SEED = 20260918
FRAC = dict(test=0.2, dev=0.1)   # rest is the enrollment pool
SPLIT_VERSION = "split-v1"


def assign(spans):
    ok = [r for r in spans if r["ok"]]
    blocks = {}
    for r in ok:
        blocks.setdefault(r["idx"] // BLOCK, []).append(r)
    keys = sorted(blocks)
    rnd = random.Random(SEED)
    rnd.shuffle(keys)
    n = len(keys)
    n_test = max(1, round(n * FRAC["test"]))
    n_dev = max(1, round(n * FRAC["dev"]))
    split_of = {}
    for i, k in enumerate(keys):
        split_of[k] = "test" if i < n_test else ("dev" if i < n_test + n_dev else "enroll")
    for r in ok:
        r["split"] = split_of[r["idx"] // BLOCK]
    seen = {s: {r["norm"] for r in ok if r.get("split") == s} for s in ("test", "dev")}
    for r in ok:
        if r["split"] in ("dev", "enroll") and r["norm"] in seen["test"]:
            r["split"], r["dedup"] = "dropped", "text_in_test"
        elif r["split"] == "enroll" and r["norm"] in seen["dev"]:
            r["split"], r["dedup"] = "dropped", "text_in_dev"
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="b0b_16k/manifest.json")
    ap.add_argument("--hyp", required=True, help="hyp_b_large-v3.json")
    ap.add_argument("--wavdir", required=True, help="b0b_16k/")
    ap.add_argument("--ids", nargs="+", required=True)
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    a = ap.parse_args()

    man = {i["id"]: i for i in json.load(open(a.manifest, encoding="utf-8"))["items"]}
    hyp = json.load(open(a.hyp, encoding="utf-8"))
    segdir = os.path.join(a.out, "segments")
    items, report = [], {}
    for fid in a.ids:
        it = man[fid]
        spans = segment(it["ref_text"], hyp[fid], it["play_sec"])
        kept = assign(spans)
        names = cut(os.path.join(a.wavdir, it["out"]), kept, segdir, fid)
        assert len(names) == len(kept)
        reasons = {}
        for r in spans:
            if not r["ok"]:
                reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
        for r, name in zip(kept, names):
            items.append(dict(
                seg_id=name[:-4], file=name, speaker=it["speaker"], task=it["task"],
                src_id=fid, sent_idx=r["idx"], start=r["start"], end=r["end"],
                sec=round(r["end"] - r["start"], 3), text=r["text"], norm=r["norm"],
                split=r["split"], dedup=r.get("dedup"), asr_agree=r["asr_agree"],
                verified=False))
        sp = [x for x in items if x["src_id"] == fid]
        report[fid] = dict(
            sentences=len(spans), kept=len(kept), rejected=reasons,
            by_split={s: dict(n=sum(1 for x in sp if x["split"] == s),
                              sec=round(sum(x["sec"] for x in sp if x["split"] == s), 1))
                      for s in ("enroll", "dev", "test", "dropped")},
            mean_asr_agree_kept=round(sum(r["asr_agree"] for r in kept) / max(1, len(kept)), 3),
            mean_asr_agree_rejected=round(
                sum(r["asr_agree"] for r in spans if not r["ok"]) /
                max(1, sum(1 for r in spans if not r["ok"])), 3))
    out = dict(name="b1_pilot", seg_version=SEG_VERSION, split_version=SPLIT_VERSION,
               seed=SEED, block=BLOCK, frac=FRAC,
               note=("Boundaries from large-v3 word timestamps, targets from the label "
                     "transcript ('+','*' removed). Not listened to: verified=false. "
                     "Pipeline pilot on two low-baseline speakers; not an effect claim."),
               report=report, items=items)
    json.dump(out, open(os.path.join(a.out, "pilot_manifest.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
