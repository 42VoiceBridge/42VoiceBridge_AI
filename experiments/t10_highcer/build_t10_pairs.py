#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T10 option (a): KEJ + DTH enrollment pairs from the frozen seg-v1 segmenter (ANCHOR=2, untuned).

Split rule, fixed before any training and independent of model output:
  - 02-03 accepted pairs -> enroll / dev. Sentences grouped in blocks of 10 consecutive indices,
    blocks shuffled with SEED, blocks go to dev until dev holds >= 20% of the pairs.
  - 02-04 accepted pairs -> "test". These are the 4 edge-anchored sentences; b1_train decodes
    them as a SMOKE check only. The real evaluation is eval_longform.py on the whole 02-04 file.
  - normalized text present in test is removed from enroll/dev; text in dev removed from enroll.

Scope to state with any result: enrollment is the subset of 02-03 whose sentence edges large-v3
got right (asr_agree is reported). Not listened to: verified=false.

    python3 build_t10_pairs.py            # writes t10_pairs_manifest.json + segments/*.wav
"""
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "b1_pilot"))
from segment_by_asr import ANCHOR, SEG_VERSION, cut, segment  # noqa: E402

SPEAKERS = ("KEJ", "DTH")   # DTH added 2026-09-22; per-speaker split is independent
SEED = 20260921
BLOCK = 10
DEV_FRAC = 0.2


def main():
    man = {x["id"]: x for x in json.load(open(os.path.join(HERE, "t10_manifest.json"),
                                              encoding="utf-8"))["items"]}
    hyp = json.load(open(os.path.join(HERE, "results", "hyp_large-v3.json"), encoding="utf-8"))
    segdir = os.path.join(HERE, "segments")
    items, report = [], {}
    for fid, x in man.items():
        if x["speaker"] not in SPEAKERS:
            continue
        spans = segment(x["ref_text"], hyp[fid], hyp[fid]["duration"])
        ok = [r for r in spans if r["ok"]]
        if x["role"] == "test":
            for r in ok:
                r["split"] = "test"
        else:
            blocks = sorted({r["idx"] // BLOCK for r in ok})
            random.Random(SEED).shuffle(blocks)
            dev_blocks = set()
            for b in blocks:
                if sum(1 for r in ok if r["idx"] // BLOCK in dev_blocks) >= DEV_FRAC * len(ok):
                    break
                dev_blocks.add(b)
            for r in ok:
                r["split"] = "dev" if r["idx"] // BLOCK in dev_blocks else "enroll"
        names = cut(os.path.join(HERE, "t10_16k", x["out"]), ok, segdir, fid)
        for r, name in zip(ok, names):
            items.append(dict(seg_id=name[:-4], file=name, speaker=x["speaker"], task=x["task"],
                              src_id=fid, sent_idx=r["idx"], start=r["start"], end=r["end"],
                              sec=round(r["end"] - r["start"], 3), text=r["text"],
                              norm=r["norm"], split=r["split"], dedup=None,
                              asr_agree=r["asr_agree"], verified=False))
        report[fid] = dict(sentences=len(spans), kept=len(ok))
    # dedup WITHIN a speaker only: both speakers read the same scripts, and the same sentence
    # from another speaker is not a leak (it dropped 4 KEJ pairs when done globally).
    test_txt = {(x["speaker"], x["norm"]) for x in items if x["split"] == "test"}
    dev_txt = {(x["speaker"], x["norm"]) for x in items if x["split"] == "dev"}
    for x in items:
        key = (x["speaker"], x["norm"])
        if x["split"] in ("enroll", "dev") and key in test_txt:
            x["split"], x["dedup"] = "dropped", "text_in_test"
        elif x["split"] == "enroll" and key in dev_txt:
            x["split"], x["dedup"] = "dropped", "text_in_dev"
    # persisted invariant (GPT T10b review §7): no enroll/dev sentence occurs anywhere in the
    # speaker's WHOLE held-out long-form reference, not just in the 4 smoke-test sentences
    import re
    held = {x["speaker"]: re.sub(r"[^가-힣]", "", x["ref_text"])
            for x in man.values() if x["role"] == "test" and x["speaker"] in SPEAKERS}
    overlap = {spk: sum(1 for x in items if x["speaker"] == spk and x["split"] in ("enroll", "dev")
                        and x["norm"] in held[spk]) for spk in SPEAKERS}
    assert not any(overlap.values()), overlap
    by = {s: dict(n=sum(1 for x in items if x["split"] == s),
                  sec=round(sum(x["sec"] for x in items if x["split"] == s), 1))
          for s in ("enroll", "dev", "test", "dropped")}
    for spk in SPEAKERS:
        assert any(x["speaker"] == spk and x["split"] == "dev" for x in items), spk
    json.dump(dict(name="t10_option_a", seg_version=SEG_VERSION, split_version="t10-split-v1",
                   anchor=ANCHOR, seed=SEED,
                   block=BLOCK, dev_frac=DEV_FRAC, report=report, by_split=by,
                   eval_reference_overlap=overlap,
                   note=("Enrollment = edge-anchored subset of KEJ 02-03 (option a). test = 4 "
                         "edge-anchored 02-04 sentences, smoke only; the evaluation is "
                         "file-level long-form on all of 02-04. Not listened to."),
                   items=items),
              open(os.path.join(HERE, "t10_pairs_manifest.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(json.dumps(dict(report=report, by_split=by), ensure_ascii=False))


if __name__ == "__main__":
    main()
