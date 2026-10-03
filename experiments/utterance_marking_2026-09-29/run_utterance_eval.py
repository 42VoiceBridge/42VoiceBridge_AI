#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Short-utterance evaluation, frozen 2026-09-29 before any result exists.

The question: every number this project has comes from decoding a 40-minute file. The product
receives short recorded messages. Does per-speaker adaptation help in THAT condition?

Design, fixed here:
  - crops come from utterance_manifest.json; windows were picked by the timeline alone, never by
    ASR success, and boundaries were marked by a listener (see RESULT_2026-09-29.md)
  - systems: b0 (no adapter) + the five H100 training seeds for that speaker
  - decoders: default and nr5, applied symmetrically to b0 and every adapter
  - speakers are never pooled; report the per-seed spread, never the best seed
  - decoding is single-shot greedy, i.e. the SERVING path. These numbers are NOT comparable
    cell-for-cell with the long-form matrix; that is the point of running them.
  - every crop's hypothesis is saved, so the numbers can be re-derived without a GPU

Reads and writes nothing outside --out. Resumable: an existing result file with a matching
fingerprint is skipped.

    python3 run_utterance_eval.py --adapters DIR --out DIR [--dry-run]
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DEMO = os.path.join(os.path.dirname(os.path.dirname(HERE)), "demo")
SEEDS = (0, 1, 2, 3, 4)
DECODERS = [dict(name="default"), dict(name="nr5", no_repeat_ngram=5)]
BASE = "openai/whisper-small"
BASE_REVISION = "973afd24965f72e36ca33b3055d56a652f456b4d"
CODE_VERSION = "utterance-eval-v1"


def sha_dir(d):
    h = hashlib.sha256()
    for f in sorted(os.listdir(d)):
        if f.startswith("adapter_model"):
            h.update(open(os.path.join(d, f), "rb").read())
    return h.hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", default=os.path.join(HERE, "adapters"),
                    help="dir holding <SPK>_nall_s<k>/adapter")
    ap.add_argument("--manifest", default=os.path.join(HERE, "utterance_manifest.json"))
    ap.add_argument("--crops", default=os.path.join(HERE, "crops"))
    ap.add_argument("--out", default=os.path.join(HERE, "results_utterance"))
    ap.add_argument("--gate-eval", default=os.path.join(DEMO, "gate_eval.py"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    man = json.load(open(a.manifest, encoding="utf-8"))
    os.makedirs(a.out, exist_ok=True)
    ledger = os.path.join(a.out, "ledger.jsonl")

    def log(**kw):
        kw["t"] = time.strftime("%Y-%m-%d %H:%M:%S")
        line = "LEDGER " + json.dumps(kw, ensure_ascii=False)
        print(line, flush=True)
        if not a.dry_run:
            open(ledger, "a", encoding="utf-8").write(json.dumps(kw, ensure_ascii=False) + "\n")

    speakers = sorted({i["speaker"] for i in man["items"]})
    for spk in speakers:
        items = [i for i in man["items"] if i["speaker"] == spk]
        cands = {"b0": None}
        for s in SEEDS:
            p = os.path.join(a.adapters, "%s_nall_s%d" % (spk, s), "adapter")
            if os.path.isdir(p):
                cands["%s_nall_s%d" % (spk, s)] = p
            else:
                log(status="missing_adapter", speaker=spk, seed=s, path=p)
        out = os.path.join(a.out, "%s.json" % spk)
        fp = dict(code_version=CODE_VERSION, base_revision=BASE_REVISION,
                  n_items=len(items),
                  adapters={k: (sha_dir(v) if v and not a.dry_run else None)
                            for k, v in cands.items() if v},
                  decoders=[d["name"] for d in DECODERS])
        if os.path.isfile(out):
            have = json.load(open(out, encoding="utf-8"))
            if have.get("fingerprint") == fp:
                log(status="skipped", speaker=spk)
                continue
        spec = dict(base=BASE, base_revision=BASE_REVISION, decoders=DECODERS,
                    candidates=cands,
                    items=[dict(wav=os.path.join(a.crops, i["id"] + ".wav"), ref=i["ref_text"])
                           for i in items])
        sp = os.path.join(a.out, "%s_spec.json" % spk)
        cmd = [sys.executable, a.gate_eval, sp]
        print("$ " + " ".join(cmd))
        if a.dry_run:
            log(status="dry", speaker=spk, systems=len(cands), items=len(items),
                cells=len(cands) * len(DECODERS))
            continue
        json.dump(spec, open(sp, "w", encoding="utf-8"), ensure_ascii=False)
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            log(status="failed", speaker=spk, returncode=r.returncode, tail=(r.stderr or "")[-600:])
            continue
        res = json.loads(r.stdout.strip().splitlines()[-1])
        res.update(fingerprint=fp, speaker=spk, code_version=CODE_VERSION,
                   manifest_note=man.get("note"), wall_sec=round(time.time() - t0))
        json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        log(status="done", speaker=spk, wall_sec=res["wall_sec"],
            cells=len(cands) * len(DECODERS))
    print("DONE. summarise with summarize_utterance.py")


if __name__ == "__main__":
    main()
