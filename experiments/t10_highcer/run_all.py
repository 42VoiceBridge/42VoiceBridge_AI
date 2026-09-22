#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T10 weekend batch for the Flightbase H100 JOB (2026-09-26/27). v2 after the GPT T10b review.

A frozen EXPLORATORY follow-up on already-examined test recordings (not a confirmatory test).
Same data split and hyper-parameters as T10a; nothing is chosen from these results.

Order (priority from the review, §6):
  1. platform check: b0 KEJ, default decoder, no VAD, eval seed 0 - text hash vs the Colab output
  2. archived Colab adapters (KEJ run 1, run 2): both decoders. Run 1 + nr5 is the missing cell
  3. train KEJ seeds 0-4 and DTH seeds 0-4 (dev-selected epoch, rule unchanged)
  4. evaluation matrix, all systems x decoder {default, nr5} x VAD {off, silero} x eval seed {0, 1}
     (main block first: VAD off, eval seed 0)
Every cell goes to results_h100/ledger.jsonl: done / failed / fallback (no epoch beat b0 on dev,
recorded as B1 := B0 for every arm - never silently dropped) / skipped (identical config exists).

JOB: python /root/project/datasets_rw/<dataset>/run_all.py        (--dry-run prints the plan)
"""
import hashlib
import json
import os
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "results_h100")
LONG = os.path.join(OUT, "longform")
EVAL = {"KEJ": "ID-01-13-N-KEJ-02-04-F-36-KK", "DTH": "ID-01-13-N-DTH-02-04-M-85-KK"}
SEEDS = (0, 1, 2, 3, 4)
ARCHIVED = {"KEJ": {"KEJ_colab_run1": "archived/KEJ_colab_run1/adapter",
                    "KEJ_colab_run2": "archived/KEJ_colab_run2/adapter"}}
NO_REPEAT = 5          # fixed 2026-09-22 before any control result existed
DECODERS = (0, NO_REPEAT)
VADS = (False, True)
EVAL_SEEDS = (0, 1)


def sha_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def ledger(dry, **kw):
    kw["t"] = time.strftime("%Y-%m-%d %H:%M:%S")
    print("LEDGER " + json.dumps(kw, ensure_ascii=False), flush=True)
    if not dry:
        with open(os.path.join(OUT, "ledger.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(kw, ensure_ascii=False) + "\n")


def sh(cmd, dry):
    print("\n$ " + " ".join(cmd), flush=True)
    return 0 if dry else subprocess.run(cmd, cwd=BASE).returncode


def evaluate(spk, system, adapter, dec, vad, es, dry):
    sys.path.insert(0, BASE)
    from eval_longform import CODE_VERSION, out_name
    fid = EVAL[spk]
    mask = os.path.join(BASE, "masks", fid + ".json")
    out = os.path.join(LONG, out_name(fid, system, dec, vad, es))
    cell = dict(speaker=spk, system=system, decoder="nr%d" % dec if dec else "default",
                vad="silero" if vad else "off", eval_seed=es)
    want = dict(adapter_sha256=sha_file(os.path.join(adapter, "adapter_model.safetensors"))
                if adapter and not dry else None, eval_seed=es, code_version=CODE_VERSION,
                vad_mask_sha256=json.load(open(mask))["mask_sha256"] if vad else None,
                no_repeat_ngram=dec)
    if os.path.exists(out):
        have = json.load(open(out, encoding="utf-8"))
        if all(have.get(k) == v for k, v in want.items()):
            return ledger(dry, status="skipped", **cell)
    cmd = [sys.executable, "eval_longform.py", "--id", fid, "--tag", system, "--wavdir", "t10_16k",
           "--out", LONG, "--eval-seed", str(es)]
    if adapter:
        cmd += ["--adapter", adapter]
    if dec:
        cmd += ["--no-repeat", str(dec)]
    if vad:
        cmd += ["--vad-mask", mask]
    rc = sh(cmd, dry)
    ledger(dry, status="done" if rc == 0 else "failed", returncode=rc, **cell)


def train(spk, seed, dry):
    rid = "%s_nall_s%d" % (spk, seed)
    d = os.path.join(OUT, rid)
    fp = dict(code=sha_file(os.path.join(BASE, "b1_train.py")),
              manifest=sha_file(os.path.join(BASE, "t10_pairs_manifest.json")), seed=seed)
    fpp = os.path.join(d, "train_fingerprint.json")
    if os.path.exists(os.path.join(d, "summary.json")) and os.path.exists(fpp) \
            and json.load(open(fpp)) == fp:
        ledger(dry, status="skipped", stage="train", run=rid)
    else:
        rc = sh([sys.executable, "b1_train.py", "--speaker", spk, "--seed", str(seed),
                 "--manifest", "t10_pairs_manifest.json", "--segdir", "segments", "--out", OUT], dry)
        if rc == 0 and not dry:
            json.dump(fp, open(fpp, "w"))
        ledger(dry, status="done" if rc == 0 else "failed", stage="train", run=rid, returncode=rc)
    adapter = os.path.join(d, "adapter")
    if dry or os.path.isdir(adapter):
        return rid, adapter
    ledger(dry, status="fallback", stage="train", run=rid,
           note="no epoch beat b0 on dev -> B1 := B0 for every decoder/VAD/eval-seed cell")
    return rid, None


def check_platform(dry):
    """Text of b0 KEJ (default, no VAD, es0) vs the archived Colab output. Informational: the
    eval seed is now reset later than in v1, so only fallback-free windows must match."""
    colab = os.path.join(BASE, "archived", "colab_b0_KEJ_text.txt")
    from eval_longform import out_name
    mine = os.path.join(LONG, out_name(EVAL["KEJ"], "b0", 0, False, 0))
    if dry or not (os.path.exists(colab) and os.path.exists(mine)):
        return
    a = open(colab, encoding="utf-8").read().strip()
    b = json.load(open(mine, encoding="utf-8"))["text"].strip()
    ledger(dry, status="platform_check", identical=a == b, colab_chars=len(a), h100_chars=len(b),
           note="differences are hardware or seed-timing effects; keep H100 and Colab numbers apart")


def main():
    dry = "--dry-run" in sys.argv
    cache = os.path.join(BASE, "hf_cache")
    if os.path.isdir(cache):
        os.environ["HF_HOME"], os.environ["HF_HUB_OFFLINE"] = cache, "1"
    os.makedirs(LONG, exist_ok=True)
    t0 = time.time()
    # 1-2. platform check + archived adapters
    evaluate("KEJ", "b0", "", 0, False, 0, dry)
    check_platform(dry)
    for name, ad in ARCHIVED["KEJ"].items():
        for dec in DECODERS:
            evaluate("KEJ", name, os.path.join(BASE, ad), dec, False, 0, dry)
    # 3. training
    systems = {spk: [("b0", "")] + [(n, os.path.join(BASE, a)) for n, a in
                                    ARCHIVED.get(spk, {}).items()] for spk in EVAL}
    for spk in EVAL:
        for seed in SEEDS:
            rid, adapter = train(spk, seed, dry)
            if adapter:
                systems[spk].append((rid, adapter))
        print("elapsed %.0f min" % ((time.time() - t0) / 60), flush=True)
    # 4. matrix, main block first
    blocks = [(v, es) for es in EVAL_SEEDS for v in VADS]
    for vad, es in blocks:
        for spk in EVAL:
            for system, adapter in systems[spk]:
                for dec in DECODERS:
                    evaluate(spk, system, adapter, dec, vad, es, dry)
        print("block vad=%s es=%d done, elapsed %.0f min" % (vad, es, (time.time() - t0) / 60),
              flush=True)
    print("\nDONE. Download %s (ledger.jsonl lists every cell)" % OUT)


if __name__ == "__main__":
    main()
