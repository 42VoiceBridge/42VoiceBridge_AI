#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B1 pilot - personal LoRA on whisper-small, one speaker per run. Colab T4.

    !python /content/drive/MyDrive/b1_pilot/b1_train.py --speaker CYU
    !python /content/drive/MyDrive/b1_pilot/b1_train.py --speaker KJW --n 30

What this run is, and is not
  - A PIPELINE test: verified-boundary-free segments (verified=false), two low-baseline
    speakers, one seed by default. Every number it prints is [pilot]. Not a headline.
  - The split (enroll / dev / test) is fixed in pilot_manifest.json before any training.
    Epoch selection uses DEV only. TEST is scored once, after selection.
  - B0 (zero-shot small) and B1 (small + LoRA) are decoded by the SAME code path with the
    SAME settings, on the SAME segments. large-v3 (faster-whisper) is an optional reference
    on the same segments; it uses a different runtime, so treat small-vs-large as indicative.
  - Scoring imports cer()/norm_syl() from b0_run.py unchanged, so the metric is the B0 metric,
    including its known problems (digits deleted, non-Hangul deleted). Number formatting is a
    confound here: the adapted model can learn to spell numbers in Hangul like the labels,
    which lowers CER without better recognition. The summary reports CER with and without
    segments where any system wrote a digit.

Outputs (default /content/drive/MyDrive/b1_pilot/results/<run_id>/)
  hyp_<system>.json   per-segment hypotheses
  per_segment.csv     paired scores
  summary.json        pooled CER per split and system, versions, config, adapter size
  adapter/            PEFT adapter (the thing the demo server loads)
"""
import argparse
import csv
import json
import math
import os
import platform
import random
import sys
import time
import wave

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.dirname(HERE), "/content/drive/MyDrive"):
    if p not in sys.path:
        sys.path.append(p)

RUN_VERSION = "b1-pilot-v1"


def die(msg):
    sys.exit("\n[STOP] " + msg)


try:
    from b0_run import (PAIRED_SYSTEMS, SCORING_VERSION, cer, count_digits,  # noqa: F401
                        norm_syl, number_bearing, selftest, to_jamo)
except ImportError:
    die("b0_run.py not found. Put it at MyDrive/ (it already is if you ran B0) or next to "
        "this script. The scorer must be the B0 scorer, unchanged.")


def read_wav(path):
    import numpy as np
    w = wave.open(path, "rb")
    sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
    if (sr, ch, sw) != (16000, 1, 2):
        die("%s is not 16 kHz mono 16-bit (%s)" % (path, (sr, ch, sw)))
    x = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    w.close()
    return x


def pooled(rows, key_ref, key_hyp, jamo):
    """Pooled CER = total edits / total reference tokens (same normalization as B0)."""
    E = N = 0
    for r in rows:
        c, n = cer(r[key_ref], r[key_hyp], jamo=jamo)
        if n == 0:
            continue
        E += c * n
        N += n
    return (E / N if N else float("nan")), N


def runaway(ref, hyp):
    """Degenerate repetition loop: a hypothesis far longer than its reference.

    Measured 2026-09-18: with num_beams=5, whisper-small turned `아, 그래요?` (4 syllables) into
    200 tokens of `아`, which alone produced 199 of the KJW dev split's 232 syllable edits (86%)
    and moved dev syllable CER from 0.160 (greedy) to 0.928. Hangul repetition is NOT removed by
    norm_syl, unlike foreign script, so it enters the score as insertions and a 20-segment pooled
    CER can be dominated by one segment. Reproduced on CPU/fp32, so it is not a GPU or fp16
    effect. Flagged, never dropped silently - the summary reports CER with and without.
    """
    r, h = len(norm_syl(ref)), len(norm_syl(hyp))
    return h > 30 and h > 3 * max(r, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speaker", required=True)
    ap.add_argument("--n", default="all", help="enrollment budget in sentences, or 'all'")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base", default="openai/whisper-small")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--targets", default="q_proj,v_proj")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--beams", type=int, default=1,
                    help="1 = greedy (default since 2026-09-18: beam search produced repetition "
                         "loops that dominated pooled CER; see HO §5.2)")
    ap.add_argument("--with-large", action="store_true",
                    help="also score faster-whisper large-v3 zero-shot on the same segments")
    ap.add_argument("--manifest", default=os.path.join(HERE, "pilot_manifest.json"))
    ap.add_argument("--segdir", default=os.path.join(HERE, "segments"))
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--cpu-ok", action="store_true", help="allow CPU (very slow; for smoke tests)")
    a = ap.parse_args()

    selftest()
    try:
        import numpy as np
        import torch
        import transformers
        import peft
        from peft import LoraConfig, get_peft_model
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
    except ImportError as e:
        die("missing package (%s). Run the install cell first." % e)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu" and not a.cpu_ok:
        die("no GPU. Runtime > Change runtime type > T4 GPU. (--cpu-ok only for smoke tests)")
    amp = dev == "cuda"

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    man = json.load(open(a.manifest, encoding="utf-8"))
    items = [x for x in man["items"] if x["speaker"] == a.speaker]
    if not items:
        die("speaker %s not in manifest" % a.speaker)
    enroll = sorted([x for x in items if x["split"] == "enroll"], key=lambda x: x["sent_idx"])
    devset = [x for x in items if x["split"] == "dev"]
    test = [x for x in items if x["split"] == "test"]
    if a.n != "all":
        k = int(a.n)
        rnd = random.Random(1000 + a.seed)      # subset randomness separate from training seed
        enroll = sorted(rnd.sample(enroll, min(k, len(enroll))), key=lambda x: x["sent_idx"])
    run_id = "%s_n%s_s%d" % (a.speaker, a.n, a.seed)
    out = os.path.join(a.out, run_id)
    os.makedirs(out, exist_ok=True)
    print("run %s | enroll %d (%.1f min) dev %d test %d | device %s" % (
        run_id, len(enroll), sum(x["sec"] for x in enroll) / 60, len(devset), len(test), dev))

    audio = {}
    for x in enroll + devset + test:
        audio[x["seg_id"]] = read_wav(os.path.join(a.segdir, x["file"]))

    proc = WhisperProcessor.from_pretrained(a.base, language="korean", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(a.base).to(dev)
    base_rev = getattr(model.config, "_commit_hash", None)
    try:
        model.generation_config.forced_decoder_ids = None
    except Exception:
        pass

    def feats(batch):
        f = proc.feature_extractor([audio[x["seg_id"]] for x in batch], sampling_rate=16000,
                                   return_tensors="pt").input_features
        return f.to(dev)

    def transcribe(m, rows, beams):
        m.eval()
        hyps = {}
        for i in range(0, len(rows), a.bs):
            b = rows[i:i + a.bs]
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16,
                                                 enabled=amp):
                ids = m.generate(input_features=feats(b), language="korean", task="transcribe",
                                 num_beams=beams, max_new_tokens=200)
            for x, t in zip(b, proc.batch_decode(ids, skip_special_tokens=True)):
                hyps[x["seg_id"]] = t.strip()
        return hyps

    def score(rows, hyps):
        rr = [dict(ref=x["text"], hyp=hyps[x["seg_id"]]) for x in rows]
        s, ns = pooled(rr, "ref", "hyp", False)
        j, nj = pooled(rr, "ref", "hyp", True)
        return dict(cer_syl=round(s, 4), cer_jamo=round(j, 4), n_syl=ns, n_jamo=nj, n_seg=len(rows))

    # ------------------------------------------------ B0: zero-shot small, same code path
    t0 = time.time()
    hyp = {"small_b0": {}}
    hyp["small_b0"].update(transcribe(model, devset + test, a.beams))
    print("B0 small  dev %s  test %s  (%.0fs)" % (score(devset, hyp["small_b0"]),
                                                score(test, hyp["small_b0"]), time.time() - t0))

    # ------------------------------------------------ B1: LoRA
    cfg = LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=0.05, bias="none",
                     target_modules=a.targets.split(","))
    model = get_peft_model(model, cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print("LoRA trainable %d / %d (%.2f%%)" % (trainable, total, 100 * trainable / total))

    tok = proc.tokenizer
    sot = model.config.decoder_start_token_id
    lab = {}
    for x in enroll:
        ids = tok(x["text"]).input_ids
        if ids and ids[0] == sot:
            ids = ids[1:]                      # the model prepends decoder_start itself
        lab[x["seg_id"]] = ids
    feat_cache = {}
    for i in range(0, len(enroll), a.bs):
        b = enroll[i:i + a.bs]
        f = feats(b).to("cpu", torch.float16)
        for x, fx in zip(b, f):
            feat_cache[x["seg_id"]] = fx

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr,
                            weight_decay=0.0)
    steps = a.epochs * math.ceil(len(enroll) / a.bs)
    warm = max(1, int(0.1 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warm, max(0.0, (steps - s) / max(1, steps - warm))))
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=amp)
    pad = -100
    rng = random.Random(a.seed)
    b0_dev_greedy = transcribe(model, devset, 1)   # selection compares greedy with greedy
    best = dict(epoch=0, dev=score(devset, b0_dev_greedy), state=None)
    history = [dict(epoch=0, loss=None, dev=best["dev"])]
    print("epoch 0 (no training) dev jamo %.4f" % best["dev"]["cer_jamo"])
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        order = enroll[:]
        rng.shuffle(order)
        tot, cnt = 0.0, 0
        for i in range(0, len(order), a.bs):
            b = order[i:i + a.bs]
            x = torch.stack([feat_cache[r["seg_id"]] for r in b]).to(dev, torch.float32)
            L = max(len(lab[r["seg_id"]]) for r in b)
            y = torch.full((len(b), L), pad, dtype=torch.long)
            for k, r in enumerate(b):
                ids = lab[r["seg_id"]]
                y[k, :len(ids)] = torch.tensor(ids)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                loss = model(input_features=x, labels=y.to(dev)).loss
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item() * len(b)
            cnt += len(b)
        dh = transcribe(model, devset, 1)            # greedy on dev for selection speed
        ds = score(devset, dh)
        history.append(dict(epoch=ep, loss=round(tot / cnt, 4), dev=ds))
        print("epoch %d loss %.4f dev jamo %.4f syl %.4f (%.0fs)" % (
            ep, tot / cnt, ds["cer_jamo"], ds["cer_syl"], time.time() - t0))
        if ds["cer_jamo"] < best["dev"]["cer_jamo"]:
            best = dict(epoch=ep, dev=ds, state={k: v.detach().to("cpu").clone()
                                                 for k, v in model.state_dict().items()
                                                 if "lora_" in k})
    print("selected epoch %d on DEV" % best["epoch"])
    if best["state"] is None:
        # No epoch beat zero-shot on dev. The honest B1 is then "no adapter": identical to B0.
        print("no epoch beat zero-shot on dev -> B1 := B0; last-epoch adapter kept only for "
              "inspection, NOT for serving")
        hyp["small_b1"] = dict(hyp["small_b0"])
        adir = os.path.join(out, "adapter_last_epoch_NOT_SELECTED")
    else:
        model.load_state_dict(best["state"], strict=False)
        hyp["small_b1"] = transcribe(model, devset + test, a.beams)
        adir = os.path.join(out, "adapter")
    model.save_pretrained(adir)
    asize = sum(os.path.getsize(os.path.join(adir, f)) for f in os.listdir(adir)
                if f.startswith("adapter_model"))

    if a.with_large:
        try:
            from faster_whisper import WhisperModel
            fw = WhisperModel("large-v3", device=dev, compute_type="float16" if amp else "int8")
            hyp["large_b0"] = {}
            for x in devset + test:
                segs, _ = fw.transcribe(audio[x["seg_id"]], language="ko", beam_size=a.beams,
                                        condition_on_previous_text=False, vad_filter=False)
                hyp["large_b0"][x["seg_id"]] = "".join(s.text for s in segs).strip()
            del fw
        except Exception as e:                                           # noqa: BLE001
            print("large-v3 reference skipped: %r" % e)

    # ------------------------------------------------ scoring
    systems = list(hyp)
    summ = dict(run_id=run_id, version=RUN_VERSION, scoring_version=SCORING_VERSION, speaker=a.speaker, n=a.n, seed=a.seed,
                enroll=dict(n=len(enroll), sec=round(sum(x["sec"] for x in enroll), 1),
                            seg_ids=[x["seg_id"] for x in enroll]),
                selected_epoch=best["epoch"], history=history,
                config=dict(base=a.base, base_revision=base_rev, rank=a.rank, alpha=a.alpha,
                            targets=a.targets, lr=a.lr, epochs=a.epochs, bs=a.bs, beams=a.beams,
                            dev_decoding="greedy", test_decoding=("greedy" if a.beams == 1 else "beam%d" % a.beams)),
                adapter=dict(dir=adir, bytes=asize, trainable_params=trainable),
                versions=dict(python=platform.python_version(), torch=torch.__version__,
                              transformers=transformers.__version__, peft=peft.__version__),
                manifest=dict(seg_version=man["seg_version"], split_version=man["split_version"]),
                results={}, caveats=[
                    "pilot: segments not listened to (verified=false)",
                    "segment edges were accepted only where large-v3 matched the first/last "
                    "syllables - selection bias toward easier edges",
                    "low-baseline speakers; not the target population",
                    "single seed unless repeated",
                    "B0 scorer: digits and non-Hangul are deleted before scoring; segments where a digit appears on either side are reported separately as *_no_number (score-v2)"])
    for split, rows in (("dev", devset), ("test", test)):
        summ["results"][split] = {s: score(rows, hyp[s]) for s in systems}
        # score-v2: membership depends on the reference and the PAIRED systems only, never on
        # whether an optional reference pass (large_b0) happened to run in this invocation.
        paired = [s for s in PAIRED_SYSTEMS if s in hyp]
        clean = [x for x in rows
                 if not number_bearing(x["text"], [hyp[s][x["seg_id"]] for s in paired])]
        summ["results"][split + "_no_number"] = {s: score(clean, hyp[s]) for s in systems}
        # legacy (score-v1) kept for traceability: hypothesis-only, all systems. Do not headline.
        legacy = [x for x in rows if all(count_digits(hyp[s][x["seg_id"]]) == 0 for s in systems)]
        summ["results"][split + "_no_digit_outputs__legacy_v1"] = {
            s: score(legacy, hyp[s]) for s in systems}
        ok = [x for x in rows if not any(runaway(x["text"], hyp[s][x["seg_id"]]) for s in systems)]
        summ["results"][split + "_no_runaway"] = {s: score(ok, hyp[s]) for s in systems}
    summ["runaway"] = {s: sorted(x["seg_id"] for x in devset + test
                                 if runaway(x["text"], hyp[s][x["seg_id"]])) for s in systems}

    with open(os.path.join(out, "per_segment.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seg_id", "split", "sec", "ref"] + sum(
            [[s, s + "_cer_jamo", s + "_digits", s + "_runaway"] for s in systems], []))
        for x in devset + test:
            row = [x["seg_id"], x["split"], x["sec"], x["text"]]
            for s in systems:
                h = hyp[s][x["seg_id"]]
                c, n = cer(x["text"], h, jamo=True)
                row += [h, round(c, 4) if n else "", count_digits(h), int(runaway(x["text"], h))]
            w.writerow(row)
    json.dump(hyp, open(os.path.join(out, "hyps.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    json.dump(summ, open(os.path.join(out, "summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    # meta.json binds the adapter to one user for the demo server. An adapter that did not
    # beat zero-shot on DEV gets user_id=null, so the server refuses to load it.
    t = summ["results"]["test"]
    meta = dict(user_id=a.speaker if best["epoch"] > 0 else None, run_id=run_id,
                base_model=a.base, base_revision=base_rev, created_at=time.strftime("%Y-%m-%d %H:%M"),
                label="pilot adapter n=%s seed=%d" % (a.n, a.seed),
                evidence="[pilot] dev-selected epoch %d; test jamo CER %.3f -> %.3f (small zero-shot -> "
                         "adapted), %d segments, not listened to" % (
                             best["epoch"], t["small_b0"]["cer_jamo"], t["small_b1"]["cer_jamo"],
                             t["small_b0"]["n_seg"]))
    json.dump(meta, open(os.path.join(adir, "meta.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\n=== %s  [pilot] ===" % run_id)
    for split in ("test", "test_no_number", "test_no_runaway"):
        print(split)
        for s in systems:
            r = summ["results"][split][s]
            print("  %-9s jamo %.4f  syl %.4f  (n_seg %d)" % (s, r["cer_jamo"], r["cer_syl"],
                                                            r["n_seg"]))
    for sysname, segs in summ["runaway"].items():
        if segs:
            print("RUNAWAY %s: %d segment(s) %s  <- pooled CER for that split is dominated by "
                  "these; see *_no_runaway" % (sysname, len(segs), segs))
    print("adapter %.2f MB -> %s" % (asize / 1e6, adir))


if __name__ == "__main__":
    main()
