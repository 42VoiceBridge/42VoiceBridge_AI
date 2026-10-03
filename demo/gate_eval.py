#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Score a held-out gate set under several adapter options, in one process (2026-09-29).

Used by train_worker.py to decide promotion. Separate from b1_train.py on purpose: that script
reports the BASE as its comparison, and promotion must beat the **incumbent** - the adapter this
user is actually being served today - which may not be the base.

Loads the pinned base once, then applies each candidate adapter in turn. Decoding matches the
serving path (greedy, D9). Prints one JSON object on stdout; everything else goes to stderr.

Decoding comes from `decoding.settings()`, the SAME configuration the server generates with
(2026-10-03, review item A2) — including `max_new_tokens`, which this script previously omitted, so
a gate score was not a measured serving score. It is still deliberately not eval_longform.py's
sequential long-form loop with a sampling fallback: that exists for long files. Numbers from here
are NOT comparable cell-for-cell with the 112-cell long-form matrix, nor with results produced
before `generation_version` gen-v1.

    python3 gate_eval.py gate.json   # {"items":[{"wav":...,"ref":...}],
                                     #  "candidates":{"incumbent":null|path,"new":path},
                                     #  "decoders":[{"name":"default"},{"name":"nr5","no_repeat_ngram":5}]}
    python3 gate_eval.py --selftest
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "experiments", "b0"))
import decoding                      # the one generation configuration (A2, 2026-10-03)


def log(*a):
    print(*a, file=sys.stderr)


def run(spec):
    from b0_run import cer                                   # noqa: E402
    import numpy as np
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    base = spec.get("base", "openai/whisper-small")
    rev = spec.get("base_revision")
    proc = WhisperProcessor.from_pretrained(base, revision=rev, language="korean",
                                            task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(base, revision=rev)
    try:
        model.generation_config.forced_decoder_ids = None
    except Exception:                                        # noqa: BLE001
        pass
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev).eval()
    loaded = getattr(model.config, "_commit_hash", None)
    if rev and loaded not in (None, rev):
        raise RuntimeError("base revision mismatch: %s vs %s" % (loaded, rev))

    import wave
    def read(p):
        w = wave.open(p, "rb")
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 16000), p
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
        w.close()
        return x

    audio = [(it["wav"], read(it["wav"]), it["ref"]) for it in spec["items"]]
    decoders = spec.get("decoders") or [dict(name="default")]
    out = {}
    peft_model = None
    for name, path in spec["candidates"].items():
        m = model
        if path:
            from peft import PeftModel
            peft_model = PeftModel.from_pretrained(model, path)
            m = peft_model.to(dev).eval()
        out[name] = {}
        for d in decoders:
            gen = decoding.settings(d.get("no_repeat_ngram", 0))
            per = []
            for wav, x, ref in audio:
                f = proc(x, sampling_rate=16000, return_tensors="pt").input_features.to(dev)
                with torch.no_grad():
                    ids = m.generate(f, **gen)
                hyp = proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
                c, n = cer(ref, hyp, jamo=True)
                per.append(dict(wav=os.path.basename(wav), ref=ref, hyp=hyp,
                                cer_jamo=None if n == 0 else round(c, 4), n_jamo=n,
                                edits=0 if n == 0 else int(round(c * n))))
                log("  [%s/%s] %s cer=%s" % (name, d["name"], os.path.basename(wav),
                                             per[-1]["cer_jamo"]))
            # 2026-10-03 review: pooling ROUNDED per-item rates can shift a promotion decision near
            # the threshold. Pool integer edit counts over integer reference lengths instead.
            num = sum(p["edits"] for p in per if p["cer_jamo"] is not None)
            den = sum(p["n_jamo"] for p in per)
            out[name][d["name"]] = dict(
                adapter=path, decoder=d["name"],
                generation=decoding.describe(d.get("no_repeat_ngram", 0)),
                pooled_cer_jamo=None if not den else round(num / den, 4),
                n_items=len(per), n_jamo=den, per_item=per)
        if peft_model is not None:                            # unload before the next candidate
            model = peft_model.unload()
            peft_model = None
    import transformers, peft as peftlib
    return dict(device=dev, base=base, base_revision=loaded, results=out,
                versions=dict(transformers=transformers.__version__, peft=peftlib.__version__,
                              torch=torch.__version__))


def selftest():
    """Pooled CER must weight by reference length, not average per-item rates."""
    per = [dict(cer_jamo=0.5, n_jamo=2, edits=1), dict(cer_jamo=0.0, n_jamo=98, edits=0)]
    num = sum(p["edits"] for p in per)
    den = sum(p["n_jamo"] for p in per)
    assert round(num / den, 4) == 0.01, num / den        # not (0.5+0.0)/2 == 0.25
    # integer edits, not rounded rates: 1/3 of 3 jamo must pool as 1 edit, never as 0.3333*3
    p2 = [dict(cer_jamo=round(1/3, 4), n_jamo=3, edits=1)] * 3
    assert sum(x["edits"] for x in p2) == 3 and sum(x["n_jamo"] for x in p2) == 9
    print("selftest ok")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    elif len(sys.argv) < 2:
        print(__doc__)
    else:
        selftest()
        print(json.dumps(run(json.load(open(sys.argv[1], encoding="utf-8"))), ensure_ascii=False))
