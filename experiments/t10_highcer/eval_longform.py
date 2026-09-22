#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""File-level evaluation of whisper-small with or without a LoRA adapter (T10).

v2 (2026-09-22, after the GPT T10b review):
  - decoding is GREEDY-FIRST WITH SAMPLING FALLBACK (temperature 0.0..1.0), not pure greedy
  - the eval seed is reset immediately before generate(), not before model construction
  - base model revision pinned; provenance (versions, device, dtype, hashes, resolved settings) saved
  - decoder segments (start, end, text) saved, so text can later be overlaid on a VAD mask
  - --vad-mask: post-hoc sensitivity arm. Mask intervals (VAD-positive, not verified speech) are
    concatenated with exactly 0.5 s of zeros between them; scored against the SAME whole-file
    reference; the processed->source mapping is saved
  - --no-repeat: symmetric robustness factor (no_repeat_ngram_size), NOT a hallucination remover

Both systems go through this same function. Reference policy v1 = label scored with b0_run.cer
(non-Hangul deleted; repeated attempts written in the label stay). Never compare these numbers
with faster-whisper large-v3 numbers.

    python3 eval_longform.py --id ID-01-13-N-KEJ-02-04-F-36-KK
    python3 eval_longform.py --id ... --adapter X/adapter --no-repeat 5 --vad-mask masks/ID.json
"""
import argparse
import hashlib
import json
import os
import sys
import time
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from b0_run import cer, count_digits  # noqa: E402

BASE = "openai/whisper-small"
BASE_REVISION = "973afd24965f72e36ca33b3055d56a652f456b4d"   # the revision every run so far used
CODE_VERSION = "eval-longform-v2"
GAP = 0.5


def sha(b):
    return hashlib.sha256(b).hexdigest()


def load16k(path, max_sec=0):
    w = wave.open(path, "rb")
    assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2), path
    x = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768
    return x[:int(max_sec * 16000)] if max_sec else x


def apply_mask(x, mask):
    """Concatenate mask intervals with GAP s of zeros between them. Returns audio + mapping rows
    (processed_start_sec, source_start_sec, length_sec); gaps are synthetic and unmapped."""
    gap = np.zeros(int(GAP * 16000), np.float32)
    parts, mapping, pos = [], [], 0
    for k, (a, b) in enumerate(mask["intervals"]):
        if k:
            parts.append(gap)
            pos += len(gap)
        parts.append(x[a:b])
        mapping.append((round(pos / 16000, 3), round(a / 16000, 3), round((b - a) / 16000, 3)))
        pos += b - a
    return (np.concatenate(parts) if parts else np.zeros(0, np.float32)), mapping


def settings(no_repeat):
    g = dict(language="korean", task="transcribe", num_beams=1, return_timestamps=True,
             condition_on_prev_tokens=False, return_segments=True,
             temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),   # stock OpenAI long-form safeguards
             compression_ratio_threshold=2.4, logprob_threshold=-1.0, no_speech_threshold=0.6)
    if no_repeat:
        g["no_repeat_ngram_size"] = no_repeat
    return g


def transcribe(audio, adapter="", no_repeat=0, eval_seed=0):
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = WhisperProcessor.from_pretrained(BASE, revision=BASE_REVISION,
                                            language="korean", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(BASE, revision=BASE_REVISION)
    model.generation_config.forced_decoder_ids = None
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.to(dev).eval()
    dtype = str(next(model.parameters()).dtype)
    if len(audio) == 0:                          # no VAD-positive interval: empty text, still scored
        return "", [], dev, dtype
    f = proc.feature_extractor(audio, sampling_rate=16000, return_tensors="pt", truncation=False,
                               padding="longest", return_attention_mask=True)
    torch.manual_seed(eval_seed)                 # immediately before generation (review §3)
    if dev == "cuda":
        torch.cuda.manual_seed_all(eval_seed)
    with torch.no_grad():
        out = model.generate(input_features=f.input_features.to(dev),
                             attention_mask=f.attention_mask.to(dev), **settings(no_repeat))
    segs = []
    for s in (out["segments"][0] if isinstance(out, dict) else []):
        rec = dict(start=round(float(s["start"]), 3), end=round(float(s["end"]), 3),
                   text=proc.decode(s["tokens"], skip_special_tokens=True).strip())
        for k, v in s.items():                   # keep any scalar extras (e.g. fallback info)
            if k not in rec and isinstance(v, (int, float, str, bool)):
                rec[k] = v
        segs.append(rec)
    ids = out["sequences"] if isinstance(out, dict) else out
    text = proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
    return text, segs, dev, dtype


def out_name(fid, tag, no_repeat, vad, eval_seed, max_sec=0):
    return "%s__%s__dec-%s__vad-%s__es%d%s.json" % (
        fid, tag, "nr%d" % no_repeat if no_repeat else "default", "silero" if vad else "off",
        eval_seed, "__%ds" % max_sec if max_sec else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--adapter", default="")
    ap.add_argument("--tag", default="", help="system name in the output file")
    ap.add_argument("--no-repeat", type=int, default=0)
    ap.add_argument("--vad-mask", default="")
    ap.add_argument("--eval-seed", type=int, default=0)
    ap.add_argument("--max-sec", type=float, default=0, help="plumbing checks only")
    ap.add_argument("--wavdir", default=os.path.join(HERE, "t10_16k"))
    ap.add_argument("--out", default=os.path.join(HERE, "results", "longform"))
    a = ap.parse_args()
    item = {x["id"]: x for x in json.load(open(os.path.join(HERE, "t10_manifest.json"),
                                               encoding="utf-8"))["items"]}[a.id]
    src = load16k(os.path.join(a.wavdir, item["out"]), a.max_sec)
    mask = json.load(open(a.vad_mask)) if a.vad_mask else None
    if mask:
        assert mask["id"] == a.id and mask["n_samples"] == len(src), "mask does not fit this audio"
    audio, mapping = apply_mask(src, mask) if mask else (src, None)
    tag = a.tag or (os.path.basename(os.path.dirname(a.adapter.rstrip("/"))) if a.adapter else "b0")
    adapter_sha = (sha(open(os.path.join(a.adapter, "adapter_model.safetensors"), "rb").read())
                   if a.adapter else None)
    t0 = time.time()
    text, segs, dev, dtype = transcribe(audio, a.adapter, a.no_repeat, a.eval_seed)
    import peft
    import torch
    import transformers
    dur = len(audio) / 16000
    g = settings(a.no_repeat)
    g["temperature"] = list(g["temperature"])
    rec = dict(
        id=a.id, system=tag, adapter=a.adapter or None, adapter_sha256=adapter_sha,
        decoder="nr%d" % a.no_repeat if a.no_repeat else "default", no_repeat_ngram=a.no_repeat,
        vad="silero" if mask else "off", vad_mask_sha256=mask["mask_sha256"] if mask else None,
        eval_seed=a.eval_seed, code_version=CODE_VERSION, base=BASE, base_revision=BASE_REVISION,
        generation=g, device=dev,
        device_name=torch.cuda.get_device_name(0) if dev == "cuda" else "cpu", dtype=dtype,
        versions=dict(python=sys.version.split()[0], torch=torch.__version__,
                      transformers=transformers.__version__, peft=peft.__version__),
        audio_sha256=sha(src.tobytes()), ref_sha256=sha(item["ref_text"].encode()),
        source_sec=round(len(src) / 16000, 1), decoded_sec=round(dur, 1),
        vad_retained_sec=mask["retained_sec"] if mask else None, vad_mapping=mapping,
        wall_sec=round(time.time() - t0), n_segments=len(segs),
        last_segment_end=segs[-1]["end"] if segs else None,
        endpoint_reach=round(segs[-1]["end"] / dur, 3) if segs and dur else None,
        hyp_chars=len(text), digits_hyp=count_digits(text), text=text,
        text_sha256=sha(text.encode()), segments=segs)
    if not a.max_sec:   # a truncated excerpt has no matching reference; never score it
        for unit, jamo in (("syl", False), ("jamo", True)):
            c, n = cer(item["ref_text"], text, jamo=jamo)
            rec["cer_" + unit], rec["n_" + unit] = round(c, 4), n
    os.makedirs(a.out, exist_ok=True)
    p = os.path.join(a.out, out_name(a.id, tag, a.no_repeat, mask, a.eval_seed, a.max_sec))
    json.dump(rec, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: rec.get(k) for k in ("system", "decoder", "vad", "eval_seed", "cer_jamo",
                                              "cer_syl", "n_segments", "wall_sec")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
