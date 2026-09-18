#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B1 pilot: cut long single-file recordings into sentence-level training pairs (<= 30 s).

The AI-Hub labels carry ONE transcript per file and no per-sentence timestamps. This script
locates each reference sentence in the audio by aligning the reference text to the word
timestamps of an existing large-v3 hypothesis (hyp_*.json from the B0/B0b runs).

What it trusts and what it does not
  - Only the sentence BOUNDARIES come from the recognizer. The training/eval target is always
    the reference sentence, never the recognizer output.
  - A sentence is kept only if its first and last Hangul characters (within ANCHOR chars)
    align exactly to recognizer characters, and its boundary words are not shared with a
    neighbouring sentence. Middle-of-sentence errors do NOT cause rejection, so the kept set
    is not filtered on overall recognizer accuracy. It IS filtered on edge accuracy - that is
    a selection bias toward sentences whose first/last syllables large-v3 got right. It is
    reported as `asr_agree` and must be stated with any result.
  - Nothing here is listened to. Every segment carries verified=false.

Usage
  python3 segment_by_asr.py --wav X.wav --hyp hyp.json --id FILE_ID --ref-text "..." \
      --speaker CYU --task 06-01 --out outdir
  (normally called through build_pilot.py)
"""
import argparse
import json
import os
import re
import unicodedata
import wave

import numpy as np

SEG_VERSION = "seg-v1"
ANCHOR = 2          # first/last N hangul chars of a sentence must contain an exact match
PAD_START = 0.25    # seconds of padding before the first boundary word
PAD_END = 0.35      # seconds after the last boundary word
MAX_SEC = 30.0
MIN_SEC = 0.4

HANGUL = re.compile(r"[^가-힣]")


def norm(s):
    return HANGUL.sub("", unicodedata.normalize("NFC", s))


def clean_ref(s):
    """Same rule as the B0 scorer: drop '+' and '*' markers, keep the words."""
    return re.sub(r"[+*]", "", s)


def split_sentences(text):
    parts = re.split(r"(?<=[.?!])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def hyp_chars(hyp):
    """Flatten word timestamps into a normalized char string + per-char word index."""
    words, chars, owner = [], [], []
    for seg in hyp["segments"]:
        ws = seg.get("words") or [dict(w=seg["text"], s=seg["start"], e=seg["end"])]
        for w in ws:
            t = norm(w["w"])
            if not t:
                continue
            wi = len(words)
            words.append(dict(s=float(w["s"]), e=float(w["e"]), text=w["w"]))
            for c in t:
                chars.append(c)
                owner.append(wi)
    return "".join(chars), owner, words


def align(ref, hyp):
    """ref char index -> hyp char index for exact matches (None otherwise).

    Plain Levenshtein DP with a full int32 table (numpy, no third-party aligner, so the
    result does not depend on which library is installed). Backtrace tie-break order:
    diagonal (match/substitution) > deletion from ref > insertion into ref.
    """
    n, m = len(ref), len(hyp)
    D = np.zeros((n + 1, m + 1), dtype=np.int32)
    D[0] = np.arange(m + 1)
    hb = np.frombuffer(hyp.encode("utf-32-le"), dtype=np.uint32)
    ar = np.arange(m + 1, dtype=np.int32)
    for i in range(1, n + 1):
        cost = (hb != ord(ref[i - 1])).astype(np.int32)
        t = np.empty(m + 1, dtype=np.int32)
        t[0] = D[i - 1, 0] + 1
        t[1:] = np.minimum(D[i - 1, 1:] + 1, D[i - 1, :-1] + cost)
        D[i] = np.minimum.accumulate(t - ar) + ar
    out = [None] * n
    i, j = n, m
    while i > 0 and j > 0:
        c = 0 if ref[i - 1] == hyp[j - 1] else 1
        if D[i, j] == D[i - 1, j - 1] + c:
            if c == 0:
                out[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif D[i, j] == D[i - 1, j] + 1:
            i -= 1
        else:
            j -= 1
    return out


def segment(ref_text, hyp, dur_total):
    sents = split_sentences(clean_ref(ref_text))
    H, owner, words = hyp_chars(hyp)
    R, sid = [], []
    for i, s in enumerate(sents):
        for c in norm(s):
            R.append(c)
            sid.append(i)
    R = "".join(R)
    amap = align(R, H)

    spans = [None] * len(sents)
    pos = 0
    for i, s in enumerate(sents):
        n = len(norm(s))
        a, b = pos, pos + n
        pos = b
        if n == 0:
            continue
        head = [amap[k] for k in range(a, min(a + ANCHOR, b)) if amap[k] is not None]
        tail = [amap[k] for k in range(max(b - ANCHOR, a), b) if amap[k] is not None]
        matched = sum(1 for k in range(a, b) if amap[k] is not None)
        rec = dict(idx=i, text=s, norm=norm(s), n=n, asr_agree=round(matched / n, 3),
                   ok=False, reason="")
        if not head or not tail:
            rec["reason"] = "edge_not_anchored"
        else:
            rec["w0"], rec["w1"] = owner[head[0]], owner[tail[-1]]
            if rec["w1"] < rec["w0"]:
                rec["reason"] = "non_monotonic"
            else:
                rec["ok"] = True
        spans[i] = rec

    live = [r for r in spans if r is not None]
    # boundary words must not be shared with the neighbour
    for j, r in enumerate(live):
        if not r["ok"]:
            continue
        prev = next((x for x in reversed(live[:j]) if "w1" in x), None)
        nxt = next((x for x in live[j + 1:] if "w0" in x), None)
        if prev is not None and prev["w1"] >= r["w0"]:
            r["ok"], r["reason"] = False, "shared_start_word"
        if nxt is not None and nxt["w0"] <= r["w1"]:
            r["ok"], r["reason"] = False, "shared_end_word"

    # times, padding clipped at midpoints to neighbours
    for j, r in enumerate(live):
        if "w0" not in r:
            continue
        r["t0"], r["t1"] = words[r["w0"]]["s"], words[r["w1"]]["e"]
    for j, r in enumerate(live):
        if not r["ok"]:
            continue
        prev = next((x for x in reversed(live[:j]) if "t1" in x), None)
        nxt = next((x for x in live[j + 1:] if "t0" in x), None)
        lo = (prev["t1"] + r["t0"]) / 2 if prev else 0.0
        hi = (r["t1"] + nxt["t0"]) / 2 if nxt else dur_total
        r["start"] = round(max(lo, r["t0"] - PAD_START), 3)
        r["end"] = round(min(hi, r["t1"] + PAD_END), 3)
        d = r["end"] - r["start"]
        if d > MAX_SEC:
            r["ok"], r["reason"] = False, "over_30s"
        elif d < MIN_SEC:
            r["ok"], r["reason"] = False, "under_min"
    return live


def cut(wav_path, spans, out_dir, file_id):
    os.makedirs(out_dir, exist_ok=True)
    w = wave.open(wav_path, "rb")
    sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
    assert (sr, ch, sw) == (16000, 1, 2), "expects the 16 kHz mono 16-bit copies from prep_*.py"
    out = []
    for r in spans:
        if not r["ok"]:
            continue
        a, b = int(r["start"] * sr), int(r["end"] * sr)
        w.setpos(a)
        frames = w.readframes(b - a)
        name = "%s__s%03d.wav" % (file_id, r["idx"])
        o = wave.open(os.path.join(out_dir, name), "wb")
        o.setnchannels(1); o.setsampwidth(2); o.setframerate(sr)
        o.writeframes(frames); o.close()
        out.append(name)
    w.close()
    return out
