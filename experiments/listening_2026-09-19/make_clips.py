#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cut the listening-review clips (2026-09-19). Reuses the B0 multi-format reader.

The source recordings are 44.1k/48k, 16/24/32-bit, mono or stereo (HO §2), so `wave` alone
cannot read them; prep_04_b0.read_wav_mono_float handles that. Clips are written 16 kHz mono
16-bit so any browser plays them.

    python3 make_clips.py
"""
import json
import os
import sys
import unicodedata
import wave

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(PROJECT, "experiments", "b0"))
from prep_04_b0 import read_wav_mono_float  # noqa: E402

SRC = os.path.join(PROJECT, "013.구음장애 음성인식 데이터")
OUT = os.path.join(HERE, "clips")

# (clip_id, filename fragment, start_sec, end_sec or None for end-of-file)
CLIPS = [
    ("A1a_KEJ-02-04_start",   "KEJ-02-04",   0,    75),
    ("A1b_KEJ-02-04_mid",     "KEJ-02-04",   480,  555),
    ("A1c_KEJ-02-04_end",     "KEJ-02-04",   -75,  None),
    ("A2_CYW-04-1000_tail",   "CYW-04-1000", 18,   None),
    ("B1_KEJ-04-6000_all",    "KEJ-04-6000", 0,    None),
    ("B2_KEJ-04-5000_tail",   "KEJ-04-5000", 193,  None),
    ("B3_KEJ-04-8000_start",  "KEJ-04-8000", 0,    60),
    ("C1_KEJ-02-03_start",    "KEJ-02-03",   0,    90),
    ("D1_CYW-04-5000_halluc", "CYW-04-5000", 103,  126),
]


def index():
    out = {}
    for root, _, fs in os.walk(SRC):
        for f in fs:
            if f.lower().endswith(".wav"):
                out[unicodedata.normalize("NFC", f)] = os.path.join(root, f)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    idx = index()
    meta = []
    for cid, frag, a, b in CLIPS:
        hit = [k for k in idx if frag in k]
        if not hit:
            sys.exit("[STOP] source not found: %s" % frag)
        x, sr, _ch, _sw = read_wav_mono_float(idx[hit[0]])
        dur = len(x) / sr
        a = dur + a if a < 0 else a
        b = dur if b is None else min(b, dur)
        seg = x[int(a * sr):int(b * sr)]
        # resample to 16 kHz by linear interpolation; adequate for listening, not for scoring
        n = int(len(seg) / sr * 16000)
        import numpy as np
        y = np.interp(np.linspace(0, len(seg) - 1, n), np.arange(len(seg)), seg)
        y = np.clip(y, -1, 1)
        p = os.path.join(OUT, cid + ".wav")
        w = wave.open(p, "wb")
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes((y * 32767).astype("<i2").tobytes())
        w.close()
        meta.append(dict(id=cid, source=hit[0], start=round(a, 2), end=round(b, 2),
                         sec=round(b - a, 1), source_sec=round(dur, 1)))
        print("%-26s %6.1fs  <- %s [%.1f, %.1f] of %.1fs" % (cid, b - a, hit[0], a, b, dur))
    json.dump(meta, open(os.path.join(HERE, "clips.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
