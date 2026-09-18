#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cut the pilot segments listed in pilot_manifest.json out of the 16 kHz source files.

The split and the boundaries were fixed on 2026-09-18 (build_pilot.py) before any training.
This script does not re-align anything; it only cuts [start, end) from the source WAVs, so
the segments on Drive are byte-identical to the ones the manifest was built from.

    python cut_segments.py --wavdir /content/drive/MyDrive/b0b_16k
"""
import argparse
import json
import os
import sys
import wave

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(HERE, "pilot_manifest.json"))
    ap.add_argument("--wavdir", required=True, help="folder with the b0b 16 kHz WAVs")
    ap.add_argument("--out", default=os.path.join(HERE, "segments"))
    a = ap.parse_args()
    man = json.load(open(a.manifest, encoding="utf-8"))
    os.makedirs(a.out, exist_ok=True)
    by_src = {}
    for x in man["items"]:
        by_src.setdefault(x["src_id"], []).append(x)
    n = 0
    for src, rows in by_src.items():
        p = os.path.join(a.wavdir, src + ".wav")
        if not os.path.isfile(p):
            sys.exit("[STOP] missing source %s - is b0b_16k on Drive?" % p)
        w = wave.open(p, "rb")
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (16000, 1, 2):
            sys.exit("[STOP] %s is not 16 kHz mono 16-bit" % p)
        for x in rows:
            a0, b0 = int(x["start"] * 16000), int(x["end"] * 16000)
            w.setpos(a0)
            fr = w.readframes(b0 - a0)
            o = wave.open(os.path.join(a.out, x["file"]), "wb")
            o.setnchannels(1); o.setsampwidth(2); o.setframerate(16000)
            o.writeframes(fr); o.close()
            n += 1
        w.close()
    by_split = {}
    for x in man["items"]:
        by_split[(x["speaker"], x["split"])] = by_split.get((x["speaker"], x["split"]), 0) + 1
    print("cut %d segments -> %s" % (n, a.out))
    for k in sorted(by_split):
        print("  %s %-7s %d" % (k[0], k[1], by_split[k]))


if __name__ == "__main__":
    main()
