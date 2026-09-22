#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T10 step 1 — transcribe the high-CER manifest with large-v3, keeping word timestamps.

The segmenter (b1_pilot/segment_by_asr.py) places sentence boundaries by aligning large-v3 word
timestamps to the label text, so it needs this pass first. KEJ-02-03 and KEJ-02-04 have never been
transcribed by anything; DTH-02-03/02-04 neither.

Decoding matches B0 exactly so the B0 numbers stay comparable: beam 5, no condition_on_previous,
VAD on, min_silence 500 ms, word timestamps. (D9's greedy default governs the *demo server* and the
adaptation test pass, not this alignment pass — changing it here would break comparability with the
B0 set, and alignment only needs timings.)

    !cd /content/drive/MyDrive/t10_highcer && python t10_transcribe.py
"""
import argparse
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.dirname(HERE), "/content/drive/MyDrive"):
    if p not in sys.path:
        sys.path.append(p)
try:
    from b0_run import cer, selftest
except ImportError:
    sys.exit("\n[STOP] b0_run.py not found. Put it next to this script or at MyDrive/.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(HERE, "t10_manifest.json"))
    ap.add_argument("--wavdir", default="",
                    help="folder holding the 16 kHz wavs. Default: t10_16k/ next to this script, "
                         "falling back to this script's own folder — a Drive upload often drops "
                         "the files one level up, so both are searched before giving up")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--beams", type=int, default=5)
    ap.add_argument("--speaker", default="", help="run one speaker only, e.g. KEJ. "
                    "Useful when the Drive upload is large: KEJ is 69 MB, DTH 135 MB")
    a = ap.parse_args()
    selftest()
    m = json.load(open(a.manifest, encoding="utf-8"))
    items = [x for x in m["items"] if not a.speaker or x["speaker"] == a.speaker]
    if not items:
        sys.exit("[STOP] no items for speaker %r" % a.speaker)
    if a.wavdir:
        cands = [a.wavdir]
    else:
        cands = [os.path.join(HERE, "t10_16k"), HERE]
    for d in cands:
        have = sorted(os.path.basename(f) for f in glob.glob(os.path.join(d, "*.wav")))
        if all(x["out"] in have for x in items):
            a.wavdir = d
            break
    else:
        a.wavdir = cands[0]
        have = sorted(os.path.basename(f) for f in glob.glob(os.path.join(a.wavdir, "*.wav")))
    missing = [x["out"] for x in items if x["out"] not in have]
    if missing:
        sys.exit("[STOP] %d of %d wav files are missing from %s\n"
                 "  found %d wav: %s\n  missing: %s\n"
                 "  A Drive folder drag often creates the folder and silently drops the large\n"
                 "  files. Upload the wav FILES into that folder directly, or run one speaker at a\n"
                 "  time with --speaker KEJ (69 MB) / --speaker DTH (135 MB).\n  searched: %s"
                 % (len(missing), len(items), a.wavdir, len(have), have or "(none)", missing, cands))
    print("wav %d개 확인 (%s), 전사 시작" % (len(items), a.wavdir))
    os.makedirs(a.out, exist_ok=True)
    from faster_whisper import WhisperModel
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        print("GPU 없음 — large-v3 106분 분량은 CPU에서 몇 시간이다. 런타임을 T4로.")

    model = WhisperModel(a.model, device=dev, compute_type="float16" if dev == "cuda" else "int8")
    hyp, rows = {}, []
    for x in items:
        p = os.path.join(a.wavdir, x["out"])
        t0 = time.time()
        segs, info = model.transcribe(p, language="ko", beam_size=a.beams,
                                      condition_on_previous_text=False, vad_filter=True,
                                      vad_parameters=dict(min_silence_duration_ms=500),
                                      word_timestamps=True)
        S = []
        for s in segs:
            S.append(dict(start=round(s.start, 2), end=round(s.end, 2), text=s.text,
                          words=[dict(w=w.word, s=round(w.start, 2), e=round(w.end, 2))
                                 for w in (s.words or [])]))
        text = "".join(s["text"] for s in S).strip()
        hyp[x["id"]] = dict(segments=S, text=text, duration=info.duration,
                            duration_after_vad=getattr(info, "duration_after_vad", None))
        c, n = cer(x["ref_text"], text, jamo=True)
        rows.append((x["id"], x["role"], round(c, 4), n, len(S), round(time.time() - t0)))
        print("%-32s %-10s jamo %.4f  구간 %4d  %4ds" % (x["id"], x["role"], c, len(S),
                                                        time.time() - t0))
        json.dump(hyp, open(os.path.join(a.out, "hyp_%s%s.json" % (a.model, ("_" + a.speaker) if a.speaker else "")), "w", encoding="utf-8"),
                  ensure_ascii=False)   # checkpoint after every file
    tag = ("_" + a.speaker) if a.speaker else ""
    json.dump(dict(model=a.model, beams=a.beams, device=dev, speaker=a.speaker or "all",
                   files=[dict(zip(("id", "role", "cer_jamo", "n_jamo", "n_seg", "sec"), r))
                          for r in rows]),
              open(os.path.join(a.out, "t10_b0_summary%s.json" % tag), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\n-> %s" % a.out)


if __name__ == "__main__":
    main()
