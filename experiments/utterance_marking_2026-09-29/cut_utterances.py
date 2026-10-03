#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cut the marked utterances into an eval set (2026-09-29).

Input: the marks JSON pasted back from index.html, plus windows.json and the T10 manifest.
Output: crops/<id>.wav and utterance_manifest.json in the SAME shape as t10_manifest.json, so
`eval_longform.py` scores them unchanged — same normaliser, same provenance, same pinned revision.
That costs a model load per crop and buys an identical code path; the previous 112-cell matrix took
53 min, so ~240 short crops is comparable.

Rules, so the eval set cannot be quietly improved:
  - a mark is INCLUDED only if it has both boundaries, end > start, and a ref_index that is not
    "none". Everything else is written to excluded[] with the reason, never dropped silently.
  - `confidence: "unsure"` is INCLUDED and flagged. It is reported as a separate stratum, not
    removed - dropping the hard ones is how an eval set drifts easy.
  - PAD_SEC of context is added on each side, clamped to the file.
  - the reference text is the corpus label for that sentence, taken verbatim. Nothing is
    transcribed, edited or re-segmented here.

    python3 cut_utterances.py marks.json
    python3 cut_utterances.py --selftest
"""
import json
import os
import sys
import wave

HERE = os.path.dirname(os.path.abspath(__file__))
T10 = os.path.join(os.path.dirname(HERE), "t10_highcer")
CROPS = os.path.join(HERE, "crops")
PAD_SEC = 0.25
SR = 16000


def classify(m):
    """(include?, reason). Reason is None when included."""
    if m.get("ref_index") == "none":
        return False, "listener found no usable utterance in this window"
    if m.get("ref_index") is None:
        return False, "no reference sentence chosen"
    a, b = m.get("start_sec"), m.get("end_sec")
    if a is None or b is None:
        return False, "boundary not marked"
    if b <= a:
        return False, "end is not after start (%.2f -> %.2f)" % (a, b)
    return True, None


def cut(src, dst, a_sec, b_sec, n_frames):
    a = max(0, int((a_sec - PAD_SEC) * SR))
    b = min(n_frames, int((b_sec + PAD_SEC) * SR))
    w = wave.open(src, "rb")
    w.setpos(a)
    raw = w.readframes(b - a)
    w.close()
    o = wave.open(dst, "wb")
    o.setnchannels(1); o.setsampwidth(2); o.setframerate(SR)
    o.writeframes(raw)
    o.close()
    return (b - a) / SR


def selftest():
    ok = [dict(ref_index=3, start_sec=1.0, end_sec=2.0)]
    assert classify(ok[0]) == (True, None)
    assert classify(dict(ref_index="none"))[0] is False
    assert classify(dict(ref_index=1, start_sec=None, end_sec=2.0))[0] is False
    assert classify(dict(ref_index=1, start_sec=5.0, end_sec=5.0))[0] is False
    assert classify(dict(ref_index=0, start_sec=0.0, end_sec=0.5)) == (True, None)  # index 0 is valid
    print("selftest ok")


def main(path):
    marks = json.load(open(path, encoding="utf-8"))["marks"]
    wins = json.load(open(os.path.join(HERE, "windows.json"), encoding="utf-8"))
    refs = wins["refs"]
    t10 = json.load(open(os.path.join(T10, "t10_manifest.json"), encoding="utf-8"))
    src_id = {spk: next(i["id"] for i in t10["items"] if "-%s-02-04-" % spk in i["id"])
              for spk in refs}
    os.makedirs(CROPS, exist_ok=True)

    items, excluded = [], []
    for m in marks:
        spk = m["speaker"]
        inc, why = classify(m)
        if not inc:
            excluded.append(dict(speaker=spk, window=m["window"], reason=why))
            continue
        fid = src_id[spk]
        wav = os.path.join(T10, "t10_16k", fid + ".wav")
        w = wave.open(wav, "rb"); n = w.getnframes(); w.close()
        uid = "%s-UTT-%s" % (fid, m["window"])
        dur = cut(wav, os.path.join(CROPS, uid + ".wav"), m["start_sec"], m["end_sec"], n)
        items.append(dict(id=uid, speaker=spk, task="02-04-utterance", role="test",
                          ref_text=refs[spk][m["ref_index"]],
                          ref_index=m["ref_index"], source_id=fid,
                          start_sec=m["start_sec"], end_sec=m["end_sec"],
                          marked_sec=round(m["end_sec"] - m["start_sec"], 2),
                          crop_sec=round(dur, 2), pad_sec=PAD_SEC,
                          confidence=m.get("confidence"), note=m.get("note")))

    out = dict(name="t10_utterances", created="2026-09-29",
               note="Short-utterance eval set. Windows chosen by the timeline only (never by ASR "
                    "success); boundaries marked by a listener; reference text is the corpus label "
                    "verbatim. Excluded items are listed, not dropped.",
               pad_sec=PAD_SEC, items=items, excluded=excluded)
    json.dump(out, open(os.path.join(HERE, "utterance_manifest.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    print("included %d, excluded %d" % (len(items), len(excluded)))
    for spk in refs:
        got = [i for i in items if i["speaker"] == spk]
        uns = sum(1 for i in got if i["confidence"] == "unsure")
        print("  %s: %d crops, %.1f s total, median %.1f s, %d marked 'unsure'"
              % (spk, len(got), sum(i["crop_sec"] for i in got),
                 sorted(i["crop_sec"] for i in got)[len(got) // 2] if got else 0, uns))
    for e in excluded:
        print("  excluded %s %s: %s" % (e["speaker"], e["window"], e["reason"]))
    dupes = [i["ref_index"] for i in items]
    for spk in refs:
        ix = [i["ref_index"] for i in items if i["speaker"] == spk]
        if len(ix) != len(set(ix)):
            print("  NOTE %s: the same reference sentence is used by more than one crop "
                  "(repeated reading attempts) - expected, but do not treat them as independent" % spk)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    elif len(sys.argv) < 2:
        print(__doc__)
    else:
        selftest()
        main(sys.argv[1])
