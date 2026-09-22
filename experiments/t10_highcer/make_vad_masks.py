#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Frozen VAD masks for the post-hoc VAD sensitivity arm (specified 2026-09-22, GPT T10b review §5).

The mask is a property of the audio, not of any model, so it is computed ONCE here (Silero is a small
CPU model) and shipped as JSON; every system and decoder on the H100 then uses byte-identical
intervals. Setting proposed by the reviewer, not validated for dysarthric speech.

Output masks/<id>.json: sample-index intervals after padding, clamped and unioned, plus parameters,
package version and model SHA-256. Intervals are "VAD-positive", not verified speech.

    python3 make_vad_masks.py
"""
import hashlib
import importlib.metadata
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from eval_longform import load16k  # noqa: E402

PARAMS = dict(threshold=0.5, neg_threshold=0.35, min_speech_duration_ms=100,
              min_silence_duration_ms=500, max_speech_duration_s=float("inf"), speech_pad_ms=300,
              window_size_samples=512, return_seconds=False)
EVAL_IDS = ("ID-01-13-N-KEJ-02-04-F-36-KK", "ID-01-13-N-DTH-02-04-M-85-KK")


def union(iv, n):
    out = []
    for a, b in sorted((max(0, a), min(n, b)) for a, b in iv):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def main():
    import silero_vad
    import torch
    torch.set_num_threads(2)
    ver = importlib.metadata.version("silero-vad")
    if ver != "6.2.0":
        sys.exit("[STOP] silero-vad %s, spec requires 6.2.0 - do not substitute" % ver)
    jit = os.path.join(os.path.dirname(silero_vad.__file__), "data", "silero_vad.jit")
    model_sha = hashlib.sha256(open(jit, "rb").read()).hexdigest()
    model = silero_vad.load_silero_vad(onnx=False)
    man = {x["id"]: x for x in json.load(open(os.path.join(HERE, "t10_manifest.json"),
                                              encoding="utf-8"))["items"]}
    os.makedirs(os.path.join(HERE, "masks"), exist_ok=True)
    for fid in EVAL_IDS:
        x = load16k(os.path.join(HERE, "t10_16k", man[fid]["out"]))
        ts = silero_vad.get_speech_timestamps(torch.from_numpy(x), model, sampling_rate=16000,
                                              **PARAMS)
        iv = union([(t["start"], t["end"]) for t in ts], len(x))
        rec = dict(id=fid, n_samples=len(x), sr=16000, intervals=iv,
                   retained_sec=round(sum(b - a for a, b in iv) / 16000, 2),
                   source_sec=round(len(x) / 16000, 2),
                   params={k: (str(v) if v == float("inf") else v) for k, v in PARAMS.items()},
                   silero_vad=ver, model_file="silero_vad.jit", model_sha256=model_sha,
                   gap_sec=0.5)
        rec["mask_sha256"] = hashlib.sha256(json.dumps(iv).encode()).hexdigest()
        json.dump(rec, open(os.path.join(HERE, "masks", fid + ".json"), "w"), indent=1)
        print("%s  intervals %d  retained %.1f / %.1f s (%.1f%%)" % (
            fid, len(iv), rec["retained_sec"], rec["source_sec"],
            100 * rec["retained_sec"] / rec["source_sec"]))
    assert union([(5, 10), (8, 12), (-3, 2), (20, 99)], 30) == [[0, 2], [5, 12], [20, 30]]


if __name__ == "__main__":
    main()
