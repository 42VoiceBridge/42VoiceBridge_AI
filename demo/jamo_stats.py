# -*- coding: utf-8 -*-
"""
Jamo-level MODEL-error statistics (metric jamo-err-v1).

This is what the proposal used to call "weak phoneme diagnosis". It is neither phonemes nor a
property of the speaker:
  - tokens are Unicode jamo with their position (I = initial, M = medial, F = final);
    orthographic, not phonetic. Silent initial 'ㅇ' is kept as its own token (ㅇ/I).
  - it counts how often THE MODEL gets a reference token wrong. The reference is intent-based,
    so the speaker's mispronunciation rate is not recoverable from it.
Definitions (fixed for v1):
  - input pairs: (reference text, model output). The reference must come from a known
    enrollment prompt or an independent human correction - never from the model itself.
  - normalization: NFC, Hangul syllables only (same as the B0 scorer).
  - alignment: Levenshtein over position-tagged jamo tokens; tie-break diagonal > deletion >
    insertion (same DP as the segmenter).
  - errors of a reference token = substitutions + deletions aligned to it.
  - denominator = occurrences of the token in the references (sample_count counts token
    occurrences, not utterances).
  - insertions are reported per inserted token type, separately; they have no reference token.
  - a token with fewer than min_support occurrences gets status "insufficient_data" and no rate.
"""
import re
import unicodedata

import numpy as np

METRIC_VERSION = "jamo-err-v1"
CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
JONG = " ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"


def tokens(s):
    s = re.sub(r"[^가-힣]", "", unicodedata.normalize("NFC", s))
    out = []
    for c in s:
        o = ord(c) - 0xAC00
        out.append(CHO[o // 588] + "/I")
        out.append(JUNG[(o % 588) // 28] + "/M")
        if o % 28:
            out.append(JONG[o % 28] + "/F")
    return out


def align_ops(r, h):
    """Return list of (op, ref_token_or_None, hyp_token_or_None); op in eq/sub/del/ins."""
    voc = {t: i for i, t in enumerate(sorted(set(r) | set(h)))}
    ri = np.array([voc[t] for t in r], dtype=np.int32)
    hi = np.array([voc[t] for t in h], dtype=np.int32)
    n, m = len(r), len(h)
    D = np.zeros((n + 1, m + 1), dtype=np.int32)
    D[0] = np.arange(m + 1)
    ar = np.arange(m + 1, dtype=np.int32)
    for i in range(1, n + 1):
        cost = (hi != ri[i - 1]).astype(np.int32)
        t = np.empty(m + 1, dtype=np.int32)
        t[0] = D[i - 1, 0] + 1
        t[1:] = np.minimum(D[i - 1, 1:] + 1, D[i - 1, :-1] + cost)
        D[i] = np.minimum.accumulate(t - ar) + ar
    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            c = 0 if r[i - 1] == h[j - 1] else 1
            if D[i, j] == D[i - 1, j - 1] + c:
                ops.append(("eq" if c == 0 else "sub", r[i - 1], h[j - 1]))
                i, j = i - 1, j - 1
                continue
        if i > 0 and D[i, j] == D[i - 1, j] + 1:
            ops.append(("del", r[i - 1], None))
            i -= 1
        else:
            ops.append(("ins", None, h[j - 1]))
            j -= 1
    ops.reverse()
    return ops


def compute(pairs, min_support=20):
    occ, err, ins = {}, {}, {}
    used = 0
    for p in pairs:
        r, h = tokens(p.get("ref", "")), tokens(p.get("hyp", ""))
        if not r:
            continue
        used += 1
        for op, rt, ht in align_ops(r, h):
            if rt is not None:
                occ[rt] = occ.get(rt, 0) + 1
                if op in ("sub", "del"):
                    err[rt] = err.get(rt, 0) + 1
            elif op == "ins":
                ins[ht] = ins.get(ht, 0) + 1
    rows = []
    for t, n in occ.items():
        jamo, pos = t.split("/")
        e = err.get(t, 0)
        ok = n >= min_support
        rows.append(dict(token=jamo, position={"I": "initial", "M": "medial", "F": "final"}[pos],
                         errors=e, sample_count=n,
                         error_rate=round(e / n, 4) if ok else None,
                         status="ok" if ok else "insufficient_data",
                         silent_initial=(t == "ㅇ/I")))
    rows.sort(key=lambda x: (x["status"] != "ok", -(x["error_rate"] or 0), -x["sample_count"]))
    return dict(metric_version=METRIC_VERSION, min_support=min_support, pairs_used=used,
                reference_tokens=sum(occ.values()),
                insertions=[dict(token=k.split("/")[0], count=v,
                                 position={"I": "initial", "M": "medial", "F": "final"}[k.split("/")[1]])
                            for k, v in sorted(ins.items(), key=lambda kv: -kv[1])],
                tokens=rows)


if __name__ == "__main__":
    assert tokens("각") == ["ㄱ/I", "ㅏ/M", "ㄱ/F"]
    assert tokens("아 1") == ["ㅇ/I", "ㅏ/M"]
    ops = align_ops(tokens("가"), tokens("카"))
    assert ops == [("sub", "ㄱ/I", "ㅋ/I"), ("eq", "ㅏ/M", "ㅏ/M")], ops
    out = compute([dict(ref="가나다", hyp="카나")], min_support=1)
    by = {(x["token"], x["position"]): x for x in out["tokens"]}
    assert by[("ㄱ", "initial")]["errors"] == 1 and by[("ㄷ", "initial")]["errors"] == 1
    assert by[("ㄴ", "initial")]["errors"] == 0
    out2 = compute([dict(ref="가", hyp="가나")], min_support=1)
    assert out2["insertions"] and out2["tokens"][0]["errors"] == 0
    print("jamo_stats selftest ok")
