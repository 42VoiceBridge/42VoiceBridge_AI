#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Listening kit for the VAD-mask audit (2026-09-26, GPT review §10 item 1).

Why this exists: on 2026-09-26 I argued from deletion counts that Silero had removed more than half
of KEJ's speech. That argument is circular — it certifies the input using the recogniser's own
errors. A deletion cannot be attributed to cropping rather than to recognition failure without
listening against source time. This kit collects that independent evidence.

THE SAMPLING RULE IS FROZEN HERE AND IS BLIND TO EVERY MODEL OUTPUT.
It uses only the mask intervals and the timeline. No hypothesis text, no CER, no reference text is
read, and none is shown to the listener — the 2026-09-19 review found that showing the label changes
what the listener hears, so the clips are presented bare.

  G-clips (gaps the mask REMOVED): split the file into GAP_BINS equal bins by time; in each bin take
    the LONGEST gap whose midpoint falls in that bin and which lasts at least MIN_GAP_SEC. Highlight
    at most HIGHLIGHT_MAX_SEC from the middle of that gap; add CONTEXT_SEC of audio on both sides.
  B-clips (edges of intervals the mask KEPT): split into EDGE_BINS equal bins; in each take the kept
    interval whose midpoint is nearest the bin centre; audit its onset in even bins and its offset in
    odd bins, with EDGE_SEC either side of that edge.

Known bias, stated before any answer is collected: taking the longest gap per bin over-samples long
gaps. Speech found there establishes local mask failure. Speech NOT found there is weak evidence
about short gaps, and a sample can never certify the whole recording.

    python3 make_mask_audit.py            # writes clips/ and index.html
    python3 make_mask_audit.py --selftest
"""
import json
import os
import sys
import wave

HERE = os.path.dirname(os.path.abspath(__file__))
T10 = os.path.join(os.path.dirname(HERE), "t10_highcer")
OUT = os.path.join(HERE, "clips")

GAP_BINS = 12
EDGE_BINS = 8
MIN_GAP_SEC = 1.0
HIGHLIGHT_MAX_SEC = 10.0
CONTEXT_SEC = 2.0
EDGE_SEC = 1.5
SPEAKERS = ("KEJ", "DTH")


def gaps_of(intervals, n):
    """Complement of the kept intervals inside [0, n)."""
    out, prev = [], 0
    for a, b in intervals:
        if a > prev:
            out.append((prev, a))
        prev = max(prev, b)
    if prev < n:
        out.append((prev, n))
    return out


def pick_per_bin(spans, n, bins, key):
    """One span per time bin, chosen by `key` among spans whose midpoint falls in the bin."""
    picked = []
    for i in range(bins):
        lo, hi = n * i // bins, n * (i + 1) // bins
        here = [s for s in spans if lo <= (s[0] + s[1]) // 2 < hi]
        if here:
            picked.append((i, max(here, key=key)))
    return picked


def plan(mask):
    sr, n = mask["sr"], mask["n_samples"]
    kept = [tuple(x) for x in mask["intervals"]]
    clips = []

    long_gaps = [g for g in gaps_of(kept, n) if (g[1] - g[0]) / sr >= MIN_GAP_SEC]
    for i, (a, b) in pick_per_bin(long_gaps, n, GAP_BINS, key=lambda g: g[1] - g[0]):
        full = (b - a) / sr
        cap = int(HIGHLIGHT_MAX_SEC * sr)
        if b - a > cap:                     # middle of the gap
            mid = (a + b) // 2
            ha, hb = mid - cap // 2, mid - cap // 2 + cap
        else:
            ha, hb = a, b
        pad = int(CONTEXT_SEC * sr)
        clips.append(dict(kind="gap", bin=i, cid="G%02d" % i,
                          clip=(max(0, ha - pad), min(n, hb + pad)),
                          highlight=(ha, hb), gap_sec=round(full, 2),
                          gap_at=round(a / sr, 1),
                          truncated=bool(b - a > cap)))

    # nearest-to-bin-centre needs the bin centre, so this one does not use pick_per_bin
    for i in range(EDGE_BINS):
        lo, hi = n * i // EDGE_BINS, n * (i + 1) // EDGE_BINS
        here = [iv for iv in kept if lo <= (iv[0] + iv[1]) // 2 < hi]
        if not here:
            continue
        centre = (lo + hi) // 2
        iv = min(here, key=lambda x: abs((x[0] + x[1]) // 2 - centre))
        onset = (i % 2 == 0)
        edge = iv[0] if onset else iv[1]
        pad = int(EDGE_SEC * sr)
        clips.append(dict(kind="onset" if onset else "offset", bin=i, cid="B%02d" % i,
                          clip=(max(0, edge - pad), min(n, edge + pad)),
                          highlight=(edge, min(n, edge + pad)) if onset else (max(0, edge - pad), edge),
                          kept_sec=round((iv[1] - iv[0]) / sr, 2), edge_at=round(edge / sr, 1)))
    return clips


def cut(src, dst, a, b):
    w = wave.open(src, "rb")
    assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 16000), src
    w.setpos(a)
    data = w.readframes(b - a)
    w.close()
    o = wave.open(dst, "wb")
    o.setnchannels(1)
    o.setsampwidth(2)
    o.setframerate(16000)
    o.writeframes(data)
    o.close()


def selftest():
    assert gaps_of([(0, 10), (20, 30)], 40) == [(10, 20), (30, 40)]
    assert gaps_of([(5, 10)], 10) == [(0, 5)]
    assert gaps_of([(0, 10)], 10) == []
    assert gaps_of([(0, 4), (2, 8)], 10) == [(8, 10)]          # overlapping kept spans
    n = 1000
    g = [(0, 100), (400, 450), (900, 990)]
    got = pick_per_bin(g, n, 2, key=lambda s: s[1] - s[0])
    assert got == [(0, (0, 100)), (1, (900, 990))], got
    print("selftest ok")


def main():
    os.makedirs(OUT, exist_ok=True)
    manifest = []
    for spk in SPEAKERS:
        mp = [p for p in os.listdir(os.path.join(T10, "masks")) if "-%s-" % spk in p][0]
        mask = json.load(open(os.path.join(T10, "masks", mp), encoding="utf-8"))
        wav = os.path.join(T10, "t10_16k", mask["id"] + ".wav")
        sr = mask["sr"]
        for c in plan(mask):
            name = "%s_%s.wav" % (spk, c["cid"])
            cut(wav, os.path.join(OUT, name), *c["clip"])
            c.update(speaker=spk, file=name, source_id=mask["id"],
                     clip_sec=round((c["clip"][1] - c["clip"][0]) / sr, 2),
                     hl_start_in_clip=round((c["highlight"][0] - c["clip"][0]) / sr, 2),
                     hl_sec=round((c["highlight"][1] - c["highlight"][0]) / sr, 2))
            manifest.append(c)
    json.dump(dict(created="2026-09-26", rule=dict(
        gap_bins=GAP_BINS, edge_bins=EDGE_BINS, min_gap_sec=MIN_GAP_SEC,
        highlight_max_sec=HIGHLIGHT_MAX_SEC, context_sec=CONTEXT_SEC, edge_sec=EDGE_SEC),
        clips=manifest), open(os.path.join(HERE, "clips.json"), "w", encoding="utf-8"),
        ensure_ascii=False, indent=1)
    write_html(manifest)
    tot = sum(c["clip_sec"] for c in manifest)
    print("%d clips, %.1f min of audio" % (len(manifest), tot / 60))
    for spk in SPEAKERS:
        cs = [c for c in manifest if c["speaker"] == spk]
        print("  %s: %d gap + %d edge, %.1f s of removed audio covered"
              % (spk, sum(c["kind"] == "gap" for c in cs), sum(c["kind"] != "gap" for c in cs),
                 sum(c["hl_sec"] for c in cs if c["kind"] == "gap")))


def write_html(manifest):
    H = ['<meta charset="utf-8"><title>VAD 마스크 청취 감사 2026-09-26</title>', """<style>
body{font:15px/1.65 -apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo",sans-serif;
 max-width:820px;margin:0 auto;padding:24px 20px 120px;color:#111;background:#fff}
h1{font-size:22px;margin:0 0 6px} h2{font-size:15px;margin:30px 0 8px;color:#666;
 border-bottom:1px solid #ddd;padding-bottom:4px}
.card{border:1px solid #ddd;border-radius:6px;padding:12px 14px;margin:12px 0}
.card h3{font-size:15px;margin:0 0 4px}
.hl{background:#fff4d6;padding:7px 9px;border-radius:4px;font-size:14px;margin:6px 0}
audio{width:100%;margin:6px 0} .meta{font-size:12px;color:#888}
label{display:block;padding:3px 0;font-size:14px;cursor:pointer}
input[type=text]{width:100%;font:13px inherit;border:1px solid #ccc;border-radius:4px;
 padding:5px 7px;box-sizing:border-box;margin-top:5px}
#bar{position:fixed;bottom:0;left:0;right:0;background:#fff;border-top:2px solid #111;
 padding:10px 20px;text-align:center}
button{font:14px inherit;padding:8px 18px;border:1px solid #111;background:#111;color:#fff;
 border-radius:4px;cursor:pointer}
#n{font-size:13px;color:#666;margin-right:12px}
</style>""",
         "<h1>VAD 마스크 청취 감사 — 2026-09-26</h1>",
         """<p>질문은 하나다: <b>표시된 구간에 말소리가 있었나?</b> 라벨(정답 텍스트)은 일부러 안 보여준다
— 2026-09-19 검수에서 라벨을 보면 들리는 내용이 달라진다는 게 확인됐다.</p>
<p><b>모르겠으면 "모르겠음"이 정답이다.</b> 추측이 데이터가 되면 이 감사 자체가 무의미해진다.
표본이라 "여기서 말소리가 나왔다" = 마스크 결함 확인이지만, "안 나왔다"가 마스크 정상을 증명하지는 않는다.</p>
<p class=meta>다 채우고 맨 아래 <b>복사</b> → 대화창에 붙여넣기.</p>"""]
    for spk in SPEAKERS:
        H.append("<h2>%s — 제거된 구간 (G)</h2>" % spk)
        for c in [x for x in manifest if x["speaker"] == spk and x["kind"] == "gap"]:
            H.append(card(c))
        H.append("<h2>%s — 남긴 구간의 경계 (B)</h2>" % spk)
        for c in [x for x in manifest if x["speaker"] == spk and x["kind"] != "gap"]:
            H.append(card(c))
    H.append("""<div id=bar><span id=n></span><button onclick="cp()">복사</button></div>
<script>
const R=[...document.querySelectorAll('.card')];
function upd(){const d=R.filter(c=>c.querySelector('input[type=radio]:checked')).length;
 document.getElementById('n').textContent=d+' / '+R.length;}
document.addEventListener('change',upd); upd();
function cp(){const o=R.map(c=>{const s=c.querySelector('input[type=radio]:checked');
 return {id:c.dataset.cid,speaker:c.dataset.spk,kind:c.dataset.kind,
  answer:s?s.value:null,note:c.querySelector('input[type=text]').value||null};});
 const t=JSON.stringify({audit:'vad_mask_2026-09-26',answers:o},null,1);
 navigator.clipboard.writeText(t).then(()=>{alert('복사됐다. 대화창에 붙여넣어라.');},
  ()=>{const w=window.open();w.document.body.innerHTML='<pre>'+t.replace(/</g,'&lt;')+'</pre>';});}
</script>""")
    open(os.path.join(HERE, "index.html"), "w", encoding="utf-8").write("\n".join(H))


GAP_OPTS = [("speech_clear", "또렷한 말소리가 있다"),
            ("speech_unclear", "사람 소리는 있으나 말인지 불분명하다"),
            ("nonspeech", "말소리 없다 (침묵·숨·잡음·기계음)"),
            ("unsure", "모르겠음")]
EDGE_OPTS = [("clipped", "잘렸다 — 말이 시작/끝나는데 오디오가 끊긴다"),
             ("intact", "안 잘렸다"),
             ("unsure", "모르겠음")]


def card(c):
    if c["kind"] == "gap":
        head = "%s %s — 제거된 구간 (원본 %.1f초 지점, 이 gap 전체 %.1f초%s)" % (
            c["speaker"], c["cid"], c["gap_at"], c["gap_sec"],
            ", 가운데 %.0f초만 들려준다" % c["hl_sec"] if c["truncated"] else "")
        hl = ("클립 <b>%.1f초 ~ %.1f초</b>가 마스크가 <b>제거한</b> 구간이다. 앞뒤 %.1f초는 남긴 구간(맥락용)." % (
            c["hl_start_in_clip"], c["hl_start_in_clip"] + c["hl_sec"], CONTEXT_SEC))
        opts = GAP_OPTS
    else:
        head = "%s %s — 남긴 구간의 %s (원본 %.1f초 지점, 그 구간 %.1f초)" % (
            c["speaker"], c["cid"], "시작" if c["kind"] == "onset" else "끝", c["edge_at"], c["kept_sec"])
        hl = ("클립 <b>%.1f초</b>에서 마스크 경계가 있다. %s쪽이 <b>남긴</b> 구간이다." % (
            c["hl_start_in_clip"] if c["kind"] == "onset" else EDGE_SEC,
            "뒤" if c["kind"] == "onset" else "앞"))
        opts = EDGE_OPTS
    rs = "".join('<label><input type=radio name=%s value=%s> %s</label>' % (
        c["speaker"] + c["cid"], v, t) for v, t in opts)
    return ('<div class=card data-cid="%s" data-spk="%s" data-kind="%s"><h3>%s</h3>'
            '<div class=hl>%s</div><audio controls preload=none src="clips/%s"></audio>'
            '<div class=meta>%s · 클립 %.1f초</div>%s'
            '<input type=text placeholder="메모(선택) — 들린 내용, 이상한 점">'
            '</div>') % (c["cid"], c["speaker"], c["kind"], head, hl, c["file"],
                         c["source_id"], c["clip_sec"], rs)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        selftest()
        main()
