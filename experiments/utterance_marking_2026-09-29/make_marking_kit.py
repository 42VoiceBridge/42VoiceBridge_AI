#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Boundary-marking kit for the short-utterance evaluation (2026-09-29, GPT review §6 / Q5).

Why: every number we have comes from decoding a 40-minute file. The product receives short recorded
messages. Neither the VAD-off arm nor the VAD-masked arm is that condition, so "does adaptation help
in deployment" is unmeasured. This kit produces the crops for that measurement.

SELECTION IS FROZEN HERE AND IS INDEPENDENT OF ASR CORRECTNESS.
The reviewer's objection to the obvious shortcut — reusing seg-v1, which only accepts sentences the
recogniser already found — is that it biases the eval set toward easy utterances. So windows are
chosen by the timeline alone: the recording is split into BINS equal bins and one WINDOW_SEC window
is cut at each bin centre. Whatever is in that window is what gets marked, including weak, repeated
and failed attempts. A window with nothing usable is recorded as `none`, never silently dropped.

The listener marks boundaries and IDENTIFIES which reference sentence is being read; they never type
a transcript. The reference text is the corpus label, unchanged — we do not create a reference by
listening (HO §4: "that is editing the reference to match ASR output").

    python3 make_marking_kit.py            # writes windows/ and index.html
    python3 make_marking_kit.py --selftest
"""
import array
import json
import os
import sys
import wave

HERE = os.path.dirname(os.path.abspath(__file__))
T10 = os.path.join(os.path.dirname(HERE), "t10_highcer")
OUT = os.path.join(HERE, "windows")

BINS = 10
WINDOW_SEC = 50.0
ENV_MS = 25                      # RMS envelope resolution
SR = 16000
EVAL = {"KEJ": "ID-01-13-N-KEJ-02-04-F-36-KK", "DTH": "ID-01-13-N-DTH-02-04-M-85-KK"}


def windows(n_samples, bins=BINS, win=WINDOW_SEC, sr=SR):
    """One window per time bin, centred on the bin centre, clamped inside the file."""
    w = int(win * sr)
    out = []
    for i in range(bins):
        centre = n_samples * (2 * i + 1) // (2 * bins)
        a = max(0, min(centre - w // 2, n_samples - w))
        out.append((i, a, min(n_samples, a + w)))
    return out


def envelope(samples, ms=ENV_MS, sr=SR):
    """Peak-absolute per ms bin, normalised to 0..1. Peak, not RMS: quiet consonants stay visible."""
    step = max(1, sr * ms // 1000)
    out = []
    for i in range(0, len(samples), step):
        chunk = samples[i:i + step]
        out.append(max((abs(v) for v in chunk), default=0))
    top = max(out) or 1
    return [round(v / top, 3) for v in out]


def read_slice(path, a, b):
    w = wave.open(path, "rb")
    assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, SR), path
    w.setpos(a)
    raw = w.readframes(b - a)
    w.close()
    s = array.array("h")
    s.frombytes(raw)
    return s, raw


def sentences(ref):
    return [p.strip() for p in ref.split(".") if p.strip()]


def selftest():
    assert windows(1000, bins=2, win=0.02, sr=10000) == [(0, 150, 350), (1, 650, 850)]
    # a window wider than the file is clamped to the file
    assert windows(100, bins=1, win=1.0, sr=1000) == [(0, 0, 100)]
    e = envelope(array.array("h", [0, 100, -200, 50] * 40), ms=1, sr=4000)
    assert len(e) == 40 and max(e) == 1.0, (len(e), max(e))
    assert sentences("가. 나다.  . 라") == ["가", "나다", "라"]
    print("selftest ok")


def main():
    os.makedirs(OUT, exist_ok=True)
    man = json.load(open(os.path.join(T10, "t10_manifest.json"), encoding="utf-8"))
    refs = {it["id"]: it["ref_text"] for it in man["items"]}
    pages, refmap = [], {}
    for spk, fid in EVAL.items():
        wav = os.path.join(T10, "t10_16k", fid + ".wav")
        w = wave.open(wav, "rb")
        n = w.getnframes()
        w.close()
        refmap[spk] = sentences(refs[fid])
        for i, a, b in windows(n):
            s, raw = read_slice(wav, a, b)
            name = "%s_W%02d.wav" % (spk, i)
            o = wave.open(os.path.join(OUT, name), "wb")
            o.setnchannels(1); o.setsampwidth(2); o.setframerate(SR)
            o.writeframes(raw)
            o.close()
            pages.append(dict(speaker=spk, bin=i, wid="W%02d" % i, file=name, source_id=fid,
                              window_start=round(a / SR, 2), window_sec=round((b - a) / SR, 2),
                              env=envelope(s)))
    json.dump(dict(created="2026-09-29", rule=dict(bins=BINS, window_sec=WINDOW_SEC, env_ms=ENV_MS,
                                                   selection="timeline only, ASR-independent"),
                   refs=refmap, pages=[{k: v for k, v in p.items() if k != "env"} for p in pages]),
              open(os.path.join(HERE, "windows.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    pre = None
    pp = os.path.join(HERE, "prefill.json")
    if os.path.isfile(pp):
        pre = json.load(open(pp, encoding="utf-8"))
        print("prefilled %d sentence choices from prefill.json" % len(pre.get("marks", [])))
    write_html(pages, refmap, pre)
    print("%d windows (%d per speaker), %.1f min of audio"
          % (len(pages), BINS, sum(p["window_sec"] for p in pages) / 60))
    for spk in EVAL:
        print("  %s: %d reference sentences to choose from" % (spk, len(refmap[spk])))


def write_html(pages, refmap, prefill=None):
    pre = {(m["speaker"], m["window"]): m for m in (prefill or {}).get("marks", [])}
    H = ['<meta charset="utf-8"><title>발화 경계 표시 2026-09-29</title>', """<style>
body{font:15px/1.6 -apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo",sans-serif;
 max-width:980px;margin:0 auto;padding:22px 18px 110px;color:#111;background:#fff}
h1{font-size:21px;margin:0 0 6px} h2{font-size:15px;margin:28px 0 8px;color:#666;
 border-bottom:1px solid #ddd;padding-bottom:4px}
.card{border:1px solid #ddd;border-radius:6px;padding:12px 14px;margin:14px 0}
.card h3{font-size:15px;margin:0 0 8px}
svg{width:100%;height:110px;display:block;background:#fafafa;border:1px solid #e3e3e3;
 border-radius:4px;cursor:crosshair;touch-action:none}
.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:8px}
audio{flex:1 1 320px;min-width:260px}
select{max-width:100%;font:13px inherit;padding:5px;border:1px solid #ccc;border-radius:4px}
input[type=text]{flex:1 1 220px;font:13px inherit;border:1px solid #ccc;border-radius:4px;padding:5px 7px}
button{font:13px inherit;padding:6px 12px;border:1px solid #111;background:#fff;color:#111;
 border-radius:4px;cursor:pointer}
button.p{background:#111;color:#fff}
.sel{font:12px ui-monospace,Menlo,monospace;color:#444;background:#f4f4f4;padding:3px 7px;border-radius:3px}
#bar{position:fixed;bottom:0;left:0;right:0;background:#fff;border-top:2px solid #111;
 padding:10px 18px;text-align:center} #n{font-size:13px;color:#666;margin-right:12px}
label.cf{font-size:13px;margin-right:10px}
</style>""",
         "<h1>발화 경계 표시 — 2026-09-29</h1>",
         """<p>창 20개(화자당 10개), 각 50초. <b>창마다 발화 하나를 골라 경계를 표시하고, 그게 정답
문장 중 어느 것인지 고르면 됩니다.</b> 전사를 타이핑하는 게 아닙니다 — 정답 텍스트는 원래 라벨을 그대로 씁니다.</p>
<p><b>경계 잡는 법 두 가지 — 편한 쪽으로.</b></p>
<ol style="margin:6px 0 10px 20px">
<li><b>재생하면서 버튼</b>(추천): 재생하다가 발화가 시작되는 지점에서 <b>◀ 여기가 시작</b>,
끝나는 지점에서 <b>여기가 끝 ▶</b>. 일시정지 상태에서도 현재 재생 위치 기준으로 잡힙니다.</li>
<li><b>파형 드래그</b>: 파형 위를 가로로 <b>끌면</b> 됩니다. 한 번 클릭만 하면 시작만 잡히고
빨간 글씨로 "끝을 지정하세요"가 뜹니다.</li>
</ol>
<p><b>선택 재생</b>으로 잡은 구간만 들어보고 확인하세요. 경계는 ±0.3초면 충분합니다.
앞뒤 여백은 괜찮고, 말이 잘리는 게 문제입니다. 카드 테두리가 <b>초록</b>이면 그 창은 완료입니다.</p>
<p>중요: <b>잘 안 읽힌 발화, 반복해서 읽은 발화, 실패한 발화도 그대로 표시하세요.</b> 잘 된 것만 고르면
평가셋이 쉬운 쪽으로 편향되고, 그게 이 측정을 무의미하게 만듭니다.
쓸 만한 발화가 없는 창은 문장 선택에서 <b>해당 없음</b>을 고르면 됩니다 — 그것도 기록됩니다.</p>
<p style="font-size:13px;color:#888">다 하고 맨 아래 <b>복사</b> → 대화창에 붙여넣기.</p>"""]
    for spk in EVAL:
        H.append("<h2>%s — 창 10개</h2>" % spk)
        for p in [x for x in pages if x["speaker"] == spk]:
            H.append(card(p, refmap[spk], pre.get((spk, p["wid"]))))
    H.append("""<div id=bar><span id=n></span><button class=p onclick="cp()">복사</button></div>
<script>
const CARDS=[...document.querySelectorAll('.card')];
function done(c){const a=parseFloat(c.dataset.s),b=parseFloat(c.dataset.e),v=c.querySelector('select').value;
 if(v==='none')return true;                       // no usable utterance: no boundary needed
 return v!==''&&!isNaN(a)&&!isNaN(b)&&b>a;}
function upd(){const d=CARDS.filter(done).length;
 CARDS.forEach(c=>{c.style.borderColor=done(c)?'#8bc34a':'#ddd';});
 document.getElementById('n').textContent=d+' / '+CARDS.length+' 완료';}
CARDS.forEach(c=>{
 const svg=c.querySelector('svg'), au=c.querySelector('audio'), out=c.querySelector('.sel');
 const W=parseFloat(c.dataset.w); let dragging=false, x0=0;
 const px=e=>{const r=svg.getBoundingClientRect();
   return Math.max(0,Math.min(1,((e.touches?e.touches[0].clientX:e.clientX)-r.left)/r.width));};
 const draw=()=>{const a=parseFloat(c.dataset.s||'NaN'),b=parseFloat(c.dataset.e||'NaN');
   const r=svg.querySelector('.selrect');
   if(isNaN(a)||isNaN(b)||b<=a){r.setAttribute('width',0);
     out.textContent=isNaN(a)?'구간 없음':'시작만 잡힘 ('+a.toFixed(2)+'s) — 끝을 지정하세요';
     out.style.background=isNaN(a)?'#f4f4f4':'#ffe6e6';return;}
   out.style.background='#e8f5e9';
   r.setAttribute('x',1000*a/W); r.setAttribute('width',1000*(b-a)/W);
   out.textContent=a.toFixed(2)+' – '+b.toFixed(2)+' s  (길이 '+(b-a).toFixed(2)+'s)';};
 const start=e=>{dragging=true;x0=px(e)*W;c.dataset.s=x0;c.dataset.e=x0;draw();e.preventDefault();};
 const move=e=>{if(!dragging)return;const t=px(e)*W;
   c.dataset.s=Math.min(x0,t);c.dataset.e=Math.max(x0,t);draw();e.preventDefault();};
 const end=()=>{if(dragging){dragging=false;upd();}};
 svg.addEventListener('mousedown',start); svg.addEventListener('touchstart',start);
 window.addEventListener('mousemove',move); window.addEventListener('touchmove',move,{passive:false});
 window.addEventListener('mouseup',end); window.addEventListener('touchend',end);
 c.querySelector('.play').onclick=()=>{const a=parseFloat(c.dataset.s),b=parseFloat(c.dataset.e);
   if(isNaN(a)||b<=a){alert('먼저 파형을 드래그해서 구간을 잡으세요.');return;}
   au.currentTime=a; au.play();
   const stop=()=>{if(au.currentTime>=b){au.pause();au.removeEventListener('timeupdate',stop);}};
   au.addEventListener('timeupdate',stop);};
 const setEdge=which=>()=>{const t=au.currentTime;
   if(!t&&au.paused){alert('먼저 재생해서 원하는 지점까지 들은 다음 누르세요.');return;}
   if(which==='a'){c.dataset.s=t; if(parseFloat(c.dataset.e||'NaN')<=t) c.dataset.e='';}
   else{ if(c.dataset.s===''){alert('시작을 먼저 잡으세요.');return;}
         if(t<=parseFloat(c.dataset.s)){alert('끝이 시작보다 앞입니다.');return;} c.dataset.e=t;}
   draw();upd();};
 c.querySelector('.setA').onclick=setEdge('a');
 c.querySelector('.setB').onclick=setEdge('b');
 c.querySelector('.clr').onclick=()=>{c.dataset.s='';c.dataset.e='';draw();upd();};
 c.querySelector('select').addEventListener('change',upd);
 draw();
});
upd();
function cp(){const bad=CARDS.filter(c=>!done(c));
 if(bad.length&&!confirm(bad.length+'개가 아직 미완입니다('+bad.map(c=>c.dataset.spk+c.dataset.wid).join(', ')
   +').\\n\\n그래도 복사할까요? 경계가 없는 창은 평가에서 빠집니다.')) return;
 const o=CARDS.map(c=>{const sel=c.querySelector('select');
 const a=parseFloat(c.dataset.s),b=parseFloat(c.dataset.e),w=parseFloat(c.dataset.w0);
 return {speaker:c.dataset.spk,window:c.dataset.wid,
  ref_index:sel.value===''?null:(sel.value==='none'?'none':parseInt(sel.value)),
  start_sec:isNaN(a)?null:+(w+a).toFixed(2), end_sec:isNaN(b)?null:+(w+b).toFixed(2),
  confidence:(c.querySelector('input[type=radio]:checked')||{}).value||null,
  note:c.querySelector('input[type=text]').value||null};});
 const t=JSON.stringify({kit:'utterance_marking_2026-09-29',marks:o},null,1);
 navigator.clipboard.writeText(t).then(()=>alert('복사됐다. 대화창에 붙여넣어라.'),
  ()=>{const w=window.open();w.document.body.innerHTML='<pre>'+t.replace(/</g,'&lt;')+'</pre>';});}
</script>""")
    open(os.path.join(HERE, "index.html"), "w", encoding="utf-8").write("\n".join(H))


def card(p, refs, pre=None):
    env = p["env"]
    n = len(env)
    pts = " ".join("%.1f,%.1f" % (1000.0 * i / n, 55 - 53 * v) for i, v in enumerate(env))
    pts2 = " ".join("%.1f,%.1f" % (1000.0 * i / n, 55 + 53 * v) for i, v in reversed(list(enumerate(env))))
    ticks = "".join(
        '<line x1="%.1f" y1="0" x2="%.1f" y2="110" stroke="#ddd"/>'
        '<text x="%.1f" y="12" font-size="9" fill="#999">%ds</text>'
        % (1000.0 * t / p["window_sec"], 1000.0 * t / p["window_sec"],
           1000.0 * t / p["window_sec"] + 2, int(p["window_start"] + t))
        for t in range(0, int(p["window_sec"]) + 1, 10))
    sel = (pre or {}).get("ref_index")
    def o(v, label):
        return '<option value="%s"%s>%s</option>' % (
            v, " selected" if str(sel) == str(v) and sel is not None else "", label)
    opts = ('<option value=""%s>— 정답 문장 선택 —</option>' % ("" if sel is not None else " selected")
            + o("none", "해당 없음 (쓸 만한 발화 없음)")
            + "".join(o(i, "%d. %s" % (i + 1, r.replace("<", "&lt;")[:60])) for i, r in enumerate(refs)))
    note = (pre or {}).get("note") or ""
    return ('<div class="card" data-spk="%s" data-wid="%s" data-w="%.2f" data-w0="%.2f" '
            'data-s="" data-e="">'
            '<h3>%s %s — 원본 %.1f초부터 %.0f초 구간</h3>'
            # the selection rect paints AFTER the waveform so it tints it instead of hiding behind it
            '<svg viewBox="0 0 1000 110" preserveAspectRatio="none">%s'
            '<polygon points="%s %s" fill="#2a6fb0"/>'
            '<rect class="selrect" x="0" y="0" width="0" height="110" fill="#ffb300" opacity="0.38"/>'
            '</svg>'
            '<div class="row"><audio controls preload=none src="windows/%s"></audio>'
            '<button class="setA">◀ 여기가 시작</button><button class="setB">여기가 끝 ▶</button>'
            '<button class="play">선택 재생</button><button class="clr">지우기</button>'
            '<span class="sel">구간 없음</span></div>'
            '<div class="row"><select>%s</select></div>'
            '<div class="row">'
            '<label class="cf"><input type=radio name="cf_%s%s" value="sure"> 확실</label>'
            '<label class="cf"><input type=radio name="cf_%s%s" value="unsure"> 애매</label>'
            '<input type=text value="%s" placeholder="메모(선택) — 반복 낭독, 실패한 시도, 이상한 점"></div>'
            '</div>') % (p["speaker"], p["wid"], p["window_sec"], p["window_start"],
                         p["speaker"], p["wid"], p["window_start"], p["window_sec"],
                         ticks, pts, pts2, p["file"], opts,
                         p["speaker"], p["wid"], p["speaker"], p["wid"],
                         note.replace('"', "&quot;"))


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        selftest()
        main()
