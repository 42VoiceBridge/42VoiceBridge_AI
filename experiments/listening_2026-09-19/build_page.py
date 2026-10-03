#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build index.html for the 2026-09-19 listening review. Reads clips.json + labels + hypotheses."""
import glob
import json
import os
import re
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(os.path.dirname(HERE))

TASKS = {
 "A1a_KEJ-02-04_start": ("A. 설계 차단 해제", "KEJ 02-04 — 앞 75초 (전체 16.2분)",
   "라벨 대비 오디오 길이 비율이 1.788이다. 라벨이 1,091자인데 녹음이 그보다 훨씬 길다.",
   ["들리는 말이 라벨 문장들과 대체로 이어지는가",
    "같은 문장을 두 번 이상 반복해 읽는가",
    "읽기가 아닌 것(지시·잡담·장비 소리)이 섞여 있는가"]),
 "A1b_KEJ-02-04_mid": ("A. 설계 차단 해제", "KEJ 02-04 — 중간 75초 (8:00~9:15)",
   "중간이 통째로 비었거나 중복됐는지 본다.", ["말이 계속 이어지는가, 긴 무음이 있는가", "앞 클립과 같은 내용이 또 나오는가"]),
 "A1c_KEJ-02-04_end": ("A. 설계 차단 해제", "KEJ 02-04 — 마지막 75초",
   "끝이 잘렸는지, 라벨에 없는 내용이 붙었는지 본다.", ["문장 중간에 녹음이 끊기는가", "끝부분이 읽기가 아닌 다른 것인가"]),
 "A2_CYW-04-1000_tail": ("A. 설계 차단 해제", "CYW 04-1000 — 뒤 29초 (전체 47.4초)",
   "라벨이 127자로 동료 중앙값 180자보다 짧다. large-v3는 47.4초 파일에서 46.74초에 "
   "'미영이는 음악에 어울리는'으로 문장 중간에 멈췄다.",
   ["녹음이 문장 중간에 그냥 끊기는가", "끊기기 전까지 발화가 정상인가"]),
 "B1_KEJ-04-6000_all": ("B. KEJ 0.520의 정체", "KEJ 04-6000 — 전체 64.7초 (최악 파일)",
   "자모 CER 0.520이 무엇에서 오는지 귀로 확인한다. 측정치는 있지만 아무도 들어본 적이 없다.",
   ["발음이 뭉개져서 못 알아듣는가, 아니면 알아들을 만한데 인식기가 틀리는가",
    "본인은 몇 퍼센트나 알아들을 수 있는가 (대략)",
    "녹음 품질 문제(잡음·클리핑·마이크 거리)가 있는가"]),
 "B2_KEJ-04-5000_tail": ("B. KEJ 0.520의 정체", "KEJ 04-5000 — 뒤 33.6초",
   "인식기가 라벨에 없는 구절을 뱉었다(CYW·KJW·KEJ 세 화자 공통). 네 가지가 열려 있다: "
   "라벨 누락 / 모델 환각 / 다른 사람 목소리 / 판정 불가.",
   ["라벨에 없는 내용을 실제로 읽고 있는가 (→ 라벨 누락)",
    "그 구간이 무음이거나 잡음뿐인가 (→ 모델 환각)",
    "다른 사람 목소리인가"]),
 "B3_KEJ-04-8000_start": ("B. KEJ 0.520의 정체", "KEJ 04-8000 — 앞 60초 (전체 92.7초)",
   "large-v3가 '모리 히베르 하루 부시간맛 도암맛 비빌리버스…'처럼 한국어가 아닌 소리를 뱉었다.",
   ["실제 발화가 있는가, 무음/잡음인가", "발화가 있다면 라벨 내용과 관계가 있는가"]),
 "C1_KEJ-02-03_start": ("C. T10 실현 가능성", "KEJ 02-03 — 앞 90초 (전체 19.7분, 문장 낭독)",
   "**이게 가장 중요하다.** KEJ에는 06-01(파일럿에서 쓴 대화체 낭독)이 없다. 고CER 화자 "
   "적응을 하려면 02-03 또는 02-04를 써야 하는데, 문장 단위로 자동 분절이 되는지가 관건이다.",
   ["문장과 문장 사이에 무음이 뚜렷한가 (0.5초 이상)",
    "한 문장을 여러 번 고쳐 읽는가",
    "문장 경계를 사람이 듣고 찍을 수 있겠는가"]),
 "D1_CYW-04-5000_halluc": ("D. 환각 확인", "CYW 04-5000 — 103~115초",
   "large-v3가 112.87초에 '영상편집 및 자료조사 김재경 기상캐스터'를 뱉었다. 방송 크레딧 환각으로 보인다.",
   ["그 시점에 소리가 있는가, 무음인가", "있다면 무슨 말인가"]),
}


def labels():
    out = {}
    for f in glob.glob(os.path.join(PROJECT, "013.구음장애 음성인식 데이터", "**", "*.json"),
                       recursive=True):
        n = unicodedata.normalize("NFC", os.path.basename(f))
        if "-13-" in n:
            out[n[:-5]] = f
    return out


def main():
    clips = json.load(open(os.path.join(HERE, "clips.json"), encoding="utf-8"))
    lab = labels()
    hyp = json.load(open(os.path.join(PROJECT, "experiments/b0/b0_result/hyp_large-v3.json"),
                         encoding="utf-8"))
    rows = []
    for c in clips:
        sid = unicodedata.normalize("NFC", c["source"])[:-4]
        ref = ""
        if sid in lab:
            d = json.load(open(lab[sid], encoding="utf-8"))
            ref = re.sub(r"[+*]", "", d.get("Transcript", "")).strip()
        h = ""
        if sid in hyp:
            segs = [s for s in hyp[sid]["segments"] if s["end"] > c["start"] and s["start"] < c["end"]]
            h = " ".join("[%.1f] %s" % (s["start"], s["text"].strip()) for s in segs)
        grp, title, why, qs = TASKS[c["id"]]
        rows.append(dict(c, group=grp, title=title, why=why, questions=qs, ref=ref, hyp=h))

    html = ["""<meta charset="utf-8"><title>청취 검수 2026-09-19</title>
<style>
body{font:15px/1.65 -apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo",sans-serif;
 max-width:860px;margin:0 auto;padding:24px;color:#111;background:#fff}
h1{font-size:22px;margin:0 0 4px} h2{font-size:15px;margin:32px 0 8px;color:#666;
 border-bottom:1px solid #ddd;padding-bottom:4px;font-weight:600}
.card{border:1px solid #ddd;border-radius:6px;padding:14px 16px;margin:14px 0}
.card h3{font-size:16px;margin:0 0 6px}
.why{background:#f6f6f6;padding:8px 10px;border-radius:4px;font-size:14px;margin:8px 0}
audio{width:100%;margin:8px 0}
.txt{font-size:13px;color:#444;background:#fafafa;border-left:3px solid #ccc;
 padding:6px 10px;margin:6px 0;max-height:130px;overflow:auto;white-space:pre-wrap}
.lbl{font-size:12px;color:#888;font-weight:600}
ul{margin:8px 0 6px 0;padding-left:20px} li{margin:3px 0}
textarea{width:100%;min-height:64px;font:13px/1.5 ui-monospace,Menlo,monospace;
 border:1px solid #ccc;border-radius:4px;padding:8px;box-sizing:border-box}
#out{position:sticky;bottom:0;background:#fff;border-top:2px solid #111;padding:10px 0;margin-top:30px}
button{font:14px inherit;padding:7px 14px;border:1px solid #111;background:#111;color:#fff;
 border-radius:4px;cursor:pointer}
.meta{font-size:12px;color:#888}
</style>
<h1>청취 검수 — 2026-09-19</h1>
<p class=meta>클립 9개, 약 8분 분량. 판단만 적으면 된다. <b>안 들리거나 모르겠으면 "모르겠음"이 정답이다</b>
— 추측해서 적으면 그게 데이터가 되어버린다. 다 적고 맨 아래 <b>복사</b>를 눌러 붙여넣으면 된다.</p>"""]
    last = None
    for i, r in enumerate(rows):
        if r["group"] != last:
            html.append("<h2>%s</h2>" % r["group"])
            last = r["group"]
        html.append('<div class=card><h3>%s</h3>' % r["title"])
        html.append('<div class=why>%s</div>' % r["why"])
        html.append('<audio controls preload=none src="clips/%s.wav"></audio>' % r["id"])
        html.append('<div class=meta>%s · %.1f초 구간 [%.1f–%.1f] / 원본 %.1f초</div>'
                    % (r["source"], r["sec"], r["start"], r["end"], r["source_sec"]))
        if r["ref"]:
            html.append('<div class=lbl>라벨(의도 기준 전사, 파일 전체)</div><div class=txt>%s</div>'
                        % r["ref"][:1200])
        if r["hyp"]:
            html.append('<div class=lbl>large-v3 인식 결과(이 구간)</div><div class=txt>%s</div>' % r["hyp"])
        html.append('<div class=lbl>판단할 것</div><ul>%s</ul>'
                    % "".join("<li>%s</li>" % q for q in r["questions"]))
        html.append('<textarea id="a%d" placeholder="여기에 답. 모르겠으면 모르겠음."></textarea></div>' % i)
    html.append("""<div id=out><button onclick="cp()">복사</button>
 <span id=msg class=meta></span></div><script>
function cp(){const rs=%s;let s="청취 검수 결과 2026-09-19\\n";
rs.forEach((r,i)=>{const v=document.getElementById('a'+i).value.trim();
s+="\\n["+r+"]\\n"+(v||"(미기입)")+"\\n";});
navigator.clipboard.writeText(s).then(()=>{document.getElementById('msg').textContent="복사됨 — 붙여넣으세요";});}
</script>""" % json.dumps([r["id"] for r in rows], ensure_ascii=False))
    open(os.path.join(HERE, "index.html"), "w", encoding="utf-8").write("\n".join(html))
    print("index.html written,", len(rows), "clips")


if __name__ == "__main__":
    main()
