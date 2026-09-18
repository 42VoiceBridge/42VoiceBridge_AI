#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B0 후속 측정 2건. 코랩에서 실행한다.

    !python /content/drive/MyDrive/b0b_run.py

━━ A. KEJ VAD ON/OFF ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
KEJ의 자모 CER 0.520이 정말 인식 실패인지, 아니면 VAD가 약한 발화를 잘라낸
탓인지 가른다. KEJ의 발화비율이 37.9%로 낮았다 (CYW 70%, KJW 78%).
파일별 개선/악화 개수와 자모·음절 두 지표를 함께 낸다. 자동 판정은 하지 않는다 —
임계값은 통계 검정이 아니고, 한 화자의 8파일은 8개 독립 표본이 아니다.
기존 b0_16k 폴더를 그대로 쓴다. 새로 올릴 것 없다.

━━ B. 자유발화 · 대화체 B0 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
04 낭독에서 large-v3가 이미 잘한다(CTK 0.020). 그래서 낭독이 아닌 발화를 잰다.
  03    자기소개 — 고정 질문 25개 응답. 템플릿이 있어 완전 자유발화가 아니다.  CTK · JMS · KEJ
  06-01 대화체 짧은 문장 **낭독**. 자연 대화가 아니다.                    CYU · KJW
CTK · KEJ · KJW는 04에도 있어 **같은 화자 안에서 낭독 대 비낭독**을 비교할 수 있다.
이게 이번 측정의 요점이다.

입력
  /content/drive/MyDrive/b0_16k/      (A용, 이미 올려둔 것)
  /content/drive/MyDrive/b0b_16k/     (B용, 새로 올릴 것. wav 5개 + manifest.json)
  /content/drive/MyDrive/b0_run.py    (CER 유틸을 여기서 가져온다)
출력
  /content/drive/MyDrive/b0b_results/
"""

import csv
import gc
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

DATA_A = os.environ.get("B0_DATA", "/content/drive/MyDrive/b0_16k")
DATA_B = os.environ.get("B0B_DATA", "/content/drive/MyDrive/b0b_16k")
OUT = os.environ.get("B0B_OUT", "/content/drive/MyDrive/b0b_results")
PREV = os.environ.get("B0_PREV", "/content/drive/MyDrive/b0_results/b0_per_file.csv")

MODEL_A = "large-v3"          # VAD 비교는 헤드라인 모델로
MODELS_B = ["small", "large-v3"]


def load_utils():
    try:
        from b0_run import (cer, norm_syl, to_jamo, save_json,
                            detect_device, selftest)     # noqa: F401
        return cer, norm_syl, save_json, detect_device, selftest
    except ImportError as e:
        sys.exit("b0_run.py 를 못 찾았다 (%s).\n"
                 "  이 스크립트와 같은 폴더(드라이브 최상위)에 b0_run.py 가 있어야 한다." % e)


cer, norm_syl, save_json, detect_device, selftest = load_utils()


def foreign(s):
    """한글이 아닌 글자 수. norm_syl 이 지워버리는 출력을 원문 기준으로 센다."""
    import re as _re
    return len(_re.findall(r"[\u3040-\u30ff\u4e00-\u9fffA-Za-z]", s))


def transcribe(model, path, vad):
    kw = dict(language="ko", task="transcribe", beam_size=5,
              condition_on_previous_text=False, word_timestamps=True)
    if vad:
        kw["vad_filter"] = True
        kw["vad_parameters"] = dict(min_silence_duration_ms=500)
    else:
        kw["vad_filter"] = False
    segs, info = model.transcribe(path, **kw)
    S = [dict(start=s.start, end=s.end, text=s.text,
              words=[dict(w=w.word, s=w.start, e=w.end) for w in (s.words or [])])
         for s in segs]
    return dict(segments=S, text="".join(x["text"] for x in S),
                duration=getattr(info, "duration", None),
                duration_after_vad=getattr(info, "duration_after_vad", None))


def new_model(size, device, compute):
    from faster_whisper import WhisperModel
    return WhisperModel(size, device=device, compute_type=compute,
                        cpu_threads=os.cpu_count() or 2)


def drop(device):
    """호출부가 먼저 참조를 끊고(model = None) 이걸 부른다.
    인자로 받아 지우면 호출부 참조가 남아 다음 모델을 올릴 때 둘이 동시에 메모리에 있다."""
    gc.collect()
    if device == "cuda":
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


# ══════════════════════════════════════════════════════ A. KEJ VAD ON/OFF

def phase_a(device, compute):
    mp = os.path.join(DATA_A, "manifest.json")
    if not os.path.exists(mp):
        print("  b0_16k/manifest.json 이 없다. A단계를 건너뛴다.")
        return None
    items = [i for i in json.load(open(mp, encoding="utf-8"))["items"]
             if i["speaker"] == "KEJ" and not i["defect"]]
    if not items:
        print("  KEJ 파일이 없다. A단계를 건너뛴다.")
        return None
    print("\n" + "=" * 70)
    print("A. KEJ VAD ON/OFF  (%s, %d파일 %.1f분)"
          % (MODEL_A, len(items), sum(i["play_sec"] for i in items) / 60))
    print("=" * 70)
    model = new_model(MODEL_A, device, compute)
    res = {"on": {}, "off": {}}
    t0 = time.time()
    for n, it in enumerate(items, 1):
        p = os.path.join(DATA_A, it["out"])
        for mode, vad in (("on", True), ("off", False)):
            res[mode][it["id"]] = transcribe(model, p, vad)
        save_json(res, os.path.join(OUT, "partial_kej_vad.json"))
        a = res["on"][it["id"]]
        b = res["off"][it["id"]]
        print("  [%d/%d] %-8s ON 구간%3d / OFF 구간%3d   누적 %.1f분"
              % (n, len(items), it["task"], len(a["segments"]), len(b["segments"]),
                 (time.time() - t0) / 60), flush=True)
    model = None
    drop(device)
    save_json(res, os.path.join(OUT, "hyp_kej_vad.json"))

    rows = []
    print("\n  %-9s %8s %8s %8s %8s %9s" % ("과제", "정답자", "ON자", "OFF자", "ON CER", "OFF CER"))
    for it in items:
        ref = it["ref_text"]
        ca, nj = cer(ref, res["on"][it["id"]]["text"], jamo=True)
        cb, _ = cer(ref, res["off"][it["id"]]["text"], jamo=True)
        sa1, _ = cer(ref, res["on"][it["id"]]["text"])
        sb1, _ = cer(ref, res["off"][it["id"]]["text"])
        ha = len(norm_syl(res["on"][it["id"]]["text"]))
        hb = len(norm_syl(res["off"][it["id"]]["text"]))
        d = res["on"][it["id"]]
        rows.append(dict(file=it["id"], task=it["task"], ref_jamo=nj,
                         ref_chars=it["ref_chars"], hyp_on=ha, hyp_off=hb,
                         cer_on=ca, cer_off=cb, cer_syl_on=sa1, cer_syl_off=sb1,
                         foreign_on=foreign(res["on"][it["id"]]["text"]),
                         foreign_off=foreign(res["off"][it["id"]]["text"]),
                         vad_ratio=(round(d["duration_after_vad"] / d["duration"], 3)
                                    if d.get("duration") and d.get("duration_after_vad") else "")))
        print("  %-9s %8d %8d %8d %8.3f %9.3f" % (it["task"], it["ref_chars"], ha, hb, ca, cb))
    pa = sum(r["cer_on"] * r["ref_jamo"] for r in rows) / sum(r["ref_jamo"] for r in rows)
    pb = sum(r["cer_off"] * r["ref_jamo"] for r in rows) / sum(r["ref_jamo"] for r in rows)
    # 음절 기준도 함께 낸다. 자모와 부호가 갈릴 수 있다.
    sa = sum(r["cer_syl_on"] * r["ref_chars"] for r in rows) / sum(r["ref_chars"] for r in rows)
    sb = sum(r["cer_syl_off"] * r["ref_chars"] for r in rows) / sum(r["ref_chars"] for r in rows)
    imp = sum(1 for r in rows if r["cer_off"] < r["cer_on"])
    print("\n  pooled 자모  ON %.4f  OFF %.4f  %+.4f" % (pa, pb, pb - pa))
    print("  pooled 음절  ON %.4f  OFF %.4f  %+.4f" % (sa, sb, sb - sa))
    print("  파일별: OFF가 좋아진 파일 %d / 나빠진 파일 %d (총 %d)" % (imp, len(rows) - imp, len(rows)))
    # 인과·유의성 판정을 자동으로 쓰지 않는다. 임계값은 통계 검정이 아니고
    # 한 화자의 여러 파일은 독립 표본이 아니다. 해석은 사람이 한다.
    v = ("자모 %+.4f / 음절 %+.4f, OFF 개선 %d개·악화 %d개. "
         "원인이나 ON/OFF 선택 근거로 쓰지 말 것. 잘려나간 구간을 들어야 한다."
         % (pb - pa, sb - sa, imp, len(rows) - imp))
    print("  요약: " + v)
    with open(os.path.join(OUT, "kej_vad.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            r2 = dict(r)
            for k in ("cer_on", "cer_off", "cer_syl_on", "cer_syl_off"):
                r2[k] = round(r[k], 4)
            w.writerow(r2)
    return dict(cer_jamo_on=round(pa, 4), cer_jamo_off=round(pb, 4),
                cer_syl_on=round(sa, 4), cer_syl_off=round(sb, 4),
                delta_jamo=round(pb - pa, 4), delta_syl=round(sb - sa, 4),
                files_improved_off=imp, files_worsened_off=len(rows) - imp,
                note=v, n_files=len(rows))


# ══════════════════════════════════════════════════════ B. 자유발화 · 대화체

def read_prev():
    """04 낭독 B0 결과를 읽어 화자별 자모 CER(가중)을 돌려준다. 없으면 빈 dict."""
    if not os.path.exists(PREV):
        print("  (이전 결과 %s 없음 — 낭독 대비 비교는 생략)" % PREV)
        return {}
    out = {}
    rows = list(csv.DictReader(open(PREV, encoding="utf-8")))
    for m in set(r["model"] for r in rows):
        for sp in set(r["speaker"] for r in rows):
            g = [r for r in rows if r["model"] == m and r["speaker"] == sp
                 and r["defect"] != "True"]
            if not g:
                continue
            tot = sum(float(r["ref_jamo"]) for r in g)
            if tot:
                out[(m, sp)] = sum(float(r["cer_jamo"]) * float(r["ref_jamo"])
                                   for r in g) / tot
    return out


def phase_b(device, compute):
    mp = os.path.join(DATA_B, "manifest.json")
    if not os.path.exists(mp):
        print("  b0b_16k/manifest.json 이 없다. B단계를 건너뛴다.")
        return None
    items = json.load(open(mp, encoding="utf-8"))["items"]
    missing = [i["out"] for i in items
               if not os.path.exists(os.path.join(DATA_B, i["out"]))]
    if missing:
        print("  wav 누락: %s" % missing[:5])
        sys.exit("b0b_16k 업로드가 끝났는지 확인할 것.")
    print("\n" + "=" * 70)
    print("B. 자유발화 · 대화체 B0  (%d파일 %.1f분)"
          % (len(items), sum(i["play_sec"] for i in items) / 60))
    print("=" * 70)
    todo = MODELS_B if device == "cuda" else ["small"]
    res = {}
    for size in todo:
        print("\n  --- %s ---" % size)
        model = new_model(size, device, compute)
        r, t0 = {}, time.time()
        for n, it in enumerate(items, 1):
            r[it["id"]] = transcribe(model, os.path.join(DATA_B, it["out"]), True)
            save_json(r, os.path.join(OUT, "partial_b_%s.json" % size))
            d = r[it["id"]]
            ratio = ("%5.1f%%" % (100.0 * d["duration_after_vad"] / d["duration"])
                     if d.get("duration") and d.get("duration_after_vad") else "    -")
            print("    [%d/%d] %-4s %-7s %5.1f분  구간%4d  발화%s  누적 %.1f분"
                  % (n, len(items), it["speaker"], it["task"], it["play_sec"] / 60,
                     len(d["segments"]), ratio, (time.time() - t0) / 60), flush=True)
        model = None
        drop(device)
        res[size] = r
        save_json(r, os.path.join(OUT, "hyp_b_%s.json" % size))

    prev = read_prev()
    rows = []
    for size, r in res.items():
        for it in items:
            c, nj = cer(it["ref_text"], r[it["id"]]["text"], jamo=True)
            cs, _ = cer(it["ref_text"], r[it["id"]]["text"])
            d = r[it["id"]]
            rows.append(dict(model=size, file=it["id"], speaker=it["speaker"],
                             task=it["task"], kind=it["kind"],
                             minutes=round(it["play_sec"] / 60, 1),
                             ref_chars=it["ref_chars"], ref_jamo=nj,
                             hyp_chars=len(norm_syl(d["text"])),
                             vad_ratio=(round(d["duration_after_vad"] / d["duration"], 3)
                                        if d.get("duration") and d.get("duration_after_vad") else ""),
                             cer_jamo=c, cer_syl=cs,
                             cer_jamo_04=round(prev.get((size, it["speaker"]), float("nan")), 4)))
    with open(os.path.join(OUT, "b0b_per_file.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for x in rows:
            y = dict(x)
            y["cer_jamo"] = round(x["cer_jamo"], 4)
            y["cer_syl"] = round(x["cer_syl"], 4)
            w.writerow(y)

    print("\n  %-5s %-7s %-14s %7s %8s %9s %11s %9s"
          % ("화자", "과제", "유형", "정답자", "발화%", "자모CER", "04낭독CER", "배수"))
    for size in todo:
        print("  --- %s ---" % size)
        for x in [y for y in rows if y["model"] == size]:
            p = x["cer_jamo_04"]
            mult = ("%9.1fx" % (x["cer_jamo"] / p)) if p == p and p > 0 else "%9s" % "-"
            pp = ("%11.3f" % p) if p == p else "%11s" % "-"
            print("  %-5s %-7s %-14s %7d %7s%% %9.3f %s %s"
                  % (x["speaker"], x["task"],
                     "준자유발화" if x["kind"] == "semi_spontaneous" else "대화체낭독",
                     x["ref_chars"],
                     ("%.1f" % (100 * x["vad_ratio"])) if x["vad_ratio"] != "" else "-",
                     x["cer_jamo"], pp, mult))

    ref_m = "large-v3" if "large-v3" in todo else todo[0]
    print("\n  같은 화자 안에서 낭독 대 비낭독 (%s 기준)" % ref_m)
    for sp in ("CTK", "KEJ", "KJW"):
        g = [x for x in rows if x["speaker"] == sp and x["model"] == ref_m]
        if not g:
            continue
        x = g[0]
        p = x["cer_jamo_04"]
        if p == p:
            print("    %-5s 04 낭독 %.3f  →  %s %.3f   (%.1f배)"
                  % (sp, p, x["task"], x["cer_jamo"], x["cer_jamo"] / p if p else 0))
        else:
            print("    %-5s %s %.3f   (04 값 없음)" % (sp, x["task"], x["cer_jamo"]))

    print("\n  샘플 (앞 140자)")
    for it in items:
        print("\n  ───── %s / %s (%s) ─────" % (it["speaker"], it["task"], it["kind"]))
        print("    정답 :", it["ref_text"][:140])
        for size in todo:
            print("    %-9s:" % size, res[size][it["id"]]["text"][:140])

    def pooled(rs):
        t = sum(x["ref_jamo"] for x in rs)
        return sum(x["cer_jamo"] * x["ref_jamo"] for x in rs) / t if t else float("nan")

    summary = {}
    for size in todo:
        g = [x for x in rows if x["model"] == size]
        summary[size] = dict(
            pooled_all=round(pooled(g), 4),
            pooled_semi_spontaneous=round(pooled([x for x in g if x["kind"] == "semi_spontaneous"]), 4),
            pooled_dialogue=round(pooled([x for x in g if x["kind"] != "semi_spontaneous"]), 4),
            by_file={x["file"]: round(x["cer_jamo"], 4) for x in g})
    return summary


# ══════════════════════════════════════════════════════════════ 본체

def main():
    print("=" * 70)
    print("B0 후속 — A. KEJ VAD ON/OFF   B. 자유발화·대화체")
    print("=" * 70)
    try:
        import rapidfuzz, faster_whisper  # noqa: F401
        print("\n라이브러리 OK")
    except ImportError as e:
        sys.exit("설치 셀이 실패했다: %s" % e)
    selftest()
    device, compute, name = detect_device()
    print("장치: %s (%s / %s)" % (name, device, compute))
    if device == "cpu":
        print("  GPU 없음 — A는 large-v3라 CPU에서 몇 시간 걸린다. 권장하지 않는다.")
    os.makedirs(OUT, exist_ok=True)

    a = phase_a(device, compute)
    b = phase_b(device, compute)

    save_json(dict(date=time.strftime("%Y-%m-%d"), device=name,
                   kej_vad=a, free_and_dialogue=b),
              os.path.join(OUT, "b0b_summary.json"))
    print("\n" + "=" * 70)
    print("저장 위치: %s" % OUT)
    for fn in ["b0b_summary.json", "kej_vad.csv", "b0b_per_file.csv"]:
        print("  %s" % fn)
    print("이 셋을 내려받아 sw_challenge 폴더에 넣으면 된다.")
    print("=" * 70)


if __name__ == "__main__":
    main()
