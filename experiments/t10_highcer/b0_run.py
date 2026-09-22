#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B0 측정 — 적응 없는 Whisper가 구음장애 04 지문 세트에서 얼마나 틀리는지.

코랩에서 이렇게 실행한다:
    !python /content/drive/MyDrive/b0_run.py

v2 (2026-09-18). 외부 코드 검증에서 나온 지적을 반영했다.
  - 자모 CER 가중평균에 음절 수를 쓰던 버그 수정 (BUG)
  - 조각 종료 시각이 한 문자 앞에서 끊기던 버그 수정 (BUG)
  - 빈 items / 전부 결함 / 빈 정답에서 죽던 문제 방어 (BUG)
  - 파일별 체크포인트 저장 (자동 재개는 아니다)
  - VAD 전후 길이를 기록해 발화비율을 산출
  - 자동 경계는 "검수 후보"로만 표기. 학습용 정답 경계가 아니다.

정답(ref_text)에 대하여
  라벨 전사는 구축가이드라인상 **발화자의 의도 기준**으로 적힌 것이다. 실제로 난
  소리를 음성학적으로 옮긴 것이 아니다. `+`(반복) `*`(묻힘) 기호를 지웠다고 해서
  "실제 발화"가 되지 않는다. 여기서 재는 CER은 "ASR이 의도된 문장을 되살렸는가"이며,
  그것이 제품 관점에서 필요한 값이다. 청취 검증은 하지 않았다.

입력  : /content/drive/MyDrive/b0_16k/  (wav 31개 + manifest.json)
출력  : /content/drive/MyDrive/b0_results/
          hyp_<model>.json          인식 결과 (구간·단어 타임스탬프 포함)
          partial_<model>.json      진행 중 체크포인트
          b0_per_file.csv           파일별 CER
          b0_summary.json           요약
          combined_boundaries.json  04 합본 10조각 경계 (검수 후보)
"""

import csv
import difflib
import gc
import json
import os
import re
import sys
import time
import unicodedata

DATA = os.environ.get("B0_DATA", "/content/drive/MyDrive/b0_16k")
OUT = os.environ.get("B0_OUT", "/content/drive/MyDrive/b0_results")

COV_MIN = 0.4          # 조각 정렬 인정 최소 커버리지. 검증된 기준이 아니라 임의값이다.


# ══════════════════════════════════════════════════════════ 1. CER 유틸

CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"          # 19
JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"        # 21
JONG = " ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"  # 28 (첫 칸은 종성 없음)


def to_jamo(s):
    """한글 음절을 초성·중성·종성으로 분해한다.

    주의: 초성과 종성이 같은 호환 자모로 표현된다(예: 초성 ㄱ과 종성 ㄱ이 동일 문자).
    지금 정의의 자모 CER에는 문제없지만, 위치별 발음 혼동을 분석할 표현은 아니다.
    """
    out = []
    for c in s:
        o = ord(c) - 0xAC00
        if 0 <= o < 11172:
            out.append(CHO[o // 588])
            out.append(JUNG[(o % 588) // 28])
            j = JONG[o % 28]
            if j != " ":
                out.append(j)
        else:
            out.append(c)
    return "".join(out)


def norm_syl(s):
    """NFC 정규화 후 한글 음절만 남긴다. 공백·문장부호·숫자·영문 제거.

    숫자를 지우므로 표기 차이가 양방향으로 왜곡된다.
      정답 '한 시 삼십 분' / 인식 '1시 30분'  → '한시삼십분' vs '시분'  (과대평가)
      정답 '가'           / 인식 '가123'      → '가' vs '가'            (삽입이 사라짐)
    그래서 숫자가 있는 파일은 numeric_review 로 표시만 하고 점수는 건드리지 않는다.
    """
    return re.sub(r"[^가-힣]", "", unicodedata.normalize("NFC", s))


def count_digits(s):
    return len(re.findall(r"\d", s))


def _lev(a, b):
    """편집거리. rapidfuzz가 있으면 그걸 쓰고 없으면 직접 계산한다.

    직접 계산은 5,000자 쌍에서 4초쯤 걸린다. main() 앞에서 rapidfuzz 존재를
    확인하므로 실전에서는 이 경로로 오지 않는다.
    """
    try:
        from rapidfuzz.distance import Levenshtein
        return Levenshtein.distance(a, b)
    except ImportError:
        pass
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(ref, hyp, jamo=False):
    """(CER, 정답 토큰 수). jamo=True면 토큰 수는 자모 수다. 정답이 비면 (nan, 0)."""
    r, h = norm_syl(ref), norm_syl(hyp)
    if jamo:
        r, h = to_jamo(r), to_jamo(h)
    if not r:
        return float("nan"), 0
    return _lev(r, h) / len(r), len(r)


def selftest():
    assert (len(CHO), len(JUNG), len(JONG)) == (19, 21, 28)
    assert to_jamo("한글") == "ㅎㅏㄴㄱㅡㄹ", to_jamo("한글")
    assert to_jamo("아") == "ㅇㅏ"
    assert norm_syl("안녕, 1시 30분!") == "안녕시분"
    assert cer("안녕하세요", "안녕하세요")[0] == 0.0
    assert cer("안녕하세요", "")[0] == 1.0
    assert cer("가", "가", jamo=True)[1] == 2          # ㄱ+ㅏ
    assert cer("각", "각", jamo=True)[1] == 3          # ㄱ+ㅏ+ㄱ
    a = cer("안녕하세요", "안녕하십니다")[0]
    b = cer("안녕하세요", "안녕하십니다", jamo=True)[0]
    assert 0 < b < a, (a, b)
    print("  자기검사 통과  (초성19 중성21 종성28, 음절 CER %.3f / 자모 CER %.3f)" % (a, b))


# ══════════════════════════════════════════════════════════ 2. 저장 보조

def save_json(obj, path):
    """임시 파일에 쓰고 교체한다. 저장 중 죽어도 기존 파일이 잘리지 않는다."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)


def detect_device():
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda", "float16", torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "cpu", "int8", "CPU"


# ══════════════════════════════════════════════════════════ 3. 인식

def run_model(size, items, data_dir, device, compute):
    from faster_whisper import WhisperModel
    print("\n=== %s  (%s / %s) ===" % (size, device, compute))
    model = WhisperModel(size, device=device, compute_type=compute,
                         cpu_threads=os.cpu_count() or 2)
    ckpt = os.path.join(OUT, "partial_%s.json" % size)
    res, t0 = {}, time.time()
    for n, it in enumerate(items, 1):
        segs, info = model.transcribe(
            os.path.join(data_dir, it["out"]),
            language="ko", task="transcribe", beam_size=5,
            # 구음장애 발화에서 이 옵션이 켜져 있으면 한 번 잘못 인식한 뒤
            # 그 문맥에 갇혀 같은 말을 무한 반복 생성한다.
            condition_on_previous_text=False,
            # 무음이 많아 걸러내지 않으면 헛말을 지어낸다. 다만 VAD가 약한 발화를
            # 잘라내면 그 손실도 CER에 들어간다. 여기 값은 Whisper+VAD 파이프라인 성능이다.
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            word_timestamps=True,
        )
        S = []
        for s in segs:                       # 제너레이터라 여기서 실제 연산이 돈다
            S.append(dict(start=s.start, end=s.end, text=s.text,
                          words=[dict(w=w.word, s=w.start, e=w.end)
                                 for w in (s.words or [])]))
        dur = getattr(info, "duration", None)
        dur_vad = getattr(info, "duration_after_vad", None)
        res[it["id"]] = dict(segments=S, text="".join(x["text"] for x in S),
                             duration=dur, duration_after_vad=dur_vad)
        save_json(res, ckpt)                 # 파일마다 체크포인트. 자동 재개는 아니다.
        ratio = ("%5.1f%%" % (100.0 * dur_vad / dur)) if (dur and dur_vad) else "    -"
        print("  [%2d/%2d] %-4s %-8s %5.1f분  구간%4d  발화%s  누적 %.1f분"
              % (n, len(items), it["speaker"], it["task"],
                 it["play_sec"] / 60, len(S), ratio, (time.time() - t0) / 60),
              flush=True)
    del model
    gc.collect()
    try:
        import torch
        if device == "cuda":
            torch.cuda.empty_cache()
    except Exception:
        pass
    return res


# ══════════════════════════════════════════════════════════ 4. 04 합본 경계

def char_timeline(segs):
    """인식 결과를 (한글만 남긴 문자열, 각 문자의 시작 시각, 끝 시각) 으로 편다.

    단어 안에서 균등 보간한 값이다. 실제 음절 경계가 아니다.
    """
    chars, starts, ends = [], [], []
    for s in segs:
        ws = s["words"] or [dict(w=s["text"], s=s["start"], e=s["end"])]
        for w in ws:
            t = norm_syl(w["w"])
            if not t:
                continue
            span = w["e"] - w["s"]
            for k, c in enumerate(t):
                chars.append(c)
                starts.append(w["s"] + span * k / len(t))
                ends.append(w["s"] + span * (k + 1) / len(t))
    return "".join(chars), starts, ends


def locate(hyp_chars, piece, start=0, min_block=5):
    """piece 가 hyp_chars[start:] 의 어디에 있는지 찾는다.

    조각은 순서대로 읽혔으므로 직전 조각이 끝난 지점부터만 본다. 그러지 않으면
    짧은 조각(04-5500, 62자)이 뒤쪽 엉뚱한 자리의 우연한 일치까지 끌어와
    구간이 터무니없이 길어진다.

    반환 (a, b, cov). b 는 exclusive end 다.

    한계: 인식이 중간에 길게 헛말을 끼워 넣으면 span 필터가 뒤쪽 정상 일치를
    잘라내 커버리지가 낮게 나온다. 그 경우 그 조각은 미검출로 처리된다.
    """
    q = norm_syl(piece)
    win = hyp_chars[start:]
    if not win or not q:
        return None
    sm = difflib.SequenceMatcher(None, win, q, autojunk=False)
    b = [x for x in sm.get_matching_blocks() if x.size >= min_block]
    if not b:
        return None
    a0 = min(x.a for x in b)
    span = int(len(q) * 2.0) + 30
    b = [x for x in b if x.a - a0 <= span]
    cov = sum(x.size for x in b) / len(q)
    return a0 + start, max(x.a + x.size for x in b) + start, cov


def boundaries(items, res, canon, order):
    out = {}
    for it in items:
        if it["kind"] != "combined":
            continue
        hc, starts, ends = char_timeline(res[it["id"]]["segments"])
        print("\n───── %s   인식 %d자 / 정답 %d자 ─────"
              % (it["speaker"], len(hc), it["ref_chars"]))
        if not hc:
            print("  인식 결과가 비었다. 정렬 불가.")
            out[it["speaker"]] = []
            continue
        rec, prev, cursor = [], -1, 0
        for k in order:
            r = locate(hc, canon[k], start=cursor)
            if r is None:
                print("  %-8s 미검출" % k)
                rec.append(dict(piece=k, ok=False, reason="no_match"))
                continue
            a, b, cov = r
            if cov <= COV_MIN:
                # 커버리지가 낮은 후보로 커서를 전진시키면 뒤 조각까지 놓친다.
                print("  %-8s 커버 %5.1f%%  낮아서 버림" % (k, cov * 100))
                rec.append(dict(piece=k, ok=False, cov=round(cov, 3),
                                reason="low_coverage"))
                continue
            t0 = starts[min(a, len(starts) - 1)]
            t1 = ends[min(b - 1, len(ends) - 1)]
            mono = a > prev
            prev = a
            cursor = max(cursor, b - 10)
            rec.append(dict(piece=k, ok=True, cov=round(cov, 3),
                            start=round(t0, 3), end=round(t1, 3),
                            mono=mono, needs_review=True))
            print("  %-8s 커버 %5.1f%%  %6.2f~%6.2f분  %s"
                  % (k, cov * 100, t0 / 60, t1 / 60, "" if mono else "순서역전"))
        hit = sum(1 for r in rec if r.get("ok"))
        mono_all = all(r.get("mono", True) for r in rec)
        verdict = ("검수 후보" if (hit >= 8 and mono_all)
                   else ("부분 후보" if hit >= 5 else "정렬 실패"))
        print("  => 정렬 %d/10, 순서 단조 %s  →  %s" % (hit, mono_all, verdict))
        out[it["speaker"]] = rec
    return out


# ══════════════════════════════════════════════════════════ 5. 본체

def main():
    print("=" * 70)
    print("B0 측정 — 적응 없는 Whisper")
    print("=" * 70)

    print("\n[1] 라이브러리")
    try:
        from rapidfuzz.distance import Levenshtein  # noqa: F401
        print("  rapidfuzz OK")
    except ImportError:
        sys.exit("rapidfuzz 가 없다. 설치 셀이 실패했다.\n"
                 "  !pip install rapidfuzz  을 먼저 실행할 것.")
    try:
        import faster_whisper  # noqa: F401
        print("  faster-whisper OK")
    except ImportError:
        sys.exit("faster-whisper 가 없다. 설치 셀이 실패했다.")
    selftest()

    print("\n[2] 데이터")
    mp = os.path.join(DATA, "manifest.json")
    if not os.path.exists(mp):
        sys.exit("manifest.json 을 못 찾았다: %s\n"
                 "  드라이브에 b0_16k 폴더를 올렸는지, 위치가 맞는지 확인할 것." % mp)
    man = json.load(open(mp, encoding="utf-8"))
    items, canon, order = man["items"], man["canon"], man["canon_order"]

    if not items:
        sys.exit("manifest 의 items 가 비었다.")
    dup = [i for i in set(x["id"] for x in items)
           if sum(1 for x in items if x["id"] == i) > 1]
    if dup:
        sys.exit("id 가 중복됐다: " + ", ".join(sorted(dup)[:5]))
    if not any(not i["defect"] for i in items):
        sys.exit("결함 표시가 없는 파일이 하나도 없다. 평가할 대상이 없다.")
    blank = [i["id"] for i in items if not i["defect"] and not norm_syl(i["ref_text"])]
    if blank:
        sys.exit("정규화 후 정답이 비는 파일: " + ", ".join(blank))
    missing = [i["out"] for i in items
               if not os.path.exists(os.path.join(DATA, i["out"]))]
    print("  매니페스트 %d개 / 누락 %d개 / 총 %.2f시간"
          % (len(items), len(missing), sum(i["play_sec"] for i in items) / 3600))
    if missing:
        print("  누락:", missing[:5])
        sys.exit("wav 가 빠졌다. 업로드가 끝났는지 확인할 것.")

    print("\n[3] 장치")
    device, compute, name = detect_device()
    print("  %s  (%s / %s)" % (name, device, compute))
    if device == "cpu":
        print("  GPU 없음 — small 만 돌린다. large-v3 는 CPU로 며칠 걸린다.")

    os.makedirs(OUT, exist_ok=True)

    todo = ["small"] + (["large-v3"] if device == "cuda" else [])
    results = {}
    for size in todo:
        results[size] = run_model(size, items, DATA, device, compute)
        save_json(results[size], os.path.join(OUT, "hyp_%s.json" % size))
        print("  저장: hyp_%s.json" % size)

    # ---------------------------------------------------------------- 집계
    print("\n" + "=" * 70)
    print("결과")
    print("=" * 70)
    rows = []
    for tag, res in results.items():
        for it in items:
            hyp = res[it["id"]]["text"]
            c_syl, n_syl = cer(it["ref_text"], hyp)
            c_jamo, n_jamo = cer(it["ref_text"], hyp, jamo=True)
            dur, dvad = res[it["id"]].get("duration"), res[it["id"]].get("duration_after_vad")
            rows.append(dict(
                model=tag, file=it["id"], speaker=it["speaker"], task=it["task"],
                kind=it["kind"], batch16k=it["batch16k"], defect=it["defect"],
                minutes=round(it["play_sec"] / 60, 1),
                ref_chars=n_syl, ref_jamo=n_jamo,
                hyp_chars=len(norm_syl(hyp)),
                ref_digits=count_digits(it["ref_text"]), hyp_digits=count_digits(hyp),
                numeric_review=bool(count_digits(it["ref_text"]) or count_digits(hyp)),
                vad_sec=round(dvad, 1) if dvad else "",
                vad_ratio=round(dvad / dur, 3) if (dur and dvad) else "",
                cer_syl=c_syl, cer_jamo=c_jamo))      # 집계 전 반올림하지 않는다

    with open(os.path.join(OUT, "b0_per_file.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            r2 = dict(r)
            r2["cer_syl"] = round(r["cer_syl"], 4)
            r2["cer_jamo"] = round(r["cer_jamo"], 4)
            w.writerow(r2)

    good = [r for r in rows if not r["defect"]]
    speakers = sorted(set(r["speaker"] for r in good))

    def pooled(rs):
        """자모 수로 가중한 CER. 파일 단순 평균이 아니다."""
        tot = sum(r["ref_jamo"] for r in rs)
        if not tot:
            return float("nan")
        return sum(r["cer_jamo"] * r["ref_jamo"] for r in rs) / tot

    print("\n화자별 자모 CER  (자모 수 가중, 결함 파일 제외, 낮을수록 잘 알아들은 것)")
    print("  %-6s%s  %s" % ("화자", "".join("%12s" % m for m in todo), " 파일"))
    for sp in speakers:
        line = "  %-6s" % sp
        for m in todo:
            rs = [r for r in good if r["speaker"] == sp and r["model"] == m]
            line += ("%12.3f" % pooled(rs)) if rs else ("%12s" % "-")
        line += "  %4d" % len([r for r in good
                               if r["speaker"] == sp and r["model"] == todo[0]])
        print(line)

    print("\n모델별 전체")
    summary = dict(date=time.strftime("%Y-%m-%d"), n_files=len(items),
                   n_speakers=len(speakers),
                   hours=round(sum(i["play_sec"] for i in items) / 3600, 2),
                   device=name, cov_min=COV_MIN, models={},
                   note=("B0 = 적응 없음. 정답은 라벨 전사(의도 기준, +/* 제거). "
                         "청취 검증 없음. CER은 Whisper+VAD 파이프라인 성능이다."))
    for m in todo:
        g = [r for r in good if r["model"] == m]
        mean = sum(r["cer_jamo"] for r in g) / len(g)
        syl = sum(r["cer_syl"] for r in g) / len(g)
        print("  %-9s 자모 가중 %.3f  파일평균 %.3f  음절 파일평균 %.3f  범위 %.3f~%.3f"
              % (m, pooled(g), mean, syl,
                 min(r["cer_jamo"] for r in g), max(r["cer_jamo"] for r in g)))
        summary["models"][m] = dict(
            cer_jamo_pooled=round(pooled(g), 4),
            cer_jamo_file_mean=round(mean, 4),
            cer_syl_file_mean=round(syl, 4),
            cer_jamo_min=round(min(r["cer_jamo"] for r in g), 4),
            cer_jamo_max=round(max(r["cer_jamo"] for r in g), 4),
            by_speaker_pooled={
                sp: round(pooled([r for r in g if r["speaker"] == sp]), 4)
                for sp in speakers})

    vr = [r for r in good if r["vad_ratio"] != ""]
    if vr:
        print("\n발화비율 (VAD 통과 시간 / 전체 길이) — 슬롯 가설 검증용")
        print("  %-6s%s" % ("화자", "".join("%12s" % m for m in todo)))
        for sp in speakers:
            line = "  %-6s" % sp
            for m in todo:
                rs = [r for r in vr if r["speaker"] == sp and r["model"] == m]
                line += ("%11.1f%%" % (100 * sum(r["vad_ratio"] for r in rs) / len(rs))
                         ) if rs else ("%12s" % "-")
            print(line)

    print("\n인식 길이 / 정답 길이  (1.0이 정상, 0에 가까우면 아예 못 받아적음)")
    print("  %-6s%s" % ("화자", "".join("%12s" % m for m in todo)))
    for sp in speakers:
        line = "  %-6s" % sp
        for m in todo:
            g = [r for r in good if r["speaker"] == sp and r["model"] == m]
            rc = sum(r["ref_chars"] for r in g)
            line += ("%12.2f" % (sum(r["hyp_chars"] for r in g) / rc)) if rc else ("%12s" % "-")
        print(line)

    nr = [r for r in good if r["numeric_review"]]
    if nr:
        print("\n숫자가 섞인 파일 %d개. 숫자를 지우고 재기 때문에 표기 차이가" % len(nr))
        print("양방향으로 왜곡된다(정답의 한글 수사는 오류로, 인식의 숫자 삽입은 무시로).")
        print("점수는 그대로 두었다. 해당 파일의 원문을 직접 볼 것.")
        for r in nr:
            print("  %-9s %s  정답숫자 %d  인식숫자 %d"
                  % (r["model"], r["file"], r["ref_digits"], r["hyp_digits"]))

    hall = [r for r in good if r["cer_jamo"] > 1.0]
    if hall:
        print("\nCER 1.0 초과 (삽입이 많다 = 헛말 생성):")
        for r in hall:
            print("  %-9s %s  CER %.3f  길이비 %.2f"
                  % (r["model"], r["file"], r["cer_jamo"],
                     r["hyp_chars"] / max(r["ref_chars"], 1)))

    # 경계 계산 전에 저장한다. 뒤에서 죽어도 CER 결과는 남는다.
    save_json(summary, os.path.join(OUT, "b0_summary.json"))
    print("\n  저장: b0_summary.json, b0_per_file.csv")

    # ---------------------------------------------------------------- 육안
    print("\n" + "=" * 70)
    print("샘플 (화자별 첫 파일, 앞 140자)")
    print("=" * 70)
    for sp in speakers:
        it = next((i for i in items if i["speaker"] == sp and not i["defect"]), None)
        if not it:
            continue
        print("\n───── %s / %s ─────" % (sp, it["task"]))
        print("  정답 :", it["ref_text"][:140])
        for m in todo:
            print("  %-9s:" % m, results[m][it["id"]]["text"][:140])

    # ---------------------------------------------------------------- 경계
    ref_model = "large-v3" if "large-v3" in results else "small"
    print("\n" + "=" * 70)
    print("04 합본 10조각 경계 — %s 기준" % ref_model)
    print("자동 정렬 결과는 **검수 후보**다. 학습용 정답 경계로 쓰지 말 것.")
    print("=" * 70)
    bd = boundaries(items, results[ref_model], canon, order)
    save_json(bd, os.path.join(OUT, "combined_boundaries.json"))

    print("\n" + "=" * 70)
    print("저장 위치: %s" % OUT)
    for fn in ["b0_summary.json", "b0_per_file.csv", "combined_boundaries.json"]:
        print("  %s" % fn)
    print("이 셋을 내려받아 sw_challenge 폴더에 넣으면 된다.")
    print("자동 재개는 없다. 다시 실행하면 처음부터 추론한다.")
    print("=" * 70)


if __name__ == "__main__":
    main()

# --- scoring rule frozen 2026-09-18 (score-v2). Additive: cer()/norm_syl() are unchanged. ---
SCORING_VERSION = "score-v2"

PAIRED_SYSTEMS = ("small_b0", "small_b1")   # the b0-vs-b1 comparison. Reference systems
                                            # (e.g. large_b0) must NEVER change the denominator.


def number_bearing(ref, hyps):
    """True if this segment's score is distorted by norm_syl deleting digits.

    norm_syl strips digits from reference AND hypothesis, so any segment where a digit appears
    on either side is scored against a mutilated string: a digit-writing system gets a free pass
    where the label spelled the number out, and vice versa. Measured 2026-09-18: the labels use
    BOTH conventions, even within one speaker (CYU 13 segments with digits, 13 with money spelled
    in Hangul; KJW 0 with digits) - so the distortion has no fixed direction and cannot be
    corrected by choosing one convention.

    `hyps` = the hypotheses of the PAIRED systems only. This is the part that was wrong before:
    the old filter tested whichever systems a run happened to include, so adding a large-v3
    reference pass silently removed segments and the n30 and nall conditions ended up scored on
    different sets (KJW 49 vs 48, CYU 100 vs 99).

    Not fixed here, deliberately: converting between digits and Hangul numerals needs the
    Sino-Korean/native distinction and the labels do not follow it consistently (HO §4:
    `육십 이 살` where the counter demands `예순두 살`). A converter would add its own error rate
    to the metric. Excluding and counting is honest; guessing the reading is not.
    """
    return bool(count_digits(ref)) or any(count_digits(h) for h in hyps)
