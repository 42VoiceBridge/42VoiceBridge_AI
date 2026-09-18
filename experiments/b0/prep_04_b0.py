#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B0 측정용 04 지문 세트 전처리 — 로컬(맥)에서 실행한다.

하는 일
  1. b0_manifest.json에 적힌 31개 wav를 찾아
  2. 16kHz / 모노 / 16bit 로 변환해서 b0_16k/ 에 저장
  3. 변환 결과를 검증하고 b0_16k/manifest.json 에 실측값을 덧붙여 다시 쓴다

주의
  - 맥에서 한글 파일명이 NFD로 저장되어 있어 NFC 문자열로는 못 찾는다.
    그래서 glob을 쓰지 않고 os.listdir로 받은 이름을 NFC로 정규화해 매칭한다.
  - 원본은 44.1k/48k, 16/24/32bit, 모노/스테레오가 섞여 있다. 선언된
    SamplingRate(48000)는 믿지 말고 wav 헤더에서 읽는다. 아래 코드가 그렇게 한다.
  - 이미 16kHz인 파일도 그대로 복사하지 않고 같은 경로를 태워 포맷을 통일한다.
    (업샘플링은 하지 않는다. 16k → 16k는 변환 없음)

실행
    cd "/Users/yujemin/Desktop/sw_challenge"
    python3 experiments/b0/prep_04_b0.py

변환기
    ffmpeg 가 있으면 ffmpeg를 쓴다 (권장).
    없으면 scipy가 있으면 scipy.signal.resample_poly 로 처리한다.
    둘 다 없으면 안내를 띄우고 멈춘다.
        brew install ffmpeg      또는      python3 -m pip install scipy
"""

import json
import os
import shutil
import subprocess
import sys
import unicodedata
import wave

ROOT = os.path.dirname(os.path.abspath(__file__))          # experiments/b0/
# manifest 의 src 는 프로젝트 루트(sw_challenge/) 기준 상대경로다. 2026-09-18 폴더 재편 후
# 스크립트가 experiments/b0/ 로 옮겨졌으므로 원본 색인은 두 단계 위에서 한다.
PROJECT = os.path.dirname(os.path.dirname(ROOT))
MANIFEST = os.path.join(ROOT, "b0_manifest.json")
OUTDIR = os.path.join(ROOT, "b0_16k")
TARGET_RATE = 16000


# ---------------------------------------------------------------- 파일 찾기

def nfc(s):
    return unicodedata.normalize("NFC", s)


def build_index(base):
    """base 아래 모든 파일을 {NFC 경로: 실제 경로} 로 색인한다."""
    idx = {}
    for dirpath, dirnames, filenames in os.walk(base):
        for fn in filenames:
            real = os.path.join(dirpath, fn)
            rel = os.path.relpath(real, base)
            idx[nfc(rel)] = real
    return idx


# ---------------------------------------------------------------- 변환

def have_ffmpeg():
    return shutil.which("ffmpeg") is not None


def convert_ffmpeg(src, dst):
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-i", src, "-ac", "1", "-ar", str(TARGET_RATE),
           "-sample_fmt", "s16", dst]
    subprocess.run(cmd, check=True)


def read_wav_mono_float(path):
    """비트수·채널 무관하게 모노 float32 배열과 원본 샘플레이트를 돌려준다."""
    import numpy as np
    w = wave.open(path, "rb")
    sr, ch, sw, nf = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
    chunks = []
    while True:
        raw = w.readframes(sr * 30)
        if not raw:
            break
        if sw == 1:
            a = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
        elif sw == 2:
            a = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        elif sw == 3:
            r = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
            v = (r[:, 0] | (r[:, 1] << 8) | (r[:, 2] << 16))
            v = ((v << 8).astype(np.int32)) >> 8          # 부호 확장
            a = v.astype(np.float32) / (2 ** 23)
        elif sw == 4:
            a = np.frombuffer(raw, dtype="<i4").astype(np.float32) / (2 ** 31)
        else:
            raise RuntimeError("지원하지 않는 sampwidth: %d" % sw)
        if ch > 1:
            a = a.reshape(-1, ch).mean(axis=1)
        chunks.append(a)
    w.close()
    return (np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)), sr, ch, sw


def convert_scipy(src, dst):
    import numpy as np
    from math import gcd
    from scipy.signal import resample_poly
    x, sr, ch, sw = read_wav_mono_float(src)
    if sr != TARGET_RATE:
        g = gcd(sr, TARGET_RATE)
        x = resample_poly(x, TARGET_RATE // g, sr // g)
    x = np.clip(x, -1.0, 1.0)
    pcm = (x * 32767.0).astype("<i2")
    w = wave.open(dst, "wb")
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(TARGET_RATE)
    w.writeframes(pcm.tobytes())
    w.close()


# ---------------------------------------------------------------- 검증

def probe(path):
    w = wave.open(path, "rb")
    info = dict(rate=w.getframerate(), ch=w.getnchannels(),
                bits=w.getsampwidth() * 8,
                sec=round(w.getnframes() / w.getframerate(), 2))
    w.close()
    return info


# ---------------------------------------------------------------- 본체

def main():
    if not os.path.exists(MANIFEST):
        sys.exit("b0_manifest.json 이 없다. 이 스크립트와 같은 폴더에 두고 다시 실행한다.")

    man = json.load(open(MANIFEST, encoding="utf-8"))
    items = man["items"]
    print("대상 %d 파일" % len(items))

    if have_ffmpeg():
        convert, how = convert_ffmpeg, "ffmpeg"
    else:
        try:
            import numpy, scipy  # noqa: F401
            convert, how = convert_scipy, "scipy"
        except ImportError:
            sys.exit("ffmpeg 도 scipy 도 없다.\n"
                     "  brew install ffmpeg\n"
                     "또는\n"
                     "  python3 -m pip install numpy scipy\n"
                     "둘 중 하나를 설치하고 다시 실행한다.")
    print("변환기: %s" % how)

    print("원본 색인 중...")
    idx = build_index(PROJECT)
    os.makedirs(OUTDIR, exist_ok=True)

    ok, skipped, failed = 0, 0, []
    for it in items:
        src = idx.get(nfc(it["src"]))
        dst = os.path.join(OUTDIR, it["out"])
        if src is None:
            failed.append((it["id"], "원본 없음"))
            continue
        if os.path.exists(dst) and os.path.getsize(dst) > 1000:
            it["probe"] = probe(dst)
            skipped += 1
            ok += 1
            continue
        try:
            convert(src, dst)
            it["probe"] = probe(dst)
            ok += 1
            print("  [%2d/%2d] %s  %s  %.1f분" % (ok, len(items), it["speaker"],
                                                 it["task"], it["probe"]["sec"] / 60))
        except Exception as e:                                    # noqa: BLE001
            failed.append((it["id"], repr(e)))

    # 검증: 원본 playTime 과 변환본 길이가 맞는지
    print("\n검증")
    bad = []
    for it in items:
        p = it.get("probe")
        if not p:
            continue
        if p["rate"] != TARGET_RATE or p["ch"] != 1 or p["bits"] != 16:
            bad.append((it["id"], "포맷 불일치 %s" % p))
        drift = abs(p["sec"] - it["play_sec"])
        if drift > 1.0:
            bad.append((it["id"], "길이 차이 %.1f초 (원본 %.1f / 변환 %.1f)"
                        % (drift, it["play_sec"], p["sec"])))
    if bad:
        for i, m in bad:
            print("  주의  %s  %s" % (i, m))
    else:
        print("  전부 16kHz 1ch 16bit, 길이 오차 1초 이내")

    total = sum(it.get("probe", {}).get("sec", 0) for it in items)
    size = sum(os.path.getsize(os.path.join(OUTDIR, it["out"]))
               for it in items if os.path.exists(os.path.join(OUTDIR, it["out"])))
    print("\n변환 %d / 건너뜀 %d / 실패 %d" % (ok - skipped, skipped, len(failed)))
    for i, m in failed:
        print("  실패  %s  %s" % (i, m))
    print("오디오 %.2f시간, %.2f GB" % (total / 3600, size / 1e9))

    man["prepared_with"] = how
    json.dump(man, open(os.path.join(OUTDIR, "manifest.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\n결과 폴더: %s" % OUTDIR)
    print("이 폴더를 통째로 구글 드라이브에 올린다. (manifest.json 포함)")


if __name__ == "__main__":
    main()
