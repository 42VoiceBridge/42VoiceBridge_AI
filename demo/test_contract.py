#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contract tests for AI_BACKEND_CONTRACT v1. Run against any server that claims v1.

    ASR_ENGINE=mock ASR_ADAPTERS=./test_adapters python3 server.py &      # or the real engine
    python3 test_contract.py http://127.0.0.1:8000 KJW

The second argument is a user_id that has an adapter on that server (or omit to skip the
adapter-binding checks). Exit code 1 if any check fails.
"""
import io
import json
import struct
import sys
import urllib.error
import urllib.request
import wave

B = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
ADAPTER_USER = sys.argv[2] if len(sys.argv) > 2 else None
FAIL = []
RUN = __import__("uuid").uuid4().hex[:8]   # keys must be unique per run: server state persists


def call(method, path, body=None, ctype="application/json"):
    data = body if isinstance(body, (bytes, type(None))) else json.dumps(body).encode()
    req = urllib.request.Request(B + path, data=data, method=method,
                                 headers={"Content-Type": ctype} if data is not None else {})
    try:
        r = urllib.request.urlopen(req)
        return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        FAIL.append(msg)


def wav(sec, rate=16000, ch=1, amp=3000):
    bio = io.BytesIO()
    w = wave.open(bio, "wb")
    w.setnchannels(ch); w.setsampwidth(2); w.setframerate(rate)
    import math
    n = int(sec * rate)
    w.writeframes(b"".join(struct.pack("<h", int(amp * math.sin(i / 8.0))) * ch for i in range(n)))
    w.close()
    return bio.getvalue()


s, h = call("GET", "/v1/health")
check(s == 200 and h.get("contract_version") == "ai-contract-v1", "health / contract_version")
tone = wav(1.5)
s, t = call("POST", "/v1/asr/transcribe?user_id=guest", tone, "audio/wav")
check(s == 200, "transcribe 200")
for k in ("transcription_id", "request_id", "user_id", "revision", "status", "text", "alternatives",
          "score", "score_type", "model", "decoding", "audio", "latency_ms", "created_at"):
    check(k in t, "result has field %s" % k)
check(t.get("alternatives") == [] and t.get("score") is None and t.get("score_type") is None,
      "v1: alternatives [] and score/score_type null")
check(t["model"].get("adapter_id") is None, "guest -> no adapter")
if not ADAPTER_USER:
    # "0 failed" must not read as full coverage: this branch holds the only HTTP checks for the
    # adapter path, and it is skipped unless a user with an active adapter is named on argv.
    # test_activation.py covers the same ground against a running server (groups 2, 3, 8).
    print("SKIP adapter-path checks: pass a user_id with an active adapter as argv[2]")
if ADAPTER_USER:
    s, ta = call("POST", "/v1/asr/transcribe?user_id=%s" % ADAPTER_USER, tone, "audio/wav")
    check(ta["model"]["adapter_id"] is not None and ta["model"]["adapter_revision"], "adapter user gets own adapter")
    s, tb = call("POST", "/v1/asr/transcribe?user_id=%s&use_adapter=false" % ADAPTER_USER, tone, "audio/wav")
    check(tb["model"]["adapter_id"] is None, "use_adapter=false -> base")
    s, tc = call("POST", "/v1/asr/transcribe?user_id=guest&adapter_id=%s" % ta["model"]["adapter_id"],
                 tone, "audio/wav")
    check(tc["model"]["adapter_id"] is None, "client cannot select another user's adapter")
s, e = call("POST", "/v1/asr/transcribe", tone, "audio/wav")
check(s == 400 and e["error"]["code"] == "missing_user_id", "missing user_id -> 400")
s, e = call("POST", "/v1/asr/transcribe?user_id=a", b"not audio", "audio/wav")
check(s == 415 and e["error"]["code"] == "unsupported_media_type", "non-WAV -> 415")
s, e = call("POST", "/v1/asr/transcribe?user_id=a", wav(1, rate=44100, ch=2), "audio/wav")
check(s == 422 and e["error"]["code"] == "bad_audio_format", "44.1k stereo -> 422 bad_audio_format")
s, e = call("POST", "/v1/asr/transcribe?user_id=a", wav(0.1), "audio/wav")
check(s == 422 and e["error"]["code"] == "audio_too_short", "0.1 s -> 422 audio_too_short")
s, e = call("POST", "/v1/asr/transcribe?user_id=a", wav(31, amp=500), "audio/wav")
check(s == 413 and e["error"]["code"] == "audio_too_long", "31 s -> 413 audio_too_long")
s, e = call("POST", "/v1/asr/transcribe?user_id=a", wav(1, amp=0), "audio/wav")
check(s == 200 and e["status"] == "no_speech" and e["text"] == "", "digital silence -> no_speech")

tid = t["transcription_id"]
s, e = call("POST", "/v1/tts", {"confirmation_id": "nope", "idempotency_key": RUN + "k0"})
check(s == 404, "tts needs an existing confirmation")
s, e = call("POST", "/v1/transcriptions/%s/confirm" % tid, {"revision": 99, "confirmed_text": "x"})
check(s == 409 and e["error"]["code"] == "stale_revision", "stale revision -> 409")
s, e = call("POST", "/v1/transcriptions/%s/confirm" % tid, {"revision": 1, "confirmed_text": " "})
check(s == 422, "empty confirmed_text -> 422")
s, c1 = call("POST", "/v1/transcriptions/%s/confirm" % tid, {"revision": 1, "confirmed_text": "물 주세요"})
check(s == 200 and c1["consent"] == {"store_audio": False, "use_for_training": False},
      "consent defaults to false/false")
s, r1 = call("POST", "/v1/tts", {"confirmation_id": c1["confirmation_id"], "idempotency_key": RUN + "k1"})
check(s == 200 and r1["text"] == "물 주세요" and r1["replayed"] is False, "tts speaks exactly the confirmed text")
s, r2 = call("POST", "/v1/tts", {"confirmation_id": c1["confirmation_id"], "idempotency_key": RUN + "k1"})
check(r2.get("replayed") is True and r2["tts_id"] == r1["tts_id"], "retry with same key does not create a new TTS")
s, c2 = call("POST", "/v1/transcriptions/%s/confirm" % tid, {"revision": 1, "confirmed_text": "물 좀 주세요"})
check(c2["supersedes"] == c1["confirmation_id"], "new confirmation supersedes the old one")
s, e = call("POST", "/v1/tts", {"confirmation_id": c1["confirmation_id"], "idempotency_key": RUN + "k2"})
check(s == 409 and e["error"]["code"] == "confirmation_superseded", "superseded text cannot be spoken")
s, e = call("POST", "/v1/tts", {"confirmation_id": c2["confirmation_id"], "idempotency_key": RUN + "k1"})
check(s == 409 and e["error"]["code"] == "idempotency_key_reused", "idempotency key bound to one text")

s, j = call("POST", "/v1/analysis/jamo-errors",
            {"pairs": [{"ref": "반찬 좀 더 주세요", "hyp": "반창 좀 더 주세요"}], "min_support": 1})
top = j["tokens"][0]
check(s == 200 and (top["token"], top["position"], top["error_rate"]) == ("ㄴ", "final", 0.5),
      "jamo: 찬->창 is one final-ㄴ error out of two final-ㄴ occurrences")
s, j = call("POST", "/v1/analysis/jamo-errors", {"pairs": [{"ref": "가", "hyp": "가"}]})
check(j["tokens"][0]["status"] == "insufficient_data" and j["tokens"][0]["error_rate"] is None,
      "below min_support -> insufficient_data, error_rate null")
s, p = call("POST", "/v1/enroll/next-prompts", {"user_id": "u", "n": 3, "seed": 1})
s2, p2 = call("POST", "/v1/enroll/next-prompts", {"user_id": "u", "n": 3, "seed": 1})
check(s == 200 and len(p["prompts"]) == 3 and p["prompts"] == p2["prompts"], "prompts: n honoured, seeded")
s, e = call("POST", "/v1/enroll/next-prompts", {"user_id": "u", "strategy": "error_based"})
check(s == 501, "error_based strategy is 501, not faked")
# ---- enrollment ingest over HTTP (backend T3, 2026-10-03) -------------------------------
import base64 as _b64
EU = "ct_" + RUN
_pid = p["prompts"][0]["prompt_id"]
_pid2 = p["prompts"][1]["prompt_id"]
s, r = call("POST", "/v1/enroll/recordings?user_id=%s&prompt_id=%s" % (EU, _pid),
            wav(1.0), "audio/wav")
check(s == 201 and r["n_enrolled"] == 1 and r["item"]["reviewed"] is False,
      "ingest: 201, label from the prompt pool, not reviewed")
check(r["item"]["text"] == p["prompts"][0]["text"] and r["item"]["text_source"] == "prompt_pool",
      "ingest: the label is the pool text for that prompt_id, and says so")
s, r2 = call("POST", "/v1/enroll/recordings?user_id=%s&prompt_id=%s&text_b64=%s"
             % (EU, _pid2, _b64.urlsafe_b64encode("읽은 문장".encode()).decode().rstrip("=")),
             wav(1.0), "audio/wav")
check(s == 201 and r2["item"]["text"] == "읽은 문장" and r2["item"]["reviewed"] is True,
      "ingest: a reviewed transcript overrides the prompt text and marks itself reviewed")
s, r3 = call("POST", "/v1/enroll/recordings?user_id=%s&prompt_id=%s" % (EU, _pid),
             wav(1.3), "audio/wav")
check(s == 201 and r3["replaced_previous_take"] is True and r3["n_enrolled"] == 2,
      "ingest: re-recording a prompt replaces the take instead of duplicating the sentence")
s, li = call("GET", "/v1/enroll/recordings?user_id=%s" % EU)
check(s == 200 and li["n_enrolled"] == 2 and all("text" not in i for i in li["items"]),
      "enroll list: counts and provenance, no label text")
s, e = call("POST", "/v1/adapters/train?user_id=%s" % EU)
check(s == 409 and e["error"]["code"] == "enrollment_unreviewed",
      "train refuses while any recording has no reviewed transcript")
s, e = call("POST", "/v1/enroll/recordings?user_id=%s&prompt_id=99-99-nope" % EU,
            wav(1.0), "audio/wav")
check(s == 404 and e["error"]["code"] == "unknown_prompt_id", "ingest: prompt must be in the pool")
s, e = call("POST", "/v1/enroll/recordings?user_id=%s/../x&prompt_id=%s" % (EU, _pid),
            wav(1.0), "audio/wav")
check(s == 400 and e["error"]["code"] == "bad_user_id", "ingest: path traversal in user_id refused")
s, e = call("POST", "/v1/enroll/recordings?user_id=%s&prompt_id=%s" % (EU, _pid),
            wav(1.0, rate=44100), "audio/wav")
check(s == 422 and e["error"]["code"] == "bad_audio_format",
      "ingest: same audio limits as transcribe")
s, d1 = call("DELETE", "/v1/enroll/recordings?user_id=%s&prompt_id=%s" % (EU, _pid))
check(s == 200 and d1["audio_deleted"] is True and d1["adapters_retired"] is False
      and d1["n_enrolled"] == 1,
      "delete: audio and pair entry go; the response says adapters are NOT retired")
s, d2 = call("DELETE", "/v1/enroll/recordings?user_id=%s&prompt_id=%s" % (EU, _pid))
check(s == 404 and d2["error"]["code"] == "unknown_prompt_id", "delete is not idempotently silent")
s, d3 = call("DELETE", "/v1/enroll/recordings?user_id=%s&prompt_id=%s" % (EU, _pid2))
check(s == 200 and d3["n_enrolled"] == 0, "delete: last recording removed")

print("\n%d failed" % len(FAIL))
sys.exit(1 if FAIL else 0)
