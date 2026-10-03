#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Transport-independent contract checks (2026-09-25).

`test_contract.py` drives the HTTP server. These call `server.core_*` directly, so they survive
the move to JPyRust (Java -> Rust JNI -> this Python process), where there is no HTTP layer.

Covers the three confirm-before-speak defects found in the 2026-09-23 GPT review:
  D-1 a cached TTS authorization replayed superseded text
  D-2 consent `"false"` (a string) granted permission
  D-6 is browser-side; its guard is checked in static/index.html, not here

and the three T19 fixes (2026-09-29):
  D-3 `request_id` was echoed and otherwise ignored, so a retry ran inference twice
  D-4 the active adapter was "newest created_at wins", so an unvalidated one went live
  D-5 the base revision was recorded but never enforced
plus `contract_version` on every response, which used to be on two of seven.

    ASR_ENGINE=mock python3 test_core.py
"""
import io
import json
import os
import struct
import sys
import wave

os.environ.setdefault("ASR_ENGINE", "mock")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import server  # noqa: E402


def wav(seconds=1.0, amp=8000):
    buf = io.BytesIO()
    w = wave.open(buf, "wb")
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(16000)
    n = int(seconds * 16000)
    w.writeframes(b"".join(struct.pack("<h", amp if (i // 80) % 2 else -amp) for i in range(n)))
    w.close()
    return buf.getvalue()


def expect(code, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except server.ApiError as e:
        assert e.http == code, (code, e.http, e.code, e.message)
        return e
    raise AssertionError("expected ApiError %s, nothing raised" % code)


def main():
    server.ENG = server.MockEngine()
    ok = 0

    tx = server.core_transcribe(wav(), "u1")
    for k in ("contract_version", "transcription_id", "status", "text", "alternatives", "score",
              "score_type", "model", "decoding", "audio", "latency_ms", "created_at"):
        assert k in tx, k
    assert tx["alternatives"] == [] and tx["score"] is None and tx["score_type"] is None, tx
    assert tx["status"] == "ok" and tx["text"], tx
    ok += 1

    silent = server.core_transcribe(wav(amp=1), "u1")
    assert silent["status"] == "no_speech" and silent["text"] == "", silent
    ok += 1

    # D-2: only real JSON booleans grant permission
    c = server.core_confirm(tx["transcription_id"],
                            dict(revision=1, confirmed_text="안녕하세요"))
    assert c["consent"] == dict(store_audio=False, use_for_training=False), c["consent"]
    ok += 1
    for bad in ("false", "true", 0, 1, None, [], {}):
        tx2 = server.core_transcribe(wav(), "u1")
        expect(422, server.core_confirm, tx2["transcription_id"],
               dict(revision=1, confirmed_text="x", consent={"use_for_training": bad}))
    ok += 1
    tx3 = server.core_transcribe(wav(), "u1")
    c3 = server.core_confirm(tx3["transcription_id"],
                             dict(revision=1, confirmed_text="x",
                                  consent=dict(use_for_training=True, store_audio=False)))
    assert c3["consent"]["use_for_training"] is True, c3
    ok += 1

    # replay of a still-valid authorization stays allowed
    r1 = server.core_tts(dict(confirmation_id=c["confirmation_id"], idempotency_key="K1"))
    assert r1["replayed"] is False and r1["text"] == "안녕하세요", r1
    r2 = server.core_tts(dict(confirmation_id=c["confirmation_id"], idempotency_key="K1"))
    assert r2["replayed"] is True and r2["text"] == "안녕하세요", r2
    ok += 1

    # D-1: after a newer confirmation supersedes it, the SAME key must not replay the old text
    c_new = server.core_confirm(tx["transcription_id"], dict(revision=1, confirmed_text="고쳤어요"))
    assert c_new["supersedes"] == c["confirmation_id"], c_new
    e = expect(409, server.core_tts,
               dict(confirmation_id=c["confirmation_id"], idempotency_key="K1"))
    assert e.code == "confirmation_superseded", e.code
    ok += 1

    # the new confirmation still works, and key reuse across texts is still refused
    r3 = server.core_tts(dict(confirmation_id=c_new["confirmation_id"], idempotency_key="K2"))
    assert r3["text"] == "고쳤어요", r3
    expect(409, server.core_tts,
           dict(confirmation_id=c_new["confirmation_id"], idempotency_key="K1"))
    ok += 1

    expect(404, server.core_tts, dict(confirmation_id="nope", idempotency_key="K9"))
    expect(409, server.core_confirm, tx["transcription_id"],
           dict(revision=99, confirmed_text="x"))
    expect(422, server.core_confirm, tx["transcription_id"], dict(revision=1, confirmed_text="  "))
    expect(400, server.core_transcribe, wav(), "  ")
    ok += 1

    h = server.core_health()
    assert h["contract_version"] == server.CONTRACT_VERSION and "limits" in h, h
    ok += 1

    # A2: serving, evaluation and the promotion gate must generate with the SAME configuration,
    # and both must record it. A gate score measured under a different cap is not a serving score.
    import decoding
    srv_gen = server.core_transcribe(wav(), "ugen")["decoding"]
    assert srv_gen == decoding.describe(), srv_gen
    assert srv_gen["max_new_tokens"] == decoding.MAX_NEW_TOKENS
    assert "no_repeat_ngram_size" not in srv_gen, "nr5 must not be a serving default"
    assert h["generation"]["generation_version"] == decoding.GENERATION_VERSION, h["generation"]
    ok += 1

    # D-3: the same client request_id must not run inference twice
    a = server.core_transcribe(wav(), "u1", request_id="R1")
    b = server.core_transcribe(wav(), "u1", request_id="R1")
    assert a == b, "a retried request_id must return the first result unchanged"
    c2 = server.core_transcribe(wav(), "u1", request_id="R2")
    assert c2["transcription_id"] != a["transcription_id"]
    # ...and ids are scoped per user, so two users may reuse the same one
    d = server.core_transcribe(wav(), "u2", request_id="R1")
    assert d["transcription_id"] != a["transcription_id"], "request_id leaked across users"
    # a generated request_id is never an idempotency key
    e1 = server.core_transcribe(wav(), "u1")
    e2 = server.core_transcribe(wav(), "u1")
    assert e1["transcription_id"] != e2["transcription_id"]
    assert server.core_transcribe(wav(), "u1", request_id="  ")["request_id"] not in ("", "  ")
    ok += 1

    # A3/A5: user_adapter now returns (meta, reason); the reason explains every base fallback
    assert server.user_adapter("nobody") == (None, "no_active_adapter")

    # D-4: no active.json -> the BASE model is served, even though an adapter exists for the user
    server.ENG.adapters = {"ad1": dict(adapter_id="ad1", user_id="u1", adapter_revision="r1",
                                       created_at="2026-01-01", size_bytes=1)}
    server.ADAPTER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_t19_tmp")
    os.makedirs(server.ADAPTER_DIR, exist_ok=True)
    ap = os.path.join(server.ADAPTER_DIR, "active.json")
    if os.path.exists(ap):
        os.remove(ap)
    assert server.user_adapter("u1") == (None, "no_active_adapter"), "adapter went live with no pointer"
    open(ap, "w").write(json.dumps({"u1": "ad1"}))
    got, why = server.user_adapter("u1")
    assert got["adapter_id"] == "ad1" and why is None
    # a pointer to an unknown adapter, or to another user's, falls back to the base WITH a reason
    open(ap, "w").write(json.dumps({"u1": "nope"}))
    assert server.user_adapter("u1") == (None, "adapter_not_found")
    open(ap, "w").write(json.dumps({"u2": "ad1"}))
    assert server.user_adapter("u2") == (None, "adapter_wrong_owner"), "served another user's adapter"
    open(ap, "w").write("{ this is not json")
    assert server.user_adapter("u1") == (None, "no_active_adapter"), "unreadable pointer must not serve"
    ok += 1

    # A3: an adapter that appears AFTER startup must be picked up, not ignored until restart
    server.ENG.adapters = {}                       # as if the server started before promotion
    d2 = os.path.join(server.ADAPTER_DIR, "late1")
    os.makedirs(d2, exist_ok=True)
    open(os.path.join(d2, "adapter_config.json"), "w").write("{}")
    open(os.path.join(d2, "adapter_model.safetensors"), "wb").write(b"w")
    json.dump(dict(user_id="u3"), open(os.path.join(d2, "meta.json"), "w"))
    open(ap, "w").write(json.dumps({"u3": "late1"}))
    got, why = server.user_adapter("u3")
    assert got is not None and got["adapter_id"] == "late1" and why is None, (got, why)
    tx_late = server.core_transcribe(wav(), "u3")
    assert tx_late["model"]["adapter_id"] == "late1", tx_late["model"]
    assert tx_late["model"]["base_reason"] is None
    # and a forced-base request says so
    tx_base = server.core_transcribe(wav(), "u3", use_adapter=False)
    assert tx_base["model"]["adapter_id"] is None
    assert tx_base["model"]["base_reason"] == "base_requested", tx_base["model"]
    import shutil as _sh
    _sh.rmtree(d2)
    os.remove(ap)
    ok += 1

    # D-5 is enforced at model load (RealEngine); here we check the pin exists and is the one
    # every adapter was trained against.
    assert server.BASE_REVISION == "973afd24965f72e36ca33b3055d56a652f456b4d", server.BASE_REVISION
    ok += 1

    # contract_version on every response, not two of seven
    tx9 = server.core_transcribe(wav(), "u9")
    c9 = server.core_confirm(tx9["transcription_id"], dict(revision=1, confirmed_text="x"))
    t9 = server.core_tts(dict(confirmation_id=c9["confirmation_id"], idempotency_key="K9x"))
    for name, r in (("health", server.core_health()), ("transcribe", tx9),
                    ("confirm", c9), ("tts", t9)):
        assert r.get("contract_version") == server.CONTRACT_VERSION, name
    body = json.dumps(dict(pairs=[dict(ref="가나", hyp="가다")])).encode()
    assert server.h_jamo({}, body)[1]["contract_version"] == server.CONTRACT_VERSION
    ok += 1

    print("core contract checks passed (%d groups)" % ok)


if __name__ == "__main__":
    main()
