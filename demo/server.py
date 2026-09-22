#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI inference server - reference implementation of AI_BACKEND_CONTRACT v1 (docs/).

Standard library HTTP server + numpy. The real ASR engine additionally needs torch,
transformers and peft (see requirements.txt). A mock engine exists for contract tests.

    python3 server.py                       # real engine, whisper-small, http://127.0.0.1:8000
    ASR_ENGINE=mock python3 server.py       # no model; deterministic fake text (tests only)

Scope of v1, stated so nobody builds on more than exists:
  - Input audio: WAV, PCM 16-bit, mono, 16 kHz, 0.3-30 s. Anything else is rejected with a
    typed error. Transcoding phone formats is the backend's job (ffmpeg), not this server's.
  - alternatives is always [] and score is always null in v1. They exist in the schema so
    the backend does not have to change its tables later; nothing is fabricated to fill them.
  - Confirmation and TTS gating are implemented here only so the demo works end to end.
    In the product they belong to the backend. TTS audio is not synthesized server-side; the
    server returns the one text the client is allowed to speak.
  - State is in memory. Restarting the server forgets transcriptions and confirmations.
  - Inference is serialized with a lock: the PEFT model holds one active adapter at a time,
    so concurrent requests for different users must not interleave.
"""
import hashlib
import json
import os
import random
import sys
import threading
import time
import uuid
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from urllib.parse import parse_qs, urlparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import jamo_stats  # noqa: E402

CONTRACT_VERSION = "ai-contract-v1"
PREPROC_VERSION = "pp-v1"          # WAV PCM16 mono 16 kHz, no resampling, no VAD
MIN_SEC, MAX_SEC = 0.3, 30.0
SILENCE_DBFS = -60.0               # heuristic, not validated: below this RMS -> no_speech
BASE = os.environ.get("ASR_BASE", "openai/whisper-small")
# Greedy by default (D9, 2026-09-18). At beam_size=5 whisper-small emitted repetition loops on
# short utterances (`아, 그래요?` -> 200 tokens of `아`), and one such segment can dominate a
# pooled CER. Accuracy is NOT the reason for greedy: it is a wash (better on the KJW test split,
# worse on the 10 demo samples). The reasons are that greedy cannot produce the loop, and median
# demo latency fell 1554 -> 666 ms. See HO §5.2 and CT §0 D9.
BEAMS = int(os.environ.get("ASR_BEAMS", "1"))
ENGINE = os.environ.get("ASR_ENGINE", "hf")
ADAPTER_DIR = os.environ.get("ASR_ADAPTERS", os.path.join(HERE, "adapters"))
POOL = os.environ.get("PROMPT_POOL", os.path.join(os.path.dirname(HERE), "data", "script_pool.json"))


# ------------------------------------------------------------------ errors
class ApiError(Exception):
    def __init__(self, http, code, message, **detail):
        super().__init__(message)
        self.http, self.code, self.message, self.detail = http, code, message, detail


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def sha256(b):
    return hashlib.sha256(b).hexdigest()


# ------------------------------------------------------------------ audio
def parse_wav(data):
    if len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ApiError(415, "unsupported_media_type", "body is not a RIFF/WAVE file")
    try:
        w = wave.open(BytesIO(data), "rb")
    except Exception as e:                                   # noqa: BLE001
        raise ApiError(422, "bad_audio", "cannot parse WAV: %s" % e)
    sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
    if (sr, ch, sw) != (16000, 1, 2):
        raise ApiError(422, "bad_audio_format",
                       "v1 accepts PCM 16-bit mono 16 kHz only; transcode before calling",
                       got=dict(sample_rate=sr, channels=ch, bits=sw * 8))
    x = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    dur = len(x) / 16000.0
    if dur < MIN_SEC:
        raise ApiError(422, "audio_too_short", "shorter than %.1f s" % MIN_SEC, duration_sec=dur)
    if dur > MAX_SEC:
        raise ApiError(413, "audio_too_long", "longer than %.0f s; split into utterances" % MAX_SEC,
                       duration_sec=dur)
    if not np.isfinite(x).all():
        raise ApiError(422, "bad_audio", "non-finite samples")
    rms = float(np.sqrt(np.mean(x ** 2))) if len(x) else 0.0
    dbfs = 20 * np.log10(rms) if rms > 0 else -120.0
    return x, dur, round(dbfs, 1)


# ------------------------------------------------------------------ engines
class MockEngine:
    name = "mock"
    base_model, base_revision = "mock", "mock"

    def __init__(self):
        self.adapters = {}
        self._scan()

    def _scan(self):
        for aid, meta in scan_adapters(ADAPTER_DIR).items():
            self.adapters[aid] = meta

    def transcribe(self, x, adapter_id):
        n = int(len(x) / 16000 * 2)
        text = "모의 인식 결과" + (" 어댑터" if adapter_id else "") + " " + "가" * max(0, n - 6)
        return text.strip()


class HFEngine:
    name = "transformers+peft"

    def __init__(self):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        self.torch = torch
        self.device = os.environ.get("ASR_DEVICE", "cpu")
        self.proc = WhisperProcessor.from_pretrained(BASE, language="korean", task="transcribe")
        m = WhisperForConditionalGeneration.from_pretrained(BASE)
        try:
            m.generation_config.forced_decoder_ids = None
        except Exception:                                     # noqa: BLE001
            pass
        self.base_model = BASE
        self.base_revision = getattr(m.config, "_commit_hash", None)
        self.model = m.to(self.device).eval()
        self.peft = False
        self.adapters = {}
        for aid, meta in scan_adapters(ADAPTER_DIR).items():
            if meta.get("base_model") not in (None, BASE):
                print("skip adapter %s: trained on %s, server base is %s" % (
                    aid, meta.get("base_model"), BASE))
                continue
            from peft import PeftModel
            if not self.peft:
                self.model = PeftModel.from_pretrained(self.model, meta["path"], adapter_name=aid)
                self.peft = True
            else:
                self.model.load_adapter(meta["path"], adapter_name=aid)
            self.model.eval()
            self.adapters[aid] = meta
            print("loaded adapter %s (%s)" % (aid, meta.get("user_id")))

    def transcribe(self, x, adapter_id):
        torch = self.torch
        f = self.proc.feature_extractor(x, sampling_rate=16000, return_tensors="pt").input_features
        f = f.to(self.device)
        kw = dict(input_features=f, language="korean", task="transcribe", num_beams=BEAMS,
                  max_new_tokens=200)
        with torch.no_grad():
            if adapter_id:
                self.model.set_adapter(adapter_id)
                ids = self.model.generate(**kw)
            elif self.peft:
                with self.model.disable_adapter():
                    ids = self.model.generate(**kw)
            else:
                ids = self.model.generate(**kw)
        return self.proc.batch_decode(ids, skip_special_tokens=True)[0].strip()


def scan_adapters(root):
    """adapters/<adapter_id>/{adapter_config.json, adapter_model.safetensors, meta.json}.
    meta.json binds the adapter to exactly one user_id."""
    out = {}
    if not os.path.isdir(root):
        return out
    for aid in sorted(os.listdir(root)):
        p = os.path.join(root, aid)
        if not os.path.isfile(os.path.join(p, "adapter_config.json")):
            continue
        meta = {}
        mp = os.path.join(p, "meta.json")
        if os.path.isfile(mp):
            meta = json.load(open(mp, encoding="utf-8"))
        if not meta.get("user_id"):
            print("skip adapter %s: meta.json has no user_id" % aid)
            continue
        size = sum(os.path.getsize(os.path.join(p, f)) for f in os.listdir(p)
                   if f.startswith("adapter_model"))
        cfg = json.load(open(os.path.join(p, "adapter_config.json"), encoding="utf-8"))
        weights = sorted(f for f in os.listdir(p) if f.startswith("adapter_model"))
        h = hashlib.sha256()
        for f in weights:
            h.update(open(os.path.join(p, f), "rb").read())
        meta.update(path=p, adapter_id=aid, size_bytes=size,
                    base_model=meta.get("base_model") or cfg.get("base_model_name_or_path"),
                    adapter_revision=h.hexdigest()[:12])   # hash of the weights, not the config
        out[aid] = meta
    return out


# ------------------------------------------------------------------ state
class Store:
    def __init__(self):
        self.lock = threading.Lock()
        self.tx = {}          # transcription_id -> result
        self.conf = {}        # confirmation_id -> confirmation
        self.latest_conf = {} # transcription_id -> confirmation_id (latest valid)
        self.tts = {}         # idempotency_key -> tts record


STORE = Store()
INFER_LOCK = threading.Lock()
ENG = None
POOL_ITEMS = None


def user_adapter(user_id):
    act = [m for m in ENG.adapters.values() if m.get("user_id") == user_id]
    act.sort(key=lambda m: m.get("created_at", ""), reverse=True)
    return act[0] if act else None


def syl_edit_distance(a, b):
    import re
    a = re.sub(r"[^가-힣]", "", a)
    b = re.sub(r"[^가-힣]", "", b)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


# ------------------------------------------------------------------ handlers
def h_health(q, body):
    return 200, dict(status="ok", contract_version=CONTRACT_VERSION, engine=ENG.name,
                     base_model=ENG.base_model, base_revision=ENG.base_revision,
                     preprocessing_version=PREPROC_VERSION, beam_size=BEAMS,
                     adapters=[dict(adapter_id=m["adapter_id"], user_id=m["user_id"],
                                    adapter_revision=m["adapter_revision"],
                                    size_bytes=m["size_bytes"], label=m.get("label"),
                                    evidence=m.get("evidence"))
                               for m in ENG.adapters.values()],
                     limits=dict(min_sec=MIN_SEC, max_sec=MAX_SEC, sample_rate=16000,
                                 channels=1, bits=16))


def h_transcribe(q, body):
    user_id = (q.get("user_id") or [""])[0].strip()
    if not user_id:
        raise ApiError(400, "missing_user_id", "user_id query parameter is required")
    use_adapter = (q.get("use_adapter") or ["true"])[0].lower() != "false"
    request_id = (q.get("request_id") or [str(uuid.uuid4())])[0]
    x, dur, dbfs = parse_wav(body)
    ad = user_adapter(user_id) if use_adapter else None
    t0 = time.time()
    if dbfs < SILENCE_DBFS:
        text, status = "", "no_speech"
    else:
        with INFER_LOCK:
            text = ENG.transcribe(x, ad["adapter_id"] if ad else None)
        status = "ok" if text else "no_speech"
    res = dict(
        contract_version=CONTRACT_VERSION, transcription_id=str(uuid.uuid4()),
        request_id=request_id, user_id=user_id, revision=1, status=status, text=text,
        alternatives=[], score=None, score_type=None,
        model=dict(engine=ENG.name, base_model=ENG.base_model, base_revision=ENG.base_revision,
                   adapter_id=ad["adapter_id"] if ad else None,
                   adapter_revision=ad["adapter_revision"] if ad else None),
        decoding=dict(language="ko", task="transcribe", beam_size=BEAMS),
        audio=dict(sha256=sha256(body), duration_sec=round(dur, 3), rms_dbfs=dbfs,
                   preprocessing_version=PREPROC_VERSION),
        latency_ms=int((time.time() - t0) * 1000), created_at=now())
    with STORE.lock:
        STORE.tx[res["transcription_id"]] = res
    return 200, res


def h_confirm(tid, body):
    try:
        req = json.loads(body or b"{}")
    except ValueError:
        raise ApiError(400, "bad_json", "body must be JSON")
    with STORE.lock:
        tx = STORE.tx.get(tid)
        if not tx:
            raise ApiError(404, "unknown_transcription", "no such transcription_id")
        if req.get("revision") != tx["revision"]:
            raise ApiError(409, "stale_revision", "confirm must reference the current revision",
                           current_revision=tx["revision"])
        text = (req.get("confirmed_text") or "").strip()
        if not text:
            raise ApiError(422, "empty_text", "confirmed_text is empty; use cancel instead")
        consent = req.get("consent") or {}
        prev = STORE.latest_conf.get(tid)
        c = dict(confirmation_id=str(uuid.uuid4()), transcription_id=tid,
                 based_on_revision=tx["revision"], confirmed_text=text,
                 text_sha256=sha256(text.encode("utf-8")),
                 source="asr_unedited" if text == tx["text"] else "user_edited",
                 edit_distance_syl=syl_edit_distance(tx["text"], text),
                 consent=dict(store_audio=bool(consent.get("store_audio", False)),
                              use_for_training=bool(consent.get("use_for_training", False))),
                 supersedes=prev, valid=True, confirmed_at=now())
        if prev:
            STORE.conf[prev]["valid"] = False
        STORE.conf[c["confirmation_id"]] = c
        STORE.latest_conf[tid] = c["confirmation_id"]
    return 200, c


def h_tts(q, body):
    try:
        req = json.loads(body or b"{}")
    except ValueError:
        raise ApiError(400, "bad_json", "body must be JSON")
    cid, key = req.get("confirmation_id"), req.get("idempotency_key")
    if not cid or not key:
        raise ApiError(400, "missing_field", "confirmation_id and idempotency_key are required")
    with STORE.lock:
        if key in STORE.tts:
            r = dict(STORE.tts[key])
            if r["confirmation_id"] != cid:
                raise ApiError(409, "idempotency_key_reused", "key already used for another text")
            r["replayed"] = True
            return 200, r
        c = STORE.conf.get(cid)
        if not c:
            raise ApiError(404, "unknown_confirmation", "no such confirmation_id")
        if not c["valid"]:
            raise ApiError(409, "confirmation_superseded",
                           "text was edited after this confirmation; confirm again")
        r = dict(tts_id=str(uuid.uuid4()), confirmation_id=cid, text=c["confirmed_text"],
                 text_sha256=c["text_sha256"], engine="client_speech_synthesis",
                 replayed=False, created_at=now())
        STORE.tts[key] = r
    return 200, r


def h_jamo(q, body):
    try:
        req = json.loads(body or b"{}")
    except ValueError:
        raise ApiError(400, "bad_json", "body must be JSON")
    pairs = req.get("pairs") or []
    if not isinstance(pairs, list) or not pairs:
        raise ApiError(422, "no_pairs", "pairs must be a non-empty list of {ref, hyp}")
    return 200, jamo_stats.compute(pairs, int(req.get("min_support", 20)))


def h_prompts(q, body):
    global POOL_ITEMS
    try:
        req = json.loads(body or b"{}")
    except ValueError:
        raise ApiError(400, "bad_json", "body must be JSON")
    strat = req.get("strategy", "random")
    if strat != "random":
        raise ApiError(501, "strategy_not_implemented",
                       "v1 implements 'random' only; 'coverage' and 'error_based' are planned")
    if POOL_ITEMS is None:
        if not os.path.isfile(POOL):
            raise ApiError(503, "prompt_pool_missing", "script_pool.json not found", path=POOL)
        d = json.load(open(POOL, encoding="utf-8"))
        # sentence-level catalogue entries only (task codes 02-03, 02-04, 06-01)
        POOL_ITEMS = sorted((k, v) for k, v in d.items() if k[:5] in ("02-03", "02-04", "06-01"))
    excl = set(req.get("exclude_prompt_ids") or [])
    cand = [kv for kv in POOL_ITEMS if kv[0] not in excl]
    n = max(1, min(int(req.get("n", 10)), 50))
    rnd = random.Random(req.get("seed", 0))
    pick = rnd.sample(cand, min(n, len(cand)))
    return 200, dict(strategy="random", strategy_version="prompt-random-v1",
                     seed=req.get("seed", 0), pool_size=len(POOL_ITEMS),
                     prompts=[dict(prompt_id=k, text=v) for k, v in pick])


def h_train(q, body):
    raise ApiError(501, "not_implemented_in_demo",
                   "v1 trains offline (b1_train.py on Colab); job API is specified in the "
                   "contract but not served by this demo")


# ------------------------------------------------------------------ http plumbing
class H(BaseHTTPRequestHandler):
    server_version = "sw-ai/1"

    def _send(self, code, obj, ctype="application/json; charset=utf-8"):
        b = obj if isinstance(obj, bytes) else json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > 5 * 1024 * 1024:
            raise ApiError(413, "body_too_large", "max 5 MB")
        return self.rfile.read(n) if n else b""

    def _route(self, method):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        p = u.path
        try:
            if method == "GET" and p in ("/", "/index.html"):
                return self._send(200, open(os.path.join(HERE, "static", "index.html"), "rb").read(),
                                  "text/html; charset=utf-8")
            if method == "GET" and p.startswith("/samples/"):
                name = os.path.basename(p)
                fp = os.path.join(HERE, "samples", name)
                if not os.path.isfile(fp):
                    raise ApiError(404, "not_found", "no such sample")
                ctype = "audio/wav" if name.endswith(".wav") else "application/json; charset=utf-8"
                return self._send(200, open(fp, "rb").read(), ctype)
            if method == "GET" and p == "/v1/health":
                return self._send(*h_health(q, None))
            body = self._body()
            if method == "POST" and p == "/v1/asr/transcribe":
                return self._send(*h_transcribe(q, body))
            if method == "POST" and p.startswith("/v1/transcriptions/") and p.endswith("/confirm"):
                return self._send(*h_confirm(p.split("/")[3], body))
            if method == "POST" and p == "/v1/tts":
                return self._send(*h_tts(q, body))
            if method == "POST" and p == "/v1/analysis/jamo-errors":
                return self._send(*h_jamo(q, body))
            if method == "POST" and p == "/v1/enroll/next-prompts":
                return self._send(*h_prompts(q, body))
            if method == "POST" and p == "/v1/adapters/train":
                return self._send(*h_train(q, body))
            raise ApiError(404, "not_found", "no route %s %s" % (method, p))
        except ApiError as e:
            return self._send(e.http, dict(error=dict(code=e.code, message=e.message, **e.detail)))
        except Exception as e:                                           # noqa: BLE001
            return self._send(500, dict(error=dict(code="internal", message=repr(e))))

    def do_GET(self):
        self._route("GET")

    def do_POST(self):
        self._route("POST")

    def log_message(self, fmt, *args):
        sys.stderr.write("%s %s\n" % (time.strftime("%H:%M:%S"), fmt % args))


def main():
    global ENG
    ENG = MockEngine() if ENGINE == "mock" else HFEngine()
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    print("engine=%s base=%s adapters=%s" % (ENG.name, ENG.base_model, list(ENG.adapters)))
    print("open http://%s:%d/" % (host, port))
    ThreadingHTTPServer((host, port), H).serve_forever()


if __name__ == "__main__":
    main()
