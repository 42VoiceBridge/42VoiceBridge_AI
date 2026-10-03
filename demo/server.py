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
import base64
import hashlib
import json
import os
import random
import re
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
# D-5: every adapter we ship was trained against this snapshot. Loading any other one silently
# mismatches them, so it is pinned here and checked after load. Same hash as
# experiments/t10_highcer/eval_longform.py and b1_train.py.
BASE_REVISION = os.environ.get("ASR_BASE_REVISION", "973afd24965f72e36ca33b3055d56a652f456b4d")
# Greedy by default (D9, 2026-09-18). At beam_size=5 whisper-small emitted repetition loops on
# short utterances (`아, 그래요?` -> 200 tokens of `아`), and one such segment can dominate a
# pooled CER. Accuracy is NOT the reason for greedy: it is a wash (better on the KJW test split,
# worse on the 10 demo samples). The reasons are that greedy cannot produce the loop, and median
# demo latency fell 1554 -> 666 ms. See HO §5.2 and CT §0 D9.
import decoding                      # the one generation configuration (A2, 2026-10-03)
BEAMS = decoding.BEAMS
MAX_BODY = 5 * 1024 * 1024        # request body cap, shared by both framings
ENGINE = os.environ.get("ASR_ENGINE", "hf")
ADAPTER_DIR = os.environ.get("ASR_ADAPTERS", os.path.join(HERE, "adapters"))
POOL_VERSION = "script-pool-v1"   # bump when the pool file is replaced
POOL_SHA = None                   # content hash of the SELECTED items, filled on first load
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

    def ensure_adapter(self, aid):
        """A3: load an adapter written AFTER startup. Returns (meta, error)."""
        if aid in self.adapters:
            return self.adapters[aid], None
        meta = scan_adapters(ADAPTER_DIR).get(aid)
        if meta is None:
            return None, "not_found"
        self.adapters[aid] = meta
        return meta, None

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
        self.proc = WhisperProcessor.from_pretrained(BASE, revision=BASE_REVISION,
                                                     language="korean", task="transcribe")
        m = WhisperForConditionalGeneration.from_pretrained(BASE, revision=BASE_REVISION)
        try:
            m.generation_config.forced_decoder_ids = None
        except Exception:                                     # noqa: BLE001
            pass
        self.base_model = BASE
        self.base_revision = getattr(m.config, "_commit_hash", None)
        # D-5: the revision was recorded but never enforced, so the hub could have served a
        # different snapshot and every adapter trained against the pinned one would be mismatched.
        if BASE_REVISION and self.base_revision not in (None, BASE_REVISION):
            raise RuntimeError("base revision mismatch: loaded %s, pinned %s"
                               % (self.base_revision, BASE_REVISION))
        self.model = m.to(self.device).eval()
        self.peft = False
        self.adapters = {}
        self.failed = {}                      # adapter_id -> why it could not be loaded
        for aid, meta in scan_adapters(ADAPTER_DIR).items():
            self._load(aid, meta)

    def _load(self, aid, meta):
        """Load one adapter into the live model. Returns None on success, else a reason string."""
        if meta.get("base_model") not in (None, BASE):
            why = "trained on %s, server base is %s" % (meta.get("base_model"), BASE)
            print("skip adapter %s: %s" % (aid, why))
            self.failed[aid] = "base_mismatch"
            return "base_mismatch"
        try:
            from peft import PeftModel
            if not self.peft:
                self.model = PeftModel.from_pretrained(self.model, meta["path"], adapter_name=aid)
                self.peft = True
            else:
                self.model.load_adapter(meta["path"], adapter_name=aid)
            self.model.eval()
        except Exception as e:                                # noqa: BLE001
            print("adapter %s failed to load: %s" % (aid, e))
            self.failed[aid] = "load_error"
            return "load_error"
        self.adapters[aid] = meta
        self.failed.pop(aid, None)
        print("loaded adapter %s (%s)" % (aid, meta.get("user_id")))
        return None

    def ensure_adapter(self, aid):
        """A3: the worker writes adapters AFTER startup, so a miss must rescan, not fall back.

        Returns (meta, error). A previous hard failure is remembered so a broken adapter does not
        re-attempt a model mutation on every request.
        """
        if aid in self.adapters:
            return self.adapters[aid], None
        if aid in self.failed:
            return None, self.failed[aid]
        meta = scan_adapters(ADAPTER_DIR).get(aid)
        if meta is None:
            return None, "not_found"
        why = self._load(aid, meta)
        return (None, why) if why else (self.adapters[aid], None)

    def transcribe(self, x, adapter_id):
        torch = self.torch
        f = self.proc.feature_extractor(x, sampling_rate=16000, return_tensors="pt").input_features
        f = f.to(self.device)
        kw = dict(input_features=f, **decoding.settings())
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
        self.by_request = {}  # (user_id, request_id) -> transcription_id   (D-3)


STORE = Store()
INFER_LOCK = threading.Lock()
ENG = None
POOL_ITEMS = None


def active_map():
    """user_id -> adapter_id, from adapters/active.json. Missing file or entry means the BASE model.

    D-4: selection used to be "newest created_at wins", so an adapter that had never been validated
    became live the moment it was written. Serving the base is a supported state, not a failure:
    the 2026-09-26 run measured a speaker for whom every adapter was worse than the base.
    """
    p = os.path.join(ADAPTER_DIR, "active.json")
    if not os.path.isfile(p):
        return {}
    try:
        m = json.load(open(p, encoding="utf-8"))
    except Exception as e:                                    # noqa: BLE001
        print("active.json unreadable (%s); serving the base model for everyone" % e)
        return {}
    return m if isinstance(m, dict) else {}


def user_adapter(user_id):
    """(meta, reason). meta is None whenever the BASE model will run; reason says why.

    A5: `adapter_id: null` used to be ambiguous between "base was requested", "the pointer is
    missing" and "the adapter would not load". The reason is now reported in the response.
    """
    aid = active_map().get(user_id)
    if not aid:
        return None, "no_active_adapter"
    ad, err = ENG.ensure_adapter(aid)        # A3: rescans, so an adapter promoted after startup loads
    if ad is None:
        print("active adapter %r for %r unusable (%s); serving the base" % (aid, user_id, err))
        return None, "adapter_" + (err or "unavailable")
    if ad.get("user_id") != user_id:
        print("active.json maps %r to an adapter owned by %r; refusing" % (user_id, ad.get("user_id")))
        return None, "adapter_wrong_owner"
    return ad, None


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
# ---------------------------------------------------------------- core (no HTTP in here)
# These four functions ARE the contract. The HTTP handlers below only parse and hand over, and
# the JPyRust Python daemon calls the same functions directly (decided 2026-09-25: the backend
# is Java and reaches this process over JNI + shared memory, not HTTP). Anything transport
# specific - status codes, query strings, raw bodies - stays out of them.


def core_health():
    return dict(status="ok", contract_version=CONTRACT_VERSION, engine=ENG.name,
                     base_model=ENG.base_model, base_revision=ENG.base_revision,
                     preprocessing_version=PREPROC_VERSION, beam_size=BEAMS,
                     generation=decoding.describe(),
                     adapters=[dict(adapter_id=m["adapter_id"], user_id=m["user_id"],
                                    adapter_revision=m["adapter_revision"],
                                    size_bytes=m["size_bytes"], label=m.get("label"),
                                    evidence=m.get("evidence"))
                               for m in ENG.adapters.values()],
                     limits=dict(min_sec=MIN_SEC, max_sec=MAX_SEC, sample_rate=16000,
                                 channels=1, bits=16))


def core_transcribe(audio_wav, user_id, use_adapter=True, request_id=None):
    """audio_wav: raw WAV bytes (PCM16 mono 16 kHz). Returns the transcription record."""
    user_id = (user_id or "").strip()
    if not user_id:
        raise ApiError(400, "missing_user_id", "user_id is required")
    # D-3: request_id used to be echoed and otherwise ignored, so a client retry after a timeout
    # ran inference again and produced a second transcription_id for one user action. Now the
    # first result comes back unchanged. Keyed per user so ids cannot collide across users, and
    # only for client-supplied ids - a generated one can never be retried against.
    client_rid = (request_id or "").strip() or None
    if client_rid:
        with STORE.lock:
            prev = STORE.by_request.get((user_id, client_rid))
            if prev:
                return STORE.tx[prev]
    request_id = client_rid or str(uuid.uuid4())
    body = audio_wav
    x, dur, dbfs = parse_wav(body)
    ad, adapter_note = (user_adapter(user_id) if use_adapter else (None, "base_requested"))
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
                   adapter_revision=ad["adapter_revision"] if ad else None,
                   # A5: why the base ran, when it ran. null when an adapter ran.
                   base_reason=adapter_note),
        decoding=decoding.describe(),
        audio=dict(sha256=sha256(body), duration_sec=round(dur, 3), rms_dbfs=dbfs,
                   preprocessing_version=PREPROC_VERSION),
        latency_ms=int((time.time() - t0) * 1000), created_at=now())
    with STORE.lock:
        STORE.tx[res["transcription_id"]] = res
        if client_rid:
            STORE.by_request[(user_id, client_rid)] = res["transcription_id"]
    return res


def _consent_flag(consent, name):
    """Strict: only a real JSON boolean grants permission. `"false"` used to become True."""
    v = consent.get(name, False)
    if isinstance(v, bool):
        return v
    raise ApiError(422, "invalid_consent",
                   "consent.%s must be a JSON boolean, got %s" % (name, type(v).__name__))


def core_confirm(tid, req):
    if not isinstance(req, dict):
        raise ApiError(400, "bad_request", "body must be a JSON object")
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
        if not isinstance(consent, dict):
            raise ApiError(422, "invalid_consent", "consent must be a JSON object")
        prev = STORE.latest_conf.get(tid)
        c = dict(confirmation_id=str(uuid.uuid4()), transcription_id=tid,
                 contract_version=CONTRACT_VERSION,
                 based_on_revision=tx["revision"], confirmed_text=text,
                 text_sha256=sha256(text.encode("utf-8")),
                 source="asr_unedited" if text == tx["text"] else "user_edited",
                 edit_distance_syl=syl_edit_distance(tx["text"], text),
                 consent=dict(store_audio=_consent_flag(consent, "store_audio"),
                              use_for_training=_consent_flag(consent, "use_for_training")),
                 supersedes=prev, valid=True, confirmed_at=now())
        if prev:
            STORE.conf[prev]["valid"] = False
        STORE.conf[c["confirmation_id"]] = c
        STORE.latest_conf[tid] = c["confirmation_id"]
    return c


def core_tts(req):
    if not isinstance(req, dict):
        raise ApiError(400, "bad_request", "body must be a JSON object")
    cid, key = req.get("confirmation_id"), req.get("idempotency_key")
    if not cid or not key:
        raise ApiError(400, "missing_field", "confirmation_id and idempotency_key are required")
    with STORE.lock:
        # validity FIRST: a cached authorization must never resurrect superseded text.
        c = STORE.conf.get(cid)
        if not c:
            raise ApiError(404, "unknown_confirmation", "no such confirmation_id")
        if not c["valid"]:
            raise ApiError(409, "confirmation_superseded",
                           "text was edited after this confirmation; confirm again")
        if key in STORE.tts:
            r = dict(STORE.tts[key])
            if r["confirmation_id"] != cid:
                raise ApiError(409, "idempotency_key_reused", "key already used for another text")
            r["replayed"] = True
            return r
        r = dict(contract_version=CONTRACT_VERSION,
                 tts_id=str(uuid.uuid4()), confirmation_id=cid, text=c["confirmed_text"],
                 text_sha256=c["text_sha256"], engine="client_speech_synthesis",
                 replayed=False, created_at=now())
        STORE.tts[key] = r
    return r


# ---------------------------------------------------------------- HTTP handlers (thin)


def h_health(q, body):
    return 200, core_health()


def h_transcribe(q, body):
    return 200, core_transcribe(body, (q.get("user_id") or [""])[0],
                                (q.get("use_adapter") or ["true"])[0].lower() != "false",
                                (q.get("request_id") or [None])[0])


def _json_body(body):
    try:
        req = json.loads(body or b"{}")
    except ValueError:
        raise ApiError(400, "bad_json", "body must be JSON")
    return req


def h_confirm(tid, body):
    return 200, core_confirm(tid, _json_body(body))


def h_tts(q, body):
    return 200, core_tts(_json_body(body))


def h_jamo(q, body):
    try:
        req = json.loads(body or b"{}")
    except ValueError:
        raise ApiError(400, "bad_json", "body must be JSON")
    pairs = req.get("pairs") or []
    if not isinstance(pairs, list) or not pairs:
        raise ApiError(422, "no_pairs", "pairs must be a non-empty list of {ref, hyp}")
    out = jamo_stats.compute(pairs, int(req.get("min_support", 20)))
    out["contract_version"] = CONTRACT_VERSION
    return 200, out


def pool_items():
    """[(prompt_id, text)] from the pinned prompt pool, loaded once. Also fills POOL_SHA.

    Shared by /v1/enroll/next-prompts and enrollment ingest so a recording can never be
    labelled from a different pool than the one the prompt was recommended from.
    """
    global POOL_ITEMS, POOL_SHA
    if POOL_ITEMS is None:
        if not os.path.isfile(POOL):
            raise ApiError(503, "prompt_pool_missing", "script_pool.json not found", path=POOL)
        d = json.load(open(POOL, encoding="utf-8"))
        # sentence-level catalogue entries only (task codes 02-03, 02-04, 06-01)
        POOL_ITEMS = sorted((k, v) for k, v in d.items() if k[:5] in ("02-03", "02-04", "06-01"))
        # R5 (backend, 2026-10-03): the same seed can return different prompts if the pool changes.
        # This hashes the SELECTED items, so it identifies what recommendations are drawn from -
        # deliberately not the strategy version, which identifies the algorithm.
        POOL_SHA = hashlib.sha256(
            "\n".join("%s\t%s" % kv for kv in POOL_ITEMS).encode("utf-8")).hexdigest()[:16]
    return POOL_ITEMS


def h_prompts(q, body):
    try:
        req = json.loads(body or b"{}")
    except ValueError:
        raise ApiError(400, "bad_json", "body must be JSON")
    strat = req.get("strategy", "random")
    if strat != "random":
        raise ApiError(501, "strategy_not_implemented",
                       "v1 implements 'random' only; 'coverage' and 'error_based' are planned")
    pool_items()
    excl = set(req.get("exclude_prompt_ids") or [])
    cand = [kv for kv in POOL_ITEMS if kv[0] not in excl]
    n = max(1, min(int(req.get("n", 10)), 50))
    rnd = random.Random(req.get("seed", 0))
    pick = rnd.sample(cand, min(n, len(cand)))
    return 200, dict(contract_version=CONTRACT_VERSION,
                     strategy="random", strategy_version="prompt-random-v1",
                     seed=req.get("seed", 0), pool_size=len(POOL_ITEMS),
                     pool_version=POOL_VERSION, pool_sha256=POOL_SHA,
                     prompts=[dict(prompt_id=k, text=v) for k, v in pick])


# ------------------------------------------------------------------ enrollment ingest
# Backend question T3 (2026-10-02): the training worker reads
# demo/enroll/<user_id>/{pairs.json,*.wav} off local disk, so until this existed the backend had
# no way to deliver recordings at all and /v1/adapters/train was unreachable for them.
#
# Shape chosen: the backend PUSHES one recording per call, raw WAV body, same framing as
# /v1/asr/transcribe. The alternative (we PULL from their storage by key or presigned URL) avoids
# a second copy of user audio and is the better production shape; it needs credentials or signed
# URLs we do not have, so it is not built. Said plainly to the backend rather than assumed.
#
# The label never comes from the request. It is the canonical text of the prompt that was
# recommended, looked up in the pinned pool - their T4 asks for exactly this, because a
# recommended sentence or a confirmation string must not become a training label by accident.
# A human-reviewed transcript may override it (text_b64), and only then is the item `reviewed`.
UNREVIEWED_OK = os.environ.get("ASR_ALLOW_UNREVIEWED", "") not in ("", "0", "false", "False")


def safe_id(v, what):
    """Path components come from the network. Anything but [A-Za-z0-9._-] is refused."""
    v = (v or "").strip()
    if not v:
        raise ApiError(400, "missing_" + what, "%s is required" % what)
    if len(v) > 128 or not re.fullmatch(r"[A-Za-z0-9._-]+", v) or v.startswith("."):
        raise ApiError(400, "bad_" + what, "%s must match [A-Za-z0-9._-]{1,128}" % what)
    return v


def enroll_dir(user_id, make=False):
    import train_worker
    d = os.path.join(train_worker.ENROLL_DIR, safe_id(user_id, "user_id"))
    if make:
        os.makedirs(d, exist_ok=True)
    return d


def read_pairs(d):
    p = os.path.join(d, "pairs.json")
    if not os.path.isfile(p):
        return []
    v = json.load(open(p, encoding="utf-8"))
    return v.get("pairs", v) if isinstance(v, dict) else v


def write_pairs(d, items):
    p = os.path.join(d, "pairs.json")
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(dict(pairs=items), f, ensure_ascii=False, indent=1)
    os.replace(tmp, p)                      # atomic: the worker never reads a half-written list


def core_enroll_put(user_id, prompt_id, wav, text_b64=None):
    """Store one enrollment recording. Re-recording the same prompt REPLACES the old take."""
    d = enroll_dir(user_id, make=True)
    pid = safe_id(prompt_id, "prompt_id")
    text = dict(pool_items()).get(pid)
    if text is None:
        raise ApiError(404, "unknown_prompt_id",
                       "not in the pinned prompt pool; recommend prompts via "
                       "/v1/enroll/next-prompts and send back the prompt_id you showed",
                       pool_version=POOL_VERSION, pool_sha256=POOL_SHA)
    source, reviewed = "prompt_pool", False
    if text_b64:
        try:
            text = base64.urlsafe_b64decode(text_b64 + "=" * (-len(text_b64) % 4)).decode("utf-8")
        except Exception as e:                                       # noqa: BLE001
            raise ApiError(400, "bad_text_b64", "text_b64 must be base64url of UTF-8: %s" % e)
        if not text.strip():
            raise ApiError(422, "empty_text", "reviewed transcript is empty")
        source, reviewed = "reviewed_transcript", True

    _, dur, dbfs = parse_wav(wav)           # same limits as transcribe: PCM16 mono 16k, 0.3-30 s
    name = "e_%s.wav" % sha256(pid.encode())[:12]
    with open(os.path.join(d, name + ".tmp"), "wb") as f:
        f.write(wav)
    os.replace(os.path.join(d, name + ".tmp"), os.path.join(d, name))

    item = dict(file=name, text=text.strip(), prompt_id=pid, text_source=source,
                reviewed=reviewed, audio_sha256=sha256(wav), bytes=len(wav),
                duration_sec=round(dur, 3), dbfs=dbfs,
                created_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    items = [x for x in read_pairs(d) if x.get("prompt_id") != pid]
    replaced = len(read_pairs(d)) - len(items)
    items.append(item)
    write_pairs(d, items)

    import train_worker
    return dict(contract_version=CONTRACT_VERSION, user_id=user_id, item=item,
                replaced_previous_take=bool(replaced),
                pool_version=POOL_VERSION, pool_sha256=POOL_SHA,
                **enroll_counts(items, train_worker))


def enroll_counts(items, tw):
    n = len(items)
    unrev = sum(1 for x in items if not x.get("reviewed", True))
    return dict(n_enrolled=n, n_unreviewed=unrev,
                min_enroll=tw.MIN_ENROLL, min_gate=tw.MIN_GATE,
                trainable=n >= tw.MIN_ENROLL and (unrev == 0 or UNREVIEWED_OK))


def core_enroll_list(user_id):
    """Counts and per-item provenance. Deliberately returns no label text."""
    import train_worker
    items = read_pairs(enroll_dir(user_id))
    return dict(contract_version=CONTRACT_VERSION, user_id=user_id,
                items=[dict(prompt_id=x.get("prompt_id"), file=x["file"],
                            text_source=x.get("text_source", "manifest"),
                            reviewed=x.get("reviewed", True),
                            audio_sha256=x.get("audio_sha256"),
                            duration_sec=x.get("duration_sec"),
                            created_at=x.get("created_at")) for x in items],
                **enroll_counts(items, train_worker))


def core_enroll_delete(user_id, prompt_id):
    """Consent withdrawal for one recording: the audio file AND its pair entry both go.

    Adapters already trained from it are NOT retired by this call - that is a separate decision
    (backend T9). The response says so rather than letting the caller assume it was handled.
    """
    import train_worker
    d = enroll_dir(user_id)
    pid = safe_id(prompt_id, "prompt_id")
    items = read_pairs(d)
    keep = [x for x in items if x.get("prompt_id") != pid]
    if len(keep) == len(items):
        raise ApiError(404, "unknown_prompt_id", "this user has no recording for that prompt_id")
    for x in items:
        if x.get("prompt_id") == pid:
            try:
                os.remove(os.path.join(d, x["file"]))
            except FileNotFoundError:
                pass
    write_pairs(d, keep)
    return dict(contract_version=CONTRACT_VERSION, user_id=user_id, deleted_prompt_id=pid,
                audio_deleted=True, adapters_retired=False,
                note="existing adapters trained from this recording are not retired by this call",
                **enroll_counts(keep, train_worker))


def core_train(user_id):
    """Queue a per-user adapter run. Promotion is decided by train_worker's held-out gate."""
    user_id = (user_id or "").strip()
    if not user_id:
        raise ApiError(400, "missing_user_id", "user_id is required")
    import train_worker
    d = enroll_dir(user_id)
    if not os.path.isfile(os.path.join(d, "pairs.json")):
        raise ApiError(404, "no_enrollment",
                       "no enrollment for this user; POST recordings to "
                       "/v1/enroll/recordings first",
                       expected_path=os.path.join(d, "pairs.json"))
    items = read_pairs(d)
    unrev = [x.get("prompt_id") for x in items if not x.get("reviewed", True)]
    # Backend T4: a recommended sentence is what the user was ASKED to say, not what they said.
    # Training on it teaches the model a reading it may never have produced. Items ingested
    # without a reviewed transcript are refused by default; the override exists so integration
    # can be tested before a review UI exists, and it is reported in the response.
    if unrev and not UNREVIEWED_OK:
        raise ApiError(409, "enrollment_unreviewed",
                       "%d of %d recordings have no human-reviewed transcript; send text_b64 on "
                       "ingest, or set ASR_ALLOW_UNREVIEWED=1 to train on prompt text anyway"
                       % (len(unrev), len(items)),
                       unreviewed_prompt_ids=unrev[:20], n_unreviewed=len(unrev))
    job = train_worker.submit(user_id)
    job["trained_on_unreviewed"] = bool(unrev)
    job["contract_version"] = CONTRACT_VERSION
    return job


def core_job(job_id):
    import train_worker
    job = train_worker.read_job((job_id or "").strip())
    if not job:
        raise ApiError(404, "unknown_job", "no such job_id")
    job["contract_version"] = CONTRACT_VERSION
    return job


def h_train(q, body):
    uid = (q.get("user_id") or [""])[0]
    if not uid and body:
        try:
            uid = (json.loads(body) or {}).get("user_id", "")
        except ValueError:
            raise ApiError(400, "bad_json", "body must be JSON")
    return 202, core_train(uid)


def h_job(job_id):
    return 200, core_job(job_id)


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
        # R3 (backend, 2026-10-03): Spring's RestClient can send JSON with
        # `Transfer-Encoding: chunked` and no Content-Length. Reading Content-Length alone gave an
        # empty body and a misleading 422. Both framings are accepted now.
        if (self.headers.get("Transfer-Encoding") or "").lower().strip() == "chunked":
            out, total = [], 0
            while True:
                line = self.rfile.readline(65).strip()
                if b";" in line:
                    line = line.split(b";", 1)[0]
                try:
                    size = int(line, 16)
                except ValueError:
                    raise ApiError(400, "bad_chunked_body", "malformed chunk size")
                if size == 0:
                    while self.rfile.readline(65).strip():      # trailers
                        pass
                    break
                total += size
                if total > MAX_BODY:
                    raise ApiError(413, "body_too_large", "max 5 MB")
                out.append(self.rfile.read(size))
                self.rfile.read(2)                              # CRLF after each chunk
            return b"".join(out)
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_BODY:
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
            if method == "GET" and p == "/v1/enroll/recordings":
                return self._send(200, core_enroll_list((q.get("user_id") or [""])[0]))
            if method == "DELETE" and p == "/v1/enroll/recordings":
                return self._send(200, core_enroll_delete(
                    (q.get("user_id") or [""])[0], (q.get("prompt_id") or [""])[0]))
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
            if method == "GET" and p.startswith("/v1/jobs/"):
                return self._send(*h_job(p.split("/")[3]))
            if method == "POST" and p == "/v1/adapters/train":
                return self._send(*h_train(q, body))
            if method == "POST" and p == "/v1/enroll/recordings":
                return self._send(201, core_enroll_put(
                    (q.get("user_id") or [""])[0], (q.get("prompt_id") or [""])[0], body,
                    (q.get("text_b64") or [None])[0]))
            raise ApiError(404, "not_found", "no route %s %s" % (method, p))
        except ApiError as e:
            return self._send(e.http, dict(contract_version=CONTRACT_VERSION,
                                           error=dict(code=e.code, message=e.message, **e.detail)))
        except Exception as e:                                           # noqa: BLE001
            return self._send(500, dict(error=dict(code="internal", message=repr(e))))

    def do_GET(self):
        self._route("GET")

    def do_POST(self):
        self._route("POST")

    def do_DELETE(self):
        self._route("DELETE")

    def log_message(self, fmt, *args):
        # the query string can carry a transcript (text_b64); keep it out of the log
        line = (fmt % args).split("?")[0]
        sys.stderr.write("%s %s\n" % (time.strftime("%H:%M:%S"), line))


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
