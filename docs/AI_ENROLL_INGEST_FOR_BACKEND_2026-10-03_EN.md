# Enrollment ingest is live — AI → Backend, 2026-10-03

Status: **pushed to `42VoiceBridge_AI` `main`, commit `1d3fbed`.** Your question **T3** is answered
in code, not in prose. `POST /v1/adapters/train` is reachable now.

**That removes the AI-side reason for your `503 TRAINING_UNAVAILABLE`; it does not by itself make
it safe to drop.** Your comment in `TrainPersonalizationModelService` cites our 501 and our
missing job route, and both of those facts have changed. But accepting a public train request
still needs your client work: deliver the recordings and verify receipt, submit with an agreed
retry/idempotency rule, store the BE-job ↔ AI-job mapping, synchronise outcomes including a
normal `rejected`, and confirm actual activation before you report personalization as active.
Keep the 503 until those exist; just stop attributing it to us.

Two things to know before reading further:

1. There is still **no deployed AI server**. `voicebridge.ai.http.base-url` defaults to
   `http://127.0.0.1:8000` and that is correct — you run the server yourself (§6). Nothing
   changed about that.
2. Your repo note said our `main` was at `0e31bd3` (2026-09-22) with no newer commits. That was
   accurate at the time. Everything below, plus the training worker, the promotion gate and the
   generation-config unification, landed in `6c6ea2e` and `1d3fbed` on 2026-10-03.

---

## 1. The three routes

All take `user_id` and `prompt_id` as query values. Both are restricted to
`[A-Za-z0-9._-]{1,128}` because they become path components; anything else is `400`.

### POST /v1/enroll/recordings → 201

Raw WAV body. Same framing and the **same audio limits as `/v1/asr/transcribe`**: PCM 16-bit mono
16 kHz, 0.3–30 s, body ≤ 5 MB. Transcoding stays yours (ffmpeg), as before.

```
POST /v1/enroll/recordings?user_id=<uuid>&prompt_id=<id from next-prompts>
Content-Type: audio/wav
<wav bytes>
```

```json
{
  "contract_version": "ai-contract-v1",
  "user_id": "be_u1",
  "item": {
    "file": "e_6a4aca141243.wav",
    "text": "<the pooled prompt text>",
    "prompt_id": "06-010537",
    "text_source": "prompt_pool",
    "reviewed": false,
    "audio_sha256": "8e06067761d9...",
    "bytes": 62764, "duration_sec": 1.96, "dbfs": -23.4,
    "created_at": "2026-10-03 10:27:21"
  },
  "replaced_previous_take": false,
  "pool_version": "script-pool-v1", "pool_sha256": "…",
  "n_enrolled": 1, "n_unreviewed": 1,
  "min_enroll": 15, "min_gate": 5, "trainable": false
}
```

`trainable` tells you whether `POST /v1/adapters/train` would be accepted right now. Use it
instead of reimplementing our thresholds.

**Re-sending a `prompt_id` replaces that take** (`replaced_previous_take: true`) and
`n_enrolled` does not grow. This is deliberate, not deduplication by accident: two recordings of
one sentence cannot be split into train and gate, because the gate would then be holding a
sentence out from itself. Your T5 asked about re-recording — this is the answer. If you want both
takes kept, say so; it needs a different split rule, not a different route.

### GET /v1/enroll/recordings → 200

Counts and per-item provenance for one user. **Returns no label text.**

```json
{"n_enrolled": 2, "n_unreviewed": 1, "min_enroll": 15, "min_gate": 5, "trainable": false,
 "items": [{"prompt_id": "06-010537", "file": "e_6a4aca141243.wav",
            "text_source": "prompt_pool", "reviewed": false,
            "audio_sha256": "8e06…", "duration_sec": 1.96,
            "created_at": "2026-10-03 10:27:21"}]}
```

### DELETE /v1/enroll/recordings → 200

Consent withdrawal for one recording. The audio file **and** its manifest entry both go.

```json
{"deleted_prompt_id": "06-010537", "audio_deleted": true,
 "adapters_retired": false,
 "note": "existing adapters trained from this recording are not retired by this call",
 "n_enrolled": 1, "trainable": false}
```

`adapters_retired` is **always `false` in v1**, and it is in the response so you cannot assume
withdrawal propagated. Retiring or retraining after a withdrawal is your **T9**, still open — see
§5. A second DELETE of the same `prompt_id` is `404`, not a silent `200`.

---

## 2. The training label never comes from the request body

This is the part to read twice, because it is the answer to your **T4** and it constrains your UI.

The label is the canonical text of `prompt_id` in the **pinned prompt pool** — the same pool
`/v1/enroll/next-prompts` draws from, identified by `pool_version` + `pool_sha256`. You cannot set
it by sending text in the body. An unknown `prompt_id` is `404 unknown_prompt_id`.

Why: a recommended sentence is what the user was **asked** to say. It is not what they said. Our
target users are dysarthric speakers; training on the prompt teaches the model a reading the user
may never have produced. Your own question put it correctly — a suggested sentence or a
confirmation string must not become a training label by accident.

So a prompt-derived label is marked `reviewed: false`, and:

```
POST /v1/adapters/train?user_id=… → 409
{"error": {"code": "enrollment_unreviewed",
           "message": "1 of 1 recordings have no human-reviewed transcript; …",
           "unreviewed_prompt_ids": ["06-010537"], "n_unreviewed": 1}}
```

To supply a reviewed transcript, pass `text_b64` — base64url of the UTF-8 text of what the user
actually read:

```
POST /v1/enroll/recordings?user_id=…&prompt_id=…&text_b64=<base64url>
```

That item becomes `text_source: "reviewed_transcript"`, `reviewed: true`. It is sent as a query
value rather than a path segment, and the server strips the query string before logging, so the
transcript does not land in the access log.

**To unblock integration before you have a review UI:** start the server with
`ASR_ALLOW_UNREVIEWED=1`. Training then proceeds on prompt text and the job record carries
`trained_on_unreviewed: true`. Use it for plumbing tests. Do not use it for a model you intend to
serve to a real user, and do not display such a model as personalized without saying what it was
trained on.

---

## 3. End-to-end, as you would call it

```
POST /v1/enroll/next-prompts        {"n": 20, "seed": …}   → prompt_id + text (+ pool_sha256)
   show the prompt, record, normalise to 16 kHz mono PCM16
POST /v1/enroll/recordings          per recording           → 201, n_enrolled, trainable
   (optional) human reviews the reading → re-POST with text_b64
GET  /v1/enroll/recordings                                  → trainable: true ?
POST /v1/adapters/train                                     → 202 + job_id
GET  /v1/jobs/{job_id}                                      → state
```

Terminal states: `promoted | rejected | needs_review | failed`.

`promoted` means the candidate beat the **incumbent** on a held-out gate **and** the active
pointer was rewritten, so the next transcribe request for that user uses it. `rejected` means it
did not beat the incumbent (a tie does not promote). **`rejected` is a normal outcome, not a
failure** — our 2026-09-26 measurements include a speaker who is worse with an adapter than
without. Your UI needs a non-error way to say it.

There is still **no numeric progress**. Keep `progress: null` (your T8).

To verify *which* model ran on a given utterance, send the **same WAV twice** — once with
`use_adapter=true`, once with `use_adapter=false` — and compare `model.adapter_id`. A
client-supplied `adapter_id` query value is deliberately **ignored**: a caller must not be able to
select another user's adapter, and there is no authentication here to make that safe. That is
asserted by `test_activation.py` group 8.

Verify which model actually ran from the transcribe response, not from your DB:
`model.adapter_id` plus `model.base_reason` (`no_active_adapter`, `base_requested`,
`adapter_wrong_owner`, `adapter_not_found`, `adapter_base_mismatch`, `adapter_load_error`;
`null` means an adapter ran). That is your **A4/A5**.

---

## 4. Why push and not pull — a decision we want your view on

We built **you push to us**. The alternative — we pull from your storage by key or presigned
URL — is the better production shape, and we did not build it only because we have no credentials
or signed-URL scheme from you.

The cost of push is concrete: **user audio now exists in two places.** Consent withdrawal has to
reach both. That is why DELETE shipped in the same commit rather than later.

If you would rather give us short-lived signed GET URLs plus a manifest (`prompt_id`,
`audio_sha256`, reviewed transcript), say so and we will switch. Then your storage stays the only
copy of the audio and your existing delete path is the only one that matters. This is your call as
much as ours; we are not treating push as settled.

---

## 5. Still open, stated rather than left for you to find

| | |
|---|---|
| **Adapter retirement after withdrawal (your T9)** | Deleting a recording does not retire adapters trained from it. No policy agreed. Options: retire immediately, retrain without the item, or record the lineage and decide per request. Needs your product/legal view. |
| **No authentication** | The server has none. If anything but localhost can reach it, you own the boundary. |
| **First-request latency** | The model loads at startup, but the first inference after that is ~2.7 s on a laptop CPU (warm: 640–680 ms). Your `read-timeout-ms` is 10000, so you are fine — but do not call it during startup. |
| ~~`base_revision` reports `null` when loading from an offline cache~~ **Fixed 2026-10-03** | The value is now confirmed from the artifact: `config._commit_hash` when present, otherwise the commit sha in the `.../snapshots/<sha>/` path of the `config.json` actually loaded. `GET /v1/health` and the transcribe `model` block carry `base_revision_verified` and `base_revision_source`; if neither route confirms it you get the requested pin with `verified: false`, which you should treat as unproven. Measured on the real engine: `973afd24…`, `verified: true`, source `cache_snapshot_path`. Your **A1/A5**. |
| ~~Split membership is not permanent across rounds~~ **Fixed 2026-10-03** | Membership is now persisted per user in `splits.json` and prior assignments are never recomputed, so a second training round cannot place a past training item in the gate. Listed here rather than deleted, because an earlier version of this document told you it was broken. |
| **Prompt-pool redistribution terms** | The pool is derived from AI-Hub 013. Its redistribution conditions have not been checked. Do not expose `/v1/enroll/next-prompts` on a public endpoint until that is settled. |

---

## 6. Running the server

```
pip install -r demo/requirements.txt        # torch, transformers, peft, numpy, safetensors
python3 demo/server.py                      # real engine, downloads whisper-small on first run
```

```
ASR_ENGINE=mock python3 demo/server.py      # no model, deterministic fake text
```

Use mock for contract and plumbing work — every response field in this document is produced by
the same code in both engines, except the transcription text itself. Do **not** accept mock output
as recognition evidence.

Env knobs you may need: `ASR_ENGINE`, `ASR_ALLOW_UNREVIEWED`, `ASR_BASE_REVISION`,
`ASR_MAX_NEW_TOKENS`, `ASR_BEAMS`, `PROMPT_POOL`, `ENROLL_DIR`, `JOB_DIR`, `ADAPTER_DIR`,
`MIN_ENROLL`, `MIN_GATE`, `TRAIN_TIMEOUT`, `HOST`, `PORT`. Overriding `MIN_ENROLL`/`MIN_GATE`
changes a policy we never validated — if you lower them, say so in the job record, because a
5-item enrollment cannot support a 5-item gate.

### What the test suites do and do not prove

An earlier draft of this document said the suites were "all passing on the real engine." That was
wrong and it is corrected here, because you would otherwise read model evidence into a plumbing
result. Three of the four run on the mock engine or on stubs by construction.

| Suite | Engine | What it establishes |
|---|---|---|
| `test_contract.py` 54 checks | **real** (run against a live `transformers+peft` server, 2026-10-03) | HTTP contract: status codes, response fields, audio limits, the enrollment routes, path-traversal refusal |
| `test_core.py` 24 groups | mock (`ASR_ENGINE` defaults to mock) | the `core_*` contract functions, independent of transport |
| `test_activation.py` 7 groups | mock, in a real server subprocess | that an adapter promoted while the server is running is picked up, and rollback — adapter *selection*, not recognition |
| `test_train_worker.py` 7 groups | trainer and gate **stubbed** | job state machine, split disjointness, tie-does-not-promote, failure never touching `active.json` |

So: a real HTTP process is not a real inference test. The only real-weight evidence we have is
separate from these suites — a single-speaker adapter run and the short-utterance evaluation on
two speakers, both on rented GPUs, both reported with their own caveats. Nothing here measures
recognition quality.

`docs/openapi_ai_v1.yaml` now documents these three routes, `POST /v1/adapters/train` 202/409 and
`GET /v1/jobs/{job_id}` with the full job schema. It was behind the code when you read it on
10-02; it is not any more.
