# AI Side — Full Status for the Backend (context file for Claude) · v2

Contract `ai-contract-v1` · updated 2026-09-25 · AI owner: 제민 (AI/Data) · Team: 42VoiceBridge
**Supersedes `BACKEND_SPEC_FOR_CLAUDE_v1_EN.md` (2026-09-18)**, which overstates what is implemented.
Source of truth if this file and the AI repo disagree: `docs/AI_BACKEND_CONTRACT_v1_EN.md` (CT) and
`docs/openapi_ai_v1.yaml` in `github.com/42VoiceBridge/42VoiceBridge_AI`.

## 0. Instructions to Claude

- You are assisting a backend developer on a 5-person student team (Java 17 / Spring Boot 3 /
  hexagonal; finals 2026-10-28). The AI part is a Python process the backend calls.
- Treat §2–§8 as a contract. **Do not invent fields, endpoints, defaults or behaviour.** If a task
  needs something this file does not define, say so and have the user ask the AI owner.
- Tags: **[M]** measured · **[P]** planned · **[U]** unverified · **[pilot]** two-speaker experiment.
  Never turn a [pilot] number into a product claim in UI copy, docs or the proposal.
- Reply to the user in Korean. Code, identifiers and comments in English.
- §9 lists things the AI side got wrong earlier and corrected. If the user quotes an older document,
  check §9 before agreeing with it.

## 1. What the product is, and what the AI part actually does

Assistive speech-to-text for people with dysarthria (구음장애). The user speaks, the system shows the
recognised text, the user edits and confirms it, and **only confirmed text is spoken aloud**.

The AI side is **one stage, not two**: a frozen `openai/whisper-small` (revision pinned
`973afd24965f72e36ca33b3055d56a652f456b4d`) plus an optional **per-user LoRA adapter**
(3.53 M trainable parameters = 1.44% of the model; artifact 14.18 MB) [M].

- There is **no model adapted on public dysarthric data**. The AI-Hub corpus was used for
  measurement and experiments only. Any design that assumes a "first-stage adapted base model"
  is wrong; the two real cases are **base** (no adapter) and **personalised** (user adapter).
- An adapter is valid **only** on that exact base model and revision. It will not load on
  `whisper-tiny`, `-base`, `-medium`, or on a different revision of `small`.
- No adapter for a user is normal: the base model runs and `model.adapter_id` is `null`.

## 2. Transport: JPyRust (decided 2026-09-25), HTTP for local development

The backend is Java and reaches the recogniser through JPyRust (Spring Boot → Rust JNI → a
**persistent Python daemon** over shared memory). Facts that matter for design:

- It is **not in-process**. A separate Python process still runs; what disappears is HTTP.
- JPyRust's "~2.33 s → ~6 ms" compares against spawning a fresh interpreter per call. Our process
  keeps the model resident, so that speed-up does not apply. **Recognition itself is ~666 ms** [M];
  transport is a fraction of a percent either way. Expect operational benefit, not latency.
- Known blocker to check early: JPyRust's README states its published Docker image ships no Python
  and no `uv`, so inference does not work inside that container as shipped [U].

**The AI side keeps the contract out of the transport.** These four functions are the contract
(`demo/server.py`); the HTTP handlers are thin wrappers over them and the daemon calls the same
functions:

```
core_health()                                                      -> dict
core_transcribe(wav_bytes, user_id, use_adapter=True, request_id=None) -> dict
core_confirm(transcription_id, request_dict)                       -> dict
core_tts(request_dict)                                             -> dict
```

Errors are raised as `ApiError(http, code, message, **detail)`. The HTTP layer maps `http` to a
status code; **the JPyRust layer needs its own mapping**. Do not collapse every failure into one
"AI unavailable" error — `no_speech` is a normal outcome (§4) and must not look like an outage.

Because HTTP disappears, the backend now owns what it used to provide:

| concern | what to implement |
|---|---|
| concurrency | one model, one request at a time. Serialise calls into the daemon |
| timeouts / cancellation | the caller sets a deadline; an expired deadline does **not** stop work already running |
| errors | an explicit `ApiError` → Java exception mapping, branching on `code` |
| readiness | a live process is **not** a ready one: the adapter may have failed to load. Expose a readiness call that reports engine, base model, base revision and loaded adapters |
| adapter artifacts | place each user's 14 MB adapter where the daemon can read it, and trigger a reload |

## 3. Fixed decisions (do not relitigate in code)

| # | Decision |
|---|---|
| D1 | Server-side inference for the prototype. On-device is a later milestone |
| D2 | Base `openai/whisper-small` (pinned revision) + per-user PEFT LoRA adapter, Transformers + PEFT |
| D3 | AI input: **WAV PCM 16-bit, mono, 16 kHz, 0.3–30 s only**, body ≤ ~5 MiB |
| D4 | `alternatives` is always `[]`; `score` and `score_type` are always `null` in v1 |
| D5 | Only confirmed text may be spoken; any edit invalidates the confirmation |
| D6 | The adapter is chosen server-side from `user_id`; clients can never name an adapter |
| D7 | "Weak phoneme diagnosis" is **jamo-level model-error statistics** (`jamo-err-v1`) — it describes the model, not the speaker |
| D8 | Consent to store audio and consent to train are separate flags, both default `false` |
| D9 | Greedy decoding (`beam_size=1`); beam 5 produced repetition loops on short utterances |
| D10 | Transport is JPyRust; the contract lives in the `core_*` functions, HTTP is one wrapper |

## 4. Operations and their fields

Conventions: JSON UTF-8, `snake_case`, UUID v4, ISO-8601 with offset. `null` means *not produced* —
never coerce it to `0` or `""`. Persist the version fields with every row. Branch on the error
`code`, never on `message`.

### 4.1 health
`status`, `contract_version`, `engine`, `base_model`, `base_revision`, `preprocessing_version`,
`beam_size`, `adapters[]` (`adapter_id`, `user_id`, `adapter_revision`, `size_bytes`, `label`,
`evidence`), `limits` (`min_sec` 0.3, `max_sec` 30, `sample_rate` 16000, `channels` 1, `bits` 16).
Call at startup, log it, fail fast if `contract_version != "ai-contract-v1"`.

### 4.2 transcribe
Input: WAV bytes + `user_id` (required, selects that user's adapter), `use_adapter` (default `true`;
`false` forces the base model, used for A/B and fallback), `request_id` (echoed).

Output: `contract_version`, `transcription_id`, `request_id`, `user_id`, `revision` (1),
`status` (`ok` | `no_speech`), `text`, `alternatives` (`[]`), `score` (`null`), `score_type` (`null`),
`model.{engine, base_model, base_revision, adapter_id, adapter_revision}`,
`decoding.{language, task, beam_size}`,
`audio.{sha256, duration_sec, rms_dbfs, preprocessing_version}`, `latency_ms`, `created_at`.

- **`status=no_speech` is not an error.** Two triggers: input RMS below −60 dBFS (a heuristic, [U]),
  or the decoder returned an empty string. The second does **not** prove the user was silent.
  UI wording: "인식된 문장이 없습니다" — not "you did not speak".
- `model.adapter_id = null` = the base model was selected. On the low-RMS path inference does not
  run at all, yet model metadata is still returned; selected model and executed model are not
  distinguished today.
- `latency_ms` **includes queue wait** behind the inference lock. It is not pure inference time.
- `request_id` is echoed but **does not deduplicate retries today** (§7 D-3). Calling twice with the
  same id runs inference twice and creates two `transcription_id`s.
- Measured latency [M]: median ~0.7 s on a laptop CPU for 2–8 s utterances, ~0.85 s with an adapter.

### 4.3 confirm — backend owns this in the product
Input `{revision, confirmed_text, consent: {store_audio, use_for_training}}`. Output
`confirmation_id`, `transcription_id`, `based_on_revision`, `confirmed_text`, `text_sha256`,
`source` (`asr_unedited` | `user_edited`), `edit_distance_syl`, `consent`, `supersedes`, `valid`,
`confirmed_at`. Errors: 404 `unknown_transcription`, 409 `stale_revision`, 422 `empty_text`,
422 `invalid_consent`.

**Consent must be a real boolean.** A string like `"false"` is rejected with `invalid_consent`;
accepting it as truthy turns a refusal into permission (this was a real defect, §7 D-2).

### 4.4 tts authorisation — backend owns this in the product
Input `{confirmation_id, idempotency_key}` → `tts_id`, `confirmation_id`, `text`, `text_sha256`,
`engine` (`client_speech_synthesis`), `replayed`, `created_at`. The AI side synthesises nothing; it
returns the only text the client may speak. Errors: 404 `unknown_confirmation`,
409 `confirmation_superseded`, 409 `idempotency_key_reused`.

**Check validity before replaying a cached authorisation** (§7 D-1), and note that an idempotent
authorisation record is *not* proof of exactly-once audible playback. Keep two things separate:
(a) stale, unconfirmed or cancelled text must never start playing; (b) duplicate or lost playback
needs its own action id and client-side handling.

### 4.5 jamo-errors (F2)
Input `{pairs: [{ref, hyp}], min_support}` where `ref` is a known enrollment prompt or an
independent human correction — **never the model's own output**. Output `metric_version`
(`jamo-err-v1`), `min_support`, `pairs_used`, `reference_tokens`, `insertions[]`, `tokens[]`
(`token`, `position` initial|medial|final, `errors`, `sample_count`, `error_rate` — `null` when
`status == "insufficient_data"`, `status`, `silent_initial`). UI wording must be
"모델이 자주 놓치는 부분", never "your weak pronunciation".

### 4.6 enrollment prompts (F3)
Input `{user_id, n, strategy, seed, exclude_prompt_ids}` → `strategy`, `strategy_version`
(`prompt-random-v1`), `seed`, `pool_size`, `prompts[]`. **Only `random` exists.** `coverage` and
`error_based` return **501** and are not faked. Persist strategy + version + seed with every prompt
shown. Design the UI and schema so `random` alone works.

### 4.7 Training and promotion — `POST /v1/adapters/train`, `GET /v1/jobs/{job_id}`

**Implemented 2026-09-29** (it used to return 501). Training itself needs a GPU, so the worker is
only exercised end to end on the weekend allocation; the promotion logic is covered by
`demo/test_train_worker.py` (7 groups) with the trainer and the gate stubbed.

**Input:** `{"user_id": "<uuid>"}`. Returns **202** with a job record. Enrollment audio is read from
`ENROLL_DIR/<user_id>/` — `pairs.json` (`[{"file": "...wav", "text": "..."}]`) plus the WAVs, PCM16
mono 16 kHz like everything else. Putting the files there is the backend's job; there is no upload
endpoint yet, and that is the "offline/manual handoff" we suggested scoping to.

**Job states:** `queued → running → evaluating → promoted | rejected | needs_review | failed`.
Poll `GET /v1/jobs/{job_id}`.

**The part to build against: an adapter is not served because training succeeded.**

Enrollment is split deterministically into **train / dev / gate**. `dev` picks the epoch. **`gate`
is held out from both** and is the only thing the promotion decision sees. The new adapter is scored
against the **incumbent** — whatever that user is served today, which is often the base — and:

| outcome | meaning |
|---|---|
| `promoted` | strictly better on the gate set; `adapters/active.json` is updated and `previous_active` records what it replaced (that is your rollback) |
| `rejected` | not better, **including a tie**. The incumbent stays active. This is a normal result, not an error |
| `needs_review` | the gate set was too small to decide automatically; a human edits `active.json` |
| `failed` | enrollment too small, trainer crashed, or no adapter produced. **The active pointer is never touched** |

**`rejected` will happen to real users.** The 2026-09-26 controlled run measured a speaker for whom
all five independently trained adapters were worse than the base. Your UI must have a sane way to
say "we could not improve on the standard model for you yet" — not an error state.

**Minimum enrollment is 15 utterances** (`MIN_ENROLL`), and the gate needs at least 5 (`MIN_GATE`).
Both are **policy, not evidence**: we have never measured how little enrollment suffices, and our
two controlled runs used 18 and 54 utterances. The backend's current "minimum 5 recordings" is below
anything we have tested and leaves no room for a gate split at all — please raise it or mark it
explicitly as a placeholder.

## 5. Rules the backend must enforce (write each as a test)

1. Send only WAV PCM16 mono 16 kHz, 0.3–30 s; split or reject longer audio.
2. A user never gets another user's adapter. Pass only the authenticated `user_id`; never accept an
   adapter id from a client.
3. Speech output only for the **latest valid confirmation**; any edit creates a new confirmation and
   invalidates the previous one. Re-check validity immediately before playback.
4. `idempotency_key` → at most one TTS event; the same key returns the same event; the same key for a
   different confirmation is a 409.
5. Confirmation must reference the current `revision`; stale revision → 409.
6. Consent flags are booleans defaulting to `false`. Only recordings with `use_for_training=true` may
   be exported for training; deleting such a recording requires retiring adapters trained on it.
7. Never display `score` or `alternatives`; never display "safe", "correct", "정확합니다".
8. Store the version fields with every transcription row.
9. `no_speech` is a normal outcome, not an error.
10. A late response must not act on stale state: capture a request generation before the call and
    discard the reply if the draft changed meanwhile (§7 D-6).

## 6. Persistence, adapter lifecycle and promotion

The v1 table sketch (`recording`, `transcription`, `confirmation`, `tts_event`, `prompt_shown`,
`adapter`, `jamo_error_snapshot`) still holds; see CT §4. Three corrections:

- **`score` is nullable.** Do not model confidence as a primitive `double` anywhere — port record,
  domain object or column. A primitive cannot express "not produced", and a stored `0.0` will later
  be read as "0% confidence". Use a nullable type and do not surface it.
- **Job state and adapter state are different lifecycles.** Job: `queued → running →
  succeeded | failed`. Adapter: `candidate → validated → active → retired | rejected`. A job that
  finishes and produces a *rejected* candidate is a success, not a failure.
- **Promotion compares against the incumbent, not only the base.** CT §3.7 requires a new adapter to
  beat the base model on that user's dev items; that is necessary but not sufficient, because it can
  beat the base and still be worse than the adapter already active. Compare on the same fixed dev
  set, under the serving configuration, with tie behaviour defined in advance. Final test material
  never enters a promotion decision. Keep the previous adapter for rollback; at most one `active`
  adapter per user.

Also record, so a retrain is reproducible: recording ids and hashes, the reviewed text revision,
preprocessing version, split assignment, consent state, base model **revision**, training config and
seed, selected checkpoint, dev metric definition and result, artifact identity (weights **and**
config), and the previous/current active pointers.

## 7. Known defects in the AI demo server — do not copy these patterns

`demo/server.py` is a reference implementation of the contract's shape, not production code. An
external review on 2026-09-23 found six defects; three broke the "only confirmed text is spoken"
promise and were fixed on 2026-09-25, and the remaining three were fixed on 2026-09-29. Each was
reproduced first and is covered by a regression check in `demo/test_core.py` (14 groups, passing).

| id | defect | status |
|---|---|---|
| D-1 | a cached TTS authorisation was replayed **before** checking the confirmation was still valid, so superseded text came back with HTTP 200 | fixed |
| D-2 | consent used truthy coercion, so `"false"` became `true` | fixed |
| D-6 | browser handlers acted on replies that arrived after an edit or cancel | fixed |
| D-3 | `request_id` had no idempotency effect, so a retry after a timeout ran inference twice and produced two `transcription_id`s | **fixed 2026-09-29** — a client-supplied `request_id` now returns the first result unchanged, keyed per user; a generated one is never an idempotency key |
| D-4 | the adapter was chosen by newest `created_at`, ignoring validation and activation state | **fixed 2026-09-29** — see §7.1 |
| D-5 | the base **revision** was recorded but not enforced | **fixed 2026-09-29** — pinned to `973afd2496…` and checked after load; a mismatch refuses to start rather than silently serving adapters trained against a different snapshot |

### 7.1 How an adapter goes live — `adapters/active.json`

This answers the request for "an explicit UUID-to-adapter activation binding", so build against it.

`adapters/active.json` maps `user_id` → `adapter_id`. **Nothing else makes an adapter live.**

- No file, or no entry for that user → **the base model is served.** This is a supported, normal
  state, not a failure. A 2026-09-26 controlled run measured a speaker for whom every one of five
  adapters was worse than the base, so the base must stay selectable per user.
- An entry pointing at an unknown adapter, or at one whose `meta.json` names a different `user_id`,
  is refused and the base is served. Cross-user adapter serving is impossible by construction.
- An unreadable file serves the base for everyone rather than falling back to a guess.

Consequences for the backend: **a completed training job must never imply an active adapter.** Those
are two states and your database should hold both. Promotion writes this file; rollback rewrites it.
Never infer `modelUsed` from job state — read `model.adapter_id` from the response, which is `null`
exactly when the base ran.

## 8. Audio ingest (backend) and local development

- Keep the original upload, transcode with a real decoder:
  `ffmpeg -i in.m4a -ac 1 -ar 16000 -sample_fmt s16 out.wav`.
- Check the channel count before downmixing; averaging stereo can cancel speech or mix another voice.
- Validate finite samples and 0.3–30 s **after** transcoding; cap upload size.
- Never trust a declared sample rate; read the file header. In the AI-Hub corpus the declared rate
  was wrong for 97 of 119 files [M].
- `audio.preprocessing_version = "pp-v1"` describes **server-side** handling only (no resampling, no
  VAD). The provenance of the phone-side conversion is the backend's to record.
- Local development without models: `ASR_ENGINE=mock python3 demo/server.py`, then
  `python3 demo/test_contract.py http://127.0.0.1:8000` (expect "0 failed"). The mock enforces the
  same validation and confirmation/TTS rules; it does not reproduce recognition quality.
  `python3 demo/test_core.py` exercises the same contract with no HTTP at all.

## 9. Corrections — claims the AI side withdrew

Older documents (including the 2026-09-18 spec) contain these. If the user quotes one, correct it.

| withdrawn claim | current statement |
|---|---|
| "General ASR does not work for this population" | False for this cohort. Zero-shot large-v3 reaches 2–4% jamo error for four of eight speaker groups; the worst is 52% [M]. Say **"error rates differ enormously between speakers"** |
| "Personal adaptation is already proven" | **Not proven, and the direction depends on the audio preprocessing.** Two low-baseline speakers improved [pilot], same session, same task. A controlled run finished 2026-09-26 on two high-error speakers (112 cells, 5 training seeds x 2 decoders x 2 VAD settings, 0 failed) [M]: **without** a VAD, all 5 adapted seeds beat the base for both speakers; **with** a fixed VAD preprocessing, adaptation still reduced error for speaker A but **increased** it for speaker B (base 22.0% jamo CER, all 5 seeds 1-4 pp worse). The effect is also unstable — the paired base-vs-adapter difference varies by up to 35.7 pp across evaluation seeds, and the smallest single win is 1.12 pp. n = 2 speakers, one recording each; the acoustic cause and any transfer to short everyday messages are unverified. **Do not build anything that assumes an adapter always beats the base**: promotion must be decided per user by measurement, and the base must remain selectable |
| "Error-based prompt selection beats random" | Never tested by us; the prior-work reading it rested on was misread and retracted on 2026-09-18 |
| "The demo implements every endpoint" | Training and job status landed 2026-09-29 (§4.7). Still 501: `coverage` and `error_based` prompt strategies |
| "`contract_version` is on every response" | Today only health and transcribe carry it. Others are being aligned |
| "Raw WAV body" as a global convention | Transcribe only; every other operation takes JSON |
| "Localhost HTTP costs ~1 ms" | No such measurement exists. Any transport-latency argument from the AI side is withdrawn |
| "Greedy decoding cannot loop" | A decoding choice guarantees no such thing; the observed reduction is scoped to the cases measured |

## 10. Not in v1

Candidate lists and confidence display; streaming recognition; on-device inference; server-side TTS
audio; `coverage` and `error_based` strategies; the training API; multi-worker AI serving (v1
serialises inference and one PEFT model holds one active adapter at a time).

## 11. Open questions for the AI owner

1. **Daemon entrypoint shape.** The AI side proposes the four `core_*` functions taking bytes and
   plain values and returning dicts. If JPyRust expects a fixed signature or JSON-in/JSON-out over
   shared memory, say which and the AI side adapts.
2. **Who places per-user adapter artifacts**, and how a reload is triggered.
3. **The container question** in §2 — resolve before committing to the deployment shape.
4. Training-data export format from the backend: `recording_id`, WAV path, confirmed text,
   `user_id`, duration. Split assignment stays on the AI side.
5. Whether `request_id` should become a real idempotency key (D-3) before automatic retries exist.
6. TTS engine: OS TTS in the app, or a stock TTS API behind the backend.

## 12. One thing that is easy to get wrong in the data model

**A user's confirmed text is not automatically a transcript of what they said.** People paraphrase,
expand or replace an utterance before sending it. If confirmations are fed back as training pairs,
mislabelled pairs enter the training set silently. Store the spoken reference separately from the
message the user wanted to send, or review alignment before training.
