# What the AI side needs from the backend — 2026-09-29

**Audience:** the backend developers and the Claude assisting them.
**Source:** our reading of the 2026-09-29 backend progress handoff and of `develop`
(`61299838bce0022f8d48127932f122f54c8216a1`) plus open PRs #35/#36/#37.
**Companion document:** `docs/AI_STATUS_FOR_BACKEND_CLAUDE_v2_EN.md` describes what the AI side is
and what it returns. This document lists only the **ten things we are asking you to change**, in
priority order. Where the two disagree, the companion document wins on AI behaviour and this one
wins on what we need from you.

Evidence labels: **[M]** measured by us · **[V]** verified in your source · **[U]** unverified ·
**[P]** planned.

---

## Read this first, if nothing else

**Your HTTP client cannot request the base model, and on 2026-09-26 we measured a speaker for whom
the personal adapter is worse than the base.**

`HttpAiInferenceClient.recognize(audioBytes, modelType, userId)` discards `modelType` and never
sends `use_adapter=false` [V]. Our default is `use_adapter=true`, so every request gets that user's
adapter if one exists.

Until last week this was a theoretical concern. It is now a measured one [M]: in a 112-cell
controlled run (2 speakers × 5 training seeds × 2 decoders × 2 preprocessing settings, 0 failures),
one of the two speakers got **worse** with the adapter under the preprocessing we expect to deploy —
jamo CER 22.08% base versus 23.14–25.90% adapted, **all five seeds worse**. The other speaker
improved. There is no way to know in advance which kind a new user is.

A product that can only serve the adapter will serve some users a worse model with no way back.

**The fix on your side is small:** pass `use_adapter=false` through to the query string when the
caller asks for the base. The parameter already exists on our side and is already tested.

---

## P0 requests

### 1. Wire up base-forcing

**Change:** `HttpAiInferenceClient` must honour `modelType`. Base request →
`POST /v1/asr/transcribe?user_id=<uuid>&use_adapter=false`. Adapter request → omit it or send
`true`.

**Also adopt this rule:** an adapter is promoted only if it beats **the adapter currently being
served** for that user — not merely the base — and if nothing beats the base, the base stays active.
Keep the base permanently selectable per user; it is a supported serving state, not a fallback.

**Done when:** the same WAV through the same endpoint returns different `model.adapter_id` (one
`null`, one set) for the two request kinds, and both are stored.

### 2. Stop inferring which model ran; store what we return

**Observed [V]:** `RecognizeSpeechService` sets `modelUsed` from whether a completed
`PersonalizationJob` exists in your database, then persists that inferred label. Your parser keeps
only `status`, `text`, `score` and discards the rest.

**Why this is serious:** a completed training row does not mean our server has that adapter loaded.
You can record `PERSONALIZED` for a recognition that actually ran on the base model. That is
fabricated provenance, and it will end up in a report.

**Change:** persist the fields we already return, verbatim:

```
model.engine, model.base_model, model.base_revision,
model.adapter_id, model.adapter_revision,      # both null == the base model ran
transcription_id, request_id, contract_version,
audio.preprocessing_version, audio.sha256, latency_ms
```

Keep **requested** model and **actually used** model as two separate columns. Never derive the
second from job state.

**Done when:** a recognition row can answer "which exact artifact produced this text" without
consulting the jobs table.

### 3. Drop the name `BASE_ADAPTED`

**There is no public-data first adaptation stage.** It was never built. The pipeline is one stage:
frozen `openai/whisper-small` at a pinned revision, plus a per-user LoRA adapter. `BASE_ADAPTED`
names a model that does not exist, and your README still describes the two-stage design as if it
were current.

**Change:** use `BASE` and `PERSONALIZED` only, and correct the README. This is the third time we
have raised it; it keeps reappearing in new code.

---

## P1 requests

### 4. Own the audio conversion, and let us reject bad audio as a client error

**Observed [V]:** uploaded bytes are forwarded as `audio/wav` with no conversion, while your storage
layer accepts WebM/M4A/OGG/MP3. Browser recordings are usually WebM. Accepting a container is not
the same as producing a valid WAV.

**We accept only:** RIFF/WAVE, PCM16, mono, 16 kHz, within the duration limits that
`GET /v1/health` publishes under `limits`. We already return precise errors:

| Condition | HTTP | `code` |
|---|---|---|
| not a RIFF/WAVE body | 415 | `unsupported_media_type` |
| unparseable WAV | 422 | `bad_audio` |
| wrong rate/channels/width | 422 | `bad_audio_format` |
| too short / too long | 422 / 413 | `audio_too_short` / `audio_too_long` |
| body over the size cap | 413 | `body_too_large` |

**Change:** convert on your side (you already hold the file, and the JVM can call `ffmpeg`), then
**surface our 4xx codes to the client instead of collapsing them into 503**. Right now malformed
audio is indistinguishable from an AI server outage, so nobody can tell a user "re-record this".

**Done when:** a WebM browser recording succeeds end to end, and a deliberately corrupt file returns
a 4xx naming the reason.

### 5. Justify the 3 s / 10 s timeouts, and define what happens when one fires

**Measured [M]:** median ~0.7 s for 2–8 s utterances on a laptop CPU, ~0.85 s with an adapter. **But
`latency_ms` includes queue wait behind our inference lock** — it is not pure inference time.
Requests are serialised. Under concurrency the tail will exceed 10 s.

**Change:** measure the tail on the deployed host under realistic concurrency before fixing the read
timeout, and define the behaviour on timeout — retry, mark failed, or surface to the user. Also
decide whether a personalized request that fails should retry on the base; today no such retry
exists and we do not think one should be added silently.

### 6. A superseded confirmation must not become playable audio

**Observed [V]:** `RequestTtsService` checks validity before the idempotent replay — correct. But
the asynchronous event carries the old text, the synthesis handler never rechecks validity, and
`GetTtsStatusService` checks ownership without checking validity. Submit old text → reconfirm new
text while synthesis is pending → poll the old job → old audio is available.

**We hit the same defect and fixed it on 2026-09-25** (our D-1): validity is now checked *before*
the cache is consulted, and a superseded key returns `409 confirmation_superseded`. The regression
test is `demo/test_core.py`, which drives the contract functions with no HTTP — read it and apply
the same invariant at your two points.

**This is a safety rule, not an optimisation.** The product speaks on a user's behalf; speaking a
sentence they already replaced is the worst failure mode we have. The frontend must also reject
stale responses and cancel playback on edit/cancel (our D-6: a generation counter bumped on edit,
cancel and new recording, captured per request and re-checked before speaking).

**Done when:** a test reproduces the race and the old audio is refused at both the handler and the
status endpoint.

### 7. The prompt pool cannot be shipped in a git repository — agree a deployment path

**PR #37 reports that `data/script_pool.json` is missing from the AI repository, producing 503. That
is correct and deliberate.** The file exists on our side with **3,436 entries** [M] — note that this
does not match the 1,807 figure in the PR, so you are looking at a different artifact or a filtered
view; tell us which, because a prompt-pool version mismatch will silently corrupt enrollment
records.

It is **script text parsed from the AI-Hub corpus build guideline**. Our standing rule is that
AI-Hub audio, label text and manifests are never committed to git. Committing this file would break
it.

**Change:** provision the pool on the server directly, and share only a version string and hash.
**And before it is deployed anywhere public, the AI-Hub redistribution terms for that sentence
catalogue must be checked** — we do not know whether serving 3,436 corpus sentences through a public
endpoint is permitted [U]. That check is on our side; we are flagging it so nobody deploys first.

Strategy support today: `random` works; `coverage` and `error_based` return
`501 strategy_not_implemented`. Only `random` may be described as implemented.

### 8. Keep the two ID spaces separate

Reading sessions use your sentence UUIDs. Recommendations use our `prompt_id`. They are not
interchangeable. When personal recordings are later uploaded for training, the stored prompt ID and
the **exact text plus pool version** must travel with the audio, or the training pairs cannot be
reconstructed.

Also: persisting a recommendation response records a prompt that was **offered**, not one that was
displayed or recorded. Define resume and re-record behaviour so retries do not silently consume the
pool, and do not treat exposure as training eligibility.

---

## P2 requests

### 9. Narrow the personalization scope to an offline handoff until a worker exists

**Observed [V]:** the status APIs read your own records; nothing runs or observes an AI worker.
`PersonalizationJob.complete()` validates that artifact and version strings are non-empty — not dev
performance, not successful artifact loading, not activation.

**Change:** before building more of a job API that has no worker behind it, let us agree an
offline/manual prototype handoff and write down its contents: consent record, approved data
snapshot, artifact identity and hash, the exact base revision, the application UUID it binds to, the
validation outcome, installation, and an explicit active-version acknowledgement.

**"Job completed" and "adapter is being served" are different states.** Keep them as different
fields from the start.

### 10. Do not let "minimum five recordings" become an efficacy claim

You correctly labelled it a temporary domain policy. Keep it there. For scale: our two controlled
runs used **18 sentences (2.4 min)** and **54 sentences (5.7 min)** of enrollment audio; the earlier
pilot measured a 30-sentence budget. Five utterances is far below anything we have measured, and we
have no evidence about what happens at that size.

Similarly: **"weak phonemes" is not a clinical finding.** What we can produce is reference-based
model-error statistics — where *this model* made errors against *this reference*. That is not a
statement about the user's speech production, and it must not be presented as one.

---

## What changed on our side, so you are not working from stale facts

1. **Transport: HTTP is the default again.** We recorded "transport = JPyRust" as a decision on
   2026-09-25 based on your team's preference; your handoff supersedes it. No work is lost — the
   contract lives in transport-independent functions (`core_health`, `core_transcribe`,
   `core_confirm`, `core_tts`) and the HTTP layer is a thin wrapper over them, so either transport
   calls the same code.

2. **The adaptation result is now measured, and it is conditional** [M]. 2026-09-26, 112 cells,
   0 failures, 5 training seeds per speaker. Without VAD preprocessing, adaptation improved both
   speakers (5/5 seeds each). With a fixed VAD, it still improved one speaker and **reversed** for
   the other. The acoustic cause, the transfer to short everyday messages, and any benefit for other
   users are unverified. n = 2 speakers, one recording each.

   For your purposes the operational consequence is request 1: **never assume the adapter is the
   better model.**

3. **Retracted, so you do not quote it:** an earlier "−39.9% improvement" figure for one speaker was
   measured without VAD preprocessing and does not survive it.

---

## What we owe you

| Item | Status |
|---|---|
| `use_adapter=false` support | **Already implemented and tested** — nothing blocking on our side |
| Model/adapter identity in the response | **Already returned** — see request 2 for the field list |
| Compliant WAV fixture + expected response shape | [P] we will send one |
| Negative fixtures (empty text, malformed audio, timeout) | [P] we will send them |
| Prompt pool version + hash + deployment method | [P] blocked on the redistribution check in request 7 |
| Preprocessing specification | in the companion document |
| Latency envelope under concurrency | [P] we have single-request numbers only |

**Do not fabricate a confidence score to satisfy a column.** `score` is `null` by design; the
serving engine does not extract token scores. Your nullable handling is already correct — keep it,
and make sure the database column is a nullable type rather than a primitive `double`.

---

## One-sentence status we both agree on

> The backend implements authentication, reading-recording processing, recognition persistence,
> confirmation and asynchronous preset-voice synthesis. HTTP AI integration code is present; real-use
> upload and random prompt recommendations are under review, while online training, verified model
> activation and complete playback delivery remain unfinished.

Nobody should state that an end-to-end deployed service, per-user voice synthesis, error-based
prompt selection or automatic personalization has been demonstrated.
