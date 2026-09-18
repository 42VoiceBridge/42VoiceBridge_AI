# AI ↔ Backend Contract v1

**Version:** `ai-contract-v1` · **Date:** 2026-09-18 · **Owner:** AI/Data track (jaemyu)
**Audience:** backend track (민수) and any LLM assisting it. Frontend (채영) reads §3 and §6.
**Reference implementation:** `demo/server.py` (stdlib HTTP + numpy; real engine needs torch,
transformers, peft). **Conformance tests:** `demo/test_contract.py` — 40 checks (37 when no adapter user is given), all passing
against the mock engine on 2026-09-18. The real engine has not been run yet (see §9).
**Machine-readable:** `docs/openapi_ai_v1.yaml`.

Evidence tags as in `BACKEND_HANDOFF_EN.md`: **[M]** measured, **[P]** planned, **[U]** unverified.

---

## 0. Decisions this contract fixes

| # | Decision | Why |
|---|---|---|
| D1 | **Inference runs on a server (Python) for the 9/23 prototype.** On-device (Rust) is a later milestone, not dropped | LoRA support in whisper-rs / candle is unverified. A phone UI does not require phone-side inference |
| D2 | **Base model `openai/whisper-small`, personal adapters = PEFT LoRA**, served through Transformers + PEFT | faster-whisper (CTranslate2) cannot load PEFT adapters. Training and serving use the same code path, so a served adapter behaves as evaluated |
| D3 | **The AI server accepts WAV PCM 16-bit mono 16 kHz, 0.3–30 s, and nothing else** | Transcoding phone formats (AAC/M4A/WebM) is the backend's job with a real decoder (ffmpeg). One format at the boundary removes a class of silent bugs |
| D4 | **`alternatives` is always `[]` and `score` is always `null` in v1** | Beam search does not produce N transcripts (and v1 decodes greedily, D9), and token probabilities are not calibrated for this population. The fields exist so tables do not change later; nothing is fabricated to fill them |
| D5 | **Only confirmed text may be synthesized.** Editing after confirmation invalidates it. Retries must not duplicate speech | A fluent wrong transcript spoken aloud is taken as what the user said |
| D6 | **Adapter selection is server-side from `user_id`.** A client cannot name an adapter | Prevents using another person's model |
| D7 | **"Weak phoneme" is renamed "jamo-level model-error statistics" (`jamo-err-v1`)** and describes the model, not the speaker | References are intent-based; tokens are orthographic, not phonetic |
| D8 | **Consent to store audio and consent to train are separate flags, both default `false`** | Confirming a sentence is not consent to reuse the recording |
| D9 | **Decoding is greedy (`beam_size = 1`) for the demo server and the pilot test pass** (2026-09-18; overrides the earlier `beam_size=5`) | [M] At `beam_size=5` whisper-small emitted repetition loops on short utterances — `아, 그래요?` became 200 tokens of `아`, one segment producing 86% of a 20-segment split's errors. Reproduced identically on CUDA/fp16 and CPU/fp32, so it is a decoder effect. Accuracy is a wash, and is **not** the reason: greedy scored better on the KJW test split (syllable CER 0.1031 vs 0.1134, 50 segments) and slightly worse on the 10 demo samples (0.1009 vs 0.0917) — both differences are 1–5 syllables and neither is a result. The reasons are the failure mode and latency: greedy cannot produce the loop, and median demo latency fell from 1554 ms to 666 ms (2.3x). `decoding.beam_size` in every response reports what actually ran |

## 1. Who owns what

| Concern | AI server | Backend | App |
|---|---|---|---|
| Authentication, `user_id` issuance | — | **owns** | sends token |
| Audio upload, transcoding to WAV 16 kHz mono | validates | **owns** (ffmpeg) | records |
| Storage of audio, transcripts, confirmations | none (memory only in demo) | **owns** | — |
| Recognition | **owns** | calls | — |
| Confirmation + TTS gate | reference impl for demo | **owns in product** | enforces in UI |
| TTS audio | — | calls a stock TTS, or app uses OS TTS | plays |
| Enrollment prompts | **owns** selection logic | stores what was shown | displays |
| Adapter training | **owns** training code | owns the job queue and artifact storage | — |
| Jamo error statistics | **owns** computation | stores snapshots | displays with fixed wording |

## 2. Conventions

- JSON, UTF-8, `snake_case`. IDs are UUID v4 strings. Timestamps ISO-8601 with offset.
- Every inference result carries the versions that produced it: `contract_version`,
  `model.base_model`, `model.base_revision`, `model.adapter_id`, `model.adapter_revision`,
  `audio.preprocessing_version`, `decoding.*`. Store them with the row. A number without its
  versions cannot be compared to another number.
- `null` means "not produced", never "zero" or "unknown but probably fine".
- Errors: HTTP status + body `{"error": {"code": "...", "message": "...", ...detail}}`.
  Codes are stable identifiers (§7); messages are for humans and may change.

## 3. Endpoints

### 3.1 `GET /v1/health`

Returns engine, base model and revision, beam size, loaded adapters (with `user_id`,
`adapter_revision` = first 12 hex of SHA-256 of the adapter weights, `size_bytes`), and input
limits. Call it at startup and log the result.

### 3.2 `POST /v1/asr/transcribe`

Request: body = raw WAV bytes, `Content-Type: audio/wav`. Query parameters:

| Param | Required | Meaning |
|---|---|---|
| `user_id` | yes | authenticated user; selects that user's active adapter |
| `use_adapter` | no, default `true` | `false` forces the base model (A/B comparison, fallback) |
| `request_id` | no | echoed back; generated if absent. Use it for tracing and idempotent retries |

Response `200`:

| Field | Type | Null? | Notes |
|---|---|---|---|
| `contract_version` | string | no | `ai-contract-v1` |
| `transcription_id` | uuid | no | the backend's primary key for this result |
| `request_id` | string | no | echo |
| `user_id` | string | no | echo |
| `revision` | int | no | `1` for recognizer output. Confirmation references it |
| `status` | enum | no | `ok` · `no_speech` |
| `text` | string | no | `""` when `status=no_speech` |
| `alternatives` | array | no | **always `[]` in v1** (D4) |
| `score` | number | **yes** | **always `null` in v1** |
| `score_type` | string | **yes** | must be non-null whenever `score` is non-null (e.g. `mean_token_logprob`). Never display an untyped score |
| `model.engine` | string | no | `transformers+peft` or `mock` |
| `model.base_model` | string | no | `openai/whisper-small` |
| `model.base_revision` | string | yes | Hub commit hash if known |
| `model.adapter_id` | string | yes | `null` = base model was used |
| `model.adapter_revision` | string | yes | weights hash; `null` with `adapter_id=null` |
| `decoding` | object | no | `language`, `task`, `beam_size` |
| `audio.sha256` | hex | no | of the exact bytes received |
| `audio.duration_sec` | number | no | |
| `audio.rms_dbfs` | number | no | loudness of the input |
| `audio.preprocessing_version` | string | no | `pp-v1` = no resampling, no VAD |
| `latency_ms` | int | no | server-side recognition time only |
| `created_at` | timestamp | no | |

`no_speech` is returned when input RMS is below −60 dBFS **(heuristic, not validated)** or the
decoder returns empty text. It is not an error; show "no speech detected, record again".

### 3.3 `POST /v1/transcriptions/{transcription_id}/confirm` — backend-owned in product

Body: `{"revision": 1, "confirmed_text": "...", "consent": {"store_audio": false, "use_for_training": false}}`

Response: `confirmation_id`, `transcription_id`, `based_on_revision`, `confirmed_text`,
`text_sha256`, `source` (`asr_unedited` · `user_edited`), `edit_distance_syl` (Hangul syllable
edit distance between recognizer text and confirmed text), `consent`, `supersedes`
(previous confirmation id or `null`), `valid`, `confirmed_at`.

Rules: `revision` must equal the current revision (else `409 stale_revision`); empty text is
`422` (use cancel instead); a new confirmation of the same transcription marks the previous one
`valid=false`.

### 3.4 `POST /v1/tts` — backend-owned in product

Body: `{"confirmation_id": "...", "idempotency_key": "..."}` → returns `tts_id`, the exact
`text` the client may speak, `text_sha256`, `engine` (`client_speech_synthesis` in the demo),
`replayed`.

Rules: unknown confirmation `404`; superseded confirmation `409 confirmation_superseded`;
same key + same confirmation returns the original record with `replayed=true` and must not
produce a second utterance; same key + different confirmation `409 idempotency_key_reused`.
The client generates one key per user press and reuses it on network retry.

### 3.5 `POST /v1/analysis/jamo-errors` (F2, renamed)

Body: `{"pairs": [{"ref": "...", "hyp": "..."}], "min_support": 20}`

`ref` must be a known enrollment prompt or an independent human correction — **never the
model's own output**. Response: `metric_version` (`jamo-err-v1`), `min_support`, `pairs_used`,
`reference_tokens`, `insertions[]`, and `tokens[]` with `token`, `position`
(`initial`·`medial`·`final`), `errors`, `sample_count` (token occurrences in references),
`error_rate` (`null` when `status=insufficient_data`), `status`, `silent_initial`.

Definitions are fixed in `demo/jamo_stats.py` (alignment, tie-break, error attribution,
denominator). UI wording: **"the model often misses this"** — never "your weak pronunciation".

### 3.6 `POST /v1/enroll/next-prompts` (F3 entry point)

Body: `{"user_id": "...", "n": 10, "strategy": "random", "seed": 0, "exclude_prompt_ids": []}`
→ `prompts[]` of `{prompt_id, text}` from the guideline catalogue (task codes 02-03, 02-04,
06-01; pool 1,807 sentences) plus `strategy`, `strategy_version`, `seed`, `pool_size`.

v1 implements `random` only. `coverage` and `error_based` return `501` until built and tested
**[P]**. Store the strategy and version with every prompt you show; the enrollment curve is
meaningless without it.

### 3.7 `POST /v1/adapters/train`, `GET /v1/adapters/jobs/{job_id}` — specified, not served **[P]**

v1 trains offline (`experiments/b1_pilot/b1_train.py` on Colab). The job contract, for the
backend queue:

| Field | Notes |
|---|---|
| `job_id`, `user_id` | |
| `status` | `queued` → `running` → `failed` \| `validated` → `active` → `retired` |
| `base_model`, `base_revision` | adapter is invalid on any other base |
| `train_items[]` | `{recording_id, text, split}`; text = prompt or human-confirmed transcript; only items with `use_for_training=true` |
| `config` | rank, alpha, target modules, lr, epochs, batch, seed |
| `dev_metric`, `baseline_dev_metric` | pooled jamo/syllable CER on the user's dev items, before and after |
| `adapter_id`, `adapter_revision`, `size_bytes`, `artifact_uri` | set when `validated` |

Activation rule: **no automatic activation.** An adapter becomes `active` only if its dev
metric beats the base model on the same dev items (the training script already refuses to
bind an adapter that fails this). The previous active adapter is kept for rollback.

## 4. Entities the backend should persist

| Entity | Key fields |
|---|---|
| `recording` | `recording_id`, `user_id`, `purpose` (`enroll`·`use`), `prompt_id`?, `source_codec`, `source_sample_rate`, `source_channels`, `sha256_source`, `sha256_wav16k`, `duration_sec`, `storage_uri`, `consent_store_audio`, `consent_train`, `created_at` |
| `transcription` | every field of §3.2 response, plus `recording_id` |
| `confirmation` | every field of §3.3 response |
| `tts_event` | `tts_id`, `confirmation_id`, `idempotency_key`, `text_sha256`, `engine`, `created_at` |
| `prompt_shown` | `user_id`, `prompt_id`, `text`, `strategy`, `strategy_version`, `seed`, `shown_at` |
| `adapter` | §3.7 fields; one `active` per user at most |
| `jamo_error_snapshot` | `user_id`, `adapter_id`?, `metric_version`, `min_support`, `pairs_used`, `computed_at`, token rows |

Retention and deletion must cover derived adapters: deleting a user's recordings with
`consent_train=true` requires retiring adapters trained on them.

## 5. Invariants (all covered by `test_contract.py`)

1. A transcription result always has `alternatives=[]`, `score=null`, `score_type=null` in v1.
2. `user_id=A` never receives an adapter bound to user B, whatever parameters are sent.
3. Only the text of the latest valid confirmation can be returned by `/v1/tts`.
4. Editing after confirmation requires a new confirmation before anything is spoken.
5. The same idempotency key never produces two TTS records.
6. Consent flags default to `false`.
7. Anything but WAV PCM16 mono 16 kHz, 0.3–30 s, is rejected with a typed error.
8. `jamo-err-v1` returns `error_rate=null` below `min_support`.
9. Unbuilt strategies return `501`, not a silent fallback.

## 6. Frontend notes

- Record in any format the browser supports, decode with Web Audio, resample to 16 kHz mono,
  encode PCM16 WAV (reference: `demo/static/index.html`, function `toWav16k`). In the product
  the backend transcodes instead; either way the AI server sees D3 input.
- Disable "speak" until a confirmation exists; re-disable on any edit.
- Do not render `score` or `alternatives` in v1. Do not show "safe", "correct" or similar.

## 7. Error codes

| HTTP | code | When |
|---|---|---|
| 400 | `missing_user_id`, `bad_json`, `missing_field` | malformed request |
| 404 | `unknown_transcription`, `unknown_confirmation`, `not_found` | |
| 409 | `stale_revision`, `confirmation_superseded`, `idempotency_key_reused` | state conflicts |
| 413 | `audio_too_long`, `body_too_large` | > 30 s audio, > 5 MB body |
| 415 | `unsupported_media_type` | not RIFF/WAVE |
| 422 | `bad_audio`, `bad_audio_format`, `audio_too_short`, `empty_text`, `no_pairs` | |
| 501 | `strategy_not_implemented`, `not_implemented_in_demo` | planned features |
| 503 | `prompt_pool_missing` | server misconfigured |
| 500 | `internal` | bug; log `request_id` |

## 8. Not in v1

Candidates and confidence display; error-based prompt selection (the project's claimed
contribution — its absence here is deliberate until it is measured); streaming recognition;
on-device inference; server-side TTS audio; multi-worker serving (v1 serializes inference with
a lock because a PEFT model holds one active adapter at a time).

## 9. Verification status

| Item | Status |
|---|---|
| Contract behaviour (40 checks) against mock engine | **[M]** pass, 2026-09-18 |
| Demo page flow (sample → recognize → edit → confirm → speak gate; mic path) in headless Chromium against mock engine | **[M]** pass, 2026-09-18 |
| Real engine (`HFEngine`) loading whisper-small + PEFT adapters | **[U]** not run: Hugging Face was blocked from the build environment. First run is on the team Mac |
| Latency on the target machine | **[U]** not measured |
| Adapter size | **[P]** measured by `b1_train.py` (`summary.json → adapter.bytes`) on the first Colab run |
