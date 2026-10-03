# AI → Backend: answers to R1–R5, C1–C4, T1–T9, A1–A5

**Date:** 2026-10-03. **Scope:** answers the two Notion pages (S1 9/30, S2 10/02) as consolidated in
`BACKEND_REQUESTS_NOTION_HANDOFF_EN_2026-10-03.md`. Source IDs preserved.

Status values mean exactly this:

| Status | Meaning |
|---|---|
| **Implemented** | runs today in `demo/server.py`, with a test you can reproduce |
| **Implemented-local-only** | runs, but has never been deployed anywhere the backend can reach |
| **Proposal** | our suggestion, not built, not agreed |
| **Unsupported** | will not exist in v1 |
| **Needs your decision** | blocked on the backend, not on us |

**No dates are given.** Where a date is unknown it says unknown. Nothing below is a commitment to a
schedule.

## 0. Version reconciliation — which of your observations are stale

Current AI revision: `demo/server.py` and `demo/train_worker.py` as of **2026-10-03**. The contract
file is `docs/AI_BACKEND_CONTRACT_v1_EN.md` (`ai-contract-v1`).

| Your observation | Still true? | Evidence |
|---|---|---|
| `POST /v1/adapters/train` returns 501 | **Stale.** It returns **202** with a job record since 2026-09-29 | `demo/server.py` `core_train()`; `demo/test_train_worker.py` 7 groups |
| No job GET route | **Stale.** `GET /v1/jobs/{job_id}` exists | same |
| `docs/openapi_ai_v1.yaml` shows 501 and no job route | **True — the OpenAPI is behind the code.** That is our bug, not your misreading | C4 below |
| Recommendations 503 without `script_pool.json` | **True** | R1 |
| Only `random`; `error_based` 501 | **True** | R4 |
| Body read by `Content-Length` only → chunked request became `422 no_pairs` | **Was true; fixed today** | R3 |
| (not in your list) a promoted job meant personalized serving | **It did not, and that was our defect. Fixed and tested today** | A3 |
| Transport for the product is HTTP; JPyRust is a separate future task | **Agreed, and we are not making it a prerequisite** | §6 |

**One correction runs the other way, against us.** Until today a training job that reported
`promoted` did **not** mean personalized inference was being served: the server loaded adapters at
startup only. We found it from the 2026-10-03 external review, reproduced it against a running
server, and fixed it the same day (A3). The rule for your UI is now precise: **display active
personalization when a transcribe response returns a non-null `model.adapter_id`, never on the
strength of a promoted job alone.**

## 1. Recommendation and transport (S1)

| ID | Status | Answer |
|---|---|---|
| **R1** prompt-pool delivery | **Needs your decision + ours** | Your hypothesis is **correct**: the file is absent from the repository deliberately. `data/script_pool.json` is parsed from the AI-Hub corpus build guideline, and our standing rule is that AI-Hub audio, label text and manifests are never committed to git. **It is not withheld by accident and it will not appear in the repo.** Delivery route is unresolved and has a blocker: we have **not** confirmed that serving this sentence catalogue from a networked endpoint is permitted under the AI-Hub distribution terms. That check is ours and is not finished. Until it is, provisioning the file onto a server is a decision we cannot make unilaterally. Proposed route once cleared: place the file on the serving host out of band, point `PROMPT_POOL` at it, and validate at startup — the server already fails closed with `503 prompt_pool_missing` and reports the path it looked at. |
| **R2** `min_support=10` | **Answered, with a limitation you should display** | Your uncertainty arithmetic is **correct and we reproduced it**: for a binomial rate at the worst case p=0.5, the 95% interval half-width is **±31.0 pp at n=10** and **±21.9 pp at n=20**. So lowering the threshold trades roughly ±22 pp for roughly ±31 pp of worst-case uncertainty. We did not verify your 2.4-vs-6.3 eligible-token simulation; that is your measurement, not ours. **Our position: 10 is acceptable for an exploratory display and not acceptable for anything that looks like a verdict about a user's speech.** If you display a rate at n=10, display the interval or the raw counts beside it. `support` counts **reference-side token occurrences**, not utterances. Below the threshold we already return `error_rate: null` with `status: "insufficient_data"` and the real `sample_count`; please render that as "not enough data yet", never as 0%. |
| **R3** chunked request bodies | **Implemented today** | You were right. The handler read `Content-Length` only, so a chunked body produced an empty read and a misleading `422 no_pairs`. `demo/server.py` now accepts **both** framings, enforces the same 5 MB cap across chunks, and returns `400 bad_chunked_body` on a malformed chunk header. Reproduce: send `/v1/analysis/jamo-errors` with `Transfer-Encoding: chunked` and no `Content-Length` — verified 200 with `pairs_used: 1`, and the `Content-Length` form still returns 200. Keep your client-side buffering if you prefer; it is no longer required. |
| **R4** `error_based` | **Unsupported in v1** | `random` is the only implemented strategy. `coverage` and `error_based` return `501 strategy_not_implemented` and are **not faked**. Date unknown. If it ever ships, the request schema does not change (`strategy` already exists); the response gains strategy-specific provenance fields and **will not** silently change meaning under the same `strategy_version`. Treat `error_based` as planned, never as implemented. |
| **R5** pool version | **Implemented today** | The recommendation response now returns `pool_version` (`"script-pool-v1"`, bumped when the file is replaced) and `pool_sha256` (16 hex chars, a content hash of the **selected** items). These are deliberately separate from `strategy_version`, which identifies the algorithm. Persist all three with every offered prompt: same `strategy_version` + same `seed` + same `pool_sha256` ⇒ same prompts; if `pool_sha256` differs, the seed does not reproduce and that is expected, not a bug. **Side effect worth recording:** this also resolves the 1,807-vs-3,436 discrepancy we raised on 2026-09-29. The file holds 3,436 catalogue entries; the server selects task codes `02-03`, `02-04`, `06-01` only, which is **1,807**. Your number was right; ours was the raw file count. |

## 2. Contract contradictions (C1–C4)

| ID | Status | Decision |
|---|---|---|
| **C1** state model | **Decided — two state machines, not one** | `AI_SIDE_FULL_STATUS_v2_EN.md` §6 is correct and CT §3.7 is wrong; we will fix §3.7. **Job:** `queued → running → evaluating → promoted \| rejected \| needs_review \| failed`. **Adapter:** `candidate → active → retired`. "Training succeeded, adapter rejected" is the **normal** outcome `job=rejected`: the candidate trained fine and simply did not beat the incumbent on the held-out gate. It is not an error and must not be rendered as one. Our 2026-09-26 run measured a speaker for whom **all five** independently seeded adapters were worse than the base, so your UI will meet this. |
| **C2** split ownership | **Decided — AI assigns; you persist. Defect fixed 2026-10-03.** | AI assigns train/dev/gate and returns the membership. Do **not** send `split` in `train_items[]`; we will ignore it. **Correction to what this row said earlier today:** the splitter used to rank a hash of `user_id\|filename` and slice by proportion, so adding enrollment re-ranked everything and a later gate item could be an earlier training item. That is now fixed by persistence rather than by a better hash — membership is written to `splits.json` per user (`split_version: split-v2`), prior assignments are kept verbatim and only new items are placed. Nothing ever moves between splits. A regression test adds 22 items to an 18-item enrollment and asserts no past training item is in the new gate; it also asserts the old rank-slicing *would* have collided, so the test is not vacuous. Each job still reports `n_train`/`n_dev`/`n_gate`. |
| **C3** promotion rule | **Decided** | The comparator is the **incumbent** — whatever that user is served today, which is the base when there is no active adapter. Metric: pooled jamo CER on the gate split, pooled over **integer edit counts** (changed today; it previously pooled rounded per-item rates, which can flip a decision near the threshold). **A tie does not promote.** Regression ⇒ incumbent stays, job is `rejected`. Approval owner: automatic only when the gate has at least `MIN_GATE` items; otherwise `needs_review` and a human decides. **Caveat we must state:** on our one real run, KEJ's 5-item gate reversed sign when a single item was removed. A 5-item gate is a prototype policy, not a validated threshold. |
| **C4** executable contract | **Fixed 2026-10-03** | `docs/openapi_ai_v1.yaml` was behind the code when you read it on 10-02 and that was our bug. It now documents `POST /v1/adapters/train` 202/400/404/409, `GET /v1/jobs/{job_id}` with the full job schema, and the three `/v1/enroll/recordings` verbs. Generate DTOs from it. Where it and the code still disagree, the code wins and we want to hear about it. |

## 3. Training and job contract (T1–T9)

**Read T1 first; several answers depend on it.**

| ID | Status | Answer |
|---|---|---|
| **T1** capability/version | **Implemented-local-only** | `POST /v1/adapters/train` → **202**, `GET /v1/jobs/{job_id}` → **200**, both in `demo/server.py` today. **They have never run anywhere you can reach.** The only real executions were on a weekend GPU allocation, driven by hand. There is no deployed AI server for you to call, and we do not control one. **Do not return 202 to your users on the strength of this document.** Readiness signal: `GET /v1/health` returns `contract_version`; we will add an explicit training-capability flag when there is a deployment to report. |
| **T2** HTTP schema | **Implemented, but see T1** | `POST /v1/adapters/train`, body `{"user_id": "<uuid>"}`, `Content-Type: application/json`, no auth in the demo (you terminate auth). Success **202** with the job record: `job_id`, `user_id`, `state`, `created_at`, `min_enroll`, `min_gate`, `contract_version`. Errors: `400 missing_user_id`, `404 no_enrollment` (with the path we looked at). **`train_items[]` does not exist in the implemented route** — the server reads enrollment from a directory (T3). CT §3.7's `train_items[]` is a *design sketch*; do not build DTOs from it. Your job ID and ours are **different identifiers**; store both. |
| **T3** audio transfer | **Needs your decision** | Today the server reads `ENROLL_DIR/<user_id>/pairs.json` + WAV files from its own filesystem — PCM16 mono 16 kHz, same as `/v1/asr/transcribe`. **A storage key of yours is not readable by us**, you are right. There is no implemented transfer mechanism. Proposal, not built: you write the snapshot to a path we can read, including a manifest with per-file sha256, and we verify every hash before training and refuse the job on any mismatch. The one-sample round trip you ask for is the right first test and we agree it should come before anything else. |
| **T4** references and pool | **Decided — this one matters** | Training references **must** be human-reviewed transcriptions of what was actually read. **Never** use the suggested prompt text as the reference, and **never** use the user's confirmed outgoing message. Both are texts the user *intended*; the acoustic model needs what was *said*. Supply `prompt_id` + `pool_version` + `pool_sha256` (R5) **alongside** the reviewed reference, never instead of it. |
| **T5** selection and splits | **Partly decided** | Keep every retake, including failed ones; discarding the hard attempts is how an enrollment set drifts easy. Split assignment and seed are ours (C2). **Minimum:** our worker refuses below `MIN_ENROLL = 15` and requires `MIN_GATE = 5`. **Both are policy, not evidence** — we have never measured how little enrollment suffices; our two controlled runs used 18 and 54 utterances. Your "minimum 5" cannot support a gate split at all. Raise it or mark it a placeholder. Dev/gate behaviour across retraining is **unresolved** and blocked on the C2 defect. |
| **T6** idempotency | **Needs your decision** | **`request_id` on ASR is not a training idempotency key** — you are right to call that out, and we agree. ASR idempotency is implemented and keyed `(user_id, request_id)`. **Training submission has no idempotency key today.** Proposal: you send a submission key, we return the existing job for the same key and `409` for the same key with different inputs; retention unresolved. Until that exists, a lost response means you cannot tell whether a job started — do not auto-retry. |
| **T7** status lookup | **Implemented-local-only** | `GET /v1/jobs/{job_id}`, polling (no events). Terminal states: `promoted`, `rejected`, `needs_review`, `failed`. Suggested mapping: `queued`→PENDING; `running`/`evaluating`→IN_PROGRESS; `promoted`/`rejected`/`needs_review`→COMPLETED; `failed`→FAILED. **`rejected` maps to COMPLETED, not FAILED** — the job did its work. Unknown ID → `404 unknown_job`; we do not currently distinguish expired from unknown. No rate limit implemented; 5 s polling is fine against a single-worker server. |
| **T8** progress and failure | **Decided** | **We do not supply numeric progress.** Agree on `progress: null`; please do not synthesise a percentage. Failure carries `error` (free text) and `trainer_returncode`. Retryability is **not** currently signalled — treat `failed` as not-automatically-retryable and surface it. Timeout: the worker kills a run after `TRAIN_TIMEOUT` (default 5400 s) and the job becomes `failed`. **Jobs are in-process and do not survive a restart** — a restart strands `queued`/`running` records. That is a known gap, not a design. |
| **T9** cancellation and withdrawal | **Unsupported today** | No cancellation API, no withdrawal API, no deletion API. Withdrawal before submission is a backend-side matter (do not hand us the data). Withdrawal **after** training must retire derived artifacts, not just source audio: the adapter directory, `active.json` entry, job records and work directory. Today that is a manual deletion with no confirmation record. If consent withdrawal is in scope for the submission, this needs to be designed before the first real user, not after. |

## 4. Adapter artifacts and serving (A1–A5)

| ID | Status | Answer |
|---|---|---|
| **A1** artifact identity | **Partly implemented** | Each promoted adapter writes `meta.json` with `user_id`, `adapter_id`, `base_model`, `base_revision`, `job_id`, `created_at`, `gate_cer_jamo`, `gate_n`, `replaced`. The serving layer computes `adapter_revision` as a **sha256 of the weight files**. **Not yet included:** a hash of the adapter *config*, artifact size in the metadata, and a training-snapshot ID. We will add them; those are gaps, not disagreements. |
| **A2** evaluation and promotion | **Implemented; the configuration mismatch was fixed 2026-10-03** | AI computes candidate metrics on the held-out gate against the incumbent (C3). Evaluation, gate and serving now share **one** configuration, `demo/decoding.py`, and all three persist it: `GET /v1/health` carries `generation`, every transcribe response carries `decoding`, every gate cell carries `generation`. It is identified by `generation_version` (currently `gen-v1`) — **numbers produced under different versions are not comparable, and you should store the version with any metric you keep.** Resolved defaults: greedy (`num_beams=1`, `do_sample=false`), `max_new_tokens=200`, Korean transcription, **no** repetition constraint (that is an experiment control, not a serving policy). A test asserts the serving response equals the shared config, so the paths cannot drift apart silently. **Caveat you should hear from us rather than discover:** the evaluation numbers we quote from 2026-10-03 were measured **before** this alignment, with no token cap. They are an evaluation condition, not measured serving performance, and re-measuring needs GPU time we do not have before 2026-10-07. |
| **A3** installation and readiness | **Was a defect; fixed and tested 2026-10-03** | `adapters/active.json` maps `user_id → adapter_id` and is the only thing that makes an adapter live. Absent file, absent entry, unknown id, wrong owner or unreadable JSON all serve the **base**, which is a supported state, not a failure. **The defect you would have hit:** the server scanned the adapter directory at startup only, so a promotion during uptime was ignored and the base kept being served. It now rescans on a miss and loads on demand. **Reproduced before fixing:** `demo/test_activation.py` starts a real server process and writes the adapter from a different process while it runs; it fails on the old code and passes 7 groups on the new one, covering promotion during uptime, a second promotion replacing the first, **rollback by rewriting the pointer**, forced base on the same server, another user's adapter refused, and an unknown id. **No restart is required, and you may display active personalization once a transcribe response actually returns the adapter id** — not merely on a promoted job. |
| **A4** actual model used | **Implemented on our side; yours needs a change** | `model.adapter_id` and `model.adapter_revision` in the transcribe response identify the weights that actually ran — **`null` exactly when the base ran**. Forced base is `use_adapter=false` (implemented and tested). Your `HttpAiInferenceClient` currently discards `modelType` and never sends it, and `RecognizeSpeechService` infers `modelUsed` from whether a completed training job exists. **Persist what we return; never infer from job state.** Keep "requested" and "actually used" as two columns. Verification: send the same WAV twice, once with `use_adapter=false`, and confirm the two stored `adapter_id` values differ. |
| **A5** mismatch and fallback | **Implemented 2026-10-03** | Base revision is **pinned and enforced**: the server refuses to start if the loaded revision differs from `973afd24965f72e36ca33b3055d56a652f456b4d`, so a silent revision mismatch cannot happen. **`adapter_id: null` is no longer ambiguous.** The transcribe response now carries `model.base_reason`, which is `null` exactly when an adapter ran and otherwise one of: `base_requested` (you sent `use_adapter=false`), `no_active_adapter`, `adapter_not_found` (pointer at an id that is not there), `adapter_wrong_owner`, `adapter_base_mismatch` (trained against a different base), `adapter_load_error`. **Persist it** — it is how you tell "the user has no model yet" from "their model is broken". Still missing: a readiness endpoint listing per-user load failures ahead of a request. |

## 5. Offline-first option

**We recommend this as the route for the current demo**, and it is what we would pick if we had to
choose today. It needs no deployed AI server, which we do not have (T1).

The record chain you proposed is right. Minimum per cycle:

| Step | Owner | Record |
|---|---|---|
| consent | BE | consent row, user id, timestamp |
| reviewed reference + revision | BE | text + reviewer + revision (T4) |
| immutable input snapshot | BE | directory + `pairs.json` + per-file sha256 |
| AI receipt | AI | hash verification result, refuse on mismatch |
| training + gate evaluation | AI | job record incl. split sizes, incumbent and candidate scores |
| artifact identity | AI | `meta.json` (A1) |
| installation + activation confirmation | AI | `active.json` entry **plus a live inference check** (A3) |
| deletion / retirement | AI + BE | confirmation record (T9, not implemented) |

Manual export is an alternative route, not a prerequisite for an online API, and we are not making
JPyRust a prerequisite for anything.

## 6. Verification plan — what is already proven and what is not

**Already passing, reproducible by you:** `demo/test_core.py` (14 groups, transport-independent),
40 HTTP contract checks, `demo/test_train_worker.py` (7 groups, promotion gate with the trainer
stubbed), chunked and `Content-Length` request framing, pool identity fields.

**Never tested, in the order we think they matter:**

1. ~~**Runtime activation (A3).**~~ **Done 2026-10-03** — `demo/test_activation.py`, 7 groups
   against a real server process, including rollback. Reproduced on the pre-fix code first.
2. **One real audio transfer (T3)** with hash verification on our side.
3. **Forced base vs active adapter (A4)** on the same WAV, both stored.
4. **A real gate rejection (C3).** The gate has never refused anything in a live run; both real jobs
   promoted. The cheapest test is an existing worse artifact against a better incumbent — no
   deliberately bad training run is needed.
5. **Submission retry (T6)** once an idempotency key exists.
6. **Terminal-state mapping (T7)**, including `rejected` → COMPLETED.

## 7. What we are asking you to decide

1. **R1** — whether provisioning the prompt pool is acceptable to you pending our AI-Hub terms check.
2. **T3** — the snapshot path and manifest format.
3. **T5** — raise the "minimum 5 recordings" or mark it explicitly provisional.
4. **T6** — whether you will supply a submission idempotency key.
5. **T9** — whether consent withdrawal is in scope for this submission.
6. **C1/T7** — confirm `rejected` renders as a normal completed outcome, not an error.

## 8. Things in your document we could not confirm

- Your 2.4-vs-6.3 eligible-token simulation (R2) is **your** measurement; we reproduced only the
  ±31 pp / ±21.9 pp interval arithmetic.
- We did not audit your current branch for this reply. Statements about backend behaviour are quoted
  from your pages.
- **Correction to our own earlier document:** `AI_REQUESTS_TO_BACKEND_2026-09-29_EN.md` §7 asked you
  to reconcile a 1,807-vs-3,436 prompt count. That is resolved and **you were right** — see R5.
