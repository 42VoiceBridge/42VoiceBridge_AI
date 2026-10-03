#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The one generation configuration, shared by serving, evaluation and the promotion gate.

2026-10-03 review, A2: `gate_eval.py` generated with no `max_new_tokens` while `server.py` capped at
200, so a gate score was not a measured serving score. In that run one adapter emitted an
**880-character** output on a single short utterance under the uncapped setting; whether the server's
cap would have changed it was unknown, which is exactly the problem.

A promotion decision is only meaningful if the candidate was measured the way it will be served, so
both call `settings()` here and both record `generation_version` beside their numbers.

`no_repeat_ngram` stays **0 by default**: it is a symmetric sensitivity control for experiments, not
a serving policy, and the same review warned that choosing n after looking at a failing item would
be test-set tuning.

    python3 decoding.py --selftest
"""
import os
import sys

# Bump when any default below changes. Numbers produced under different versions are not comparable.
GENERATION_VERSION = "gen-v1"

MAX_NEW_TOKENS = int(os.environ.get("ASR_MAX_NEW_TOKENS", "200"))
BEAMS = int(os.environ.get("ASR_BEAMS", "1"))           # D9: greedy
LANGUAGE = "korean"
TASK = "transcribe"


def settings(no_repeat_ngram=0):
    """Resolved generation kwargs. The caller passes only the audio features."""
    kw = dict(language=LANGUAGE, task=TASK, num_beams=BEAMS, do_sample=False,
              max_new_tokens=MAX_NEW_TOKENS)
    if no_repeat_ngram:
        kw["no_repeat_ngram_size"] = int(no_repeat_ngram)
    return kw


def describe(no_repeat_ngram=0):
    """What to persist beside a score so two numbers can be compared later."""
    return dict(generation_version=GENERATION_VERSION, **settings(no_repeat_ngram))


def selftest():
    a = settings()
    assert a["num_beams"] == BEAMS and a["do_sample"] is False
    assert a["max_new_tokens"] == MAX_NEW_TOKENS, "serving and gate must share the cap"
    assert "no_repeat_ngram_size" not in a, "nr5 is a control arm, never a default"
    b = settings(5)
    assert b["no_repeat_ngram_size"] == 5
    assert {k: v for k, v in b.items() if k != "no_repeat_ngram_size"} == a, \
        "the control arm must differ in exactly one key"
    d = describe(5)
    assert d["generation_version"] == GENERATION_VERSION and d["no_repeat_ngram_size"] == 5
    print("selftest ok")


if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv or True else None
