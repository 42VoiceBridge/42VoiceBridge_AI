#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Write a SYNTHETIC prompt pool so the suites run without the real one.

The real pool is derived from AI-Hub 013 and must never be in this repository, in an image, or in
CI (redistribution conditions, unverified). That is not a nuisance to work around: it means any
test that needs prompt text has to bring its own. The sentences below are invented, deliberately
bland, and carry no information about the corpus.

    python3 make_prompt_fixture.py /tmp/pool.json      # then PROMPT_POOL=/tmp/pool.json

Keys must start with a task code the server selects (02-03, 02-04, 06-01), because
`pool_items()` filters on exactly that - so this fixture also exercises the filter.
"""
import json
import sys

WORDS = ("창문을 닫아 주세요", "오늘 날씨가 좋습니다", "물 한 잔 주세요", "조금 천천히 말해 주세요",
         "여기 앉아도 될까요", "약을 먹을 시간입니다", "불을 켜 주세요", "소리를 줄여 주세요")


def build(n_per_code=12):
    pool = {}
    for code in ("02-03", "02-04", "06-01"):
        for i in range(n_per_code):
            pool["%s%05d" % (code, i)] = "%s %d" % (WORDS[i % len(WORDS)], i)
    pool["99-99-ignored"] = "이 항목은 과제 코드가 달라 선택되지 않아야 한다"
    return pool


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "prompt_fixture.json"
    pool = build()
    with open(out, "w", encoding="utf-8") as f:
        json.dump(pool, f, ensure_ascii=False, indent=1)
    print("%s: %d entries (%d selectable)" % (out, len(pool), len(pool) - 1))


if __name__ == "__main__":
    main()
