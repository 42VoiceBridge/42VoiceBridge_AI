#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Promotion-gate tests (2026-09-29). No GPU, no model: the trainer and the gate are stubbed.

What matters here is not that training works - that needs a GPU - but that a worse adapter
CANNOT go live. The 2026-09-26 run measured a speaker whose five adapters were all worse than the
base, so this is the path that stops us shipping that to a user.

    python3 test_train_worker.py
"""
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import wave

HERE = os.path.dirname(os.path.abspath(__file__))
TMP = tempfile.mkdtemp(prefix="tw_")
os.environ.update(ENROLL_DIR=os.path.join(TMP, "enroll"), JOB_DIR=os.path.join(TMP, "jobs"),
                  ADAPTER_DIR=os.path.join(TMP, "adapters"))
sys.path.insert(0, HERE)
import train_worker as W  # noqa: E402


def wav(path, sec=1.0):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    w = wave.open(path, "wb")
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
    w.writeframes(b"".join(struct.pack("<h", 500) for _ in range(int(sec * 16000))))
    w.close()


def enroll(uid, n):
    d = os.path.join(W.ENROLL_DIR, uid)
    pairs = []
    for i in range(n):
        f = "u%02d.wav" % i
        wav(os.path.join(d, f))
        pairs.append(dict(file=f, text="문장 %d 입니다" % i))
    json.dump(pairs, open(os.path.join(d, "pairs.json"), "w", encoding="utf-8"),
              ensure_ascii=False)
    return d


def stub(new_cer, inc_cer, trainer_rc=0, make_adapter=True):
    """Replace both subprocesses: b1_train.py and gate_eval.py."""
    real = subprocess.run

    def fake(cmd, **kw):
        class R:
            returncode = 0
            stdout = ""
            stderr = ""
        if cmd[1].endswith("b1_train.py"):
            R.returncode = trainer_rc
            if make_adapter and trainer_rc == 0:
                out = cmd[cmd.index("--out") + 1]
                ad = os.path.join(out, "run", "adapter")
                os.makedirs(ad, exist_ok=True)
                open(os.path.join(ad, "adapter_config.json"), "w").write("{}")
                open(os.path.join(ad, "adapter_model.safetensors"), "wb").write(b"w")
            return R
        if cmd[1].endswith("gate_eval.py"):
            spec = json.load(open(cmd[2], encoding="utf-8"))
            R.stdout = json.dumps(dict(device="cpu", base="b", base_revision="r", results=dict(
                incumbent={"default": dict(adapter=spec["candidates"]["incumbent"],
                                           pooled_cer_jamo=inc_cer,
                                           n_items=len(spec["items"]), n_jamo=100)},
                new={"default": dict(adapter=spec["candidates"]["new"],
                                     pooled_cer_jamo=new_cer,
                                     n_items=len(spec["items"]), n_jamo=100)})))
            return R
        return real(cmd, **kw)
    subprocess.run = fake
    return real


def run_sync(uid):
    job = W.write_job(dict(job_id="j-" + uid, user_id=uid, state="queued"))
    return W.run_job(job)


def main():
    ok = 0
    W.selftest()

    # a better adapter is promoted and becomes active
    enroll("alice", 20)
    real = stub(new_cer=0.30, inc_cer=0.45)
    j = run_sync("alice")
    assert j["state"] == "promoted", j
    assert W.active_map()["alice"] == j["adapter_id"], W.active_map()
    meta = json.load(open(os.path.join(W.ADAPTER_DIR, j["adapter_id"], "meta.json"),
                          encoding="utf-8"))
    assert meta["user_id"] == "alice" and meta["gate_n"] == j["n_gate"], meta
    first = j["adapter_id"]
    ok += 1

    # a worse one is REJECTED and the incumbent stays active -- the 2026-09-26 failure mode
    subprocess.run = real
    stub(new_cer=0.50, inc_cer=0.30)
    j2 = run_sync("alice")
    assert j2["state"] == "rejected", j2
    assert W.active_map()["alice"] == first, "a worse adapter replaced the incumbent"
    assert j2.get("adapter_id") is None
    ok += 1

    # a tie does not promote either
    subprocess.run = real
    stub(new_cer=0.30, inc_cer=0.30)
    assert run_sync("alice")["state"] == "rejected"
    assert W.active_map()["alice"] == first
    ok += 1

    # too little enrollment: refused with the policy reason, nothing written
    subprocess.run = real
    enroll("bob", W.MIN_ENROLL - 1)
    stub(new_cer=0.1, inc_cer=0.9)
    j3 = run_sync("bob")
    assert j3["state"] == "failed" and "minimum" in j3["error"], j3
    assert "bob" not in W.active_map()
    ok += 1

    # a trainer crash never touches the active pointer
    subprocess.run = real
    enroll("carol", 20)
    stub(new_cer=0.1, inc_cer=0.9, trainer_rc=1)
    j4 = run_sync("carol")
    assert j4["state"] == "failed" and j4["trainer_returncode"] == 1, j4
    assert "carol" not in W.active_map()
    ok += 1

    # trainer exits 0 but produces nothing
    subprocess.run = real
    stub(new_cer=0.1, inc_cer=0.9, make_adapter=False)
    j5 = run_sync("carol")
    assert j5["state"] == "failed" and "no adapter" in j5["error"], j5
    assert "carol" not in W.active_map()
    ok += 1

    # the gate set never overlaps training data
    subprocess.run = real
    pairs = [dict(file="f%02d.wav" % i) for i in range(30)]
    tr, dv, gt = W.split_pairs(pairs, "dave")
    names = lambda rows: {r["file"] for r in rows}
    assert not (names(gt) & (names(tr) | names(dv))), "gate leaked"
    assert len(tr) + len(dv) + len(gt) == 30
    ok += 1

    subprocess.run = real
    shutil.rmtree(TMP, ignore_errors=True)
    print("promotion gate checks passed (%d groups)" % ok)


if __name__ == "__main__":
    main()
