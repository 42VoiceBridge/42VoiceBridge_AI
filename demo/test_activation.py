#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A3: an adapter promoted while the server is RUNNING must actually be served (2026-10-03).

The 2026-10-03 review found that `HFEngine` scanned the adapter directory at startup only, so the
worker could write an adapter, update `active.json`, and report `promoted` while the live server
kept serving the base model indefinitely. A pointer file is not evidence that inference used the
new weights.

This drives a real server process over HTTP and writes the adapter from a DIFFERENT process while
it runs, which is the actual deployment scenario. It uses the mock ASR engine: what is under test is
activation, not recognition.

    python3 test_activation.py
"""
import io
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import wave

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = int(os.environ.get("ACTIVATION_PORT", "8123"))
BASE = "http://127.0.0.1:%d" % PORT


def wav_bytes(seconds=1.0):
    buf = io.BytesIO()
    w = wave.open(buf, "wb")
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(16000)
    n = int(seconds * 16000)
    w.writeframes(b"".join(struct.pack("<h", 6000 if (i // 80) % 2 else -6000) for i in range(n)))
    w.close()
    return buf.getvalue()


def transcribe(user_id, use_adapter=True):
    q = "?user_id=%s&use_adapter=%s" % (user_id, "true" if use_adapter else "false")
    req = urllib.request.Request(BASE + "/v1/asr/transcribe" + q, data=wav_bytes(),
                                 headers={"Content-Type": "audio/wav"}, method="POST")
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.load(r)


def write_adapter(adapter_dir, aid, user_id):
    d = os.path.join(adapter_dir, aid)
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "adapter_config.json"), "w").write("{}")
    open(os.path.join(d, "adapter_model.safetensors"), "wb").write(b"weights-" + aid.encode())
    json.dump(dict(user_id=user_id, adapter_id=aid, created_at="2026-10-03"),
              open(os.path.join(d, "meta.json"), "w", encoding="utf-8"))


def set_active(adapter_dir, mapping):
    p = os.path.join(adapter_dir, "active.json")
    tmp = p + ".tmp"
    json.dump(mapping, open(tmp, "w", encoding="utf-8"))
    os.replace(tmp, p)


def main():
    tmp = tempfile.mkdtemp(prefix="act_")
    adapters = os.path.join(tmp, "adapters")
    os.makedirs(adapters)
    env = dict(os.environ, ASR_ENGINE="mock", ASR_ADAPTERS=adapters, PORT=str(PORT))
    srv = subprocess.Popen([sys.executable, os.path.join(HERE, "server.py")],
                           env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        for _ in range(60):
            try:
                urllib.request.urlopen(BASE + "/v1/health", timeout=1).read()
                break
            except Exception:                                 # noqa: BLE001
                time.sleep(0.25)
        else:
            raise SystemExit("server did not start")

        ok = 0
        # 1. nothing promoted yet -> base, and the response says why
        r = transcribe("u1")
        assert r["model"]["adapter_id"] is None, r["model"]
        assert r["model"]["base_reason"] == "no_active_adapter", r["model"]
        ok += 1

        # 2. promotion happens NOW, from another process, while the server keeps running
        write_adapter(adapters, "u1-v1", "u1")
        set_active(adapters, {"u1": "u1-v1"})
        r = transcribe("u1")
        assert r["model"]["adapter_id"] == "u1-v1", (
            "server served %r after promotion -- it only scans adapters at startup"
            % r["model"]["adapter_id"])
        assert r["model"]["base_reason"] is None, r["model"]
        ok += 1

        # 3. forced base on the same running server, same user
        r = transcribe("u1", use_adapter=False)
        assert r["model"]["adapter_id"] is None and r["model"]["base_reason"] == "base_requested"
        ok += 1

        # 4. a second promotion replaces the first without a restart
        write_adapter(adapters, "u1-v2", "u1")
        set_active(adapters, {"u1": "u1-v2"})
        assert transcribe("u1")["model"]["adapter_id"] == "u1-v2"
        ok += 1

        # 5. rollback: pointing back at the previous adapter takes effect immediately
        set_active(adapters, {"u1": "u1-v1"})
        assert transcribe("u1")["model"]["adapter_id"] == "u1-v1"
        ok += 1

        # 6. a pointer at another user's adapter is refused, and says so
        set_active(adapters, {"u2": "u1-v1"})
        r = transcribe("u2")
        assert r["model"]["adapter_id"] is None
        assert r["model"]["base_reason"] == "adapter_wrong_owner", r["model"]
        ok += 1

        # 7. a pointer at an id that does not exist is distinguishable from "base requested"
        set_active(adapters, {"u3": "ghost"})
        r = transcribe("u3")
        assert r["model"]["adapter_id"] is None
        assert r["model"]["base_reason"] == "adapter_not_found", r["model"]
        ok += 1

        print("activation checks passed (%d groups)" % ok)
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=5)
        except Exception:                                     # noqa: BLE001
            srv.kill()
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
