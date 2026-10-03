#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-user adapter training with a held-out promotion gate (2026-09-29).

Why the gate exists. On 2026-09-26 a controlled run measured a speaker for whom **all five**
independently seeded adapters were worse than the base. So "training finished" must not mean
"serve it". Every run is judged on a split that was never trained on and never used to pick the
epoch, against the **incumbent** - whatever that user is served today, which may be the base.

Splits, deterministic from the user id so a rerun reproduces them:
    gate  = max(MIN_GATE, 20%)   held out entirely; only the promotion decision sees it
    dev   = max(2, 15%)          b1_train.py picks the epoch on this
    train = the rest

Job states: queued -> running -> evaluating -> promoted | rejected | needs_review | failed.
`needs_review` means the gate set was too small to decide automatically; a human then edits
adapters/active.json. Promotion is the only thing that writes that file.

MIN_ENROLL and MIN_GATE are POLICY, not evidence. We have never measured how little enrollment
suffices; our two controlled runs used 18 and 54 utterances. The backend's "minimum 5 recordings"
is below anything we have tested and cannot support a gate split at all.

    python3 train_worker.py --selftest
"""
import hashlib
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
ENROLL_DIR = os.environ.get("ENROLL_DIR", os.path.join(HERE, "enroll"))
JOB_DIR = os.environ.get("JOB_DIR", os.path.join(HERE, "jobs"))
ADAPTER_DIR = os.environ.get("ADAPTER_DIR", os.path.join(HERE, "adapters"))
TRAINER = os.environ.get("TRAINER",
                         os.path.join(PROJECT, "experiments", "t10_highcer", "b1_train.py"))
BASE = os.environ.get("ASR_BASE", "openai/whisper-small")
BASE_REVISION = os.environ.get("ASR_BASE_REVISION", "973afd24965f72e36ca33b3055d56a652f456b4d")

MIN_ENROLL = int(os.environ.get("MIN_ENROLL", "15"))
MIN_GATE = int(os.environ.get("MIN_GATE", "5"))
TRAIN_TIMEOUT = int(os.environ.get("TRAIN_TIMEOUT", "5400"))

Q = queue.Queue()
_worker = None


# ---------------------------------------------------------------- splits
SPLIT_VERSION = "split-v2"


def split_state_path(state_dir):
    return os.path.join(state_dir, "splits.json")


def read_split_state(state_dir):
    """{file: "train"|"dev"|"gate"} assigned in earlier rounds, or {} if none/incompatible."""
    if not state_dir:
        return {}
    p = split_state_path(state_dir)
    if not os.path.isfile(p):
        return {}
    try:
        d = json.load(open(p, encoding="utf-8"))
    except ValueError:
        return {}                              # unreadable state reassigns; it never crashes a job
    if d.get("split_version") != SPLIT_VERSION:
        return {}
    a = d.get("assigned") or {}
    return {k: v for k, v in a.items() if v in ("train", "dev", "gate")}


def write_split_state(state_dir, assigned):
    p = split_state_path(state_dir)
    os.makedirs(state_dir, exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(dict(split_version=SPLIT_VERSION, assigned=assigned,
                       updated_at=time.strftime("%Y-%m-%d %H:%M:%S")), f,
                  ensure_ascii=False, indent=1)
    os.replace(tmp, p)                         # atomic: a reader never sees a half-written map


def split_pairs(pairs, user_id, state_dir=None):
    """(train, dev, gate). Membership is PERMANENT for an item once assigned.

    2026-10-03 review defect (T26 item 2): this used to hash-rank every item and slice by
    proportion, so adding enrollment RE-RANKED everything and an item that trained the previous
    adapter could land in the next round's gate. The gate would then be scoring a model on its
    own training data and every promotion after the first was unsound.

    Fix is persistence, not a better hash: prior assignments are kept verbatim and only NEW items
    are placed, into whichever split is furthest below its target. Nothing ever moves. If the set
    shrinks and a split ends up over target, it stays over - moving an item back is the very thing
    this prevents. `state_dir` None keeps the function pure (tests, one-shot splits).
    """
    n = len(pairs)
    n_gate = max(MIN_GATE, round(n * 0.20))
    n_dev = max(2, round(n * 0.15))
    if n_gate + n_dev + 2 > n:                 # never leave fewer than 2 training items
        return None, None, None

    def rank(x):
        return hashlib.sha256((user_id + "|" + x["file"]).encode("utf-8")).hexdigest()

    prior = read_split_state(state_dir)
    groups = {"train": [], "dev": [], "gate": []}
    new = []
    for p in sorted(pairs, key=rank):
        where = prior.get(p["file"])
        (groups[where] if where else new).append(p)

    target = {"gate": n_gate, "dev": n_dev, "train": n - n_gate - n_dev}
    for p in new:                              # deterministic: `new` is already in hash order
        pick = max(("gate", "dev", "train"),
                   key=lambda k: (target[k] - len(groups[k]), k == "gate", k == "dev"))
        groups[pick].append(p)

    # the same set must come back in the same order whether an item was placed this round or a
    # previous one, otherwise the trainer sees a different input order on a rerun
    for k in groups:
        groups[k].sort(key=rank)

    if len(groups["train"]) < 2 or len(groups["gate"]) < 1:
        return None, None, None
    if state_dir:
        write_split_state(state_dir, {p["file"]: k for k, v in groups.items() for p in v})
    return groups["train"], groups["dev"], groups["gate"]


def decide(incumbent_cer, new_cer, n_gate):
    """(state, reason). Auto-promotion needs a gate set big enough to mean anything."""
    if new_cer is None or incumbent_cer is None:
        return "failed", "gate evaluation produced no score"
    if n_gate < MIN_GATE:
        return "needs_review", ("gate set has %d items, below the %d needed to decide "
                                "automatically" % (n_gate, MIN_GATE))
    if new_cer < incumbent_cer:
        return "promoted", ("gate jamo CER %.4f < incumbent %.4f" % (new_cer, incumbent_cer))
    return "rejected", ("gate jamo CER %.4f is not better than the incumbent %.4f; "
                        "the incumbent stays active" % (new_cer, incumbent_cer))


# ---------------------------------------------------------------- job records
def job_path(jid):
    return os.path.join(JOB_DIR, jid + ".json")


def write_job(job):
    os.makedirs(JOB_DIR, exist_ok=True)
    job["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    tmp = job_path(job["job_id"]) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(job, f, ensure_ascii=False, indent=1)
    os.replace(tmp, job_path(job["job_id"]))         # atomic: a reader never sees a half-write
    return job


def read_job(jid):
    p = job_path(jid)
    return json.load(open(p, encoding="utf-8")) if os.path.isfile(p) else None


def active_map():
    p = os.path.join(ADAPTER_DIR, "active.json")
    if not os.path.isfile(p):
        return {}
    try:
        m = json.load(open(p, encoding="utf-8"))
        return m if isinstance(m, dict) else {}
    except Exception:                                        # noqa: BLE001
        return {}


def set_active(user_id, adapter_id):
    os.makedirs(ADAPTER_DIR, exist_ok=True)
    m = active_map()
    prev = m.get(user_id)
    m[user_id] = adapter_id
    p = os.path.join(ADAPTER_DIR, "active.json")
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=1)
    os.replace(tmp, p)
    return prev


# ---------------------------------------------------------------- the job itself
def load_pairs(user_id):
    d = os.path.join(ENROLL_DIR, user_id)
    p = os.path.join(d, "pairs.json")
    if not os.path.isfile(p):
        raise ValueError("no enrollment at %s" % p)
    pairs = json.load(open(p, encoding="utf-8"))
    pairs = pairs.get("pairs", pairs) if isinstance(pairs, dict) else pairs
    out = []
    for i, x in enumerate(pairs):
        f = os.path.join(d, x["file"])
        if not os.path.isfile(f):
            raise ValueError("missing audio %s" % f)
        if not (x.get("text") or "").strip():
            raise ValueError("empty text for %s" % x["file"])
        out.append(dict(file=x["file"], path=f, text=x["text"].strip(), idx=i))
    # 2026-10-03: a packaging bug put held-out evaluation material into an enrollment directory and
    # it silently became training data AND gate data. Duplicate texts are the cheapest tripwire:
    # the same sentence must not appear twice, because the split would then straddle it.
    seen = {}
    for r in out:
        # whitespace and trailing punctuation only: the real leak differed by one trailing period,
        # but digits must stay or "문장 1" and "문장 2" would collide.
        key = "".join(r["text"].split()).rstrip(".!?,;·…")
        if key and key in seen:
            raise ValueError("enrollment contains the same sentence twice (%s and %s); "
                             "a split cannot hold it out from itself" % (seen[key], r["file"]))
        seen[key] = r["file"]
    return out


def build_manifest(user_id, train, dev, gate, workdir):
    """b1_train.py reads split/speaker/seg_id/sec; the splits are fixed before training starts."""
    import wave
    items = []
    for split, rows in (("enroll", train), ("dev", dev), ("test", gate)):
        for r in rows:
            w = wave.open(r["path"], "rb")
            sec = w.getnframes() / float(w.getframerate())
            w.close()
            items.append(dict(seg_id=os.path.splitext(r["file"])[0], file=r["file"],
                              speaker=user_id, task="enroll", src_id=user_id,
                              sent_idx=r["idx"], start=0.0, end=round(sec, 2), sec=round(sec, 2),
                              text=r["text"], split=split, verified=False))
    man = dict(name="enroll-%s" % user_id, split_version="worker-v1", seg_version="upload",
               anchor=None, seed=0, note="splits fixed before training; 'test' is the promotion gate",
               items=items)
    p = os.path.join(workdir, "manifest.json")
    json.dump(man, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return p


def run_job(job):
    uid = job["user_id"]
    work = os.path.join(JOB_DIR, job["job_id"] + "_work")
    os.makedirs(work, exist_ok=True)
    try:
        pairs = load_pairs(uid)
        job["n_enroll"] = len(pairs)
        if len(pairs) < MIN_ENROLL:
            return write_job(dict(job, state="failed",
                                  error="only %d enrollment utterances; policy minimum is %d "
                                        "(needed for a train/dev/gate split that means anything)"
                                        % (len(pairs), MIN_ENROLL)))
        train, dev, gate = split_pairs(pairs, uid, os.path.join(ENROLL_DIR, uid))
        if train is None:
            return write_job(dict(job, state="failed",
                                  error="cannot split %d utterances into train/dev/gate" % len(pairs)))
        job.update(state="running", n_train=len(train), n_dev=len(dev), n_gate=len(gate))
        write_job(job)

        man = build_manifest(uid, train, dev, gate, work)
        segdir = os.path.join(work, "segments")
        os.makedirs(segdir, exist_ok=True)
        for r in train + dev + gate:
            shutil.copy2(r["path"], os.path.join(segdir, r["file"]))

        cmd = [sys.executable, TRAINER, "--speaker", uid, "--manifest", man,
               "--segdir", segdir, "--out", work, "--base", BASE,
               "--base-revision", BASE_REVISION]
        if os.environ.get("ALLOW_CPU_TRAIN") == "1":
            cmd.append("--cpu-ok")
        job["trainer_cmd"] = " ".join(cmd)
        write_job(job)
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=TRAIN_TIMEOUT)
        job["trainer_returncode"] = r.returncode
        job["trainer_tail"] = (r.stdout or "")[-2000:] + (r.stderr or "")[-2000:]
        if r.returncode != 0:
            return write_job(dict(job, state="failed", error="trainer exited %d" % r.returncode))

        cand = None
        for root, dirs, _ in os.walk(work):
            if "adapter" in dirs:
                cand = os.path.join(root, "adapter")
                break
        if not cand:
            return write_job(dict(job, state="failed", error="trainer produced no adapter/"))
        job.update(state="evaluating", new_adapter_path=cand)
        write_job(job)

        inc_id = active_map().get(uid)
        inc_path = os.path.join(ADAPTER_DIR, inc_id) if inc_id else None
        spec = dict(base=BASE, base_revision=BASE_REVISION,
                    items=[dict(wav=os.path.join(segdir, g["file"]), ref=g["text"]) for g in gate],
                    candidates=dict(incumbent=inc_path, new=cand))
        sp = os.path.join(work, "gate.json")
        json.dump(spec, open(sp, "w", encoding="utf-8"), ensure_ascii=False)
        g = subprocess.run([sys.executable, os.path.join(HERE, "gate_eval.py"), sp],
                           capture_output=True, text=True, timeout=TRAIN_TIMEOUT)
        if g.returncode != 0:
            return write_job(dict(job, state="failed",
                                  error="gate evaluation exited %d" % g.returncode,
                                  gate_tail=(g.stderr or "")[-2000:]))
        res = json.loads(g.stdout.strip().splitlines()[-1])
        job["gate"] = res
        inc_cer = res["results"]["incumbent"]["default"]["pooled_cer_jamo"]
        new_cer = res["results"]["new"]["default"]["pooled_cer_jamo"]
        job.update(incumbent_adapter=inc_id, incumbent_cer_jamo=inc_cer, new_cer_jamo=new_cer)

        state, reason = decide(inc_cer, new_cer, len(gate))
        job.update(state=state, decision=reason)
        if state == "promoted":
            aid = "%s-%s" % (uid, job["job_id"][:8])
            dst = os.path.join(ADAPTER_DIR, aid)
            shutil.copytree(cand, dst, dirs_exist_ok=True)
            json.dump(dict(user_id=uid, adapter_id=aid, base_model=BASE,
                           base_revision=BASE_REVISION, job_id=job["job_id"],
                           created_at=time.strftime("%Y-%m-%d %H:%M"),
                           gate_cer_jamo=new_cer, gate_n=len(gate),
                           replaced=job.get("incumbent_adapter")),
                      open(os.path.join(dst, "meta.json"), "w", encoding="utf-8"),
                      ensure_ascii=False, indent=1)
            job["previous_active"] = set_active(uid, aid)
            job["adapter_id"] = aid
        return write_job(job)
    except subprocess.TimeoutExpired:
        return write_job(dict(job, state="failed", error="timed out after %d s" % TRAIN_TIMEOUT))
    except Exception as e:                                   # noqa: BLE001
        return write_job(dict(job, state="failed", error="%s: %s" % (type(e).__name__, e)))


def _loop():
    while True:
        jid = Q.get()
        job = read_job(jid)
        if job:
            run_job(job)
        Q.task_done()


def submit(user_id):
    global _worker
    if _worker is None:
        _worker = threading.Thread(target=_loop, daemon=True)
        _worker.start()
    jid = "job-%s" % hashlib.sha256(
        ("%s|%.6f" % (user_id, time.time())).encode()).hexdigest()[:16]
    job = write_job(dict(job_id=jid, user_id=user_id, state="queued",
                         created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                         min_enroll=MIN_ENROLL, min_gate=MIN_GATE))
    Q.put(jid)
    return job


def selftest():
    pairs = [dict(file="f%02d.wav" % i) for i in range(20)]
    tr, dv, gt = split_pairs(pairs, "u1")
    assert len(gt) == 5 and len(dv) == 3 and len(tr) == 12, (len(tr), len(dv), len(gt))
    assert not (set(x["file"] for x in tr) & set(x["file"] for x in gt)), "gate leaked into train"
    assert not (set(x["file"] for x in dv) & set(x["file"] for x in gt)), "gate leaked into dev"
    assert split_pairs(pairs, "u1") == (tr, dv, gt), "split is not deterministic"

    # T26 item 2: membership must survive a later round that adds enrollment.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        a_tr, a_dv, a_gt = split_pairs(pairs, "u1", td)
        before = {}
        for name, grp in (("train", a_tr), ("dev", a_dv), ("gate", a_gt)):
            for x in grp:
                before[x["file"]] = name
        grown = pairs + [dict(file="g%02d.wav" % i) for i in range(40)]
        b_tr, b_dv, b_gt = split_pairs(grown, "u1", td)
        after = {}
        for name, grp in (("train", b_tr), ("dev", b_dv), ("gate", b_gt)):
            for x in grp:
                after[x["file"]] = name
        moved = [f for f, w in before.items() if after[f] != w]
        assert not moved, "split membership changed across rounds: %s" % moved[:5]
        # the new items filled the under-target splits, so the gate grew but kept its old members
        assert len(b_gt) == max(MIN_GATE, round(60 * 0.20)), len(b_gt)
        assert set(x["file"] for x in a_gt) <= set(x["file"] for x in b_gt)
        assert split_pairs(grown, "u1", td) == (b_tr, b_dv, b_gt), "not stable on reread"
        # a corrupt state file reassigns instead of crashing the job
        open(split_state_path(td), "w").write("{oops")
        assert split_pairs(grown, "u1", td)[0] is not None
    assert split_pairs(pairs, "u2")[2] != gt, "split does not depend on the user"
    assert split_pairs([dict(file="a.wav")] * 6, "u")[0] is None, "tiny set must refuse to split"

    assert decide(0.40, 0.30, 5)[0] == "promoted"
    assert decide(0.30, 0.30, 5)[0] == "rejected", "a tie must not promote"
    assert decide(0.30, 0.40, 5)[0] == "rejected"
    assert decide(0.40, 0.30, 3)[0] == "needs_review", "a small gate must not auto-promote"
    assert decide(None, 0.30, 5)[0] == "failed"
    print("selftest ok")


if __name__ == "__main__":
    selftest()
