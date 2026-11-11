"""python tests/test_submission_model.py -- the COMMITTED models (models/gru.pt, the submission
default, and models/lgbm.txt) end to end.

Synthetic data only (no match data). Checks that:
  1. the committed booster loads with the pinned LightGBM,
  2. its feature list matches what solution/learned.py computes today (catches silent drift),
  3. predict + HMM decode runs and is deterministic (byte-identical labels on two runs).
"""

import hashlib
import json
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

import lightgbm as lgb  # noqa: E402

import synthetic  # noqa: E402
from solution import learned, model  # noqa: E402
from solution.make_submission import DEFAULT_MODEL  # noqa: E402

LGBM_MODEL = os.path.join(ROOT, "models", "lgbm.txt")

BLOCKS = ((1, 1000, 11000), (2, 13000, 23000))
PITCH = (synthetic.L, synthetic.W)


def run_once(booster, meta, players, ball, slot_team, period, n):
    cfg = model.Config(pitch=PITCH)
    hmm = meta["hmm"]
    cfg.emission_weight = hmm["w"]
    cfg.p_turnover, cfg.p_enter_dead, cfg.p_leave_dead = (
        hmm["p_turnover"], hmm["p_enter_dead"], hmm["p_leave_dead"])
    proba = learned.predict_proba(booster, players, ball, slot_team, period, PITCH)
    blocks = [(h, a, b, a) for h, a, b in BLOCKS]
    return learned.hmm_decode(proba, np.array(meta["prior"]), n, cfg, blocks=blocks)


def main():
    assert os.path.exists(LGBM_MODEL), "missing " + LGBM_MODEL
    booster = learned.load_booster(LGBM_MODEL)
    # the same model with CRLF line endings (a Windows checkout) must load and predict identically
    with open(LGBM_MODEL, "rb") as fh:
        raw = fh.read().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    with tempfile.TemporaryDirectory() as tmp:
        crlf_path = os.path.join(tmp, "lgbm_crlf.txt")
        with open(crlf_path, "wb") as fh:
            fh.write(raw)
        crlf = learned.load_booster(crlf_path)
    probe = np.random.default_rng(0).normal(size=(50, booster.num_feature()))
    assert np.array_equal(crlf.predict(probe), booster.predict(probe)), "CRLF model differs"
    with open(LGBM_MODEL + ".json") as fh:
        meta = json.load(fh)

    players, ball, truth = synthetic.make_match(blocks=BLOCKS, seed=3)
    n = len(ball)
    slot_team = np.array([0] * 11 + [1] * 11 + [0] * 12 + [1] * 12, dtype=np.int8)
    period = np.zeros(n, dtype=np.int8)
    for h, a, b in BLOCKS:
        period[a:b + 1] = h

    _, names = learned.compute_features(players, ball, slot_team, period, PITCH)
    assert names == meta["features"], "feature list drifted from the committed model"
    assert booster.num_feature() == len(names)

    l1 = run_once(booster, meta, players, ball, slot_team, period, n)
    l2 = run_once(booster, meta, players, ball, slot_team, period, n)
    h1 = hashlib.sha256("".join(l1).encode()).hexdigest()
    h2 = hashlib.sha256("".join(l2).encode()).hexdigest()
    assert h1 == h2, "non-deterministic output"
    assert set(np.unique(l1)) <= set("HADX")

    g = truth != "X"
    acc = float((l1[g] == truth[g]).mean())
    count, mean_s = model.possession_stats(l1)
    print("lgbm model: {} features, deterministic sha256 {}...".format(len(names), h1[:12]))
    print("  toy-match frame accuracy {:.3f}, possessions {} (mean {:.1f} s)".format(
        acc, count, mean_s))

    check_gru(players, ball, truth, slot_team, period, n, g)
    print("OK")


def check_gru(players, ball, truth, slot_team, period, n, g):
    """The committed sequence model (models/gru.pt): loads on CPU, matches the feature list, and
    gives byte-identical labels on two runs."""
    import time
    from solution import sequence
    gru_path = os.path.join(ROOT, "models", "gru.pt")
    assert DEFAULT_MODEL == gru_path, "default submission model should be the GRU"
    assert os.path.exists(gru_path), "missing " + gru_path
    net, norm, meta = sequence.load(gru_path)
    spans = [(a, b) for _h, a, b in BLOCKS]
    hashes = []
    for _ in range(2):
        t = time.time()
        proba, names = sequence.predict_proba(net, norm, players, ball, slot_team, period, PITCH,
                                              spans)
        assert names == meta["features"], "feature list drifted from the committed GRU"
        cfg = model.Config(pitch=PITCH)
        hmm = meta["hmm"]
        cfg.emission_weight = hmm["w"]
        cfg.p_turnover, cfg.p_enter_dead, cfg.p_leave_dead = (
            hmm["p_turnover"], hmm["p_enter_dead"], hmm["p_leave_dead"])
        labels = learned.hmm_decode(proba, np.array(meta["prior"]), n, cfg,
                                    blocks=[(h, a, b, a) for h, a, b in BLOCKS])
        hashes.append(hashlib.sha256("".join(labels).encode()).hexdigest())
        secs = time.time() - t
    assert hashes[0] == hashes[1], "GRU output is not deterministic"
    acc = float((labels[g] == truth[g]).mean())
    print("gru model: deterministic sha256 {}..., {:.1f} s per {} frames on CPU".format(
        hashes[0][:12], secs, n))
    print("  toy-match frame accuracy {:.3f}".format(acc))


if __name__ == "__main__":
    main()
