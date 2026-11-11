"""python tests/test_learned.py -- learned pipeline on synthetic data (no match data needed).

Trains a small booster on one toy match, decodes another, checks it is sensible and deterministic.
Says nothing about performance on real football.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

import lightgbm as lgb  # noqa: E402

from solution import learned, model  # noqa: E402
import synthetic  # noqa: E402

BLOCKS = ((1, 1000, 11000), (2, 13000, 23000))
TEAMS = np.array([0] * 11 + [1] * 11 + [1 if i % 2 == 0 else 0 for i in range(22, 46)], dtype=np.int8)
PITCH = (synthetic.L, synthetic.W)


def toy(seed):
    players, ball, truth = synthetic.make_match(blocks=BLOCKS, seed=seed)
    period = np.zeros(len(truth), dtype=np.int8)
    for h, a, b in BLOCKS:
        period[a:b + 1] = h
    return players, ball, truth, period


def decode(booster, prior, players, ball, period, cfg):
    proba = learned.predict_proba(booster, players, ball, TEAMS, period, PITCH)
    return learned.hmm_decode(proba, prior, len(ball), cfg)


def main():
    p, b, truth, period = toy(0)
    X, _ = learned.compute_features(p, b, TEAMS, period, PITCH)
    ok = np.flatnonzero(truth != "X")[::5]
    y = np.array([{"H": 0, "A": 1, "D": 2}[c] for c in truth[ok]])
    params = dict(objective="multiclass", num_class=3, num_leaves=15, verbose=-1, seed=0,
                  deterministic=True, force_row_wise=True, num_threads=1)
    booster = lgb.train(params, lgb.Dataset(X[ok], y), 50)
    prior = np.bincount(y, minlength=3) / len(y)

    p2, b2, truth2, period2 = toy(1)
    cfg = model.Config(pitch=PITCH, blocks=tuple((h, a, bb, a) for h, a, bb in BLOCKS))
    pred = decode(booster, prior, p2, b2, period2, cfg)
    again = decode(booster, prior, p2, b2, period2, cfg)
    graded = truth2 != "X"
    acc = float((pred[graded] == truth2[graded]).mean())
    pc, _ = model.possession_stats(pred)
    tc, _ = model.possession_stats(truth2)
    print("frame accuracy {:.3f}  possessions pred {} truth {}".format(acc, pc, tc))
    assert (pred == again).all(), "must be deterministic"
    assert set(np.unique(pred)) <= set("HADX")
    assert (pred[~graded] == "X").all()
    assert acc > 0.7, "toy world should be easy"
    print("OK")


if __name__ == "__main__":
    main()
