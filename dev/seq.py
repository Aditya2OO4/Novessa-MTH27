"""Sequence model: BiGRU over the per-frame features, vs LightGBM, leave-one-match-out.

    python -m dev.seq                          # compare (needs the cache from `python -m dev.train`)
    python -m dev.seq --save models/gru.pt     # also train on all matches and save the model

Public IDSSE data only (other matches, CC BY 4.0). Same protocol as dev/train.py: for each held-out
match the networks are trained on the other six, and the HMM parameters are chosen on the other
six matches' out-of-fold predictions, never on the scored match. Compared, each through the HMM:

  lgbm      LightGBM per-frame classifier
  gru       BiGRU over 40 s windows of the same features (solution/sequence.py)
  avg       mean of the two probability streams

Scores use the official evaluation/possession_metrics.py. Differences below ~0.04 spellF1 are noise.
"""

import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dev import idsse, metrics, train  # noqa: E402
from solution import sequence  # noqa: E402
from solution.sequence import IGNORE, Normaliser  # noqa: E402

CODE = train.CODE


def label_codes(gt):
    y = np.full(len(gt), IGNORE, dtype=np.int64)
    for c, k in CODE.items():
        y[gt == c] = k
    return y


def swap_codes(y):
    out = y.copy()
    out[y == 0], out[y == 1] = 1, 0
    return out


def blocks_of(d):
    return [tuple(int(v) for v in b) for b in d["blocks"]]


def spans_of(d):
    return [(a, b) for _h, a, b, _l in blocks_of(d)]


def label_prior(sets):
    """[H, A, D] training frequencies with home/away swap augmentation (H == A)."""
    c = np.zeros(3)
    for d in sets:
        y = label_codes(d["gt"])
        c += np.bincount(y[y >= 0], minlength=3)
    c[0] = c[1] = (c[0] + c[1]) / 2
    return c / c.sum()


def train_net(series, n_in, steps, window, batch, seed, device, log_every=1000):
    """series: list of (Z, Zswap, y, spans). Random windows inside the halves; half the windows use
    the home/away-swapped features and labels (exact symmetry augmentation)."""
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = sequence.build(n_in, **sequence.ARCH).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps, pct_start=0.1)
    lossf = torch.nn.CrossEntropyLoss(ignore_index=IGNORE)

    spans = [(i, a, b) for i, s in enumerate(series) for a, b in s[3] if b - a + 1 > window]
    weights = np.array([b - a + 1 for _i, a, b in spans], dtype=float)
    weights /= weights.sum()
    net.train()
    t0 = time.time()
    for step in range(steps):
        xb = np.empty((batch, window, n_in), dtype=np.float32)
        yb = np.empty((batch, window), dtype=np.int64)
        for j, k in enumerate(rng.choice(len(spans), size=batch, p=weights)):
            i, a, b = spans[k]
            s = int(rng.integers(a, b - window + 2))
            Z, Zs, y, _ = series[i]
            if rng.random() < 0.5:
                xb[j], yb[j] = Z[s:s + window], y[s:s + window]
            else:
                xb[j], yb[j] = Zs[s:s + window], swap_codes(y[s:s + window])
        x = torch.from_numpy(xb).to(device)
        t = torch.from_numpy(yb).to(device)
        loss = lossf(net(x).reshape(-1, 3), t.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if log_every and (step + 1) % log_every == 0:
            print("    step {} loss {:.4f} ({:.0f}s)".format(step + 1, loss.item(), time.time() - t0),
                  flush=True)
    return net


def fit_gru(sets, args, device):
    norm = Normaliser.fit([d["X"] for d in sets])
    series = [(norm(d["X"]), norm(d["Xs"]), label_codes(d["gt"]), spans_of(d)) for d in sets]
    net = train_net(series, series[0][0].shape[1], args.steps, args.window, args.batch, args.seed,
                    device)
    return net, norm


def score_method(name, oof, priors, data, mids):
    """Nested HMM choice (other matches' OOF), then score the held-out match.
    Returns the grid scores too, for choosing the final (consensus) HMM params."""
    jobs = {m: (oof[m], priors[m], blocks_of(data[m]), data[m]["gt"]) for m in mids}
    grid, cfgs = train.grid_scores_parallel(jobs, 8)
    s, f = [], []
    for m in mids:
        best = int(np.argmax(np.mean([grid[o] for o in mids if o != m], axis=0)))
        lab = train.grid_labels(oof[m], priors[m], blocks_of(data[m]), len(data[m]["gt"]),
                                [cfgs[best]])[0][0]
        r = metrics.evaluate(data[m]["gt"], lab)
        s.append(r["spell_f1"])
        f.append(r["macro_f1"])
    s, f = np.array(s), np.array(f)
    print("  {:<6} spellF1 {:.3f} ({:.3f})  macroF1 {:.3f} ({:.3f})   per match: {}".format(
        name, s.mean(), s.std(), f.mean(), f.std(), " ".join("{:.3f}".format(v) for v in s)),
        flush=True)
    return grid, cfgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", nargs="*", default=train.MATCH_IDS)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--window", type=int, default=1000, help="frames per training window (40 s)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-lgbm", action="store_true", help="skip the LightGBM comparison")
    ap.add_argument("--save", default=None, help="train on all matches and save the GRU here")
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, torch.cuda.get_device_name(0) if device == "cuda" else "")
    print("metric:", "OFFICIAL" if metrics.OFFICIAL else "STAND-IN")

    cache_dir = os.path.join(idsse.DEFAULT_DIR, "cache")
    mids = args.matches
    data = {m: train.prepare(m, cache_dir) for m in mids}
    methods = ["gru"] if args.no_lgbm else ["lgbm", "gru", "avg"]

    oof = {k: {} for k in methods}
    priors = {}
    for m in mids:
        t = time.time()
        others = [data[o] for o in mids if o != m]
        priors[m] = label_prior(others)
        net, norm = fit_gru(others, args, device)
        oof["gru"][m] = sequence.predict_windows(net, norm(data[m]["X"]), norm(data[m]["Xs"]),
                                                 spans_of(data[m]), device)
        if not args.no_lgbm:
            booster, _ = train.fit(others)
            oof["lgbm"][m] = train.sym_proba(booster, data[m])
            oof["avg"][m] = (oof["lgbm"][m] + oof["gru"][m]) / 2
        print("fold {} done in {:.0f}s".format(m, time.time() - t), flush=True)

    print("\n=== through the HMM, mean over {} held-out matches (sd) ===".format(len(mids)))
    grids = {k: score_method(k, oof[k], priors, data, mids) for k in methods}
    print("note: differences below ~0.04 spellF1 are noise.")

    if args.save:
        grid, cfgs = grids["gru"]
        consensus = cfgs[int(np.argmax(np.mean([grid[m] for m in mids], axis=0)))]
        t = time.time()
        net, norm = fit_gru([data[m] for m in mids], args, device)
        meta = dict(n_in=int(len(norm.med) * 2), arch=sequence.ARCH, hmm=consensus,
                    prior=label_prior([data[m] for m in mids]).tolist(),
                    features=[str(x) for x in data[mids[0]]["names"]],
                    trained_on=list(mids), steps=args.steps, window=args.window,
                    batch=args.batch, seed=args.seed)
        os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
        sequence.save(args.save, net, norm, meta)
        print("saved {} in {:.0f}s with HMM params {}".format(args.save, time.time() - t, consensus))


if __name__ == "__main__":
    main()
