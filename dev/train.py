"""Train + evaluate the learned pipeline on IDSSE matches, leave-one-match-out.

    python -m dev.train                       # all 7 matches, LOMO, prints comparison table
    python -m dev.train --save models/lgbm.txt   # also fit on all matches and save the booster

For every held-out match the classifier is trained on the other matches only, and the HMM
parameters are chosen on the other matches' out-of-fold predictions (nested), so nothing about the
held-out match is used to fit or tune anything that scores it.

Compared on the degraded held-out match:
  nearest_smooth{w}   nearest-slot team + majority smoothing (approximation of starter/baseline.py)
  handHMM             solution.features + solution.model, as shipped (dead_target_share 0.2)
  handHMM_dead=train  same, dead_target_share set to the training matches' dead share
  lgbm_argmax         per-frame argmax of the classifier
  lgbm+HMM            classifier posteriors -> sticky HMM (nested-tuned)
"""

import argparse
import itertools
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dev import degrade, idsse, metrics  # noqa: E402
from solution import features as hand_features, learned, model  # noqa: E402

MATCH_IDS = ["J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9"]
CODE = {"H": 0, "A": 1, "D": 2}
PITCH = (105.0, 68.0)
LGB_PARAMS = dict(objective="multiclass", num_class=3, learning_rate=0.05, num_leaves=63,
                  min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=0, deterministic=True,
                  force_row_wise=True, num_threads=4)
N_ROUNDS = 400
SUB = 5          # train on every 5th frame (frames are strongly autocorrelated)


def seed_of(mid):
    return sum(map(ord, mid))


def prepare(mid, cache_dir):
    """Degraded features (normal + home/away swapped), labels, blocks. Cached per match."""
    path = os.path.join(cache_dir, "feat_{}_{}.npz".format(mid, seed_of(mid)))
    if os.path.isfile(path):
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    m = idsse.load(mid)
    P, B, T = degrade.degrade(m["players"], m["ball"], m["team"], m["period"], seed=seed_of(mid))
    X, names = learned.compute_features(P, B, T, m["period"], PITCH)
    Xs, _ = learned.compute_features(P, B, (1 - T).astype(np.int8), m["period"], PITCH)
    gt = m["labels"].copy()
    # one ungraded frame between the halves so spells cannot run across half-time
    cut = int(np.flatnonzero(np.diff(m["period"]) != 0)[0])
    gt[cut] = "X"
    blocks = ((1, 0, cut - 1, 0), (2, cut + 1, len(gt) - 1, cut + 1))
    # nearest-slot code for the baseline
    d = np.sqrt(((P[:, :, :2] - B[:, None, :]) ** 2).sum(axis=2))
    d = np.where(np.isnan(d), np.inf, d)
    near = np.where(np.isinf(d.min(axis=1)), -1, T[d.argmin(axis=1)]).astype(np.int8)
    graded = gt != "X"
    hcfg = model.Config(pitch=PITCH, blocks=blocks)
    hf, _ = hand_features.compute(P, B, T, graded, hcfg)
    res = dict(X=X, Xs=Xs, names=np.array(names), gt=gt, blocks=np.array(blocks), near=near,
               hand_llr=hf["llr"], hand_dead=hf["dead_logit"])
    # dead_logit without the share anchoring, so the anchor can be re-applied with another share
    hcfg0 = model.Config(pitch=PITCH, blocks=blocks, dead_target_share=0)
    hf0, _ = hand_features.compute(P, B, T, graded, hcfg0)
    res["hand_dead_raw"] = hf0["dead_logit"]
    np.savez(path, **res)
    return res


def swap_labels(y):
    return np.where(y == 0, 1, np.where(y == 1, 0, y))


def nearest_smooth(codes, w, graded):
    """The organisers' baseline: starter/baseline.py smooth() (+/- w frames, D where nobody seen).
    Codes use the same convention as the starter (0 home, 1 away, -1 nobody)."""
    from starter.baseline import smooth
    return np.where(graded, smooth(codes, w), "X")


# ------------------------------------------------------------------- batched Viterbi grid
GRID = dict(w=(0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0),
            p_turnover=(1e-4, 3e-4, 1e-3, 3e-3, 1e-2),
            p_enter_dead=(3e-4, 1e-3, 3e-3, 1e-2),
            p_leave_dead=(1e-3, 3e-3, 1e-2, 3e-2))


def grid_configs():
    keys = list(GRID)
    return [dict(zip(keys, v)) for v in itertools.product(*(GRID[k] for k in keys))]


def batched_viterbi(base, weights, log_trans):
    """base (T,3); weights (G,); log_trans (G,3,3) -> paths (G,T) int8."""
    G, T = len(weights), len(base)
    delta = weights[:, None] * base[0][None, :] - np.log(3.0)
    back = np.empty((T, G, 3), dtype=np.int8)
    wb = base.astype(np.float64)
    for t in range(1, T):
        scores = delta[:, :, None] + log_trans
        arg = scores.argmax(axis=1)
        back[t] = arg
        delta = np.take_along_axis(scores, arg[:, None, :], axis=1)[:, 0, :] + weights[:, None] * wb[t]
    paths = np.empty((G, T), dtype=np.int8)
    paths[:, -1] = delta.argmax(axis=1)
    gi = np.arange(G)
    for t in range(T - 1, 0, -1):
        paths[:, t - 1] = back[t, gi, paths[:, t]]
    return paths


def grid_labels(proba, prior, blocks, n):
    """All grid configs decoded at once: (G, n) array of letters."""
    cfgs = grid_configs()
    weights = np.array([c["w"] for c in cfgs])
    trans = np.stack([model.transition_log_matrix(model.Config(
        p_turnover=c["p_turnover"], p_enter_dead=c["p_enter_dead"],
        p_leave_dead=c["p_leave_dead"])) for c in cfgs])
    base = np.log(np.clip(proba, 1e-6, 1.0)) - np.log(prior)
    out = np.full((len(cfgs), n), "X", dtype="<U1")
    for _h, first, last, live in blocks:
        paths = batched_viterbi(base[live:last + 1], weights, trans)
        out[:, live:last + 1] = model.LETTERS[paths]
    return out, cfgs


# ------------------------------------------------------------------- main
def fit(train_sets, n_rounds=N_ROUNDS):
    import lightgbm as lgb
    Xs, ys = [], []
    for d in train_sets:
        lab = d["gt"]
        ok = np.flatnonzero(lab != "X")[::SUB]
        yy = np.array([CODE[c] for c in lab[ok]], dtype=np.int8)
        Xs += [d["X"][ok], d["Xs"][ok]]
        ys += [yy, swap_labels(yy)]
    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    prior = np.bincount(y, minlength=3) / len(y)
    booster = lgb.train(LGB_PARAMS, lgb.Dataset(X, y, free_raw_data=True), n_rounds)
    return booster, prior


def sym_proba(booster, d):
    """Average the prediction with the home/away-mirrored prediction (exact symmetry)."""
    p = booster.predict(d["X"], num_threads=4)
    ps = booster.predict(d["Xs"], num_threads=4)[:, [1, 0, 2]]
    return (p + ps) / 2


def hand_pred(d, share):
    n = len(d["gt"])
    blocks = tuple(tuple(int(v) for v in b) for b in d["blocks"])
    cfg = model.Config(pitch=PITCH, blocks=blocks)
    dead = d["hand_dead"]
    if share is not None:
        graded = d["gt"] != "X"
        raw = d["hand_dead_raw"]
        dead = raw - np.quantile(raw[graded], 1.0 - share)
    return model.decode(d["hand_llr"], dead, n, cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", nargs="*", default=MATCH_IDS)
    ap.add_argument("--save", default=None, help="fit on all matches and save booster here")
    args = ap.parse_args()
    cache_dir = os.path.join(idsse.DEFAULT_DIR, "cache")
    print("metric:", "OFFICIAL possession_metrics" if metrics.OFFICIAL else
          "STAND-IN (dev/metrics.py) -- not the official scorer")

    data = {}
    for mid in args.matches:
        t = time.time()
        data[mid] = prepare(mid, cache_dir)
        print("prepared {} in {:.0f}s".format(mid, time.time() - t), flush=True)

    # ---- out-of-fold classifier probabilities -------------------------------------------
    oof, priors = {}, {}
    for mid in args.matches:
        t = time.time()
        booster, prior = fit([data[o] for o in args.matches if o != mid])
        oof[mid], priors[mid] = sym_proba(booster, data[mid]), prior
        print("fold {} trained+predicted in {:.0f}s".format(mid, time.time() - t), flush=True)

    # ---- every grid config decoded on every match (OOF probs) ---------------------------
    grid_scores = {}
    for mid in args.matches:
        t = time.time()
        d = data[mid]
        blocks = [tuple(int(v) for v in b) for b in d["blocks"]]
        labs, cfgs = grid_labels(oof[mid], priors[mid], blocks, len(d["gt"]))
        grid_scores[mid] = np.array([metrics.evaluate(d["gt"], l)["spell_f1"] for l in labs])
        grid_labels_cache = labs
        data[mid]["_grid"] = grid_labels_cache
        print("grid decoded on {} in {:.0f}s".format(mid, time.time() - t), flush=True)

    # ---- per held-out match: nested choice of HMM params, all methods scored -------------
    rows = {}
    chosen = {}
    for mid in args.matches:
        d = data[mid]
        gt = d["gt"]
        graded = gt != "X"
        others = [o for o in args.matches if o != mid]
        best = int(np.argmax(np.mean([grid_scores[o] for o in others], axis=0)))
        chosen[mid] = cfgs[best]
        train_dead = float(np.mean([(data[o]["gt"][data[o]["gt"] != "X"] == "D").mean()
                                    for o in others]))
        preds = {}
        for w in (0, 25, 50, 125):
            preds["nearest_smooth%d" % w] = nearest_smooth(d["near"], w, graded)
        preds["handHMM"] = hand_pred(d, None)
        preds["handHMM_dead=train"] = hand_pred(d, train_dead)
        preds["lgbm_argmax"] = np.where(graded, model.LETTERS[oof[mid].argmax(axis=1)], "X")
        preds["lgbm+HMM"] = d["_grid"][best]
        rows[mid] = {k: metrics.evaluate(gt, v) for k, v in preds.items()}
        print("\n{}  (gt spells {}, mean {:.1f}s; HMM params {})".format(
            mid, rows[mid]["handHMM"]["n_gt"], rows[mid]["handHMM"]["gt_mean_s"], chosen[mid]))
        for k, r in rows[mid].items():
            print("  {:<22} spellF1 {:.3f}  macroF1 {:.3f}  spells {:>5}  mean {:5.1f}s".format(
                k, r["spell_f1"], r["macro_f1"], r["n_pred"], r["pred_mean_s"]))

    print("\n=== mean over {} held-out matches (sd across matches) ===".format(len(args.matches)))
    for k in rows[args.matches[0]]:
        s = np.array([rows[m][k]["spell_f1"] for m in args.matches])
        f = np.array([rows[m][k]["macro_f1"] for m in args.matches])
        print("  {:<22} spellF1 {:.3f} ({:.3f})  macroF1 {:.3f} ({:.3f})".format(
            k, s.mean(), s.std(), f.mean(), f.std()))
    print("note: differences below ~0.04 spellF1 are noise.")

    if args.save:
        booster, prior = fit([data[m] for m in args.matches])
        consensus = cfgs[int(np.argmax(np.mean([grid_scores[m] for m in args.matches], axis=0)))]
        os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
        booster.save_model(args.save)
        meta = dict(prior=prior.tolist(), hmm=consensus,
                    features=[str(x) for x in data[args.matches[0]]["names"]])
        import json
        with open(args.save + ".json", "w") as fh:
            json.dump(meta, fh, indent=1)
        print("saved", args.save, "with HMM params", consensus)


if __name__ == "__main__":
    main()
