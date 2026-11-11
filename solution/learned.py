"""Learned per-frame H/A/D classifier (LightGBM) + the sticky HMM on top.

The classifier is trained on OTHER matches only (IDSSE / Sportec Bundesliga, degraded to look like
the challenge data; see dev/train.py). Nothing here is fitted on the challenge match.

Features are permutation-invariant: they use each slot's team (from the metadata), never the slot
index, so slot churn, swaps and duplicates cannot leak identity. Speeds are divided by the match's
own median moving speed, so the unknown speed unit of the challenge file does not matter.
Pitch direction is inferred per half from where each team's players stand on average.
"""

import warnings

import numpy as np

from solution import model
from solution.features import fill_ball, moving_mean

WINDOWS = (12, 50, 125)
FAR = 60.0


def _team_stats(players, sel, attack_dir, pitch):
    """Robust per-team shape stats (medians resist duplicated / swapped slots)."""
    x = players[:, sel, 0]
    y = players[:, sel, 1]
    s = players[:, sel, 2]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        cx = np.nanmedian(x, axis=1)
        cy = np.nanmedian(y, axis=1)
        sx = np.nanstd(x, axis=1)
        sy = np.nanstd(y, axis=1)
        sp = np.nanmedian(s, axis=1)
        front = np.nanmax((x - pitch[0] / 2) * attack_dir[:, None], axis=1)
        back = np.nanmin((x - pitch[0] / 2) * attack_dir[:, None], axis=1)
    count = (~np.isnan(x)).sum(axis=1).astype(np.float32)
    adv = (cx - pitch[0] / 2) * attack_dir
    return cx, cy, sx, sy, sp, adv, front, back, count


def _ffill_age(values, valid):
    """Last valid value and frames since it (age capped)."""
    n = len(values)
    idx = np.where(valid, np.arange(n), -1)
    np.maximum.accumulate(idx, out=idx)
    out = np.where(idx >= 0, values[np.maximum(idx, 0)], 0.0)
    age = np.where(idx >= 0, np.arange(n) - idx, 10 ** 4)
    return out, np.minimum(age, 10 ** 4).astype(np.float32)


def compute_features(players, ball, slot_team, period, pitch=(105.0, 68.0), max_ball_gap=25):
    """(N, F) float32 feature matrix and the feature names.

    period: (N,) int, 1/2 inside the halves, 0 elsewhere (direction is fitted per half).
    """
    n = len(ball)
    L, W = pitch
    home = slot_team == 0
    away = slot_team == 1
    raw_seen = ~np.isnan(ball[:, 0])
    ball_f = fill_ball(ball, max_ball_gap)
    seen = ~np.isnan(ball_f[:, 0])

    # speed scale: median of moving players over the match (unit-free)
    spd = players[:, :, 2]
    moving = spd[np.isfinite(spd) & (spd > 0)]
    scale = float(np.median(moving)) if moving.size else 1.0
    players = players.copy()
    players[:, :, 2] /= max(scale, 1e-6)

    # attack direction of the home team per half: home players stand in their own half on average
    attack_dir = np.ones(n, dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for h in np.unique(period[period > 0]):
            m = period == h
            hx = np.nanmean(players[m][:, home, 0])
            ax = np.nanmean(players[m][:, away, 0])
            attack_dir[m] = 1.0 if hx < ax else -1.0

    f = {}
    # ---- ball-relative -------------------------------------------------------------------
    d = np.sqrt(((players[:, :, :2] - ball_f[:, None, :]) ** 2).sum(axis=2))
    d = np.where(np.isnan(d), np.inf, d)
    dh = np.sort(d[:, home], axis=1)[:, :2]
    da = np.sort(d[:, away], axis=1)[:, :2]
    dh = np.minimum(dh, FAR)
    da = np.minimum(da, FAR)
    f["dh1"], f["dh2"], f["da1"], f["da2"] = dh[:, 0], dh[:, 1], da[:, 0], da[:, 1]
    f["margin"] = da[:, 0] - dh[:, 0]
    f["margin2"] = (da[:, 0] + da[:, 1]) / 2 - (dh[:, 0] + dh[:, 1]) / 2
    f["nh5"] = (d[:, home] < 5).sum(axis=1).astype(np.float32)
    f["na5"] = (d[:, away] < 5).sum(axis=1).astype(np.float32)
    f["nh12"] = (d[:, home] < 12).sum(axis=1).astype(np.float32)
    f["na12"] = (d[:, away] < 12).sum(axis=1).astype(np.float32)
    for k in ("dh1", "dh2", "da1", "da2", "margin", "margin2", "nh5", "na5", "nh12", "na12"):
        f[k] = np.where(seen, f[k], np.nan)

    k = 2
    bv = np.full(n, np.nan, dtype=np.float32)
    bv[k:-k] = np.linalg.norm(ball_f[2 * k:] - ball_f[:-2 * k], axis=1) / (2 * k)
    f["ball_speed"] = bv
    k = 12
    disp = np.full(n, np.nan, dtype=np.float32)
    disp[k:-k] = np.linalg.norm(ball_f[2 * k:] - ball_f[:-2 * k], axis=1)
    f["ball_disp24"] = disp
    f["ball_adv"] = (ball_f[:, 0] - L / 2) * attack_dir   # + = towards away goal
    f["ball_wide"] = np.abs(ball_f[:, 1] - W / 2)
    f["ball_out"] = np.where(seen, ((ball_f[:, 0] < 0) | (ball_f[:, 0] > L)
                                    | (ball_f[:, 1] < 0) | (ball_f[:, 1] > W)).astype(np.float32),
                             np.nan)
    f["ball_seen"] = raw_seen.astype(np.float32)

    # last / next observed margin and how stale it is (inference across ball gaps)
    mg = np.where(seen, f["margin"], 0.0)
    last, age = _ffill_age(mg, seen)
    nxt, age_n = _ffill_age(mg[::-1], seen[::-1])
    f["margin_last"], f["margin_last_age"] = last, age
    f["margin_next"], f["margin_next_age"] = nxt[::-1], age_n[::-1]

    # ---- team shape (ball-free) ------------------------------------------------------------
    H = _team_stats(players, home, attack_dir, pitch)
    A = _team_stats(players, away, -attack_dir, pitch)
    names = ("cx", "cy", "sx", "sy", "spd", "adv", "front", "back", "count")
    for nm, hv, av in zip(names, H, A):
        if nm in ("cx", "cy"):
            continue
        f["h_" + nm], f["a_" + nm] = hv, av
    f["adv_diff"] = H[5] - A[5]
    f["sx_diff"] = H[2] - A[2]
    f["sy_diff"] = H[3] - A[3]
    f["spd_all"] = np.fmin(H[4], A[4])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        f["slow_frac"] = np.nanmean(np.where(np.isnan(players[:, :, 2]), np.nan,
                                             players[:, :, 2] < 0.3), axis=1)
    # team centroids moving forward (+) or back
    k = 12
    for nm, adv in (("h_vel", H[5]), ("a_vel", A[5])):
        v = np.full(n, np.nan, dtype=np.float32)
        v[k:-k] = (adv[2 * k:] - adv[:-2 * k]) / (2 * k)
        f[nm] = v
    f["vel_diff"] = f["h_vel"] - f["a_vel"]
    # ball relative to team blocks
    f["ball_minus_hcx"] = (ball_f[:, 0] - H[0]) * attack_dir
    f["ball_minus_acx"] = (ball_f[:, 0] - A[0]) * attack_dir

    # ---- windowed context ------------------------------------------------------------------
    margin_c = np.clip(f["margin"], -20, 20)
    soft = np.tanh(f["margin"] / 3.0)
    for w in WINDOWS:
        f["margin_m%d" % w] = moving_mean(margin_c, w)
        f["soft_m%d" % w] = moving_mean(soft, w)
        f["near_m%d" % w] = moving_mean(f["nh5"] - f["na5"], w)
        f["bspeed_m%d" % w] = moving_mean(f["ball_speed"], w)
        f["spd_m%d" % w] = moving_mean(f["spd_all"], w)
        f["seen_m%d" % w] = moving_mean(f["ball_seen"], w)
        f["adv_m%d" % w] = moving_mean(f["adv_diff"], w)
        f["vel_m%d" % w] = moving_mean(f["vel_diff"], w)
        f["still_m%d" % w] = moving_mean((f["ball_disp24"] < 1.5).astype(np.float32)
                                         + np.where(np.isnan(f["ball_disp24"]), np.nan, 0), w)
        f["slow_m%d" % w] = moving_mean(f["slow_frac"], w)
    # direction of change in context (before vs after)
    for w in (25, 75):
        before = moving_mean(np.concatenate([np.full(w, np.nan), soft[:-w]]), w)
        after = moving_mean(np.concatenate([soft[w:], np.full(w, np.nan)]), w)
        f["soft_before%d" % w] = before
        f["soft_after%d" % w] = after

    names = list(f.keys())
    X = np.stack([np.asarray(f[k], dtype=np.float32) for k in names], axis=1)
    return X, names


# ---------------------------------------------------------------- inference + HMM
def predict_proba(booster, players, ball, slot_team, period, pitch):
    """(N, 3) probabilities [H, A, D], averaged with the home/away-mirrored prediction so the
    model has no home bias (it was trained with swap augmentation)."""
    X, _ = compute_features(players, ball, slot_team, period, pitch)
    p = booster.predict(X, num_threads=1)
    Xs, _ = compute_features(players, ball, (1 - slot_team).astype(np.int8), period, pitch)
    return (p + booster.predict(Xs, num_threads=1)[:, [1, 0, 2]]) / 2


def hmm_decode(proba, prior, n_frames, cfg, blocks=None):
    """Viterbi over the classifier's posteriors turned into scaled likelihoods."""
    emis = cfg.emission_weight * (np.log(np.clip(proba, 1e-6, 1.0)) - np.log(prior))
    trans = model.transition_log_matrix(cfg)
    labels = np.full(n_frames, "X", dtype="<U1")
    for _half, first, last, live_from in (blocks or cfg.blocks):
        last = min(last, n_frames - 1)
        if live_from > first:
            labels[first:live_from] = "D"
        if live_from > last:
            continue
        path = model.viterbi(emis[live_from:last + 1], trans)
        labels[live_from:last + 1] = model.LETTERS[path]
    return labels
