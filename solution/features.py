"""Per-frame evidence computed from tracking. Everything here is unsupervised: no labels.

Two independent pieces of evidence feed the model (model.py):

  team_llr    log-odds that the HOME team has the ball (positive = home), from how much closer the
              nearest home player is to the ball than the nearest away player.
  dead_logit  log-odds that the ball is dead, from stoppage cues: players slow down, the ball sits
              still, the ball leaves the pitch, a Smart Tagging set piece covers the frame.

Thresholds for "players are slow" are not hard-coded in units: they are found from the match's own
speed distribution with Otsu's method, so they adapt to whatever units the speed column is in.
"""

import warnings

import numpy as np


def moving_mean(x, half_window):
    """NaN-aware centred moving average via cumulative sums (cost independent of the window)."""
    valid = ~np.isnan(x)
    v = np.where(valid, x, 0.0).astype(np.float64)
    c = np.concatenate(([0.0], np.cumsum(v)))
    n = np.concatenate(([0], np.cumsum(valid)))
    idx = np.arange(len(x))
    lo = np.maximum(0, idx - half_window)
    hi = np.minimum(len(x), idx + half_window + 1)
    count = n[hi] - n[lo]
    total = c[hi] - c[lo]
    out = np.full(len(x), np.nan)
    ok = count > 0
    out[ok] = total[ok] / count[ok]
    return out


def fill_ball(ball, max_gap):
    """Linearly interpolate the ball across gaps of at most max_gap frames. Longer gaps stay NaN."""
    n = len(ball)
    seen = ~np.isnan(ball[:, 0])
    if seen.sum() < 2:
        return ball.copy()
    idx = np.arange(n)
    prev = np.where(seen, idx, -1)
    prev = np.maximum.accumulate(prev)
    nxt = np.where(seen, idx, n + 10 ** 9)
    nxt = np.minimum.accumulate(nxt[::-1])[::-1]
    gap = nxt - prev - 1
    out = np.empty_like(ball)
    for k in range(2):
        out[:, k] = np.interp(idx, idx[seen], ball[seen, k])
    bad = (~seen) & ((prev < 0) | (nxt > n) | (gap > max_gap))
    out[bad] = np.nan
    return out


def team_min_dist(players, ball, team_of_track):
    """(d_home, d_away): distance in metres from the ball to the nearest player of each team."""
    if players.shape[1] != len(team_of_track):
        raise ValueError("tracking has {} slots but the metadata lists {}".format(
            players.shape[1], len(team_of_track)))
    diff = players[:, :, :2] - ball[:, None, :]
    d = np.sqrt((diff ** 2).sum(axis=2))
    d = np.where(np.isnan(d), np.inf, d)
    home = d[:, team_of_track == 0].min(axis=1)
    away = d[:, team_of_track == 1].min(axis=1)
    home[np.isinf(home)] = np.nan
    away[np.isinf(away)] = np.nan
    return home, away


def otsu(values, bins=128):
    """(threshold, low_mean, high_mean) of a 1-D sample by maximising between-class variance."""
    values = values[np.isfinite(values)]
    hist, edges = np.histogram(values, bins=bins)
    mid = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist).astype(np.float64)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * mid) / np.maximum(w0, 1)
    m1 = (np.sum(hist * mid) - np.cumsum(hist * mid)) / np.maximum(w1, 1)
    between = w0 * w1 * (m0 - m1) ** 2
    k = int(np.argmax(between))
    return float(edges[k + 1]), float(m0[k]), float(m1[k])


def compute(players, ball, team_of_track, graded_mask, cfg, set_piece_mask=None):
    """Returns dict of per-frame arrays plus a dict of the calibration numbers actually used."""
    n = len(ball)
    ball_f = fill_ball(ball, cfg.max_ball_gap)
    observed = ~np.isnan(ball_f[:, 0])

    # --- who is closer to the ball -------------------------------------------------------
    d_home, d_away = team_min_dist(players, ball_f, team_of_track)
    margin = d_away - d_home
    nearest = np.fmin(d_home, d_away)
    confidence = np.clip(1.0 - (nearest - cfg.near_m) / cfg.fade_m, cfg.min_confidence, 1.0)
    llr = np.clip(confidence * margin / cfg.margin_tau_m, -cfg.llr_clip, cfg.llr_clip)
    llr = np.where(np.isnan(llr), 0.0, llr)

    # --- stoppage cues ---------------------------------------------------------------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median_speed = np.nanmedian(players[:, :, 2], axis=1)
    speed = moving_mean(median_speed, cfg.speed_half_window)
    thr, lo_mean, hi_mean = otsu(speed[graded_mask])
    spread = max((hi_mean - lo_mean) / 4.0, 1e-6)
    slow_ev = np.clip((thr - np.nan_to_num(speed, nan=thr)) / spread, -cfg.slow_clip, cfg.slow_clip)

    k = cfg.still_half_window
    disp = np.full(n, np.nan)
    disp[k:-k] = np.linalg.norm(ball_f[2 * k:] - ball_f[:-2 * k], axis=1)
    still = np.where(np.isnan(disp), 0.0,
                     np.where(disp < cfg.still_m, cfg.still_weight, -cfg.moving_weight))

    pad = cfg.out_margin_m
    L, W = cfg.pitch
    outside = observed & ((ball_f[:, 0] < -pad) | (ball_f[:, 0] > L + pad)
                          | (ball_f[:, 1] < -pad) | (ball_f[:, 1] > W + pad))
    out_ev = np.where(outside, cfg.out_weight, 0.0)

    tag_ev = np.zeros(n)
    if set_piece_mask is not None:
        tag_ev = np.where(set_piece_mask, cfg.set_piece_weight, 0.0)

    dead_logit = cfg.slow_weight * slow_ev + still + out_ev + tag_ev - cfg.dead_bias
    if cfg.dead_target_share:
        # The raw evidence has no absolute scale (it was a cliff on the external game: a small
        # bias change flipped the model from "half the match dead" to "never dead"). Anchor it
        # instead: shift so that dead_logit > 0 on about `dead_target_share` of graded frames.
        # That is a football prior, not a label, and the HMM still decides where the stoppages are.
        dead_logit = dead_logit - np.quantile(dead_logit[graded_mask], 1.0 - cfg.dead_target_share)

    calib = {"speed_threshold": thr, "speed_low_mean": lo_mean, "speed_high_mean": hi_mean,
             "ball_observed_frac_graded": float(observed[graded_mask].mean())}
    return {"llr": llr, "dead_logit": dead_logit, "ball_observed": observed}, calib
