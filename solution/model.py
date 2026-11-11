"""Config, 3-state HMM (H, A, D) and Viterbi decoding.

Why an HMM: a possession is an interval, not a sequence of independent frames. Per-frame decisions
flicker, and the ranking metric (spellF1) punishes every extra fake possession. A sticky state
model turns noisy frame evidence into a few long runs, and bridges stretches where the ball is not
tracked simply by staying in the previous state -- which is exactly what the scorer's `X`
forward-fill would have done anyway.

Every number below is a hand-set prior, NOT tuned on this match (tuning on the 5 public minutes is
forbidden and misleading; see EVALUATION.md). They are meant to be sensible defaults you can
change in one place.
"""

from dataclasses import dataclass, field

import numpy as np

H, A, D = 0, 1, 2
LETTERS = np.array(["H", "A", "D"])


@dataclass
class Config:
    pitch: tuple = (105.3, 68.0)

    # ball
    max_ball_gap: int = 25            # interpolate gaps up to 1 s
    # who-has-the-ball evidence
    near_m: float = 3.0               # full confidence if the nearest player is this close
    fade_m: float = 15.0              # ...fading to min_confidence by near_m + fade_m
    min_confidence: float = 0.25
    margin_tau_m: float = 1.5         # metres of "closer" per unit of log-odds
    llr_clip: float = 4.0
    # stoppage evidence
    speed_half_window: int = 75       # 3 s window on median player speed
    slow_weight: float = 1.0
    slow_clip: float = 2.0
    still_half_window: int = 12       # compare ball position 12 frames before/after
    still_m: float = 1.5
    still_weight: float = 1.5
    moving_weight: float = 0.5
    out_margin_m: float = 0.5
    out_weight: float = 3.0
    set_piece_weight: float = 1.0
    dead_bias: float = 1.0
    dead_target_share: float = 0.2    # prior: ~1 in 5 graded frames is dead ball (0 = off)
    # HMM
    emission_weight: float = 0.25     # frames are strongly autocorrelated: down-weight evidence
    p_turnover: float = 1.5e-3        # H<->A per frame (~ 1 per 27 s)
    p_enter_dead: float = 1.5e-3      # live -> dead per frame
    p_leave_dead: float = 6e-3        # dead -> live per frame (~ 7 s stoppage)

    # Published graded blocks (DATA.md / FAQ.md): half, first graded frame, last graded frame,
    # first frame with live play. The second half is graded from 110808 but kicks off at 113742;
    # the 2,934 frames in between are all dead ball (FAQ.md).
    blocks: tuple = ((1, 11185, 90651, 11185), (2, 110808, 190197, 113742))


def transition_log_matrix(cfg):
    stay_live = 1.0 - cfg.p_turnover - cfg.p_enter_dead
    stay_dead = 1.0 - cfg.p_leave_dead
    T = np.array([
        [stay_live, cfg.p_turnover, cfg.p_enter_dead],
        [cfg.p_turnover, stay_live, cfg.p_enter_dead],
        [cfg.p_leave_dead / 2, cfg.p_leave_dead / 2, stay_dead],
    ])
    return np.log(T)


def emission_loglik(llr, dead_logit, cfg):
    """(N, 3) log-likelihoods for [H, A, D]."""
    log_dead = -np.logaddexp(0.0, -dead_logit)
    log_live = -np.logaddexp(0.0, dead_logit)
    log_h = -np.logaddexp(0.0, -llr)
    log_a = -np.logaddexp(0.0, llr)
    out = np.stack([log_live + log_h, log_live + log_a, log_dead], axis=1)
    return cfg.emission_weight * out


def viterbi(emis, log_trans):
    T = len(emis)
    delta = emis[0] - np.log(3.0)
    back = np.empty((T, 3), dtype=np.int8)
    for t in range(1, T):
        scores = delta[:, None] + log_trans
        back[t] = scores.argmax(axis=0)
        delta = scores.max(axis=0) + emis[t]
    path = np.empty(T, dtype=np.int8)
    path[-1] = int(delta.argmax())
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode(llr, dead_logit, n_frames, cfg):
    """Full-length label array of 'H'/'A'/'D'/'X'. X outside the graded halves."""
    labels = np.full(n_frames, "X", dtype="<U1")
    emis = emission_loglik(llr, dead_logit, cfg)
    trans = transition_log_matrix(cfg)
    for _half, first, last, live_from in cfg.blocks:
        last = min(last, n_frames - 1)
        if live_from > first:
            labels[first:live_from] = "D"
        if live_from > last:
            continue
        path = viterbi(emis[live_from:last + 1], trans)
        labels[live_from:last + 1] = LETTERS[path]
    return labels


def possession_stats(labels, fps=25):
    """(count, mean seconds): maximal runs of H or A, with D and X as breaks."""
    runs, prev, length = [], None, 0
    for lab in labels:
        if lab in ("H", "A") and lab == prev:
            length += 1
        else:
            if prev in ("H", "A"):
                runs.append(length)
            prev, length = (lab, 1) if lab in ("H", "A") else (None, 0)
    if prev in ("H", "A"):
        runs.append(length)
    return len(runs), (float(np.mean(runs)) / fps if runs else 0.0)
