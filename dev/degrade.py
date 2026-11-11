"""Make a clean public match look like the challenge tracking.

Challenge data (README, CHALLENGE.md): ball missing ~41% in bursts, noisy positions, 46 anonymous
slots whose team is fixed by the metadata but whose occupant changes (churn), identity swaps, and
duplicated players. Downstream features must therefore be permutation-invariant: they may use a
slot's team, never its index or occupant.

degrade() returns the same array layout the real pipeline gets from solution.atd:
    players (N, n_slots, 3)  x_m, y_m, speed  (NaN = empty slot)
    ball    (N, 2)
    slot_team (n_slots,)     0 home / 1 away, as the metadata would list it
"""

import numpy as np


def _bursts(rng, n, target, min_len, max_len, mask=None):
    """Boolean mask covering ~target of n frames with bursts of length in [min_len, max_len]."""
    mask = np.zeros(n, dtype=bool) if mask is None else mask
    for _ in range(100000):
        if mask.mean() >= target:
            break
        s = int(rng.integers(0, n))
        # log-uniform burst lengths: many short drops, some long ones
        length = int(np.exp(rng.uniform(np.log(min_len), np.log(max_len))))
        mask[s:s + length] = True
    return mask


def degrade(players, ball, team, period=None, seed=0, n_slots=46, ball_missing=0.41,
            pos_sigma=0.3, frag_mean_s=60.0, p_cross_team=0.03, dup_rate_per_min=1.0,
            player_drop=0.03, fps=25):
    rng = np.random.default_rng(seed)
    n, n_persons = players.shape[:2]

    # ---- ball: bursty dropouts, plus position noise ------------------------------------------
    ball = ball.copy()
    ball += rng.normal(0, pos_sigma, ball.shape).astype(np.float32)
    ball[_bursts(rng, n, ball_missing, 5, 400)] = np.nan

    # ---- players: noise, short dropouts ------------------------------------------------------
    src = players.copy()
    src[:, :, :2] += rng.normal(0, pos_sigma, src[:, :, :2].shape).astype(np.float32)
    src[:, :, 2] *= np.exp(rng.normal(0, 0.1, src[:, :, 2].shape)).astype(np.float32)
    for p in range(n_persons):
        drop = _bursts(rng, n, player_drop, 3, 100)
        src[drop, p] = np.nan

    # ---- anonymous slots ---------------------------------------------------------------------
    half = n_slots // 2
    slot_team = np.array([0] * half + [1] * (n_slots - half), dtype=np.int8)
    rng.shuffle(slot_team)
    out = np.full((n, n_slots, 3), np.nan, dtype=np.float32)
    occupied = np.zeros((n, n_slots), dtype=bool)

    def place(a, b, track, team_id):
        """Put track[a:b] into a free slot of team_id (or any free slot); return slot or -1."""
        cand = np.flatnonzero(slot_team == team_id)
        free = cand[~occupied[a:b, cand].any(axis=0)]
        if len(free) == 0:
            free = np.flatnonzero(~occupied[a:b].any(axis=0))
        if len(free) == 0:
            return -1
        s = int(rng.choice(free))
        out[a:b, s] = track
        occupied[a:b, s] = True
        return s

    breaks = []
    if period is not None:
        breaks = list(np.flatnonzero(np.diff(period) != 0) + 1)
    frag_len = frag_mean_s * fps
    for p in range(n_persons):
        seen = np.flatnonzero(~np.isnan(src[:, p, 0]))
        if len(seen) == 0:
            continue
        t, end = int(seen[0]), int(seen[-1]) + 1
        while t < end:
            # fragment ends: random length, forced at half-time (tracker restarts)
            b = min(end, t + max(25, int(rng.exponential(frag_len))))
            for br in breaks:
                if t < br < b:
                    b = br
            team_id = int(team[p])
            if rng.random() < p_cross_team:      # identity swap onto the other team's slot
                team_id = 1 - team_id
            place(t, b, src[t:b, p], team_id)
            t = b

    # ---- duplicated players: a ghost copy of a real track for a few seconds -----------------
    n_dup = int(dup_rate_per_min * n / (60 * fps))
    for _ in range(n_dup):
        p = int(rng.integers(0, n_persons))
        a = int(rng.integers(0, n))
        b = min(n, a + int(rng.integers(25, 250)))
        track = src[a:b, p].copy()
        if np.isnan(track[:, 0]).all():
            continue
        track[:, :2] += rng.normal(0, 0.7, 2).astype(np.float32)
        place(a, b, track, int(team[p]))

    return out, ball, slot_team
