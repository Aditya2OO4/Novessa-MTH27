"""Tiny fake match in the real ATD file format, to smoke-test the pipeline without the real data.

This only proves the code runs end to end and behaves sensibly on a toy world. It says nothing
about performance on the real match.
"""

import numpy as np

L, W = 105.3, 68.0


def make_match(n_frames=24000, blocks=((1, 1000, 11000), (2, 13000, 23000)), seed=0):
    rng = np.random.default_rng(seed)
    truth = np.full(n_frames, "X", dtype="<U1")
    for _h, a, b in blocks:
        t = a
        side = rng.choice(["H", "A"])
        while t <= b:
            dur = int(rng.integers(200, 700))
            truth[t:min(b + 1, t + dur)] = side
            t += dur
            if rng.random() < 0.6 and t <= b:
                dd = int(rng.integers(100, 250))
                truth[t:min(b + 1, t + dd)] = "D"
                t += dd
            side = "A" if side == "H" else "H"

    base = np.concatenate([rng.uniform([5, 5], [L / 2 - 5, W - 5], (11, 2)),
                           rng.uniform([L / 2 + 5, 5], [L - 5, W - 5], (11, 2))])
    players = np.full((n_frames, 46, 3), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    pos = base.copy()
    carrier, bpos = 0, base[0].copy()
    still_at = bpos.copy()
    for t in range(n_frames):
        lab = truth[t]
        live = lab in ("H", "A")
        step = rng.normal(0, 0.12 if live else 0.02, pos.shape)
        pos = np.clip(pos + step, [0, 0], [L, W])
        speed = np.abs(rng.normal(2.6 if live else 0.5, 0.4, 22))
        if live:
            if t % 60 == 0 or truth[t - 1] != lab:
                idx = np.arange(11) + (0 if lab == "H" else 11)
                carrier = int(rng.choice(idx))
            bpos = bpos + 0.2 * (pos[carrier] + rng.normal(0, 0.8, 2) - bpos)
            still_at = bpos.copy()
        else:
            bpos = still_at
        players[t, :22, :2] = pos
        players[t, :22, 2] = speed
        ball[t] = bpos
    # ball dropouts (~40%), in bursts
    t = 0
    while t < n_frames:
        t += int(rng.integers(30, 120))
        gap = int(rng.integers(15, 100))
        ball[t:t + gap] = np.nan
    return players, ball, truth


def write_files(players, ball, raw_path, meta_path):
    n = len(ball)
    with open(raw_path, "w") as f:
        for t in range(n):
            groups = []
            for s in range(players.shape[1]):
                p = players[t, s]
                if np.isnan(p[0]):
                    groups.append("NaN,NaN,NaN")
                else:
                    groups.append("{:.6f},{:.6f},{:.3f}".format(p[0] / L, p[1] / W, p[2]))
            b = ball[t]
            bs = "NaN,NaN,NaN" if np.isnan(b[0]) else "{:.6f},{:.6f},0.000".format(b[0] / L, b[1] / W)
            f.write("{}:{}:{}\n".format(t + 1, ";".join(groups), bs))
    teams = '<Team id="HOME"><Name>Home</Name></Team><Team id="AWAY"><Name>Away</Name></Team>'
    pl = "".join('<Player id="P{0}" teamId="{1}"><Name>Track {0}</Name></Player>'.format(
        i, "HOME" if i < 11 else ("AWAY" if i < 22 else ("HOME" if i % 2 else "AWAY")))
        for i in range(46))
    with open(meta_path, "w") as f:
        f.write("<Metadata><Teams>{}</Teams><Players>{}</Players></Metadata>".format(teams, pl))
