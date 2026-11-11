"""Parse the raw tracking file (ATD) into numpy arrays, indexed by *video* frame.

Raw line format (see starter/baseline.py, which this mirrors):

    <frame_id>:<x,y,speed;x,y,speed;...>:<ball_x,ball_y,...>

frame_id counts from 1, submissions from 0, so video_frame = frame_id - 1. Missing observations
are "NaN". Coordinates are normalised 0-1 and are converted to metres here.

The parsed arrays are cached next to the repo (.cache/, gitignored). The cache is *derived match
data*: never commit it or upload it anywhere (RULES.md).
"""

import os

import numpy as np

SLOT_FIELDS = 3  # x, y, speed


def _flat_floats(text):
    return np.array(text.replace(";", ",").split(","), dtype=np.float32)


def parse_atd(raw_path, n_frames, pitch):
    """Returns players (N, S, 3) = x_m, y_m, speed;  ball (N, 2) in metres.  NaN where unseen."""
    length, width = pitch
    scale = np.array([length, width], dtype=np.float32)

    players = ball = None
    n_slots = 0
    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1
            if not 0 <= frame < n_frames:
                continue

            if players is None:
                n_slots = parts[1].count(";") + 1
                players = np.full((n_frames, n_slots, SLOT_FIELDS), np.nan, dtype=np.float32)
                ball = np.full((n_frames, 2), np.nan, dtype=np.float32)

            flat = _flat_floats(parts[1])
            if flat.size == n_slots * SLOT_FIELDS:
                players[frame] = flat.reshape(n_slots, SLOT_FIELDS)
            else:  # unexpected group width: take the first three fields of each group
                for slot, group in enumerate(parts[1].split(";")[:n_slots]):
                    values = [float(v) for v in group.split(",")[:SLOT_FIELDS]]
                    players[frame, slot, :len(values)] = values

            values = [float(v) for v in parts[2].strip().split(",")[:2]]
            if len(values) == 2:
                ball[frame] = values

    if players is None:
        raise ValueError("no tracking lines parsed from {}".format(raw_path))
    players[:, :, :2] *= scale
    ball *= scale
    return players, ball


def load_cached(raw_path, n_frames, pitch, cache_dir):
    """parse_atd with an .npz cache keyed on the raw file's size and mtime."""
    stat = os.stat(raw_path)
    key = "{}_{}_{}".format(os.path.basename(raw_path), stat.st_size, int(stat.st_mtime))
    path = os.path.join(cache_dir, key + ".npz")
    if os.path.isfile(path):
        with np.load(path) as z:
            return z["players"], z["ball"]
    players, ball = parse_atd(raw_path, n_frames, pitch)
    os.makedirs(cache_dir, exist_ok=True)
    np.savez(path, players=players, ball=ball)
    return players, ball
