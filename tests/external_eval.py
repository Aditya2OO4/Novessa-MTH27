"""Score the pipeline on an OTHER game (Metrica public sample game 3) -- allowed by RULES.md.

    git clone https://github.com/metrica-sports/sample-data
    pip install kloppy
    python tests/external_eval.py --data sample-data/data/Sample_Game_3 --degrade

What it does
  1. Loads the sample game's tracking (clean, stable player ids) and converts it to the same
     arrays the real pipeline consumes (metres, per-slot speed).
  2. --degrade makes it look like the challenge data: ball dropped in bursts to ~41% missing,
     position noise, random player dropouts.
  3. Builds per-frame H/A/D labels from the sample game's EVENT data (an approximation of the
     organisers' truth: event-derived, so timing is rougher).
  4. Runs (a) the repo's nearest-player baseline with smoothing and (b) this HMM, scores both with
     the official metric code.

Treat the numbers as a sanity check and for comparing variants against each other, not as a
prediction of the real score: different league, cleaner tracking, event-derived labels.
"""

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import possession_metrics as pm  # noqa: E402
from solution import features, model  # noqa: E402
from starter.baseline import smooth  # noqa: E402

PITCH = (105.0, 68.0)
CONTROL = {"PASS", "CARRY", "SHOT", "RECOVERY", "SET PIECE"}
STOPPAGE = {"BALL OUT", "FAULT RECEIVED"}


def load_tracking(data_dir):
    from kloppy import metrica
    name = "Sample_Game_3"
    ds = metrica.load_tracking_epts(
        meta_data=os.path.join(data_dir, name + "_metadata.xml"),
        raw_data=os.path.join(data_dir, name + "_tracking.txt"))
    df = ds.to_df()
    team_ids = [str(t.team_id) for t in ds.metadata.teams]
    pids = sorted({c[:-2] for c in df.columns if c.startswith("P") and c.endswith("_x")})
    team_of = {}
    for t in ds.metadata.teams:
        for p in t.players:
            team_of[p.player_id] = 0 if str(t.team_id) == team_ids[0] else 1
    n = len(df)
    players = np.full((n, len(pids), 3), np.nan, dtype=np.float32)
    for i, pid in enumerate(pids):
        players[:, i, 0] = df[pid + "_x"].values * PITCH[0]
        players[:, i, 1] = df[pid + "_y"].values * PITCH[1]
        v = np.hypot(np.gradient(players[:, i, 0]), np.gradient(players[:, i, 1])) * 25.0
        players[:, i, 2] = v
    ball = np.stack([df.ball_x.values * PITCH[0], df.ball_y.values * PITCH[1]], 1).astype(np.float32)
    teams = np.array([team_of[p] for p in pids], dtype=np.int8)
    return players, ball, teams, df.period_id.values, df.frame_id.values


def degrade(players, ball, seed=0, ball_missing=0.41):
    rng = np.random.default_rng(seed)
    n = len(ball)
    ball = ball.copy()
    players = players.copy()
    players[:, :, :2] += rng.normal(0, 0.3, players[:, :, :2].shape)
    miss = np.zeros(n, dtype=bool)
    t = 0
    while miss.mean() < ball_missing and t < 10 * n:
        s = int(rng.integers(0, n))
        miss[s:s + int(rng.integers(10, 150))] = True
        t += 1
    ball[miss] = np.nan
    drop = rng.random(players.shape[:2]) < 0.04
    players[drop] = np.nan
    return players, ball


def event_labels(events_path, n_frames, frame_ids):
    """Per-frame H/A/D from events: last controlling event's team; D from stoppage to restart."""
    ev = json.load(open(events_path))["data"]
    ev = [e for e in ev if e["start"]["frame"] is not None]
    ev.sort(key=lambda e: e["start"]["frame"])
    home_id = ev[0]["team"]["id"] if ev[0]["team"] else None
    first_team = None
    for e in ev:
        if e["type"]["name"] == "SET PIECE" and e["team"]:
            first_team = e["team"]["id"]
            break
    home_id = first_team
    labels = np.full(n_frames, "X", dtype="<U1")
    state, start = None, None
    marks = []
    for e in ev:
        kind = e["type"]["name"]
        f = e["start"]["frame"] - int(frame_ids[0])
        if kind in STOPPAGE:
            marks.append((f, "D"))
        elif kind in CONTROL and e["team"]:
            marks.append((f, "H" if e["team"]["id"] == home_id else "A"))
    for (f, lab), nxt in zip(marks, marks[1:] + [(marks[-1][0] + 1, None)]):
        labels[max(0, f):max(0, nxt[0])] = lab
    labels[marks[-1][0]:] = "X"
    return labels, home_id


def score(name, gt, pred):
    res = pm.evaluate(pm.label_codes(gt), pm.label_codes(pred))
    print("{:<26} spellF1 {:.3f}  macroF1 {:.3f}  poss pred/gt {}/{}  mean dur {:.1f}s (gt {:.1f}s)".format(
        name, res["spell_f1_mean"], res["macro_f1"], res["n_pred_spells"], res["n_gt_spells"],
        res["mean_spell_duration_s"], res["gt_mean_spell_duration_s"]))
    return res


def nearest_codes(players, ball, teams):
    d = np.sqrt(((players[:, :, :2] - ball[:, None, :]) ** 2).sum(axis=2))
    d = np.where(np.isnan(d), np.inf, d)
    k = d.argmin(axis=1)
    codes = np.where(np.isinf(d.min(axis=1)), -1, teams[k]).astype(np.int8)
    return codes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--degrade", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    players, ball, teams, period, frame_ids = load_tracking(args.data)
    n = len(ball)
    gt, _home = event_labels(os.path.join(args.data, "Sample_Game_3_events.json"), n, frame_ids)
    if args.degrade:
        players, ball = degrade(players, ball, args.seed)

    blocks = []
    for h in (1, 2):
        idx = np.where(period == h)[0]
        a, b = int(idx[0]), int(idx[-1])
        ok = np.where(gt[a:b + 1] != "X")[0]
        blocks.append((h, a + int(ok[0]), a + int(ok[-1]), a + int(ok[0])))
    graded = np.zeros(n, dtype=bool)
    for _h, a, b, _l in blocks:
        graded[a:b + 1] = True
    gt = np.where(graded, gt, "X")
    print("ball missing on graded frames: {:.0f}%".format(100 * np.isnan(ball[graded, 0]).mean()))
    print("gt label share on graded frames:",
          {c: round(float((gt[graded] == c).mean()), 3) for c in "HAD"})

    for w in (25, 50, 125):
        codes = nearest_codes(players, ball, teams)
        base = smooth(codes, w)
        base = np.where(graded, base, "X")
        score("baseline_smooth{}".format(w), gt, base)

    cfg = model.Config(pitch=PITCH, blocks=tuple(blocks))
    feats, calib = features.compute(players, ball, teams, graded, cfg)
    print("calibration:", {k: round(v, 3) for k, v in calib.items()})
    pred = model.decode(feats["llr"], feats["dead_logit"], n, cfg)
    score("HMM (this repo)", gt, pred)


if __name__ == "__main__":
    main()
