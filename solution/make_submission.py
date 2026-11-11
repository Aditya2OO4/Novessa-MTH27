"""Build the submission CSV from the provided data.

    python -m solution.make_submission --team-name MyTeam

Writes outputs/<team_name>.csv (the contract in SUBMISSION.md), prints diagnostics, then runs the
official validator on it. Deterministic: the same data gives the same file.
"""

import argparse
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from games import GAME, GAMES, require, source_frames  # noqa: E402
from starter.baseline import team_of_each_track, team_names, write_submission  # noqa: E402
from solution import atd, features, model  # noqa: E402

# The submission model: LightGBM trained on other matches only (dev/train.py) + sticky HMM.
DEFAULT_MODEL = os.path.join(ROOT, "models", "lgbm.txt")


def set_piece_mask(n_frames, game, data_dir):
    """Boolean mask of frames inside a Smart Tagging SET PIECES interval (None if unavailable)."""
    try:
        from starter.load_smart_tagging import load
        events = load(game, data_dir)
    except (IOError, OSError, ImportError) as exc:
        print("  smart tagging unavailable ({}); continuing without it".format(exc))
        return None
    mask = np.zeros(n_frames, dtype=bool)
    for ev in events:
        if (ev["code"] or "").strip().lower() == "set pieces":
            mask[max(0, ev["start_frame"]):min(n_frames, ev["end_frame"])] = True
    return mask


def run(team_name, out_dir, data_dir=None, use_tagging=True, cache_dir=None, model_path=None):
    cfg = model.Config(pitch=GAMES[GAME]["pitch"])
    n_frames = source_frames("tactical", GAME)
    meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME, data_dir)
    print("teams: H = {}, A = {}".format(*team_names(meta_path)))

    cache_dir = cache_dir or os.path.join(ROOT, ".cache")
    players, ball = atd.load_cached(raw_path, n_frames, cfg.pitch, cache_dir)
    team_of_track = team_of_each_track(meta_path)

    graded = np.zeros(n_frames, dtype=bool)
    for _h, first, last, live_from in cfg.blocks:
        graded[live_from:last + 1] = True

    if model_path:
        # learned classifier (trained on other matches only, dev/train.py) + sticky HMM
        import json
        import lightgbm as lgb
        from solution import learned
        with open(model_path + ".json") as fh:
            meta = json.load(fh)
        booster = learned.load_booster(model_path)
        period = np.zeros(n_frames, dtype=np.int8)
        for half, first, last, _live in cfg.blocks:
            period[first:last + 1] = half
        proba = learned.predict_proba(booster, players, ball, team_of_track, period, cfg.pitch)
        hmm = meta["hmm"]
        cfg.emission_weight = hmm["w"]
        cfg.p_turnover, cfg.p_enter_dead, cfg.p_leave_dead = (
            hmm["p_turnover"], hmm["p_enter_dead"], hmm["p_leave_dead"])
        labels = learned.hmm_decode(proba, np.array(meta["prior"]), n_frames, cfg)
    else:
        sp_mask = set_piece_mask(n_frames, GAME, data_dir) if use_tagging else None
        feats, calib = features.compute(players, ball, team_of_track, graded, cfg, sp_mask)
        print("calibration:", {k: round(v, 3) for k, v in calib.items()})
        labels = model.decode(feats["llr"], feats["dead_logit"], n_frames, cfg)

    g = labels[graded]
    print("graded frames: H {} A {} D {}".format(*(int((g == c).sum()) for c in "HAD")))
    count, mean_s = model.possession_stats(labels)
    print("possessions: {}  mean duration {:.1f} s  (public GT window: mean 14.2 s)".format(
        count, mean_s))

    path = os.path.join(out_dir, "{}.csv".format(team_name))
    write_submission(path, labels)
    print("wrote", path)

    validator = os.path.join(ROOT, "evaluation", "validate_submission.py")
    subprocess.run([sys.executable, validator, path], check=False)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--team-name", required=True)
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs"))
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--no-tagging", action="store_true")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="LightGBM booster from dev/train.py --save (default: %(default)s)")
    ap.add_argument("--hand-set", action="store_true",
                    help="use the older hand-set HMM pipeline instead of the learned model")
    args = ap.parse_args()
    run(args.team_name, args.out, args.data_dir, not args.no_tagging,
        model_path=None if args.hand_set else args.model)
