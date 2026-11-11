"""python tests/test_smoke.py  -- end-to-end on synthetic data (no match data needed)."""

import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from solution import atd, features, model  # noqa: E402
from starter.baseline import team_of_each_track  # noqa: E402
import synthetic  # noqa: E402


def main():
    blocks = ((1, 1000, 11000), (2, 13000, 23000))
    players, ball, truth = synthetic.make_match(blocks=blocks)
    n = len(truth)
    with tempfile.TemporaryDirectory() as tmp:
        raw, meta = os.path.join(tmp, "a.txt"), os.path.join(tmp, "a.xml")
        synthetic.write_files(players, ball, raw, meta)
        p2, b2 = atd.parse_atd(raw, n, (synthetic.L, synthetic.W))
        assert np.allclose(np.nan_to_num(p2), np.nan_to_num(players), atol=0.05), "parse mismatch"
        teams = team_of_each_track(meta)

    cfg = model.Config(pitch=(synthetic.L, synthetic.W),
                       blocks=tuple((h, a, b, a) for h, a, b in blocks))
    graded = np.zeros(n, dtype=bool)
    for _h, a, b, _l in cfg.blocks:
        graded[a:b + 1] = True
    feats, calib = features.compute(p2, b2, teams, graded, cfg)
    pred = model.decode(feats["llr"], feats["dead_logit"], n, cfg)

    acc = float((pred[graded] == truth[graded]).mean())
    pc, pm_ = model.possession_stats(pred)
    tc, tm = model.possession_stats(truth)
    print("calibration", {k: round(v, 3) for k, v in calib.items()})
    print("frame accuracy {:.3f}".format(acc))
    print("possessions pred {} ({:.1f}s)  truth {} ({:.1f}s)".format(pc, pm_, tc, tm))
    assert set(np.unique(pred)) <= set("HADX")
    assert acc > 0.7, "toy world should be easy"
    assert 0.5 * tc <= pc <= 2.0 * tc, "possession count should be in the right ballpark"
    print("OK")


if __name__ == "__main__":
    main()
