"""Scoring for development runs on public games.

If the official scorer (evaluation/possession_metrics.py from the starter repo) is importable it is
used. Otherwise a STAND-IN is used, written from the description in the docs we have
("spellF1 counts whole possessions, matched by overlap in time"; "macroF1 per frame"). The
stand-in's exact numbers will differ from the official ones; use it only to compare variants.

Stand-in definitions (graded frames = gt != 'X'):
  * pred 'X' is forward-filled (the scorer does this, per solution/model.py).
  * a spell is a maximal run of H (or of A); D and ungraded frames break runs.
  * a predicted spell matches a GT spell of the same team when their temporal IoU > 0.5
    (IoU > 0.5 makes the matching one-to-one without any assignment step).
  * spellF1 = mean over H and A of 2*TP / (n_pred + n_gt).
  * macroF1 = mean over H, A, D of per-frame F1 on graded frames.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
try:
    sys.path.insert(0, os.path.join(ROOT, "evaluation"))
    import possession_metrics as _official  # noqa: E402
except ImportError:
    _official = None

OFFICIAL = _official is not None


def _ffill_x(pred):
    pred = np.asarray(pred).copy()
    ok = pred != "X"
    if not ok.any():
        return pred
    idx = np.where(ok, np.arange(len(pred)), 0)
    np.maximum.accumulate(idx, out=idx)
    out = pred[idx]
    out[:np.argmax(ok)] = "X"
    return out


def spells(labels, team):
    """(starts, ends) inclusive of maximal runs equal to `team`."""
    m = np.concatenate(([False], labels == team, [False])).astype(np.int8)
    d = np.diff(m)
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1


def _matches(ps, pe, gs, ge):
    """Number of predicted spells with IoU > 0.5 against some GT spell."""
    if len(ps) == 0 or len(gs) == 0:
        return 0
    tp = 0
    # candidate GT spells overlapping each pred spell
    lo = np.searchsorted(ge, ps, side="left")
    hi = np.searchsorted(gs, pe, side="right")
    for i in range(len(ps)):
        for j in range(lo[i], hi[i]):
            inter = min(pe[i], ge[j]) - max(ps[i], gs[j]) + 1
            union = max(pe[i], ge[j]) - min(ps[i], gs[j]) + 1
            if inter * 2 > union:
                tp += 1
                break
    return tp


def evaluate(gt, pred, fps=25):
    gt = np.asarray(gt)
    if OFFICIAL:
        res = _official.evaluate(_official.label_codes(gt), _official.label_codes(np.asarray(pred)))
        return {"spell_f1": float(res["spell_f1_mean"]), "macro_f1": float(res["macro_f1"]),
                "n_pred": int(res["n_pred_spells"]), "n_gt": int(res["n_gt_spells"]),
                "pred_mean_s": float(res["mean_spell_duration_s"]),
                "gt_mean_s": float(res["gt_mean_spell_duration_s"])}
    graded = gt != "X"
    pred = np.where(graded, _ffill_x(pred), "X")
    f1s, n_pred, n_gt, durs_p, durs_g = [], 0, 0, [], []
    for team in "HA":
        ps, pe = spells(pred, team)
        gs, ge = spells(gt, team)
        tp = _matches(ps, pe, gs, ge)
        f1s.append(2.0 * tp / max(len(ps) + len(gs), 1))
        n_pred += len(ps)
        n_gt += len(gs)
        durs_p.extend(pe - ps + 1)
        durs_g.extend(ge - gs + 1)
    frame_f1 = []
    g, p = gt[graded], pred[graded]
    for c in "HAD":
        tp = np.sum((g == c) & (p == c))
        denom = np.sum(g == c) + np.sum(p == c)
        frame_f1.append(2.0 * tp / denom if denom else 0.0)
    return {"spell_f1": float(np.mean(f1s)), "macro_f1": float(np.mean(frame_f1)),
            "n_pred": n_pred, "n_gt": n_gt,
            "pred_mean_s": float(np.mean(durs_p)) / fps if durs_p else 0.0,
            "gt_mean_s": float(np.mean(durs_g)) / fps if durs_g else 0.0}
