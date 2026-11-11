"""Sequence classifier (BiGRU) over the per-frame features of solution/learned.py, + sticky HMM.

Trained on OTHER matches only (IDSSE / Sportec Bundesliga, degraded to look like the challenge
data; see dev/seq.py). Nothing here is fitted on the challenge match.

Inference runs on the CPU with deterministic kernels and a fixed thread count, so the same data
gives a byte-identical submission on any machine (no GPU needed to reproduce it).
"""

import numpy as np

from solution import learned

IGNORE = -100
INFER_THREADS = 4


class Normaliser:
    """Robust per-feature scaling fitted on training matches; NaN -> 0 plus a NaN indicator."""

    def __init__(self, med, scale):
        self.med = np.asarray(med, dtype=np.float32)
        self.scale = np.asarray(scale, dtype=np.float32)

    @classmethod
    def fit(cls, Xs):
        X = np.concatenate([x[::10] for x in Xs])
        med = np.nan_to_num(np.nanmedian(X, axis=0))
        q75, q25 = np.nanpercentile(X, [75, 25], axis=0)
        scale = np.where((q75 - q25) > 1e-6, q75 - q25, 1.0)
        return cls(med, scale)

    def __call__(self, X):
        nan = np.isnan(X)
        Z = np.clip((X - self.med) / self.scale, -6, 6)
        Z[nan] = 0.0
        return np.concatenate([Z, nan.astype(np.float32)], axis=1).astype(np.float32)


ARCH = dict(proj=96, hidden=64, layers=2, dropout=0.1)


def build(n_in, proj=96, hidden=64, layers=2, dropout=0.1):
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Sequential(nn.Linear(n_in, proj), nn.GELU(), nn.Dropout(dropout))
            self.gru = nn.GRU(proj, hidden, num_layers=layers, batch_first=True,
                              bidirectional=True, dropout=dropout)
            self.out = nn.Linear(2 * hidden, 3)

        def forward(self, x):
            h, _ = self.gru(self.inp(x))
            return self.out(h)

    return Net()


def predict_windows(net, Z, Zs, spans, device="cpu", core=3000, ctx=500):
    """(N,3) [H,A,D] probabilities inside each (first, last) span, averaged with the mirrored
    (home/away-swapped) prediction. Sliding windows: each `core` frames are predicted with `ctx`
    frames of context either side and only the core is kept, so every frame sees about the context
    length the net was trained on (40 s windows)."""
    import torch
    net.eval()
    n = len(Z)
    proba = np.full((n, 3), 1.0 / 3, dtype=np.float64)
    with torch.no_grad():
        for a, b in spans:
            for s in range(a, b + 1, core):
                e = min(s + core, b + 1)
                lo, hi = max(a, s - ctx), min(b + 1, e + ctx)
                x = torch.from_numpy(np.ascontiguousarray(np.stack([Z[lo:hi], Zs[lo:hi]])))
                p = torch.softmax(net(x.to(device)), dim=-1).cpu().numpy().astype(np.float64)
                p = (p[0] + p[1][:, [1, 0, 2]]) / 2
                proba[s:e] = p[s - lo:e - lo]
    return proba


# ------------------------------------------------------------------ saved model
def save(path, net, norm, meta):
    import torch
    torch.save({"state": {k: v.cpu() for k, v in net.state_dict().items()},
                "med": norm.med.tolist(), "scale": norm.scale.tolist(), "meta": meta}, path)


def load(path):
    """-> (net on CPU, Normaliser, meta). Sets deterministic CPU inference."""
    import torch
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(INFER_THREADS)
    blob = torch.load(path, map_location="cpu", weights_only=True)
    meta = blob["meta"]
    net = build(meta["n_in"], **meta["arch"])
    net.load_state_dict(blob["state"])
    net.eval()
    return net, Normaliser(blob["med"], blob["scale"]), meta


def predict_proba(net, norm, players, ball, slot_team, period, pitch, spans):
    """Per-frame [H, A, D] probabilities for a match (features from solution/learned.py)."""
    X, names = learned.compute_features(players, ball, slot_team, period, pitch)
    Xs, _ = learned.compute_features(players, ball, (1 - slot_team).astype(np.int8), period, pitch)
    return predict_windows(net, norm(X), norm(Xs), spans), names
