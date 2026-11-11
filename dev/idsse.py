"""Load IDSSE / Sportec open Bundesliga matches (public, CC BY 4.0) into clean numpy arrays.

    Bassek, Rein, Weber et al., "An integrated dataset of spatiotemporal and event data in elite
    soccer", Sci Data 12, 195 (2025). https://doi.org/10.1038/s41597-025-04505-y
    Mirror: https://huggingface.co/datasets/pysport/idsse-data

These are OTHER matches: allowed for development and training (RULES.md). Nothing here touches
the challenge data.

kloppy's loader is not used for the labels because it collapses BallPossession to "1 -> home,
anything else -> away" and holds the whole XML in memory. This module streams the raw XML with
iterparse and keeps the raw codes:

    BallPossession  1 = home, 2 = away (per frame, from the DFL feed)
    BallStatus      1 = alive, 0 = dead

Output arrays (one match, both halves concatenated, 25 fps):
    players (N, P, 3)  x_m, y_m (origin bottom-left, 0..105 x 0..68), speed m/s; NaN when absent
    ball    (N, 2)     x_m, y_m
    team    (P,)       0 home, 1 away
    period  (N,)       1 or 2
    possession (N,)    raw BallPossession code (0 if missing)
    status  (N,)       raw BallStatus code (-1 if missing)
    labels  (N,)       'H' / 'A' / 'D' : D where dead, else possession team
"""

import os
import urllib.request

import numpy as np

MATCHES = {
    "J03WMX": "DFL-COM-000001", "J03WN1": "DFL-COM-000001",
    "J03WOH": "DFL-COM-000002", "J03WOY": "DFL-COM-000002", "J03WPY": "DFL-COM-000002",
    "J03WQQ": "DFL-COM-000002", "J03WR9": "DFL-COM-000002",
}
BASE_URL = "https://huggingface.co/datasets/pysport/idsse-data/resolve/main/"
META = "DFL_02_01_matchinformation_{c}_DFL-MAT-{m}.xml"
TRACK = "DFL_04_03_positions_raw_observed_{c}_DFL-MAT-{m}.xml"
SECTIONS = {"firstHalf": 1, "secondHalf": 2}
DEFAULT_DIR = os.environ.get("IDSSE_DIR", os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "idsse_data"))


def _download(match_id, data_dir):
    os.makedirs(data_dir, exist_ok=True)
    paths = []
    for tmpl in (META, TRACK):
        name = tmpl.format(c=MATCHES[match_id], m=match_id)
        path = os.path.join(data_dir, name)
        if not os.path.isfile(path):
            print("  downloading", name)
            urllib.request.urlretrieve(BASE_URL + name, path + ".part")
            os.replace(path + ".part", path)
        paths.append(path)
    return paths


def _teams(meta_path):
    """(home TeamId, guest TeamId)."""
    from lxml import etree
    role = {}
    for _ev, el in etree.iterparse(meta_path, tag="Team"):
        role[el.get("Role")] = el.get("TeamId")
    return role["home"], role["guest"]


def parse(match_id, data_dir=DEFAULT_DIR):
    from lxml import etree
    meta_path, track_path = _download(match_id, data_dir)
    home, guest = _teams(meta_path)

    pitch = (105.0, 68.0)
    rows = {}            # (section, person) -> (team, N array, X, Y, S)
    ball = {}            # section -> (N, X, Y, poss, status)
    for _ev, fs in etree.iterparse(track_path, tag="FrameSet"):
        sec = SECTIONS.get(fs.get("GameSection"))
        if sec is None:
            fs.clear()
            continue
        frames = fs.findall("Frame")
        n = np.array([int(f.get("N")) for f in frames], dtype=np.int64)
        x = np.array([float(f.get("X")) for f in frames], dtype=np.float32)
        y = np.array([float(f.get("Y")) for f in frames], dtype=np.float32)
        if fs.get("TeamId", "").upper() == "BALL":
            poss = np.array([int(f.get("BallPossession", 0)) for f in frames], dtype=np.int8)
            stat = np.array([int(f.get("BallStatus", -1)) for f in frames], dtype=np.int8)
            ball[sec] = (n, x, y, poss, stat)
        elif fs.get("TeamId") in (home, guest):      # skips referees
            s = np.array([float(f.get("S", "nan")) for f in frames], dtype=np.float32) / 3.6
            team = 0 if fs.get("TeamId") == home else 1
            rows[(sec, fs.get("PersonId"))] = (team, n, x, y, s)
        fs.clear()
        while fs.getprevious() is not None:
            del fs.getparent()[0]

    persons = sorted({p for _s, p in rows})
    team_of = {p: t for (_s, p), (t, *_r) in rows.items()}
    team = np.array([team_of[p] for p in persons], dtype=np.int8)
    col = {p: i for i, p in enumerate(persons)}

    out = {k: [] for k in ("players", "ball", "period", "possession", "status", "frame_n")}
    for sec in (1, 2):
        bn, bx, by, bp, bs = ball[sec]
        n0, n1 = int(bn.min()), int(bn.max())
        length = n1 - n0 + 1
        pl = np.full((length, len(persons), 3), np.nan, dtype=np.float32)
        for (s, p), (_t, n, x, y, sp) in rows.items():
            if s != sec:
                continue
            ok = (n >= n0) & (n <= n1)
            i = n[ok] - n0
            pl[i, col[p], 0] = x[ok] + pitch[0] / 2
            pl[i, col[p], 1] = y[ok] + pitch[1] / 2
            pl[i, col[p], 2] = sp[ok]
        b = np.full((length, 2), np.nan, dtype=np.float32)
        b[bn - n0, 0] = bx + pitch[0] / 2
        b[bn - n0, 1] = by + pitch[1] / 2
        poss = np.zeros(length, dtype=np.int8)
        stat = np.full(length, -1, dtype=np.int8)
        poss[bn - n0] = bp
        stat[bn - n0] = bs
        out["players"].append(pl)
        out["ball"].append(b)
        out["period"].append(np.full(length, sec, dtype=np.int8))
        out["possession"].append(poss)
        out["status"].append(stat)
        out["frame_n"].append(np.arange(n0, n1 + 1))
    res = {k: np.concatenate(v) for k, v in out.items()}
    res["team"] = team
    return res


def labels_from(possession, status):
    lab = np.full(len(status), "X", dtype="<U1")
    lab[possession == 1] = "H"
    lab[possession == 2] = "A"
    lab[status == 0] = "D"
    return lab


def load(match_id, data_dir=DEFAULT_DIR):
    """Cached parse (npz in data_dir/cache). Returns dict incl. 'labels'."""
    cache = os.path.join(data_dir, "cache", match_id + ".npz")
    if os.path.isfile(cache):
        with np.load(cache) as z:
            res = {k: z[k] for k in z.files}
    else:
        res = parse(match_id, data_dir)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez_compressed(cache, **res)
    res["labels"] = labels_from(res["possession"], res["status"])
    return res


def report(match_id, data_dir=DEFAULT_DIR):
    m = load(match_id, data_dir)
    lab, poss, stat = m["labels"], m["possession"], m["status"]
    n = len(lab)

    def counts(a):
        v, c = np.unique(a, return_counts=True)
        return {int(k) if np.issubdtype(v.dtype, np.integer) else str(k): int(x) for k, x in zip(v, c)}

    runs = np.flatnonzero(lab[1:] != lab[:-1]) + 1
    seg = np.split(lab, runs)
    spells = [len(s) for s in seg if s[0] in "HA"]
    on_pitch = (~np.isnan(m["players"][:, :, 0])).sum(axis=1)
    print("{}: {} frames ({:.1f} min), {} persons (home {}, away {})".format(
        match_id, n, n / 1500, len(m["team"]), int((m["team"] == 0).sum()), int((m["team"] == 1).sum())))
    print("  BallPossession codes:", counts(poss))
    print("  BallStatus codes:    ", counts(stat))
    print("  labels H/A/D share:  ", {c: round(float((lab == c).mean()), 3) for c in "HADX"})
    print("  ball missing frames: {:.2%}".format(float(np.isnan(m["ball"][:, 0]).mean())))
    print("  players on pitch per frame: min {} median {} max {}".format(
        int(on_pitch.min()), int(np.median(on_pitch)), int(on_pitch.max())))
    print("  possession spells (H/A runs): {}  mean {:.1f}s  median {:.1f}s".format(
        len(spells), np.mean(spells) / 25, np.median(spells) / 25))
    alive = stat == 1
    print("  possession code while alive: ", counts(poss[alive]))
    return m


if __name__ == "__main__":
    import sys
    for mid in sys.argv[1:] or ["J03WPY"]:
        report(mid)
