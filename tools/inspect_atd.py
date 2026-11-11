"""Print the STRUCTURE of a tracking file -- counts only, never any raw lines or coordinates.

    python tools/inspect_atd.py data/tactical/BAR-ATH_tactical_atd.txt \
        --meta data/tactical/BAR-ATH_tactical_atd.xml

Why: solution/atd.py assumes the format `<frame>:<x,y,speed;x,y,speed;...>:<ball_x,ball_y,...>`.
This checks that assumption on the real file and prints only format facts (separator counts, how
many slots per line, how many are NaN, value ranges). Run it on the machine that holds the data
and paste the output wherever you need help; nothing here echoes match data.
"""

import argparse
import xml.etree.ElementTree as ET
from collections import Counter


def is_nan(token):
    return token.strip().lower() in ("nan", "")


def inspect(raw_path):
    n = bad = 0
    first = last = prev = None
    contiguous = True
    colons, slots, widths, ball_widths = Counter(), Counter(), Counter(), Counter()
    tracked_per_line = Counter()
    ball_missing = 0
    lo = {"x": 1e18, "y": 1e18, "s": 1e18}
    hi = {"x": -1e18, "y": -1e18, "s": -1e18}
    with open(raw_path) as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            n += 1
            parts = line.split(":")
            colons[len(parts) - 1] += 1
            if len(parts) != 3:
                bad += 1
                continue
            try:
                frame = int(parts[0])
            except ValueError:
                bad += 1
                continue
            first = frame if first is None else first
            if prev is not None and frame != prev + 1:
                contiguous = False
            prev = last = frame

            groups = parts[1].split(";")
            slots[len(groups)] += 1
            tracked = 0
            for g in groups:
                f = g.split(",")
                widths[len(f)] += 1
                if not is_nan(f[0]):
                    tracked += 1
                    try:
                        for key, idx in (("x", 0), ("y", 1), ("s", 2)):
                            if idx < len(f) and not is_nan(f[idx]):
                                v = float(f[idx])
                                lo[key] = min(lo[key], v)
                                hi[key] = max(hi[key], v)
                    except ValueError:
                        pass
            tracked_per_line[tracked] += 1
            b = parts[2].split(",")
            ball_widths[len(b)] += 1
            if is_nan(b[0]):
                ball_missing += 1
    return dict(n=n, bad=bad, first=first, last=last, contiguous=contiguous, colons=colons,
                slots=slots, widths=widths, ball_widths=ball_widths, tracked=tracked_per_line,
                ball_missing=ball_missing, lo=lo, hi=hi)


def report(r):
    print("lines: {}   unparseable (not `int:...:...`): {}".format(r["n"], r["bad"]))
    print("frame ids: {}..{}  contiguous: {}".format(r["first"], r["last"], r["contiguous"]))
    print("':' separators per line:", dict(r["colons"]), " (expected {2: all})")
    print("player slots per line:", dict(r["slots"]))
    print("fields per player group:", dict(r["widths"]), " (expected {3: ...} = x,y,speed)")
    print("fields in ball group:", dict(r["ball_widths"]))
    tr = r["tracked"]
    total = sum(tr.values()) or 1
    mean = sum(k * v for k, v in tr.items()) / total
    print("tracked players per frame: mean {:.1f}, min {}, max {}".format(
        mean, min(tr) if tr else None, max(tr) if tr else None))
    print("ball missing: {:.1f}% of lines".format(100.0 * r["ball_missing"] / total))
    print("value ranges  x: [{:.3f}, {:.3f}]  y: [{:.3f}, {:.3f}]  speed: [{:.2f}, {:.2f}]".format(
        r["lo"]["x"], r["hi"]["x"], r["lo"]["y"], r["hi"]["y"], r["lo"]["s"], r["hi"]["s"]))
    print("  (x,y expected inside 0..1; if speed range looks huge/tiny the units differ from m/s)")


def report_meta(meta_path):
    root = ET.parse(meta_path).getroot()
    teams = [(t.get("id"), t.findtext("Name")) for t in root.iter("Team")]
    print("\nmetadata teams (first = home by FIFA convention):", teams)
    per_team = Counter(p.get("teamId") for p in root.iter("Player"))
    print("Player entries per team:", dict(per_team), " total:", sum(per_team.values()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("raw")
    ap.add_argument("--meta")
    args = ap.parse_args()
    report(inspect(args.raw))
    if args.meta:
        report_meta(args.meta)
