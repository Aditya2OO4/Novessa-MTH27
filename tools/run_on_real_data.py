"""ONE command to run everything on the REAL match. Run it BY HAND, never through an AI tool.

    python tools/run_on_real_data.py --team-name YourTeamName

It (1) checks the data files exist, (2) prints the tracking file's structure (counts only),
(3) builds outputs/<team>.csv and runs the official format validator, (4) copies the file to
evaluation/results/ and scores it on the 5 public minutes (a frame-alignment check, NOT for tuning).

The match data location comes from MTAH_DATA_DIR (or ./data). Keep it OUTSIDE the repo folder.
Everything printed is a count or a summary, never raw tracking lines.
"""

import argparse
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team-name", required=True)
    ap.add_argument("--no-tagging", action="store_true")
    ap.add_argument("--hand-set", action="store_true",
                    help="use the older hand-set HMM pipeline instead of the learned model")
    args = ap.parse_args()

    from games import GAME, require
    print("== 1. data files ==")
    atd_meta, atd_raw = require(["atd_meta", "atd_raw"], GAME)
    print("found tracking + metadata")

    print("\n== 2. tracking file structure (counts only) ==")
    import inspect_atd
    inspect_atd.report(inspect_atd.inspect(atd_raw))
    inspect_atd.report_meta(atd_meta)

    print("\n== 3. build submission ==")
    from solution import make_submission
    out_dir = os.path.join(ROOT, "outputs")
    model_path = None if args.hand_set else make_submission.DEFAULT_MODEL
    print("model:", model_path or "hand-set HMM")
    path = make_submission.run(args.team_name, out_dir, None, not args.no_tagging,
                               model_path=model_path)

    print("\n== 4. frame-alignment check on the public 5 minutes (do NOT tune on this) ==")
    results = os.path.join(ROOT, "evaluation", "results")
    shutil.copy(path, os.path.join(results, os.path.basename(path)))
    subprocess.run([sys.executable, os.path.join(ROOT, "evaluation", "run_evaluation.py")],
                   check=False)
    print("\nDone. Your file: {}".format(path))


if __name__ == "__main__":
    main()
