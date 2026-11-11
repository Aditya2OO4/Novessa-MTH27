# Possession pipeline (HMM)

Drop-in folder for your fork of the starter repo (it imports `games.py` and `starter/` from the
repo root). **Python 3.12**, dependencies pinned exactly in `requirements.txt` (generated from
`requirements.in` with `uv pip compile`).

```bash
python3.12 -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r solution/requirements.txt
python tests/test_submission_model.py                   # committed model loads, deterministic
python -m solution.make_submission --team-name YourTeam # needs the match data (MTAH_DATA_DIR)
```

The default is the learned model, `models/lgbm.txt`: LightGBM trained on other matches only, plus
the sticky HMM. `--hand-set` runs the older hand-set pipeline described below instead.
`.gitattributes` keeps `models/` byte-identical on every OS. A CRLF checkout would make LightGBM
reject the model.

Writes `outputs/YourTeam.csv` and runs the official validator on it. First run parses the tracking
(~1-2 min) and caches it in `.cache/` (derived match data, gitignored: never commit or upload it).

## How it works
1. **Parse** tracking, convert to metres, interpolate ball gaps <= 1 s.
2. **Team evidence**: log-odds that home has the ball from how much closer the nearest home player
   is than the nearest away player (confidence fades when nobody is near the ball).
3. **Dead-ball evidence**: slow players (threshold found by Otsu on the match's own speed
   distribution, so no unit assumptions), ball stationary for ~1 s, ball off the pitch, Smart
   Tagging set-piece intervals.
4. **3-state sticky HMM (H/A/D) + Viterbi** per half. Missing-ball stretches carry the previous
   state. Second-half frames 110808-113741 are `D` (published in FAQ.md). `X` outside graded halves.

## Honest caveats
- Parameters in `solution/model.py::Config` are hand-set priors. They were only sanity-checked on a
  synthetic toy match, **not** on the real data. Do not tune them on the 5 public minutes.
- Check the printed possession count / mean duration for plausibility (public window: 17
  possessions, mean 14.2 s). Wildly more or fewer means turnover/dead probabilities are off.
- Ideas, roughly by expected gain: better evidence when the ball is missing (team compactness,
  player movement toward the ball), HMM parameters fitted by EM (unsupervised, allowed),
  a learned model trained on *other* games, use of broadcast tracking.

## Testing on another game (allowed by RULES.md)
```bash
git clone https://github.com/metrica-sports/sample-data ../sample-data
pip install kloppy
python tests/external_eval.py --data ../sample-data/data/Sample_Game_3 --degrade
```
Compares the repo's nearest-player baseline (3 smoothing windows) with this HMM using the official
metric code. Ground truth there is derived from event data, so it is rougher than the organisers';
use it to compare variants, not to predict the real score.
