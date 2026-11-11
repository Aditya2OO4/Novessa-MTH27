# [TEAM NAME]: Ball Possession, Technical Document

More than a Hack 2027, Phase 1 (Initial Challenge)
Team: [members] · Repository: [public repo URL]

---

## 1. Summary

For every frame of the tactical video we output `H`, `A`, `D` or `X`. The pipeline has three
stages:

1. **Features:** we compute 78 numbers per frame from the provided automatic tracking. They
   describe who is near the ball, how the two teams are shaped and moving, and whether play looks
   stopped.
2. **Sequence classifier:** a small bidirectional GRU reads about 40 seconds of those features at
   a time and gives a probability for each of home possession, away possession and dead ball.
3. **Possession decoder:** a three-state "sticky" hidden Markov model (H / A / D) with Viterbi
   decoding turns those probabilities into a small number of long, clean possessions, which is
   what the ranking metric (spellF1) rewards.

The classifier is trained **only on other matches**: 7 public Bundesliga matches whose
per-frame possession and ball-status labels come from the official league data feed. We
deliberately degrade those matches to look like the challenge tracking. Nothing is labelled by
hand, and nothing is fitted on the provided match.

On the 7 public matches, each one held out in turn, the pipeline scores **spellF1 0.598 /
macroF1 0.900**. A re-implementation of the organisers' nearest-player baseline scores
spellF1 0.178 on the same matches. These are not scores on the challenge match, which is noisier
(see Limitations).

---

## 2. Approach and why

**Why not just "nearest player to the ball"?** The ball is missing on 41% of graded frames, and
frame-by-frame decisions flicker. Each flicker creates a fake possession, and spellF1 punishes
fake possessions heavily. The organisers' own baseline shows this: smoothing changes spellF1 by a
factor of five while macroF1 hardly moves.

**So we treat possession as a sequence problem**, in two layers:

- **The classifier sees context, not single frames.** It reads a 40-second window, so it can use
  what happened before and after a gap in the ball tracking (for example, which team moves
  forward, or who is closest when the ball reappears).
- **The decoder enforces a sensible structure.** The HMM makes switching possession or entering a
  stoppage "expensive", so isolated noisy frames cannot split a possession in two. Viterbi
  decoding gives the single most likely sequence of H / A / D for each half.

**Why we chose the GRU.** We compared three classifiers under the same protocol (Section 6). The
GRU beat a LightGBM per-frame classifier on every one of the 7 held-out matches (spellF1 0.598 vs
0.526, about 0.07 above the 0.04 noise band). Averaging the two added nothing measurable, so we
ship the GRU alone.

**Symmetry.** Every prediction is averaged with the prediction for the same frame with home and
away swapped, and training uses the same swap as augmentation. The model therefore has no built-in
preference for either team.

---

## 3. Assets used, and assets dropped

| Asset | Used? | How / why not |
|---|---|---|
| Tactical automatic tracking (ATD), players + ball | **Yes, the only input** | Positions and speeds of every slot and the ball, converted to metres. Slot team comes from the metadata. Slot index and identity are never used, because they switch. |
| Smart Tagging | No (in the shipped model) | Used only by an earlier hand-set version (set-piece intervals as dead-ball evidence). The learned model does not need it, and its throw-ins carry no team tag. |
| Homography | No | Only needed to go between pixels and the pitch. We work on the pitch only. |
| Tactical / broadcast video | No | Detecting the ball from video could fill the 41% gaps, but we judged it out of scope for the time available. |
| Broadcast tracking | No | Not synchronised to the tactical video, and a third of its frames have no usable homography. |
| Public 5-minute ground truth | Only to check the pipeline | Used to check frame alignment and file format, and once to read how short possessions are labelled (Section 5). **Never used for training, tuning or choosing between versions.** |

---

## 4. Training data

| Dataset | Source | Labels: who produced them | Amount |
|---|---|---|---|
| IDSSE / Sportec open Bundesliga data | Bassek, Rein, Weber et al., *"An integrated dataset of spatiotemporal and event data in elite soccer"*, Scientific Data 12, 195 (2025). Licence CC BY 4.0. Mirror: `huggingface.co/datasets/pysport/idsse-data` | The official DFL data feed's per-frame `BallPossession` and `BallStatus` fields, recorded by the league's data provider. **Not labelled by our team.** | 7 matches (IDs J03WMX, J03WN1, J03WOH, J03WOY, J03WPY, J03WQQ, J03WR9), 25 fps |
| Metrica Sports sample data (game 3) | `github.com/metrica-sports/sample-data` | Labels derived automatically from its event data | 1 match, used only as an extra check, not for training |

**No data about the provided match (Barcelona vs Athletic Club) was used for training or
validation.** No human at our team labelled any frame of any match.

**Labels per frame:** `D` where `BallStatus = 0` (ball dead), otherwise `H` / `A` from
`BallPossession`.

**Making the clean data look like the challenge data** (`dev/degrade.py`). The Bundesliga tracking
is clean and the challenge tracking is not, so each training match is degraded:

- **Ball:** removed on 41% of frames, in bursts of 0.2 to 16 s, plus 0.3 m position noise.
- **Players:** position and speed noise, and short dropouts.
- **Slots:** players spread over 46 anonymous slots with track breaks (about every 60 s), 3%
  cross-team identity swaps and occasional duplicated tracks.

---

## 5. Possession rules we implemented

- **Three states only.** Every graded frame is `H`, `A` or `D`.
- **Ungraded frames** (before kickoff, half-time, after the final whistle) are `X`.
- **Second half, frames 110,808 to 113,741** are written as `D`. This follows the published
  graded-frame table: the half is graded from 110,808 but kicks off at 113,742.
- **Loose balls and challenges.** We have no separate "contested" state. A possession continues
  until the evidence for the other team is strong enough to beat the decoder's switching cost, so
  a single touch or deflection does not normally change possession.
- **Short possessions are allowed.** We considered removing all opposition spells shorter than
  2 s from the training labels, to model "touches without control do not count". The public
  ground truth, however, contains genuine possessions of 0.76 s and 1.32 s, so we kept the labels
  as they are.
- **Restarts.** Dead-ball periods are learned from the `BallStatus` labels: players slowing down,
  the ball still or leaving the pitch, team shape. A possession that follows a stoppage starts
  fresh. We do not try to say who "owns" a dead ball (the challenge asks for `D`).
- **Ball not tracked.** We do not invent a ball position. The classifier uses team shape and
  movement, how long ago the ball was last seen and who was closest then and when it reappears.
  The decoder then keeps the current possession through the gap unless the evidence says
  otherwise.

---

## 6. How we evaluated without overfitting

We never scored ourselves on the challenge match. All decisions were made on the 7 public
matches, with **leave-one-match-out**:

- For each match, the classifier is trained on the other six, then scored on it.
- The decoder's settings are chosen using the other six matches only (nested), never the scored
  match.
- Scores use the organisers' official `evaluation/possession_metrics.py`.

| Method (all through the same decoder) | spellF1 (sd) | macroF1 (sd) |
|---|---|---|
| Nearest player + 2 s smoothing (organisers' baseline, re-implemented) | 0.178 (0.014) | 0.426 (0.020) |
| Hand-set rules + HMM (first version) | 0.252 (0.028) | 0.723 (0.023) |
| LightGBM per frame + HMM | 0.526 (0.055) | 0.884 (0.018) |
| **BiGRU + HMM (submitted)** | **0.598 (0.039)** | **0.900 (0.011)** |

Searching the decoder's settings more widely did not help (0.519 to 0.526, within noise). After
that, the remaining gains came from the classifier, not the decoder.

---

## 7. On the challenge match

We ran the pipeline above on the full match once and submitted that output unchanged. These are
sanity numbers only, never a score.

- Graded frames labelled H / A / D: [ ] / [ ] / [ ]
- Number of possessions and mean duration: 281 possessions, mean 13.1 s (the public window has 17
  possessions with a mean of 14.2 s, so the length is plausible)
- Validator result: Valid (199,319 rows, one per tactical-video frame)
- Public 5-minute window, used to check the pipeline only: [scoreboard line]

---

## 8. Limitations, and what we would do with more time

- **Different possession definition.** Our labels come from the Bundesliga feed, while the
  challenge ground truth comes from Metrica's corrected tracking. The two may disagree on borderline
  cases (deflections, aerial duels, very short spells).
- **Simulated noise is not the real noise.** We remove the ball at random times, while in the real
  data it goes missing mostly in crowded moments (boxes, duels, tackles). The real match is
  therefore harder than our test, and we expect a lower score on it.
- **Only 7 training matches.** More labelled public matches would help, but few exist with
  per-frame possession labels.
- **Unused inputs.** With more time, the clearest gains would be (1) detecting the ball in the
  tactical video to fill tracking gaps, (2) using Smart Tagging set-piece intervals as an extra
  input to the classifier, and (3) a more realistic simulation of ball loss.

---

## 9. Reproduction

Requirements: Python 3.12 and the match data bundle. A GPU is not needed.

**The match data is not in our repository, on purpose.** Our submission was produced by running
this pipeline on the real Barcelona vs Athletic Club match data. RULES.md forbids redistributing
that data or publishing derived data that would reconstruct it, and the repository is public. So
`.gitignore` excludes `data/`, the video, the tracking files, the parse cache (`.cache/`) and our
output CSVs (`outputs/`). Anyone with an enrolled copy of the data can rebuild the submission with
the commands below. `python tools/run_on_real_data.py --team-name [TEAM NAME]` does the same in one
step, and also scores the 5 public minutes as a frame-alignment check.

```bash
git clone [public repo URL] && cd [repo]
python3.12 -m venv .venv
. .venv/bin/activate                        # Windows: .venv\Scripts\activate
pip install -r solution/requirements.txt    # exact pinned versions
# Linux, optional: install torch==2.11.0 from https://download.pytorch.org/whl/cpu to skip the CUDA download

export MTAH_DATA_DIR=/path/to/data          # folder holding tactical/ and broadcast/
python -m solution.make_submission --team-name [TEAM NAME]
```

This writes `outputs/[TEAM NAME].csv` and runs the official validator on it. Inference runs on the
CPU with deterministic settings, and running it twice gives a byte-identical file.
`python tests/test_submission_model.py` checks this without the match data.

**Retraining the model** (optional; public data only, downloaded automatically):

```bash
pip install -r dev/requirements-dev.txt
python -m dev.train                               # builds the feature cache, LightGBM comparison
python -m dev.seq --no-lgbm --save models/gru.pt  # sequence model (about 10 min on a laptop GPU)
```

**Main files:**

- `solution/atd.py`: parses the tracking file.
- `solution/learned.py`: the 78 features and the HMM decoder.
- `solution/sequence.py`: the GRU model.
- `solution/make_submission.py`: builds the CSV.
- `dev/`: training and evaluation on public data.

**Open-source software:** NumPy, pandas, PyTorch, LightGBM, kloppy (Metrica sample loading),
lxml. Data: IDSSE / Sportec Open Data (CC BY 4.0).
