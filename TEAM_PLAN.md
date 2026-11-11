# More Than a Hack 2027 — Team Plan

**Prepared by:** Aditya  **Date:** Nov 9, 2026
**Challenge:** Initial Challenge — "who has the ball, in every frame"
**Deadline:** Nov 17, 2026 (submit via Google Form; link comes by email)

Read this once top to bottom. It tells you what we have to do, where we are, and who does what.

---

## 1. What we have to do (in plain words)

We get one full match: **FC Barcelona (home) vs Athletic Club (away)**. The match is 199,319 video frames (25 per second).

For **every frame**, we write one letter:

| Letter | Meaning |
|---|---|
| **H** | Barcelona has the ball, ball is in play |
| **A** | Athletic has the ball, ball is in play |
| **D** | Dead ball (out of play, free kick being set up, etc.) |
| **X** | No answer. Only used outside the graded parts (before kickoff, halftime, after the whistle). Inside the match we always answer H, A or D. |

The result is **one CSV file**: `frame,label`, exactly 199,319 rows.

This is not a "possession percentage" task. It is a **label for each frame** — a sequence labeling problem.

### What the data gives us
- Tracking: position of all players and the ball, every frame. It is **automatic and noisy**: gaps, identity switches, and the **ball is missing on 41% of graded frames**.
- Smart Tagging: automatic phases of play and set pieces (with team tags).
- Homography and two videos (we are not using these yet).
- 5 minutes of real ground truth (17 possessions). **Only for checking the file is correct.**

### Graded frames
- 1st half: video frames 11,185 – 90,651
- 2nd half: 110,808 – 190,197 (frames 110,808 – 113,741 are all `D`, kickoff is at 113,742)
- Everything else is ignored, so `X` there costs nothing.

---

## 2. How we are scored

- **Ranking metric: `spellF1`.** It counts how many **whole possessions** we get right (matched by overlap in time). Flickering labels (H A H A H) create fake possessions and kill the score.
- **Tie-break: `macroF1`** (per-frame score).
- The organizers' **baseline is 0.2433** on the full match. Beating that is the minimum.
- Differences below about 0.04 are noise.
- **Top 20 teams** from this stage go to the Innovation Challenge (Nov 26 – Jan 14). Top 5 of those pitch in Barcelona on Mar 2, 2027, with travel covered.
- There is **no published pass mark**. It is purely rank-based.

### We cannot see our real score
- Submission is a Google Form. **One response per team. We can edit it until the deadline. The last response before Nov 17 counts.**
- **No leaderboard, no feedback.**
- So we cannot "submit, see the score, improve". Our only checks are:
  1. `python evaluation/validate_submission.py <file>.csv` (format check).
  2. The 5 public minutes (to confirm frames line up — **never for tuning**).
  3. Our own tests on **other games** (see section 5).

---

## 3. Rules that matter (don't break these)

1. **Labels must come from an algorithm.** No hand-labelling or hand-fixing the match.
2. **No hand labels on this match** — not for training, not for validation.
3. Hand-labelled or external data from **other games is allowed**, if organizers can access it and our document explains it.
4. **No human-made data about this match** (Opta, StatsBomb, etc.).
5. **Do not commit or share the match data.** Our code repo must be public and must not contain data, or derived tracking exports. The `data/` folder is gitignored.
6. **Do NOT give the real match data to any hosted AI** (the FAQ says no third-party hosted/cloud LLMs for videos, datasets or extracts). **This includes any AI coding assistant or chatbot.**
   - Use AI tools on public data only.
   - Run anything on the real data **by hand**.
   - Don't paste real data or outputs from it into any AI tool.
   - Even summary numbers: ask the organizers first (hackathon.talentarena@mobileworldcapital.com).
7. We must give: the CSV (`<team_name>.csv`), a public GitHub repo that reproduces the same file, and a short technical document (`<team_name>.pdf` or `.md`). Team name must match the registration exactly.

---

## 4. Where we are now (Nov 9, evening)

- ✅ Everyone is registered.
- ✅ One teammate has the data.
- ✅ First version of the code is written (see below).
- ✅ Tested on fake data and on one real public game (Metrica sample game 3).
- ⬜ **Not yet run on the real match.**
- ⬜ Not yet split into owned parts.
- ⬜ No shared repo yet.
- ⬜ Haven't contacted organizers about the data/LLM rule.

### The code we have (folder `solution/`)

| File | What it does |
|---|---|
| `atd.py` | Reads the tracking file, converts to metres, caches it |
| `features.py` | Fills short ball gaps; works out which team is closer to the ball; detects stoppage signals (slow players, still ball, ball off pitch, set-piece tags) |
| `model.py` | All settings in one place, plus a 3-state smoothing model (H/A/D) that stops flickering and turns frames into long possessions |
| `make_submission.py` | Runs everything, writes `outputs/<team>.csv`, runs the official validator |
| `tests/test_smoke.py` | Checks the code runs, using a fake match |
| `tests/external_eval.py` | Scores the code on a real public game (Metrica sample), using the organizers' scoring code |

Run it: `python -m solution.make_submission --team-name <name>`

### How well does it work?
On the public Metrica game with noisy tracking (60% of ball data removed):

| Method | spellF1 |
|---|---|
| Organizer-style baseline (5 s smoothing) | 0.254 |
| **Our version 1** | **0.342** |

**Be careful with this number.** It is one game, a different league, and its "truth" is built from event data (rougher than the organizers'). It means "the idea works", not "we will score 0.34".

### Known weak points
- **Settings are guesses** (hand-set). Not trained.
- **Dead-ball detection** was fragile at first; we fixed a big bug but it is still the weakest part.
- **Frames where the ball is missing (41%)**: we just keep the previous label.
- The parser was written from the format the starter code reads. It has never seen the real file, so expect a small fix.

---

## 5. Our approach to make it better

The idea: stop relying on my hand-set rules and **learn** from other games with real labels, then smooth the result.

1. **Get labelled data from other games.**
   - **IDSSE / Sportec Open DFL dataset**: 7 Bundesliga matches, 25 fps, tracking + events, CC-BY licence (on Hugging Face as `pysport/idsse-data`, also Figshare). The Floodlight library says its DFL reader extracts **ball status and possession codes** — very close to how the organizers built their truth. **Not yet verified that those fields are in the open files — first job is to check.**
   - Metrica sample game 3 (already used).
   - SkillCorner open data (has a per-frame possession field, 10 fps, broadcast tracking).
2. **Make that data look like ours**: drop the ball in bursts (about 41% missing), add position noise, drop some players.
3. **Train a model** (e.g. gradient-boosted trees) on per-frame features: distances to the ball, speeds, ball speed, number of players near the ball, ball outside the pitch, plus a few seconds of context → probabilities for H / A / D.
4. **Keep the smoothing model** on top of those probabilities to turn them into proper possessions.
5. **Improve the missing-ball frames** using ideas from open-source work that guesses the ball carrier from player movement:
   - PathCRF — github.com/hyunsungkim-ds/pathcrf
   - Ball Radar — github.com/pientist/ballradar
   - Check their licences. Credit anything we use in the technical document.
6. **Improve dead-ball detection** with the DFL ball-status labels and the Smart Tagging set-piece windows.

Priority order: **1 → 2 → 3 → 4**, then 5 and 6 if time allows. We have about 8 days, so don't start with the heavy research models.

---

## 6. Who does what

| Role | Person | Job |
|---|---|---|
| **Data owner / integrator** | ______ | Runs the pipeline on the real data (by hand, no AI tools on the data). Checks frame alignment, runs the validator, fixes parser issues, makes the final CSV, uploads the form. Also checks that our printed numbers look plausible. |
| **Model (ball visible)** | ______ | Loads IDSSE data, builds the degraded training set, trains the learned model, compares it against version 1 on the held-out matches. |
| **Hard cases** | ______ | Missing-ball frames and dead-ball detection. Writes the technical document. Keeps the repo clean. |
| **Coordinator** | Aditya | Keeps the plan, merges, talks to organizers, final review. |

Everyone works from **one shared repo** (private until the end, then public). Each person works on their own module and branch, and we merge at least once a day. All three of us use the same **external evaluation script** as the scoreboard, so changes are compared fairly.

---

## 7. Timeline

| When | What |
|---|---|
| **Nov 9–10** | Create shared repo. Email organizers (data/LLM rule, can aggregate stats be shared, ID question is already settled). Verify IDSSE data has ball-status and possession fields. Data owner runs version 1 on the real data and fixes the parser. |
| **Nov 10–14** | Model and Hard-cases work in parallel on external matches. Daily merge. |
| **Nov 14–15** | Data owner runs the merged version on the real data. Check the sanity numbers. **Code freeze on Nov 15.** |
| **Nov 16** | Validate the exact file. Public repo ready (no data in it). Technical document done. Upload the form. |
| **Nov 17** | Buffer day only. Don't plan to use it. |

Reminder: the Innovation Challenge starts Nov 26 and needs **3–5 members**, so we're fine as a team.

---

## 8. How we sanity-check the real output (without a score)

When the data owner runs the pipeline, look at the printed summary:
- Number of possessions and average length. The public 5-minute window has 17 possessions averaging 14.2 s. If ours is way off (e.g. 900 possessions of 4 s, or 20 of 5 minutes), the smoothing settings are wrong.
- Share of frames labelled D. Football has roughly one in five frames dead; if it's 60% or 0%, the dead-ball logic is broken.
- H vs A should be roughly balanced.
- Validator says "Valid".

Do **not** tune on the 5 public minutes — it picks the wrong settings (the organizers' own tests prove this).

---

## 9. Submission checklist

- [ ] CSV named `<team_name>.csv`, exactly 199,319 rows, header `frame,label`
- [ ] `python evaluation/validate_submission.py <file>.csv` says Valid on the **exact file** being uploaded
- [ ] Technical document named `<team_name>.pdf` (or `.md`): approach, assets used and dropped, training data (source, who labelled, how much — none from this match), possession rules, limitations, how to reproduce
- [ ] Public GitHub repo with pinned dependencies and the command that regenerates the same file. **No match data inside.**
- [ ] A Google account for the upload
- [ ] Team name identical to registration
- [ ] One person owns the final upload; the rest don't submit extra responses

---

## 10. Honest risks

- Our only score evidence comes from a different game with rougher labels.
- Dead-ball logic may behave differently on the real match.
- The parser may need a fix on first contact with the real file.
- Nobody knows what score is needed to reach the top 20. Aim as high as we can.
- Deadline is close. Keep the simple version working at all times — **always keep a valid submission ready**, and only replace it with something that is clearly better on the external test.
