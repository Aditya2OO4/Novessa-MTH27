# Pending work

Deadline: **November 17, 2026**. Current best on 7 degraded IDSSE matches (leave-one-match-out,
official scorer): LightGBM + sticky HMM, spellF1 0.518 / macroF1 0.882. See `dev/RESULTS.md`.
Differences below ~0.04 spellF1 are noise.

## Modelling (public data only, re-score with `python -m dev.train` after each)

- [ ] **1. Widen + speed up the HMM search.** The chosen params sit on the grid edge
      (p_turnover 0.01 = max, p_leave_dead 0.001 = min). Widen both ranges. Make it faster:
      coarse-to-fine search, or rank settings with a cheap internal spell metric and score only
      the top few with the official scorer (the last run took ~55 min, mostly 560 official scorings).
- [ ] **2. Realistic ball dropout in `degrade()`.** Drop the ball mostly when players crowd it or
      it is contested, not at random times.
- [ ] **3. Stickier training target.** IDSSE spells average 5.5-9.2 s vs 14.2 s published for the
      BAR-ATH public window. Try absorbing very short spells so the target averages ~14 s. Use only
      that published average, never the public 5 minutes themselves.
- [ ] **4. Ball-missing inference + dead-ball / restart detection.** Better evidence when the ball
      is absent (player movement/formation, who is nearest the last known ball); detect restarts
      (players frozen, set-piece shapes, ball off pitch).
- [ ] Revisit the dead-ball share: IDSSE is 41-52% D; the hand-set 0.2 is likely too low.

## Real-match run (human only, by hand, never via an AI tool)

- [ ] Run `python -m solution.make_submission --team-name <team>` on the real data.
- [ ] Run the validator on the exact file to be uploaded.
- [ ] Check it is deterministic: run twice, compare hashes.

## Submission deliverables (SUBMISSION.md)

- [ ] `<team_name>.csv` possession file.
- [ ] Public code repository URL (make this repo public before submitting; no data in it).
- [ ] Short technical document `<team_name>.pdf` or `.md` covering:
  - approach, and why
  - assets used and dropped
  - training data: IDSSE / Sportec Bundesliga open tracking data (Bassek et al. 2025, CC BY 4.0),
    labels from DFL BallStatus / BallPossession; Metrica sample game 3 for external checks.
    Nothing from the provided game.
  - possession rules implemented (loose balls, challenges, restarts)
  - limitations
  - reproduction steps
- [ ] Credit open-source code and libraries (kloppy, LightGBM, scikit-learn, ...); check licences.
- [ ] Confirm the team has a Google account for the form; one response per team.
