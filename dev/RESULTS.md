# Development results (public data only)

All numbers below are on **other matches** (IDSSE / Sportec Bundesliga, CC BY 4.0, Bassek et al.
2025), never on BAR-ATH. Scored with the official `evaluation/possession_metrics.py`.

## Setup
- 7 IDSSE matches (J03WMX J03WN1 J03WOH J03WOY J03WPY J03WQQ J03WR9). Labels per frame:
  `D` where DFL `BallStatus == 0`, else `BallPossession` (1 home / 2 away). Referees excluded.
- Each match degraded with `dev/degrade.py` (41% ball missing in bursts, 0.3 m noise, 46
  anonymous slots with churn, 3% cross-team swaps, duplicated tracks).
- Leave-one-match-out: classifier trained on 6, scored on the 7th. HMM parameters chosen on the 6
  training matches' out-of-fold predictions (nested), never on the scored match.
- Regenerate: `python -m dev.train --save models/lgbm.txt` (~55 min on CPU, data auto-downloaded
  to `../idsse_data`).

## Results (mean over 7 held-out matches, sd in brackets), 2026-11-11

| method | spellF1 | macroF1 |
|---|---|---|
| baseline_raw (official smoother) | 0.019 (0.002) | 0.418 (0.014) |
| baseline_smooth25 | 0.151 (0.009) | 0.431 (0.018) |
| baseline_smooth50 | 0.178 (0.014) | 0.426 (0.020) |
| baseline_smooth125 | 0.166 (0.016) | 0.390 (0.026) |
| hand-set HMM (solution/features+model, dead share 0.2) | 0.226 (0.035) | 0.601 (0.045) |
| hand-set HMM, dead share from training matches | 0.252 (0.028) | 0.723 (0.023) |
| LightGBM per-frame argmax | 0.171 (0.018) | 0.880 (0.018) |
| **LightGBM + sticky HMM** | **0.518 (0.052)** | **0.882 (0.020)** |

lgbm+HMM is best on every one of the 7 matches. Differences below ~0.04 spellF1 are noise.

## Caveats
- IDSSE possession is DFL's definition: spells average 5.5-9.2 s vs 14.2 s in the BAR-ATH public
  window. The HMM params chosen (p_turnover 0.01) suit short spells and may over-split on BAR-ATH.
- IDSSE dead share is 41-52%. The hand-set `dead_target_share = 0.2` matches only the public
  5 minutes (19.6% D), which must not be used for tuning; it is likely far too low for the full match.
- `degrade()` drops the ball at random times; on the real data it is missing mostly in crowded /
  contested moments, so the real task is harder than this simulation.
- Chosen HMM params sit on the grid edge (p_turnover 0.01 = max, p_leave_dead 0.001 = min).

## Next steps
1. Widen the HMM grid (faster: coarse search, or score fewer configs with the official scorer).
2. Make degrade() drop the ball preferentially when players crowd it / it is contested.
3. Stickier target closer to the BAR-ATH style (e.g. absorb very short spells) as a variant.
4. Task 5: better inference when the ball is missing; dead-ball / restart detection.
