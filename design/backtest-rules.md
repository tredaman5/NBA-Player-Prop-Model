# Backtest rules (pre-registered)

Version 1, written 2026-10-25, before any backtest result exists.

These rules decide what counts as evidence that the model beats the sportsbook lines. They are written down first so the answer cannot be shaped by which slice of the data happens to look good. The code in `src/backtests/rules.py` copies the numbers below; a test checks that the two agree.

## Question

Does the model's projection of a player's points predict the outcome better than the median sportsbook line does?

## Unit of analysis

One **line**: a (player, game, market) with a consensus line from `consensus_lines` (median across books, at least 3 books, the latest capture of that game). Today only the points market is captured.

## Which lines count (the sample)

A line is in the sample only if all of these hold:

1. The game date is **2026-10-26 or later**. Roughly one night of games (through 2026-10-24, 35 games) was looked at informally while building the first comparison. Those games are excluded permanently.
2. The game date is after the model's `trained_through` date, so the model never saw the game or anything after it.
3. The player had at least 5 prior games, the same rule the live slate uses.
4. The player played (minutes above 0). Sportsbooks void a prop when the player does not play, so those lines are dropped and counted, not scored.
5. The line is not a push (the actual result differs from the line).

The projection for each line is what the model gives from information strictly before that game (the same feature builder the live slate uses; a test proves the two agree).

## Primary measures

- **Direction.** The model says *over* if projection > line and *under* if projection < line (projection equal to the line is dropped). A line is a *hit* if the player's actual points land on the model's side. Primary number: the hit rate across all eligible lines, with no threshold on the size of the gap.
- **Accuracy.** For each line, `d = |line - actual| - |projection - actual|`. A positive average means the model's projection was closer to the actual result than the line was.

## Verdict

The verdict is read **once**, when the sample for **one model label** reaches **1,500** non-push lines with a direction. Before that the report is labelled *interim*: it prints the numbers with their uncertainty, states plainly that it is not a verdict, and nothing built from it may use the words edge, bet, EV or pick.

At 1,500 lines the verdict is:

- **Edge shown** only if both hold:
  - the lower end of the 95% interval for the hit rate is above **0.524** (the break-even rate at -110 on both sides), and
  - the lower end of the 95% interval for average `d` is above **0**.
- **No edge shown** otherwise, including when only one of the two holds.

Intervals are a bootstrap that resamples whole games (lines in the same game are correlated): 5,000 draws, seed 2026, 2.5th and 97.5th percentiles.

## Exploratory results (never part of the verdict)

Breakdowns by size of the gap (under 1, 1 to 2, 2 to 3, 3 or more points), minutes, number of books, and time before tip-off may be reported, clearly labelled exploratory. A pattern found there is a hypothesis to test on the next sample, not a finding.

## One model, one count

Every result is tied to a model label. A retrain, new features or any change to the model starts a new sample for the new label, beginning with games after its `trained_through` date. Samples are never pooled across labels, and older results stay on record.

## Notes and limits

- The 0.524 break-even assumes -110 on both sides. Real prices vary by book and are in the log; the report also prints the average implied break-even for context, but the verdict uses 0.524.
- Lines are captured between about 2 and 240 minutes before tip-off, so the line's timing varies. It is recorded per line and reported, not corrected for.
- No injury or lineup information is available to the model. A verdict applies to the model as it is.
- The verdict is about points projections against the median line. It says nothing about any other market or any sizing of bets.

## Changing these rules

Only by adding a new dated version of this file **before** looking at results under it, and by stating what changed and why. Editing a rule after seeing results is not allowed. Earlier versions are kept.

## What the verdict allows

- Interim, or *no edge shown*: the page and the reports keep projection, line and difference only.
- *Edge shown*: edge and expected-value language may be added for that model label, with the interval stated next to it.
