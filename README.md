# Player Experience vs. Team Success — FIFA World Cup, 1930–2026

**Does a national team's on-field experience (cumulative World Cup appearances) actually predict how far it goes in the tournament?**

This project tests that question statistically across 92 years of World Cup history — 2,136 team-matches, 23,495 individual player-appearances, and every tournament from 1930 to 2026 — using head-to-head comparisons, a logistic regression, and rank-based tournament progression analysis.

**[Live interactive dashboard →](https://claude.ai/code/artifact/0415398c-ca54-4dbd-b7d8-d8875807f9a4)**

<!-- Add a screenshot of the dashboard here once you've taken one, e.g.: -->
<!-- ![Dashboard preview](assets/dashboard_preview.png) -->

## TL;DR

| Question | Finding |
|---|---|
| Does the more-experienced side win more often, head-to-head? | **Yes — 63.3%** of decisive matches (n = 762, p < .0001) |
| Does the effect scale with the size of the experience gap? | **Yes** — win rate climbs from 10% (badly out-experienced) to 89% (heavily favored); a +1 average-cap advantage raises the odds of winning by **+35.4%** (logistic regression, n = 1,738 decisive matches) |
| Do teams that go further in the tournament have more experienced squads? | **Yes, step-wise** — Early exit 0.48 → Quarter Final 0.57 → Semi Final 0.62 → Final 0.72 (mean experience percentile; Kruskal–Wallis p < .00001) |
| Are champions specifically above-average in experience? | **18 of 23 champions (78.3%)** were above their own tournament's median experience (one-sample t-test p = .0002) |

**Caveat, stated up front:** this is correlation, not causation, and experience is one contributing factor among several (FIFA ranking, host status, and squad quality all matter too and aren't controlled for here). The four low-experience champions — 1954, 1958, 1998, 2026 — are discussed in the dashboard as the interesting exceptions, not swept under the rug.

## Repo structure

```
worldcup-experience-success/
├── README.md
├── requirements.txt
├── dashboard/
│   └── index.html          # the published interactive dashboard (Vega-Lite), open directly in a browser
├── scripts/
│   └── build_charts.py     # reproduces every stat + chart above in Python (pandas/scikit-learn/Altair)
└── data/
    ├── head_to_head.csv                  # 1 row per team per match: starting-XI experience, opponent's, exp_diff, result
    ├── team_tournament_experience.csv    # 1 row per team per tournament: experience percentile, furthest stage reached
    ├── chart1_winrate_by_round.json
    ├── chart2b_decisive_merged.json
    ├── chart3_qf_stage.json
    └── chart4_champions_by_year.json
```

`data/` holds the cleaned, feature-engineered tables — not the original 14-sheet source workbook, which isn't redistributed here. `scripts/build_charts.py` reads these directly, so the whole analysis (every number in the table above) is reproducible with `pip install -r requirements.txt && cd scripts && python build_charts.py`.

## How the experience metric is built

- **Unit of analysis**: each team's **starting XI** in a given match — not the full squad — since substitutes who never played didn't affect that match's outcome.
- **Experience** = caps *entering that specific tournament* (`Matches Before Tournament Year`), never total career caps — using total caps would leak future appearances into historical tournaments the player hadn't played yet.
- **Head-to-head comparison** (`exp_diff`): a team's average starting-XI caps minus the opponent's, for every match — this is what chart 1 and chart 2 are built on.
- **Tournament-level percentile**: each team's mean starting-XI experience, ranked 0–1 against every other team in *that same tournament year* — this controls for the fact that "experienced" meant something different in 1930 than in 2026, since the tournament itself didn't exist yet for most nations to have caps in.
- **Decisive-matches-only convention**: every win-rate/odds-ratio statistic here excludes draws — a draw isn't a "loss" for the more-experienced side, and folding draws into the "not a win" bucket quietly biases every win rate downward. (This was originally a bug in an earlier iteration — see below.)

## A debugging note, because it's the most useful part of this project to talk about

An earlier version of this analysis counted draws as "not a win" when computing win rates. That's an easy mistake to make and an easy one to miss, because it doesn't crash anything — it just quietly shifts every percentage down by however many draws happened to be in that slice of the data. It surfaced when a bin that should have read ~50% (teams with *no* experience advantage at all) read 41% instead. Tracing it back: draws were ~19% of matches in that bin, and counting them as losses for both sides pulls the "win rate" below the true 50/50 baseline you'd expect at parity.

The fix was to scope every win-rate and odds-ratio statistic to decisive matches only (Win/Lose, draws excluded) — and then to audit *every other number in the dashboard* to check whether the same bug was hiding anywhere else. It wasn't: the percentile-based statistics (tournament progression, champions' average) never touched win/draw/loss counting in the first place, so they were unaffected. That audit — going back and checking whether a fix applied everywhere it needed to, not just where the symptom first appeared — is the part of this project I'd point to first in an interview.

## Statistical methods used

- Binomial test (head-to-head win rate vs. 50%)
- Logistic regression, `P(Win) ~ exp_diff`, with manually-derived standard errors and p-values via the Fisher information matrix (no `statsmodels` dependency)
- Kruskal–Wallis H-test and Spearman rank correlation (experience percentile across 4 ordered tournament-progression tiers)
- One-sample t-test and Wilcoxon signed-rank test (champions' mean experience percentile vs. 0.5)

## Tools

`pandas`, `numpy`, `scipy`, `scikit-learn` for the analysis; `Altair` (Vega-Lite) for the charts in `scripts/build_charts.py`. The published dashboard (`dashboard/index.html`) is hand-written Vega-Lite/JavaScript with the same underlying data and numbers — built that way because the analysis environment it was originally developed in didn't have PyPI access to install Altair, so the Vega-Lite spec was authored directly. `build_charts.py` is the portable, pip-installable version of the same charts.

## Author

Frank — data analysis project, 2026.
