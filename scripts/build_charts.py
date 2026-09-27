"""
World Cup Player Experience vs. Team Success
=============================================
Reproduces every statistic and chart from the published dashboard
(dashboard/index.html) using pandas + scikit-learn for the analysis and
Altair (Vega-Lite) for the charts.

This script reads the already-cleaned, feature-engineered data in ../data/
(derived from a 14-sheet raw workbook: match results, starting-XI lineups,
and each player's caps entering the tournament). See ../README.md for the
full data-engineering write-up.

Usage:
    pip install -r ../requirements.txt
    python build_charts.py
    # -> writes charts.html in this folder; prints every headline stat to stdout
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
import altair as alt

DATA = "../data"

# ---------------------------------------------------------------------------
# Load the pre-engineered datasets
# ---------------------------------------------------------------------------
# head_to_head.csv: one row per team per match, with the starting XI's mean
# caps entering the tournament, the opponent's same figure, and exp_diff
# (own avg - opponent avg) -- the core paired comparison used everywhere below.
h2h = pd.read_csv(f"{DATA}/head_to_head.csv")

# team_tournament_experience.csv: one row per team per tournament, with each
# team's mean starting-XI experience and its percentile rank (0-1) within
# that tournament year, plus how far the team went (furthest_stage 0-4).
tt = pd.read_csv(f"{DATA}/team_tournament_experience.csv")

# chart3_qf_stage.json: tt above, but re-tiered into 4 display buckets
# (Early exit / Quarter Final / Semi Final / Final) using each match's raw
# round text -- see README "Chart 3 tiering" for why this needed a second pass.
chart3_data = pd.read_json(f"{DATA}/chart3_qf_stage.json")
tier_order = ['Early exit', 'Quarter Final', 'Semi Final', 'Final']
chart3_data['new_tier'] = pd.Categorical(chart3_data['new_tier'], categories=tier_order, ordered=True)

# chart4_champions_by_year.json: the 23 champion team-tournaments only.
chart4_data = pd.read_json(f"{DATA}/chart4_champions_by_year.json").sort_values('tournament_year')

print("Loaded:", len(h2h), "team-match rows,", len(tt), "team-tournament rows,",
      len(chart4_data), "champions")

# ---------------------------------------------------------------------------
# Decisive-matches-only convention
# ---------------------------------------------------------------------------
# Every win-rate / odds-ratio statistic below is scoped to decisive matches
# (Win/Lose) with draws excluded, so an experience "advantage" is measured
# against an outcome that could actually go either way. Percentile-based
# stats (chart 3, chart 4) never touch win/draw/loss counting at all.
dec = h2h.dropna(subset=['exp_diff'])
dec = dec[dec['Match Result'].isin(['Win', 'Lose'])].copy()

# ---------------------------------------------------------------------------
# Chart 1 data: win rate of the more-experienced side, by round
# ---------------------------------------------------------------------------
dec_c1 = dec[dec['exp_diff'] != 0].copy()
dec_c1['side'] = np.where(dec_c1['exp_diff'] > 0, 'More experienced', 'Less experienced')

round_order = ['First and Second Round', 'Semi-Final', 'Final']
round_label = {'First and Second Round': 'Group / Round 1-2', 'Semi-Final': 'Semi-Final', 'Final': 'Final'}

rows = []
for rnd in round_order + ['All rounds']:
    sub = dec_c1 if rnd == 'All rounds' else dec_c1[dec_c1['Main Round'] == rnd]
    for side in ['More experienced', 'Less experienced']:
        s2 = sub[sub['side'] == side]
        n = len(s2)
        wins = (s2['Match Result'] == 'Win').sum()
        rows.append({'round': 'All rounds' if rnd == 'All rounds' else round_label[rnd],
                     'side': side, 'n': n, 'wins': int(wins), 'win_rate': wins / n if n else np.nan})
chart1_data = pd.DataFrame(rows)

overall = chart1_data[chart1_data['round'] == 'All rounds']
n_total = int(overall['n'].sum())
wr_more = overall.loc[overall['side'] == 'More experienced', 'win_rate'].iloc[0]
print(f"\nChart 1 -- head-to-head (decisive only, n={n_total}): "
      f"more-experienced side wins {wr_more:.1%}")
for _, r in chart1_data[chart1_data['round'] != 'All rounds'].iterrows():
    if r['side'] == 'More experienced':
        print(f"  {r['round']:<20} n={r['n']:<4} win_rate={r['win_rate']:.1%}")

# ---------------------------------------------------------------------------
# Chart 2 data + logistic regression: P(Win) ~ exp_diff
# ---------------------------------------------------------------------------
bin_edges = [-100, -6, -3, -1, 1, 3, 6, 100]
bin_labels = ['<=-6', '-5..-3', '-2..-1', '~0 (-1..1)', '1..2', '3..5', '>=6']
dec_c2 = dec.copy()
dec_c2['bin'] = pd.cut(dec_c2['exp_diff'], bins=bin_edges, labels=bin_labels, right=True)
chart2_data = dec_c2.groupby('bin', observed=True).agg(
    n=('Match Result', 'size'), win_rate=('Match Result', lambda s: (s == 'Win').mean())).reset_index()
chart2_data['bin'] = pd.Categorical(chart2_data['bin'], categories=bin_labels, ordered=True)

X = dec[['exp_diff']].values
y = (dec['Match Result'] == 'Win').astype(int).values
clf = LogisticRegression().fit(X, y)
b0, b1 = clf.intercept_[0], clf.coef_[0][0]

# manual standard errors via the Fisher information matrix (no statsmodels dependency)
Xd = np.column_stack([np.ones(len(X)), X[:, 0]])
p_hat = clf.predict_proba(X)[:, 1]
W = np.diag(p_hat * (1 - p_hat))
cov = np.linalg.inv(Xd.T @ W @ Xd)
se = np.sqrt(np.diag(cov))
z = np.array([b0, b1]) / se
pvals = 2 * (1 - stats.norm.cdf(np.abs(z)))

odds_ratio_per_cap = np.exp(b1)
print(f"\nChart 2 -- logistic regression P(Win) ~ exp_diff (decisive only, n={len(dec)}):")
print(f"  coef = {b1:.4f} (SE={se[1]:.4f}, p={pvals[1]:.4g})")
print(f"  odds ratio per +1 avg cap advantage = {odds_ratio_per_cap:.4f} "
      f"(+{(odds_ratio_per_cap - 1) * 100:.1f}%)  <- headline stat")
print(f"  odds ratio per +5 avg caps advantage = {np.exp(b1 * 5):.4f}")

curve_x = np.linspace(dec['exp_diff'].min(), dec['exp_diff'].max(), 200)
curve_y = clf.predict_proba(curve_x.reshape(-1, 1))[:, 1]
chart2_curve = pd.DataFrame({'exp_diff': curve_x, 'predicted_win_prob': curve_y})

# ---------------------------------------------------------------------------
# Chart 3 stats: experience percentile by 4-tier furthest stage
# ---------------------------------------------------------------------------
kw = stats.kruskal(*[g['exp_percentile_in_tournament'].values
                      for _, g in chart3_data.groupby('new_tier', observed=True)])
sp = stats.spearmanr(chart3_data['new_tier'].cat.codes, chart3_data['exp_percentile_in_tournament'])
print(f"\nChart 3 -- experience percentile by furthest stage (n={len(chart3_data)}):")
print(chart3_data.groupby('new_tier', observed=True)['exp_percentile_in_tournament']
      .agg(['mean', 'median', 'count']).reindex(tier_order))
print(f"  Kruskal-Wallis H={kw.statistic:.2f}, p={kw.pvalue:.2g}; "
      f"Spearman rho={sp.correlation:.3f}, p={sp.pvalue:.2g}")

# ---------------------------------------------------------------------------
# Chart 4 stats: champions specifically
# ---------------------------------------------------------------------------
champ_mean_pct = chart4_data['exp_percentile_in_tournament'].mean()
above_median = (chart4_data['exp_percentile_in_tournament'] > 0.5).sum()
t_p = stats.ttest_1samp(chart4_data['exp_percentile_in_tournament'], 0.5).pvalue
print(f"\nChart 4 -- champions (n={len(chart4_data)}): mean percentile = {champ_mean_pct:.3f}, "
      f"{above_median}/{len(chart4_data)} = {above_median / len(chart4_data):.1%} above their own "
      f"tournament's median (one-sample t-test p={t_p:.4f})")

# ---------------------------------------------------------------------------
# Altair charts (palette approximates the live dashboard's theme)
# ---------------------------------------------------------------------------
COLOR_MORE_EXP = "#3b82f6"
COLOR_LESS_EXP = "#94a3b8"
COLOR_BAR = "#3b82f6"
COLOR_MUTED = "#94a3b8"

chart1 = alt.Chart(chart1_data).mark_bar().encode(
    x=alt.X('side:N', title=None, axis=alt.Axis(labels=False, ticks=False)),
    y=alt.Y('win_rate:Q', title='Win rate', axis=alt.Axis(format='%'), scale=alt.Scale(domain=[0, 1])),
    color=alt.Color('side:N', title=None,
                     scale=alt.Scale(domain=['More experienced', 'Less experienced'],
                                      range=[COLOR_MORE_EXP, COLOR_LESS_EXP])),
    column=alt.Column('round:N', title=None,
                       sort=['Group / Round 1-2', 'Semi-Final', 'Final', 'All rounds'],
                       header=alt.Header(orient='bottom', labelFontWeight=600)),
    tooltip=['round', 'side', 'n', alt.Tooltip('win_rate:Q', format='.1%')],
).properties(width=95, height=220,
             title='1. Win rate of the side with more average starting-XI experience (decisive matches only)')

chart2_bars = alt.Chart(chart2_data).mark_bar(color=COLOR_BAR, cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
    x=alt.X('bin:N', title='Experience-gap bin (avg caps/player)', sort=bin_labels),
    y=alt.Y('win_rate:Q', title='Win rate', axis=alt.Axis(format='%'), scale=alt.Scale(domain=[0, 1])),
    tooltip=['bin', 'n', alt.Tooltip('win_rate:Q', format='.1%')],
).properties(width=560, height=240, title='2. Dose-response: bigger experience edge, higher win rate')

chart3 = alt.Chart(chart3_data).mark_boxplot(extent='min-max').encode(
    x=alt.X('new_tier:N', title='Furthest stage reached', sort=tier_order),
    y=alt.Y('exp_percentile_in_tournament:Q', title='Experience percentile', axis=alt.Axis(format='%'),
            scale=alt.Scale(domain=[0, 1])),
    color=alt.Color('new_tier:N', legend=None, sort=tier_order),
).properties(width=560, height=280, title='3. Teams that go further have systematically higher experience')

bars4 = alt.Chart(chart4_data).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
    x=alt.X('tournament_year:O', title='Year (champion)'),
    y=alt.Y('exp_percentile_in_tournament:Q', title='Experience percentile', axis=alt.Axis(format='%'),
            scale=alt.Scale(domain=[0, 1])),
    color=alt.Color('exp_percentile_in_tournament:Q', legend=None, scale=alt.Scale(scheme='blues')),
    tooltip=['tournament_year', 'team_name', alt.Tooltip('exp_percentile_in_tournament:Q', format='.1%'),
             alt.Tooltip('mean_caps:Q', format='.2f')],
)
rule4 = alt.Chart(pd.DataFrame({'y': [0.5]})).mark_rule(
    strokeDash=[4, 3], color=COLOR_MUTED, strokeWidth=1.3).encode(y='y:Q')
chart4 = (bars4 + rule4).properties(
    width=640, height=240, title="4. Are champions' squads above their own tournament's median experience?")

dashboard = alt.vconcat(chart1, chart2_bars, chart3, chart4).resolve_scale(color='independent')
dashboard.save('charts.html')
print("\nSaved charts.html")
