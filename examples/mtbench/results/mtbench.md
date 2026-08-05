# MT-Bench: the same six models judged by people and by GPT-4

Human votes: 3,355 from 65 annotators. GPT-4 verdicts: 2,400.

## The two leaderboards

| Rank | Human annotators | GPT-4 as judge |
| ---: | --- | --- |
| 1 | gpt-4 | gpt-4 |
| 2 | claude-v1 | claude-v1 |
| 3 | gpt-3.5-turbo | gpt-3.5-turbo |
| 4 | vicuna-13b-v1.2 | vicuna-13b-v1.2 |
| 5 | alpaca-13b | alpaca-13b |
| 6 | llama-13b | llama-13b |

Kendall tau between the two orders: **+1.00**. On the 1,612 comparisons where the humans reached a majority, GPT-4 agreed **71.3%** of the time.

## What the judge's own inconsistency looks like

The GPT-4 split records a verdict of `tie (inconsistent)` when asking in the other order produced the opposite answer. That happened on **380 of 2,400** pairs (**15.8%**).

## Audits

# Human annotators

3,355 battles over 6 models. 0 of 10 probes raised a flag, 0 did not apply.

## Leaderboard

| Rank | Model | Score | 95% CI | Rank interval | Battles |
| ---: | --- | ---: | --- | --- | ---: |
| 1 | gpt-4 | +1.020 | [+0.906, +1.142] | 1 | 1,029 |
| 2 | claude-v1 | +0.642 | [+0.530, +0.760] | 2-3 | 996 |
| 3 | gpt-3.5-turbo | +0.599 | [+0.524, +0.675] | 2-3 | 1,478 |
| 4 | vicuna-13b-v1.2 | -0.057 | [-0.158, +0.039] | 4 | 1,106 |
| 5 | alpaca-13b | -0.761 | [-0.872, -0.649] | 5 | 1,018 |
| 6 | llama-13b | -1.444 | [-1.566, -1.325] | 6 | 1,083 |

## Probes

| Probe | Value | Innocent band | Verdict |
| --- | ---: | --- | --- |
| `separation` | 0.200 | 0 to 0.25 | clean |
| `rank_volatility` | 1.000 | 0 to 2 | clean |
| `connectivity` | 1.000 | exact | clean |
| `coverage_imbalance` | 0.068 | 0 to 0.5 | clean |
| `position_bias` | 0.004 | -0.05 to 0.05 | clean |
| `position_reordering` | 0.000 | exact | clean |
| `tie_convention` | 0.000 | exact | clean |
| `intransitivity` | -0.000 | -1 to 0.02 | clean |
| `vote_clustering` | 1.086 | 0 to 1.15 | clean |
| `annotator_influence` | 0.037 | 0 to 1 | clean |

## What each probe found

- **Adjacent ranks the votes separate** (`separation`, clean) - 4 of 5 adjacent pairs are separated with family-wise control across the board (4 without it); the board collapses to 5 tiers
- **Width of the rank interval** (`rank_volatility`, clean) - the median model's 95% rank interval spans 1.0 ranks; claude-v1 spans 2
- **Comparability of the board** (`connectivity`, clean) - every model is connected to every other through a chain of opponents
- **Evenness of the comparison budget** (`coverage_imbalance`, clean) - battles per model run from 996 to 1478; 15 of 15 possible pairs were ever compared
- **Advantage of being shown first** (`position_bias`, clean) - the model shown first wins 50.2% of decisive battles (log-odds advantage +0.004)
- **Order changes after correcting for position** (`position_reordering`, clean) - 0 of the 4 separated pair(s) reverse, and 0 model(s) change rank, once the first-position advantage is fitted out
- **Order changes under another tie rule** (`tie_convention`, clean) - ties are 23.2% of battles; 0 of the 4 separated pair(s) reverse when ties are dropped or modelled instead of split
- **Preference cycles beyond what the fit predicts** (`intransitivity`, clean) - 0 of 20 decided triads are cyclic (0.0%); the fitted model produces 0.0% on the same comparison budget
- **Cost of pretending votes are independent** (`vote_clustering`, clean) - resampling voters instead of battles widens standard errors by 1.09x and costs 0 separated pair(s); the busiest 1% of voters account for 6.1% of votes
- **Largest shift from dropping one annotator** (`annotator_influence`, clean) - dropping the most influential of 65 annotators (who cast 3.2% of votes) moves claude-v1 by 0.04 of the median gap between neighbours


# GPT-4 as judge

2,400 battles over 6 models. 1 of 10 probes raised a flag, 3 did not apply.

## Leaderboard

| Rank | Model | Score | 95% CI | Rank interval | Battles |
| ---: | --- | ---: | --- | --- | ---: |
| 1 | gpt-4 | +1.808 | [+1.637, +2.006] | 1 | 800 |
| 2 | claude-v1 | +0.988 | [+0.797, +1.211] | 2 | 800 |
| 3 | gpt-3.5-turbo | +0.525 | [+0.392, +0.667] | 3 | 800 |
| 4 | vicuna-13b-v1.2 | -0.170 | [-0.307, -0.029] | 4 | 800 |
| 5 | alpaca-13b | -1.228 | [-1.418, -1.063] | 5 | 800 |
| 6 | llama-13b | -1.922 | [-2.167, -1.718] | 6 | 800 |

## Probes

| Probe | Value | Innocent band | Verdict |
| --- | ---: | --- | --- |
| `separation` | 0.000 | 0 to 0.25 | clean |
| `rank_volatility` | 1.000 | 0 to 2 | clean |
| `connectivity` | 1.000 | exact | clean |
| `coverage_imbalance` | 0.000 | 0 to 0.5 | clean |
| `position_bias` | - | - | skipped |
| `position_reordering` | - | - | skipped |
| `tie_convention` | 0.000 | exact | clean |
| `intransitivity` | 0.000 | -1 to 0.02 | clean |
| `vote_clustering` | 1.477 | 0 to 1.15 | **flag** |
| `annotator_influence` | - | - | skipped |

## What each probe found

- **Adjacent ranks the votes separate** (`separation`, clean) - 5 of 5 adjacent pairs are separated with family-wise control across the board (5 without it); the board collapses to 6 tiers
- **Width of the rank interval** (`rank_volatility`, clean) - the median model's 95% rank interval spans 1.0 ranks; alpaca-13b spans 1
- **Comparability of the board** (`connectivity`, clean) - every model is connected to every other through a chain of opponents
- **Evenness of the comparison budget** (`coverage_imbalance`, clean) - battles per model run from 800 to 800; 15 of 15 possible pairs were ever compared
- **Advantage of being shown first** (`position_bias`, skipped) - no model pair appears in both orderings (0% of 15 pairs do), so which model sits in the first slot is fixed per pair and cannot be told apart from the models themselves
- **Order changes after correcting for position** (`position_reordering`, skipped) - position is not identifiable in this data, so nothing to correct
- **Order changes under another tie rule** (`tie_convention`, clean) - ties are 25.0% of battles; 0 of the 5 separated pair(s) reverse when ties are dropped or modelled instead of split
- **Preference cycles beyond what the fit predicts** (`intransitivity`, clean) - 0 of 20 decided triads are cyclic (0.0%); the fitted model produces 0.0% on the same comparison budget
- **Cost of pretending votes are independent** (`vote_clustering`, flag) - resampling items instead of battles widens standard errors by 1.48x and costs 0 separated pair(s); the busiest 1% of items account for 0.6% of votes
- **Largest shift from dropping one annotator** (`annotator_influence`, skipped) - 1 annotators: leaving out one of so few is not a perturbation worth reading (this runs for 3-200)


