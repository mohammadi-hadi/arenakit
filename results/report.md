# arenakit demo boards

## Even board

Enough votes, spread evenly, no artefacts planted.

- planted: `nothing`
- fired: `nothing` (as planted)


## Too few votes

The same board on 200 votes instead of 6000.

- planted: `separation`, `rank_volatility`
- fired: `rank_volatility`, `separation` (as planted)

  - **separation** - 1 of 5 adjacent pairs are separated with family-wise control across the board (1 without it); the board collapses to 2 tiers
  - **rank_volatility** - the median model's 95% rank interval spans 2.5 ranks; model-b spans 3

## First slot favoured

Being shown first is worth real odds, and one model takes that slot almost always.

- planted: `position_bias`, `position_reordering`
- fired: `position_bias`, `position_reordering` (as planted)

  - **position_bias** - the model shown first wins 69.9% of decisive battles (log-odds advantage +0.725)
  - **position_reordering** - 1 of the 5 separated pair(s) reverse, and 2 model(s) change rank, once the first-position advantage is fitted out

## Draw-prone model

One weak model draws most of its battles, and half a win each is generous.

- planted: `tie_convention`
- fired: `tie_convention` (as planted)

  - **tie_convention** - ties are 36.0% of battles; 1 of the 5 separated pair(s) reverse when ties are dropped or modelled instead of split

## Rock, paper, scissors

Three models beat each other in a circle, which no single strength explains.

- planted: `intransitivity`
- fired: `intransitivity` (as planted)

  - **intransitivity** - 1 of 20 decided triads are cyclic (5.0%); the fitted model produces 0.0% on the same comparison budget

## Two separate pools

Two halves of the board that never met, ranked as though they had.

- planted: `connectivity`
- fired: `connectivity` (as planted)

  - **connectivity** - the comparison graph has 2 components, so scores across them are not comparable; the audit kept the largest (3 models) and dropped 3

## Lopsided budget

The comparison budget is spent mostly on the models at the top.

- planted: `coverage_imbalance`
- fired: `coverage_imbalance` (as planted)

  - **coverage_imbalance** - battles per model run from 117 to 5465; 14 of 15 possible pairs were ever compared

## A few power voters

Six people with strong tastes cast a third of all votes.

- planted: `separation`, `vote_clustering`
- fired: `separation`, `vote_clustering` (as planted)

  - **separation** - 2 of 5 adjacent pairs are separated with family-wise control across the board (4 without it); the board collapses to 3 tiers
  - **vote_clustering** - resampling voters instead of battles widens standard errors by 2.30x and costs 2 separated pair(s); the busiest 1% of voters account for 18.4% of votes

## One loud annotator

Two equally good models, and one annotator with a firm view about them.

- planted: `annotator_influence`, `separation`, `vote_clustering`
- fired: `annotator_influence`, `separation`, `vote_clustering` (as planted)

  - **separation** - 2 of 5 adjacent pairs are separated with family-wise control across the board (2 without it); the board collapses to 3 tiers
  - **vote_clustering** - resampling voters instead of battles widens standard errors by 3.42x and costs 2 separated pair(s); the busiest 1% of voters account for 34.7% of votes
  - **annotator_influence** - dropping the most influential of 40 annotators (who cast 34.7% of votes) moves model-c by 1.12 of the median gap between neighbours
