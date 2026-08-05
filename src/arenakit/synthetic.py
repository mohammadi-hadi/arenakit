"""Arenas with known truth and implantable defects.

Every probe in this package claims to detect something. The way that claim is
checked here is by construction: build a leaderboard where the defect is planted
at a known size, run the audit, and require the matching probe to fire - and,
just as importantly, require a clean board to fire nothing.

One mechanism drives most of the dials. Each voter carries an offset per model,
drawn once, so a voter is not simply noisy but consistently opinionated. Set
``voter_sigma`` to zero and voters vanish as a source of correlation; raise it
and concentrate the votes on a few people and the difference between resampling
battles and resampling voters becomes visible, which is exactly the mistake
:func:`arenakit.probes.probe_vote_clustering` looks for.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .schema import Arena, Battle


def simulate(
    n_models: int = 8,
    n_battles: int = 4000,
    seed: int = 0,
    spread: float = 1.0,
    tie_rate: float = 0.10,
    position_bias: float = 0.0,
    order_skew: float = 0.5,
    skewed_model: int = 1,
    hedger: int | None = None,
    hedger_tie_rate: float = 0.6,
    imbalance: float = 0.0,
    disconnect: bool = False,
    cycle: bool = False,
    cycle_size: int = 5,
    cycle_strength: float = 1.0,
    skill_offsets: Sequence[float] | None = None,
    n_voters: int = 400,
    n_items: int = 200,
    voter_sigma: float = 0.0,
    heavy_voters: int = 0,
    heavy_share: float = 0.0,
    heavy_sigma: float = 0.0,
    loud_offset: tuple[int, int, float] | None = None,
    loud_share: float = 0.0,
) -> Arena:
    """Generate an arena.

    Args:
        n_models: models on the board.
        n_battles: votes cast.
        seed: PCG64 seed.
        spread: skills run from ``+spread`` to ``-spread``, evenly.
        tie_rate: chance any battle is called a tie.
        position_bias: log-odds advantage of whichever model is shown first.
        order_skew: chance ``skewed_model`` takes the first slot in its own
            battles. At 0.5 the assignment is fair; higher and a global position
            advantage turns into an advantage for one model in particular.
        skewed_model: which model gets the favourable slot.
        hedger: a model that draws far more often than its skill implies.
        hedger_tie_rate: tie chance in that model's battles.
        imbalance: concentration of the pairing budget, 0 even, 1 very skewed.
        disconnect: split the board into two pools that never meet.
        cycle: give the first ``cycle_size`` models circular preferences that no
            single strength per model can explain.
        cycle_size: how many models take part in the cycle.
        cycle_strength: log-odds of each edge in that cycle.
        skill_offsets: added to the even skill ladder, one per model. Use it
            to place two models a hair apart, so a small artefact is enough to
            put them in the wrong order.
        n_voters: distinct voters.
        n_items: distinct prompts.
        voter_sigma: spread of each voter's per-model taste.
        heavy_voters: how many voters cast an outsized share of the votes.
        heavy_share: the share they cast between them.
        heavy_sigma: their taste spread, usually stronger than everyone else's.
        loud_offset: ``(i, j, delta)`` - one annotator who prefers model ``i``
            over model ``j`` by ``delta`` log-odds regardless of what they are
            shown.
        loud_share: the share of all votes that annotator casts.
    """
    rng = np.random.Generator(np.random.PCG64(seed))
    models = [f"model-{chr(ord('a') + i)}" for i in range(n_models)]
    skill = np.linspace(spread, -spread, n_models)
    if skill_offsets is not None:
        skill = skill + np.asarray(skill_offsets, dtype=float)

    offsets = rng.normal(0.0, voter_sigma, size=(n_voters, n_models)) if voter_sigma else None
    heavy = np.arange(min(heavy_voters, n_voters))
    if heavy.size and heavy_sigma:
        strong = rng.normal(0.0, heavy_sigma, size=(heavy.size, n_models))
        if offsets is None:
            offsets = np.zeros((n_voters, n_models))
        offsets[heavy] = strong
    loud = n_voters - 1 if loud_offset is not None else None
    if loud_offset is not None:
        if offsets is None:
            offsets = np.zeros((n_voters, n_models))
        first, second, delta = loud_offset
        offsets[loud] = 0.0
        offsets[loud, first] = delta / 2.0
        offsets[loud, second] = -delta / 2.0

    weight = np.ones(n_models)
    if imbalance > 0:
        weight = np.exp(-imbalance * np.arange(n_models) * 8.0 / max(n_models - 1, 1))
    pools = (
        [list(range(n_models // 2)), list(range(n_models // 2, n_models))]
        if disconnect
        else [list(range(n_models))]
    )

    battles: list[Battle] = []
    for _ in range(n_battles):
        pool = pools[rng.integers(0, len(pools))]
        probability = weight[pool] / weight[pool].sum()
        first, second = rng.choice(pool, size=2, replace=False, p=probability)
        if order_skew != 0.5 and skewed_model in (first, second):
            wants_first = rng.random() < order_skew
            if (first == skewed_model) != wants_first:
                first, second = second, first

        if loud is not None and rng.random() < loud_share:
            voter = loud
        elif heavy.size and rng.random() < heavy_share:
            voter = int(heavy[rng.integers(0, heavy.size)])
        else:
            voter = int(rng.integers(0, max(n_voters - (1 if loud is not None else 0), 1)))

        gap = skill[first] - skill[second]
        if cycle and max(int(first), int(second)) < cycle_size:
            step = (int(second) - int(first)) % cycle_size
            forward = 1 <= step <= cycle_size // 2
            gap = cycle_strength if forward else -cycle_strength
        if offsets is not None:
            gap += offsets[voter, first] - offsets[voter, second]
        gap += position_bias

        chance = tie_rate
        if hedger is not None and hedger in (first, second):
            chance = hedger_tie_rate
        if rng.random() < chance:
            winner = "tie"
        else:
            winner = "a" if rng.random() < 1.0 / (1.0 + np.exp(-gap)) else "b"

        battles.append(
            Battle(
                model_a=models[first],
                model_b=models[second],
                winner=winner,  # type: ignore[arg-type]
                item_id=f"item-{rng.integers(0, n_items)}",
                voter_id=f"voter-{voter}",
            )
        )
    return Arena(battles, models=models)


def true_skill(
    n_models: int = 8, spread: float = 1.0, skill_offsets: Sequence[float] | None = None
) -> np.ndarray:
    """The skills :func:`simulate` uses, centred like a fitted board."""
    skill = np.linspace(spread, -spread, n_models)
    if skill_offsets is not None:
        skill = skill + np.asarray(skill_offsets, dtype=float)
    return skill - skill.mean()
