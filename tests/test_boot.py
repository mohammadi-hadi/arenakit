"""Resampling, rank intervals, and the separation test."""

from __future__ import annotations

import numpy as np
import pytest

from arenakit.boot import (
    battles_needed,
    bootstrap,
    bootstrap_values,
    percentile_ci,
    separation,
)
from arenakit.bt import fit_bt
from arenakit.synthetic import simulate


def test_same_seed_same_draws():
    arena = simulate(n_models=5, n_battles=1500, seed=1)
    first = bootstrap(arena, n_resamples=100, seed=0)
    second = bootstrap(arena, n_resamples=100, seed=0)
    other = bootstrap(arena, n_resamples=100, seed=1)
    assert first.scores.tobytes() == second.scores.tobytes()
    assert first.scores.tobytes() != other.scores.tobytes()


def test_rank_interval_contains_the_fitted_rank():
    arena = simulate(n_models=6, n_battles=4000, seed=2)
    boot = bootstrap(arena, n_resamples=300, seed=0)
    fitted = fit_bt(arena.counts()).ranks
    bounds = boot.rank_ci()
    assert np.all(bounds[:, 0] <= fitted)
    assert np.all(fitted <= bounds[:, 1])


def test_overlapping_intervals_do_not_mean_unresolved():
    """Why this package does not decide separation by interval overlap.

    Bradley-Terry scores move together under resampling, so two models can have
    visibly overlapping score intervals while the gap between them is still
    significant. Judging separation by overlap would call those pairs
    unresolved, which is wrong in the conservative direction.

    The comparison here is against the marginal test, since that is the one the
    overlap rule is a substitute for. The simultaneous test is stricter than
    overlap by construction and is not the claim being made.
    """
    arena = simulate(n_models=12, n_battles=6000, spread=1.4, tie_rate=0.0, seed=3)
    boot = bootstrap(arena, n_resamples=500, seed=0)
    sep = separation(boot, arena.models)
    intervals = boot.score_ci()

    overlapping_but_resolved = [
        pair
        for pair in sep.pairs
        if pair["resolved_marginal"]
        and intervals[pair["lower_index"], 1] > intervals[pair["upper_index"], 0]
    ]
    assert overlapping_but_resolved, "expected at least one overlapping yet separated pair"


def test_simultaneous_control_is_never_looser_than_marginal():
    arena = simulate(n_models=8, n_battles=1200, seed=4)
    boot = bootstrap(arena, n_resamples=400, seed=0)
    sep = separation(boot, arena.models)
    assert sep.n_simultaneous <= sep.n_marginal
    assert all(pair["resolved_marginal"] for pair in sep.pairs if pair["resolved_simultaneous"])
    assert sep.critical_simultaneous >= max(p["critical_marginal"] for p in sep.pairs) - 1e-9


def test_tiers_partition_the_board():
    arena = simulate(n_models=7, n_battles=700, seed=5)
    boot = bootstrap(arena, n_resamples=400, seed=0)
    sep = separation(boot, arena.models)
    flat = [index for tier in sep.tiers for index in tier]
    assert sorted(flat) == list(range(arena.n_models))
    assert len(sep.tiers) == sep.n_simultaneous + 1


def test_plenty_of_votes_separates_everything():
    arena = simulate(n_models=5, n_battles=20000, spread=1.5, tie_rate=0.0, seed=6)
    sep = separation(bootstrap(arena, n_resamples=300, seed=0), arena.models)
    assert sep.n_simultaneous == sep.n_pairs
    assert sep.unresolved_share == 0.0


def test_opinionated_heavy_voters_widen_clustered_intervals():
    arena = simulate(
        n_models=6,
        n_battles=4000,
        n_voters=200,
        heavy_voters=5,
        heavy_share=0.5,
        heavy_sigma=1.2,
        seed=7,
    )
    plain = bootstrap(arena, n_resamples=300, seed=0, cluster="battle")
    clustered = bootstrap(arena, n_resamples=300, seed=0, cluster="voter")
    assert clustered.scores.std(axis=0).mean() > 1.5 * plain.scores.std(axis=0).mean()
    assert clustered.n_clusters < plain.n_clusters


def test_clustering_on_a_missing_column_is_refused():
    arena = simulate(n_models=4, n_battles=400, seed=8)
    arena.voter = None
    with pytest.raises(ValueError, match="voter_id"):
        bootstrap(arena, n_resamples=10, cluster="voter")


def test_bootstrap_values_and_percentile_interval():
    arena = simulate(n_models=5, n_battles=3000, position_bias=0.5, tie_rate=0.0, seed=9)
    draws = bootstrap_values(
        arena, lambda counts: fit_bt(counts, order=True).order_log or 0.0, n_resamples=200
    )
    low, high = percentile_ci(draws)
    assert low < 0.5 < high
    assert low < high


def test_battles_needed_scales_with_the_square_of_the_shortfall():
    pair = {"gap": 0.1, "se": 0.1}
    assert battles_needed(pair, 1000, 2.0) == pytest.approx(4000.0)
    assert battles_needed({"gap": 0.0, "se": 0.1}, 1000, 2.0) is None
