"""The fits, checked against closed forms and against statsmodels."""

from __future__ import annotations

import math

import numpy as np
import pytest

from arenakit.bt import (
    MAX_ITER,
    BTFit,
    DisconnectedBoardError,
    UnboundedFitError,
    components,
    fit_bt,
    fit_rao_kupper,
    leaderboard,
    to_elo,
)
from arenakit.schema import Arena, Battle
from arenakit.synthetic import simulate, true_skill


def _battles(pairs):
    return Arena([Battle(model_a=a, model_b=b, winner=w) for a, b, w in pairs])


def test_two_models_match_the_closed_form():
    """With two models the maximum likelihood estimate is the logit of the win rate."""
    pairs = [("a", "b", "a")] * 30 + [("a", "b", "b")] * 10
    fit = fit_bt(_battles(pairs).counts(), prior=0.0)
    gap = fit.scores[0] - fit.scores[1]
    assert gap == pytest.approx(math.log(30 / 10), abs=1e-9)


def test_half_credit_ties_sit_between_the_two_conventions():
    decisive = [("a", "b", "a")] * 20 + [("a", "b", "b")] * 10
    with_ties = decisive + [("a", "b", "tie")] * 20
    dropped = fit_bt(_battles(with_ties).counts(), ties="drop", prior=0.0)
    halved = fit_bt(_battles(with_ties).counts(), ties="half", prior=0.0)
    # Ties pull the two models together without reordering them.
    assert 0 < halved.scores[0] - halved.scores[1] < dropped.scores[0] - dropped.scores[1]


def test_matches_statsmodels_logistic_regression():
    """Bradley-Terry is logistic regression on signed model indicators."""
    statsmodels = pytest.importorskip("statsmodels.api")
    arena = simulate(n_models=5, n_battles=3000, tie_rate=0.0, seed=3)
    fit = fit_bt(arena.counts(), ties="drop", prior=0.0)

    design = np.zeros((arena.n_battles, arena.n_models - 1))
    rows = np.arange(arena.n_battles)
    for sign, index in ((1.0, arena.a), (-1.0, arena.b)):
        keep = index < arena.n_models - 1
        design[rows[keep], index[keep]] = sign
    response = (arena.y == 0).astype(float)
    reference = statsmodels.Logit(response, design).fit(disp=0).params

    ours = fit.scores[:-1] - fit.scores[-1]
    assert np.allclose(ours, reference, atol=1e-5)


def test_position_term_matches_statsmodels():
    """The first-position advantage is the intercept of the same regression."""
    statsmodels = pytest.importorskip("statsmodels.api")
    arena = simulate(n_models=5, n_battles=4000, tie_rate=0.0, position_bias=0.4, seed=4)
    fit = fit_bt(arena.counts(), ties="drop", order=True, prior=0.0)

    design = np.zeros((arena.n_battles, arena.n_models))
    design[:, 0] = 1.0  # the position slot
    rows = np.arange(arena.n_battles)
    for sign, index in ((1.0, arena.a), (-1.0, arena.b)):
        keep = index < arena.n_models - 1
        design[rows[keep], index[keep] + 1] = sign
    response = (arena.y == 0).astype(float)
    reference = statsmodels.Logit(response, design).fit(disp=0).params

    assert fit.order_log == pytest.approx(reference[0], abs=1e-5)
    assert np.allclose(fit.scores[:-1] - fit.scores[-1], reference[1:], atol=1e-5)


def test_recovers_planted_skill():
    arena = simulate(n_models=6, n_battles=20000, tie_rate=0.0, seed=5)
    fit = fit_bt(arena.counts())
    assert np.allclose(fit.scores, true_skill(6, 1.0), atol=0.08)


def test_rao_kupper_reduces_to_bradley_terry_without_ties():
    arena = simulate(n_models=5, n_battles=2000, tie_rate=0.0, seed=6)
    plain = fit_bt(arena.counts(), ties="drop")
    tied = fit_rao_kupper(arena.counts())
    assert tied.tie_theta == pytest.approx(1.0)
    assert np.allclose(plain.scores, tied.scores, atol=1e-8)


def test_rao_kupper_finds_a_tie_parameter_above_one():
    arena = simulate(n_models=5, n_battles=4000, tie_rate=0.35, seed=7)
    fit = fit_rao_kupper(arena.counts())
    assert fit.tie_theta is not None and fit.tie_theta > 1.2


def test_prior_keeps_an_undefeated_model_finite():
    pairs = [("a", "b", "a")] * 40 + [("b", "c", "b")] * 40
    counts = _battles(pairs).counts()
    fit = fit_bt(counts, prior=0.5)
    assert np.all(np.isfinite(fit.scores))
    with pytest.raises(UnboundedFitError, match="strongly connected"):
        fit_bt(counts, prior=0.0)


def test_prior_barely_moves_a_well_fed_board():
    arena = simulate(n_models=5, n_battles=8000, tie_rate=0.0, seed=8)
    with_prior = fit_bt(arena.counts(), prior=0.5)
    without = fit_bt(arena.counts(), prior=0.0)
    assert np.allclose(with_prior.scores, without.scores, atol=5e-3)


def test_disconnected_boards_are_refused():
    pairs = [("a", "b", "a")] * 20 + [("c", "d", "a")] * 20
    counts = _battles(pairs).counts()
    with pytest.raises(DisconnectedBoardError, match="never"):
        fit_bt(counts)
    with pytest.raises(DisconnectedBoardError):
        fit_rao_kupper(counts)
    assert [len(group) for group in components(counts.comparisons)] == [2, 2]


def test_refits_are_bit_for_bit_identical():
    arena = simulate(n_models=6, n_battles=3000, seed=9)
    first = fit_bt(arena.counts())
    second = fit_bt(arena.counts())
    assert first.scores.tobytes() == second.scores.tobytes()
    assert first.n_iter == second.n_iter


def test_iteration_cap_is_a_real_cap():
    assert MAX_ITER >= 1000
    arena = simulate(n_models=4, n_battles=500, seed=10)
    assert fit_bt(arena.counts()).n_iter < MAX_ITER


def test_elo_is_monotone_and_ranking_is_stable():
    arena = simulate(n_models=6, n_battles=3000, seed=11)
    fit = fit_bt(arena.counts())
    elo = to_elo(fit.scores)
    assert np.array_equal(np.argsort(-elo), np.argsort(-fit.scores))
    board = leaderboard(arena.models, fit)
    assert [row["rank"] for row in board] == list(range(1, arena.n_models + 1))
    assert board[0]["score"] >= board[-1]["score"]


def test_ranks_are_one_based_and_complete():
    fit = BTFit(scores=np.array([0.5, -0.5, 0.0]))
    assert fit.ranks.tolist() == [1, 3, 2]
    assert fit.ranking.tolist() == [0, 2, 1]


def test_unknown_tie_rule_is_rejected():
    arena = simulate(n_models=3, n_battles=200, seed=12)
    with pytest.raises(ValueError, match="tie rule"):
        fit_bt(arena.counts(), ties="ignore")  # type: ignore[arg-type]
