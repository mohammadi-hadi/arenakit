"""The probes, and the planted boards they have to keep finding."""

from __future__ import annotations

import numpy as np
import pytest

from arenakit.demo import DEMOS, check_demo
from arenakit.probes import (
    INNOCENT_UNRESOLVED,
    LOO_JUDGE_RANGE,
    MIN_BATTLES_PER_MODEL,
    _cyclic_share,
    _gini,
    run_audit,
)
from arenakit.result import ProbeResult
from arenakit.schema import Arena, Battle
from arenakit.synthetic import simulate

FAST = 400


def test_every_planted_board_still_behaves_as_planted():
    assert check_demo(n_resamples=FAST) == []


def test_the_demo_covers_every_probe():
    """No probe may ship without a planted board that makes it fire."""
    audit = run_audit(simulate(n_models=5, n_battles=800, seed=1), n_resamples=FAST)
    planted = {flag for demo in DEMOS for flag in demo.expect}
    assert planted == {probe.name for probe in audit.probes}


def test_a_clean_board_raises_nothing():
    arena = simulate(n_models=6, n_battles=6000, spread=1.2, seed=0)
    audit = run_audit(arena, n_resamples=FAST)
    assert audit.flags == []
    assert audit.n_models == 6


def test_gini_matches_hand_computed_cases():
    assert _gini(np.array([1.0, 1.0, 1.0, 1.0])) == pytest.approx(0.0)
    assert _gini(np.array([0.0, 0.0, 0.0, 4.0])) == pytest.approx(0.75)
    assert _gini(np.array([0.0, 0.0])) == 0.0


def test_cyclic_share_on_a_hand_built_tournament():
    # a beats b beats c beats a: one triad, and it is a cycle.
    wins = np.array([[0, 3, 0], [0, 0, 3], [3, 0, 0]], dtype=float)
    share, cycles, triads = _cyclic_share(wins)
    assert (share, cycles, triads) == (1.0, 1, 1)

    # a beats both, b beats c: one triad, transitive.
    wins = np.array([[0, 3, 3], [0, 0, 3], [0, 0, 0]], dtype=float)
    assert _cyclic_share(wins) == (0.0, 0, 1)


def test_disconnected_boards_are_audited_on_the_largest_component():
    arena = simulate(n_models=6, n_battles=3000, disconnect=True, seed=2)
    audit = run_audit(arena, n_resamples=FAST)
    connectivity = audit["connectivity"]
    assert connectivity.triggered
    assert connectivity.detail["n_components"] == 2
    assert len(connectivity.detail["dropped_from_audit"]) == 3
    assert audit.n_models == 6  # reported over the whole board
    assert len(audit.leaderboard) == 3  # ranked only where ranking means something


def test_under_compared_models_are_flagged_not_dropped():
    arena = simulate(n_models=5, n_battles=6000, imbalance=0.95, seed=3)
    audit = run_audit(arena, n_resamples=FAST)
    thin = audit["connectivity"].detail["under_compared"]
    assert thin
    assert audit["connectivity"].detail["min_battles"] < MIN_BATTLES_PER_MODEL
    assert {row["model"] for row in audit.leaderboard} >= set(thin)


def test_position_probes_skip_a_canonically_ordered_board():
    arena = simulate(n_models=5, n_battles=2000, seed=4)
    ordered = Arena(
        [
            Battle(
                model_a=arena.models[min(a, b)],
                model_b=arena.models[max(a, b)],
                winner="a" if y == 0 else ("b" if y == 1 else "tie"),
            )
            for a, b, y in zip(arena.a.tolist(), arena.b.tolist(), arena.y.tolist(), strict=True)
        ]
    )
    audit = run_audit(ordered, n_resamples=FAST)
    assert audit["position_bias"].skipped
    assert audit["position_reordering"].skipped
    assert "name-sorted" in audit["position_bias"].reason


def test_tie_probe_skips_a_board_without_ties():
    arena = simulate(n_models=5, n_battles=2000, tie_rate=0.0, seed=5)
    audit = run_audit(arena, n_resamples=FAST)
    assert audit["tie_convention"].skipped


def test_annotator_probe_skips_when_there_are_too_many_annotators():
    arena = simulate(n_models=5, n_battles=2000, n_voters=LOO_JUDGE_RANGE[1] + 50, seed=6)
    probe = run_audit(arena, n_resamples=FAST)["annotator_influence"]
    assert probe.skipped
    assert "not a perturbation" in probe.reason


def test_clustering_probe_skips_without_identifiers():
    arena = simulate(n_models=5, n_battles=1500, seed=7)
    arena.voter = None
    arena.item = None
    audit = run_audit(arena, n_resamples=FAST)
    assert audit["vote_clustering"].skipped
    assert audit["annotator_influence"].skipped


def test_audit_is_reproducible():
    arena = simulate(n_models=6, n_battles=2500, seed=8)
    first = run_audit(arena, n_resamples=FAST)
    second = run_audit(arena, n_resamples=FAST)
    assert first.to_dict() == second.to_dict()


def test_probe_result_triggers_only_outside_the_band():
    inside = ProbeResult(name="x", title="x", value=0.1, ci=(0.05, 0.2), innocent=(0.0, 0.25))
    straddling = ProbeResult(name="x", title="x", value=0.3, ci=(0.2, 0.4), innocent=(0.0, 0.25))
    outside = ProbeResult(name="x", title="x", value=0.5, ci=(0.4, 0.6), innocent=(0.0, 0.25))
    assert not inside.triggered
    assert not straddling.triggered  # the interval still touches the band
    assert outside.triggered
    assert ProbeResult(name="x", title="x", flagged=True).triggered
    assert not ProbeResult(name="x", title="x", flagged=True, skipped=True).triggered


def test_audit_lookup_and_serialisation():
    audit = run_audit(simulate(n_models=4, n_battles=800, seed=9), n_resamples=FAST)
    assert audit["separation"].name == "separation"
    with pytest.raises(KeyError):
        audit["no-such-probe"]
    payload = audit.to_dict()
    assert payload["n_models"] == 4
    assert payload["probes"][0]["innocent"] == list(INNOCENT_UNRESOLVED)
    assert [row["rank"] for row in payload["leaderboard"]] == [1, 2, 3, 4]
    assert len({row["model"] for row in payload["leaderboard"]}) == 4
    assert payload["flags"] == audit.flags
