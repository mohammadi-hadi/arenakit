"""Loading battles and indexing them."""

from __future__ import annotations

import json

import numpy as np
import pytest
from pydantic import ValidationError

from arenakit.schema import Arena, Battle, load_battles
from arenakit.synthetic import simulate


def _arena(rows):
    return Arena([Battle(model_a=a, model_b=b, winner=w) for a, b, w in rows])


def test_winner_aliases_are_accepted():
    for value in ("a", "A", "model_a", "left", " Model_A "):
        assert Battle(model_a="x", model_b="y", winner=value).winner == "a"
    for value in ("tie", "draw", "TIE"):
        assert Battle(model_a="x", model_b="y", winner=value).winner == "tie"


def test_unmapped_tie_variants_are_refused_with_advice():
    with pytest.raises(ValidationError, match="will not guess"):
        Battle(model_a="x", model_b="y", winner="tie (bothbad)")


def test_self_play_and_empty_names_are_refused():
    with pytest.raises(ValueError, match="against itself"):
        _arena([("x", "x", "a")])
    with pytest.raises(ValidationError):
        Battle(model_a=" ", model_b="y", winner="a")


def test_counts_are_the_sufficient_statistic():
    arena = _arena([("a", "b", "a"), ("a", "b", "b"), ("b", "a", "a"), ("a", "b", "tie")])
    counts = arena.counts()
    ia, ib = arena.models.index("a"), arena.models.index("b")
    # a shown first three times: one win, one loss, one tie.
    assert counts.wf[ia, ib] == 1
    assert counts.ws[ia, ib] == 1
    assert counts.tt[ia, ib] == 1
    # b shown first once and won it, which is a loss for a.
    assert counts.wf[ib, ia] == 1
    assert counts.wins[ia, ib] == 1  # a beat b once, in either position
    assert counts.wins[ib, ia] == 2
    assert counts.ties[ia, ib] == counts.ties[ib, ia] == 1
    assert counts.comparisons[ia, ib] == 4
    assert counts.n_battles == 4


def test_order_randomisation_spots_a_canonical_assignment():
    randomised = simulate(n_models=5, n_battles=800, seed=1).order_randomization()
    assert randomised.looks_positional
    assert 0.4 < randomised.forward_share < 0.6

    canonical = _arena([("a", "b", "a"), ("a", "c", "b"), ("b", "c", "a")] * 30)
    evidence = canonical.order_randomization()
    assert evidence.forward_share == 1.0
    assert not evidence.looks_positional


def test_components_and_restriction():
    arena = _arena([("a", "b", "a")] * 5 + [("c", "d", "a")] * 5)
    groups = arena.components()
    assert [len(group) for group in groups] == [2, 2]

    kept = arena.restrict(groups[0])
    assert kept.n_models == 2
    assert kept.n_battles == 5
    assert set(kept.models) <= set(arena.models)
    with pytest.raises(ValueError, match="empty"):
        arena.restrict([])


def test_subset_keeps_the_model_index():
    arena = simulate(n_models=5, n_battles=400, seed=2)
    mask = arena.voter != 0
    smaller = arena.subset(mask)
    assert smaller.models == arena.models
    assert smaller.n_battles == int(mask.sum())
    with pytest.raises(ValueError, match="no battles"):
        arena.subset(np.zeros(arena.n_battles, dtype=bool))


def test_battles_per_model_counts_both_slots():
    arena = _arena([("a", "b", "a"), ("b", "c", "a"), ("a", "c", "b")])
    counts = dict(zip(arena.models, arena.battles_per_model().tolist(), strict=True))
    assert counts == {"a": 2, "b": 2, "c": 2}


def test_models_missing_from_the_index_are_refused():
    battles = [Battle(model_a="a", model_b="b", winner="a")]
    with pytest.raises(ValueError, match="missing from"):
        Arena(battles, models=["a"])
    with pytest.raises(ValueError, match="at least one battle"):
        Arena([])


def test_round_trip_through_csv_and_jsonl(tmp_path):
    rows = [
        {"model_a": "a", "model_b": "b", "winner": "model_a", "voter_id": "v1"},
        {"model_a": "b", "model_b": "c", "winner": "tie", "voter_id": "v2"},
    ]
    csv_path = tmp_path / "battles.csv"
    csv_path.write_text(
        "model_a,model_b,winner,voter_id\n"
        + "\n".join(f"{r['model_a']},{r['model_b']},{r['winner']},{r['voter_id']}" for r in rows)
        + "\n",
        encoding="utf-8",
    )
    jsonl_path = tmp_path / "battles.jsonl"
    jsonl_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    for path in (csv_path, jsonl_path):
        loaded = load_battles(path)
        assert [b.winner for b in loaded] == ["a", "tie"]
        assert [b.voter_id for b in loaded] == ["v1", "v2"]

    with pytest.raises(ValueError, match="unsupported"):
        load_battles(tmp_path / "battles.parquet")


def test_optional_columns_are_all_or_nothing(tmp_path):
    path = tmp_path / "battles.csv"
    path.write_text("model_a,model_b,winner,voter_id\na,b,a,v1\nb,c,b,\n", encoding="utf-8")
    arena = Arena(load_battles(path))
    assert arena.voter is None  # one row lacks it, so it cannot be clustered on
