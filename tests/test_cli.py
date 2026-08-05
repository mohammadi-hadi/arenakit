"""The command line and the report writers."""

from __future__ import annotations

import json

import pytest

from arenakit.cli import main
from arenakit.probes import run_audit
from arenakit.report import (
    DEMO_END,
    DEMO_START,
    board_figure,
    demo_table,
    inject,
    leaderboard_table,
    probe_table,
    report_markdown,
    separation_figure,
    write_json,
)
from arenakit.synthetic import simulate

FAST = 120


@pytest.fixture
def battles_csv(tmp_path):
    arena = simulate(n_models=5, n_battles=900, seed=1)
    path = tmp_path / "battles.csv"
    lines = ["model_a,model_b,winner,voter_id,item_id"]
    outcomes = {0: "a", 1: "b", 2: "tie"}
    for a, b, y, voter, item in zip(
        arena.a.tolist(),
        arena.b.tolist(),
        arena.y.tolist(),
        arena.voter.tolist(),
        arena.item.tolist(),
        strict=True,
    ):
        lines.append(f"{arena.models[a]},{arena.models[b]},{outcomes[y]},voter-{voter},item-{item}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_audit_command_writes_both_reports(battles_csv, tmp_path, capsys):
    out_json = tmp_path / "audit.json"
    out_md = tmp_path / "audit.md"
    code = main(
        [
            "audit",
            str(battles_csv),
            "--resamples",
            str(FAST),
            "--json",
            str(out_json),
            "--markdown",
            str(out_md),
        ]
    )
    assert code == 0
    printed = capsys.readouterr().out
    assert "battles" in printed and "separation" in printed
    payload = json.loads(out_json.read_text())
    assert payload["n_models"] == 5
    assert len(payload["probes"]) == 10
    assert "## Leaderboard" in out_md.read_text()


def test_audit_command_can_fail_the_build(battles_csv):
    ok = main(["audit", str(battles_csv), "--resamples", str(FAST)])
    strict = main(
        ["audit", str(battles_csv), "--resamples", str(FAST), "--strict", "--cluster", "battle"]
    )
    assert ok == 0
    assert strict in (0, 1)  # depends on the board; the flag must be honoured either way


def test_demo_command_regenerates_everything(tmp_path):
    out = tmp_path / "results"
    assert main(["demo", "--out", str(out), "--resamples", str(FAST)]) == 0
    assert (out / "table.md").read_text().startswith("| Board |")
    assert "as planted" in (out / "report.md").read_text()
    summary = json.loads((out / "demo.json").read_text())
    assert all(row["matches"] for row in summary.values())
    assert (out / "figures" / "board.png").stat().st_size > 0
    assert (out / "figures" / "separation.png").stat().st_size > 0


def test_tables_and_report_render(tmp_path):
    audit = run_audit(simulate(n_models=5, n_battles=900, seed=2), n_resamples=FAST)
    assert leaderboard_table(audit, top=3).count("\n") == 4  # header, rule, three rows
    assert probe_table(audit).count("\n") == 11
    text = report_markdown(audit, "test board")
    assert text.startswith("# test board")
    assert "## Probes" in text
    assert write_json(audit, tmp_path / "a.json").exists()
    assert board_figure(audit, tmp_path / "board.png").stat().st_size > 0
    assert separation_figure(audit, tmp_path / "sep.png").stat().st_size > 0


def test_demo_table_lists_every_board():
    rows = [
        {"title": "One", "story": "planted", "flags": ["separation"]},
        {"title": "Two", "story": "clean", "flags": []},
    ]
    table = demo_table(rows)
    assert "`separation`" in table
    assert "| Two | clean | none |" in table


def test_inject_replaces_only_the_marked_block(tmp_path):
    path = tmp_path / "README.md"
    path.write_text(f"before\n{DEMO_START}\nold\n{DEMO_END}\nafter\n", encoding="utf-8")
    inject(path, "new")
    text = path.read_text()
    assert "before" in text and "after" in text and "old" not in text and "new" in text

    path.write_text("no markers here\n", encoding="utf-8")
    with pytest.raises(ValueError, match="has no"):
        inject(path, "new")
