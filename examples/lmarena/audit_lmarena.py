"""Audit 106,134 real LMArena votes over 55 models.

Run ``fetch_data.py`` first.

Two decisions are made here rather than in the library, because both change the
answer and neither has an obviously right setting:

* The dump distinguishes "tie" from "tie (bothbad)". They are collapsed into one
  tie outcome for the main audit, which is what the standard Bradley-Terry
  treatment of arena data does. ``tie_convention`` then reports what the ranking
  would look like under two other conventions.
* ``judge_hash`` identifies the voter, so the bootstrap resamples voters. The
  same audit is repeated resampling battles and prompts, and the three are
  compared, because the difference between them is the point of the exercise.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from arenakit.boot import N_RESAMPLES, SEED, bootstrap, separation
from arenakit.probes import run_audit
from arenakit.report import board_figure, report_markdown, separation_figure, write_json
from arenakit.schema import Arena, Battle

HERE = Path(__file__).parent
DATA = HERE / "data" / "battles.csv"
RESULTS = HERE / "results"

#: Both tie labels are treated as one outcome. See the module docstring.
WINNERS = {
    "model_a": "a",
    "model_b": "b",
    "tie": "tie",
    "tie (bothbad)": "tie",
}


def load() -> tuple[Arena, dict[str, int]]:
    if not DATA.exists():
        raise SystemExit(f"{DATA} is missing - run examples/lmarena/fetch_data.py first")
    battles, raw_outcomes = [], {}
    with DATA.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            raw_outcomes[row["winner"]] = raw_outcomes.get(row["winner"], 0) + 1
            battles.append(
                Battle(
                    model_a=row["model_a"],
                    model_b=row["model_b"],
                    winner=WINNERS[row["winner"]],
                    voter_id=row["voter_id"],
                    item_id=row["item_id"],
                )
            )
    return Arena(battles), raw_outcomes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resamples", type=int, default=N_RESAMPLES)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    arena, raw_outcomes = load()
    total = sum(raw_outcomes.values())
    print(f"{arena.n_battles:,} battles, {arena.n_models} models")
    for label, count in sorted(raw_outcomes.items(), key=lambda kv: -kv[1]):
        print(f"  {label:<18} {count:>7,} ({count / total:.1%})")

    audit = run_audit(arena, n_resamples=args.resamples, seed=args.seed)

    # The same board under each resampling scheme, so the cost of assuming
    # independence is visible rather than argued about.
    schemes = {}
    for cluster in ("battle", "item", "voter"):
        boot = bootstrap(arena, n_resamples=args.resamples, seed=args.seed, cluster=cluster)
        sep = separation(boot, arena.models)
        schemes[cluster] = {
            "n_clusters": boot.n_clusters,
            "mean_se": float(boot.scores.std(axis=0, ddof=1).mean()),
            "separated": sep.n_simultaneous,
            "separated_marginal": sep.n_marginal,
            "pairs": sep.n_pairs,
            "tiers": len(sep.tiers),
        }
        note = (
            " (every battle has its own prompt id, so this is battle resampling)"
            if cluster == "item" and boot.n_clusters == arena.n_battles
            else ""
        )
        schemes[cluster]["note"] = note.strip()
        print(
            f"  resampling {cluster:<7} {boot.n_clusters:>7,} units  "
            f"mean se {schemes[cluster]['mean_se']:.4f}  "
            f"separated {sep.n_simultaneous}/{sep.n_pairs}{note}"
        )

    RESULTS.mkdir(parents=True, exist_ok=True)
    write_json(audit, RESULTS / "lmarena.json")
    (RESULTS / "resampling.json").write_text(
        json.dumps(schemes, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        report_markdown(audit, "LMArena, 106k public votes"),
        "",
        "## What the votes were",
        "",
        "| Outcome | Battles | Share |",
        "| --- | ---: | ---: |",
    ]
    for label, count in sorted(raw_outcomes.items(), key=lambda kv: -kv[1]):
        lines.append(f"| `{label}` | {count:,} | {count / total:.1%} |")
    lines += [
        "",
        "## What it costs to assume votes are independent",
        "",
        "| Resampling | Units | Mean standard error | Adjacent pairs separated | |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for cluster in ("battle", "item", "voter"):
        row = schemes[cluster]
        lines.append(
            f"| {cluster} | {row['n_clusters']:,} | {row['mean_se']:.4f} | "
            f"{row['separated']} of {row['pairs']} | {row['note']} |"
        )
    (RESULTS / "lmarena.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    board_figure(audit, RESULTS / "figures" / "board.png")
    separation_figure(audit, RESULTS / "figures" / "separation.png")
    print(f"\nwrote {RESULTS}/lmarena.md, lmarena.json, resampling.json and 2 figures")


if __name__ == "__main__":
    main()
