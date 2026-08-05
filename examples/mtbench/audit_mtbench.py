"""Two leaderboards over the same six models: one built by people, one by GPT-4.

Run ``fetch_data.py`` first.

The MT-Bench judgment set is unusual and useful. The same model pairs on the
same questions were judged by 65 people and, separately, by GPT-4 acting as a
judge. That makes it possible to audit the judge's leaderboard the same way as
the humans' and put the two next to each other.

The GPT-4 split was collected by asking twice, once in each presentation order.
Where the two answers disagreed, the record says ``tie (inconsistent)``. That
label is a direct measurement of how often swapping the two answers changes the
judge's mind, so it is counted here rather than quietly folded into ties.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from arenakit.boot import N_RESAMPLES, SEED
from arenakit.probes import run_audit
from arenakit.report import board_figure, report_markdown, write_json
from arenakit.schema import Arena, Battle

HERE = Path(__file__).parent
DATA = HERE / "data"
RESULTS = HERE / "results"

WINNERS = {
    "model_a": "a",
    "model_b": "b",
    "tie": "tie",
    # GPT-4 gave opposite answers depending on which response came first.
    "tie (inconsistent)": "tie",
}


def load(split: str) -> tuple[list[Battle], Counter]:
    path = DATA / f"{split}.csv"
    if not path.exists():
        raise SystemExit(f"{path} is missing - run examples/mtbench/fetch_data.py first")
    battles, labels = [], Counter()
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            labels[row["winner"]] += 1
            battles.append(
                Battle(
                    model_a=row["model_a"],
                    model_b=row["model_b"],
                    winner=WINNERS[row["winner"]],
                    voter_id=row["voter_id"],
                    item_id=row["item_id"],
                )
            )
    return battles, labels


def kendall_tau(first: list[str], second: list[str]) -> float:
    """Rank correlation between two orderings of the same models."""
    rank = {name: i for i, name in enumerate(second)}
    order = [rank[name] for name in first]
    concordant = discordant = 0
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            if order[i] < order[j]:
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total else 0.0


def judge_agreement(human: list[Battle], judge: list[Battle]) -> dict[str, float | int]:
    """How often GPT-4 lands on the human majority verdict for the same comparison."""
    votes: dict[tuple[str, str, str], Counter] = defaultdict(Counter)
    for battle in human:
        pair = tuple(sorted((battle.model_a, battle.model_b)))
        winner = (
            "tie"
            if battle.winner == "tie"
            else (battle.model_a if battle.winner == "a" else battle.model_b)
        )
        votes[(battle.item_id or "", *pair)][winner] += 1

    matched = agree = 0
    for battle in judge:
        pair = tuple(sorted((battle.model_a, battle.model_b)))
        key = (battle.item_id or "", *pair)
        tally = votes.get(key)
        if not tally:
            continue
        top, count = tally.most_common(1)[0]
        if sum(v == count for v in tally.values()) > 1:
            continue  # the humans were split, so there is no majority to match
        matched += 1
        verdict = (
            "tie"
            if battle.winner == "tie"
            else (battle.model_a if battle.winner == "a" else battle.model_b)
        )
        agree += verdict == top
    return {
        "comparisons": matched,
        "agreements": agree,
        "rate": agree / matched if matched else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resamples", type=int, default=N_RESAMPLES)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    boards, audits, summary = {}, {}, {}
    for split, title in (("human", "65 human annotators"), ("gpt4_pair", "GPT-4 as judge")):
        battles, labels = load(split)
        arena = Arena(battles)
        audit = run_audit(arena, n_resamples=args.resamples, seed=args.seed)
        audits[split] = audit
        boards[split] = [row["model"] for row in audit.leaderboard]
        inconsistent = labels.get("tie (inconsistent)", 0)
        summary[split] = {
            "votes": arena.n_battles,
            "annotators": len(set(b.voter_id for b in battles)),
            "labels": dict(labels),
            "order_inconsistent": inconsistent,
            "order_inconsistent_share": inconsistent / arena.n_battles,
            "flags": audit.flags,
            "separated": audit["separation"].detail["n_resolved_simultaneous"],
            "pairs": audit["separation"].detail["n_pairs"],
        }
        print(f"{title}: {arena.n_battles:,} votes, flags {audit.flags}")
        if inconsistent:
            print(
                f"  swapping the two answers changes the verdict on "
                f"{inconsistent:,} of {arena.n_battles:,} pairs "
                f"({inconsistent / arena.n_battles:.1%})"
            )
        write_json(audit, RESULTS / f"{split}.json")
        board_figure(audit, RESULTS / "figures" / f"{split}.png")

    human_battles, _ = load("human")
    judge_battles, _ = load("gpt4_pair")
    agreement = judge_agreement(human_battles, judge_battles)
    tau = kendall_tau(boards["human"], boards["gpt4_pair"])
    summary["comparison"] = {
        "kendall_tau": tau,
        "human_order": boards["human"],
        "gpt4_order": boards["gpt4_pair"],
        "vote_agreement": agreement,
    }
    print(
        f"\nthe two leaderboards agree at tau = {tau:+.2f}; on the same comparisons "
        f"GPT-4 matches the human majority {agreement['rate']:.1%} of the time "
        f"({agreement['agreements']:,} of {agreement['comparisons']:,})"
    )
    (RESULTS / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# MT-Bench: the same six models judged by people and by GPT-4",
        "",
        f"Human votes: {summary['human']['votes']:,} from "
        f"{summary['human']['annotators']} annotators. "
        f"GPT-4 verdicts: {summary['gpt4_pair']['votes']:,}.",
        "",
        "## The two leaderboards",
        "",
        "| Rank | Human annotators | GPT-4 as judge |",
        "| ---: | --- | --- |",
    ]
    for index, (left, right) in enumerate(
        zip(boards["human"], boards["gpt4_pair"], strict=True), start=1
    ):
        mark = "" if left == right else "  <-"
        lines.append(f"| {index} | {left} | {right}{mark} |")
    lines += [
        "",
        f"Kendall tau between the two orders: **{tau:+.2f}**. On the "
        f"{agreement['comparisons']:,} comparisons where the humans reached a majority, "
        f"GPT-4 agreed **{agreement['rate']:.1%}** of the time.",
        "",
        "## What the judge's own inconsistency looks like",
        "",
        f"The GPT-4 split records a verdict of `tie (inconsistent)` when asking in the "
        f"other order produced the opposite answer. That happened on "
        f"**{summary['gpt4_pair']['order_inconsistent']:,} of "
        f"{summary['gpt4_pair']['votes']:,}** pairs "
        f"(**{summary['gpt4_pair']['order_inconsistent_share']:.1%}**).",
        "",
        "## Audits",
        "",
    ]
    for split, title in (("human", "Human annotators"), ("gpt4_pair", "GPT-4 as judge")):
        lines += [report_markdown(audits[split], title), ""]
    (RESULTS / "mtbench.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {RESULTS}/mtbench.md, per-split JSON and 2 figures")


if __name__ == "__main__":
    main()
