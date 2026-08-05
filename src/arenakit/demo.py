"""Nine boards, one planted defect each, and the flags they must produce.

This is the package's own regression test against wishful thinking. Each row
below builds a leaderboard with a known problem - or, in the first row, no
problem at all - and records which probes fire. The expectations are asserted,
so a change that makes a probe more eager shows up here as a clean board that
suddenly raises a flag, and a change that makes one blind shows up as a planted
defect nobody noticed.

Where a row expects more than one flag, that is not slack. Sparse votes really
do make ranks volatile as well as unresolved; opinionated heavy voters really do
widen every interval. A defect that only ever produced one symptom would be a
tidier demo and a less honest one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .boot import N_RESAMPLES, SEED
from .probes import run_audit
from .schema import Arena
from .synthetic import simulate

#: Shared settings. Six models spread over 2.4 log-odds, ten percent ties, and
#: enough votes that a clean board has no excuse to raise a flag.
BASE: dict[str, Any] = {
    "n_models": 6,
    "n_battles": 6000,
    "spread": 1.2,
    "tie_rate": 0.10,
}


@dataclass(frozen=True)
class DemoRow:
    """One planted board."""

    key: str
    title: str
    story: str
    expect: tuple[str, ...]
    dials: dict[str, Any] = field(default_factory=dict)

    def build(self, seed: int = SEED) -> Arena:
        return simulate(seed=seed, **{**BASE, **self.dials})


DEMOS: tuple[DemoRow, ...] = (
    DemoRow(
        key="even-board",
        title="Even board",
        story="Enough votes, spread evenly, no artefacts planted.",
        expect=(),
    ),
    DemoRow(
        key="too-few-votes",
        title="Too few votes",
        story="The same board on 200 votes instead of 6000.",
        expect=("separation", "rank_volatility"),
        dials={"n_battles": 200},
    ),
    DemoRow(
        key="first-slot-favoured",
        title="First slot favoured",
        story=(
            "Being shown first is worth real odds, and one model takes that slot almost always."
        ),
        expect=("position_bias", "position_reordering"),
        dials={
            "position_bias": 0.9,
            "order_skew": 0.98,
            "skewed_model": 2,
            "skill_offsets": [0, 0, 0.33, 0, 0, 0],
        },
    ),
    DemoRow(
        key="draw-prone-model",
        title="Draw-prone model",
        story="One weak model draws most of its battles, and half a win each is generous.",
        expect=("tie_convention",),
        dials={"hedger": 4, "hedger_tie_rate": 0.85},
    ),
    DemoRow(
        key="rock-paper-scissors",
        title="Rock, paper, scissors",
        story="Three models beat each other in a circle, which no single strength explains.",
        expect=("intransitivity",),
        dials={"cycle": True, "cycle_size": 3},
    ),
    DemoRow(
        key="two-separate-pools",
        title="Two separate pools",
        story="Two halves of the board that never met, ranked as though they had.",
        expect=("connectivity",),
        dials={"disconnect": True},
    ),
    DemoRow(
        key="lopsided-budget",
        title="Lopsided budget",
        story="The comparison budget is spent mostly on the models at the top.",
        expect=("coverage_imbalance",),
        dials={"imbalance": 0.6},
    ),
    DemoRow(
        key="a-few-power-voters",
        title="A few power voters",
        story="Six people with strong tastes cast a third of all votes.",
        expect=("separation", "vote_clustering"),
        dials={
            "n_voters": 300,
            "heavy_voters": 6,
            "heavy_share": 0.35,
            "heavy_sigma": 0.8,
            "voter_sigma": 0.15,
        },
    ),
    DemoRow(
        key="one-loud-annotator",
        title="One loud annotator",
        story="Two equally good models, and one annotator with a firm view about them.",
        expect=("annotator_influence", "separation", "vote_clustering"),
        dials={
            "n_voters": 40,
            "skill_offsets": [0, 0, 0, 0.48, 0, 0],
            "loud_offset": (3, 2, 4.0),
            "loud_share": 0.35,
        },
    ),
)


def run_demo(n_resamples: int = N_RESAMPLES, seed: int = SEED) -> list[dict[str, Any]]:
    """Audit every planted board and collect what fired."""
    rows = []
    for demo in DEMOS:
        audit = run_audit(demo.build(seed), n_resamples=n_resamples, seed=seed)
        rows.append(
            {
                "key": demo.key,
                "title": demo.title,
                "story": demo.story,
                "expect": list(demo.expect),
                "flags": sorted(audit.flags),
                "matches": sorted(audit.flags) == sorted(demo.expect),
                "n_battles": audit.n_battles,
                "n_models": audit.n_models,
                "reasons": {p.name: p.reason for p in audit.triggered},
                "skipped": {p.name: p.reason for p in audit.skipped},
                "audit": audit,
            }
        )
    return rows


def check_demo(n_resamples: int = N_RESAMPLES, seed: int = SEED) -> list[str]:
    """Return a description of every row whose flags are not exactly as planted."""
    problems = []
    for row in run_demo(n_resamples=n_resamples, seed=seed):
        if not row["matches"]:
            problems.append(f"{row['key']}: expected {row['expect']}, got {row['flags']}")
    return problems
