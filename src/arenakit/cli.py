"""Command line entry point.

    arenakit audit battles.csv
    arenakit demo --out results

``audit`` reads battles from CSV or JSONL and prints the board, the probes and
what each one found. ``demo`` regenerates the planted-defect diagonal and the
committed results, and fails if any board stops behaving as planted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TextIO

from .boot import ALPHA, N_RESAMPLES, SEED
from .demo import DEMOS, run_demo
from .probes import run_audit
from .report import (
    board_figure,
    demo_table,
    report_markdown,
    separation_figure,
    write_json,
)
from .result import AuditResult
from .schema import Arena, load_battles


def _print_audit(audit: AuditResult, stream: TextIO = sys.stdout) -> None:
    print(
        f"{audit.n_battles:,} battles, {audit.n_models} models, {len(audit.triggered)} flag(s)",
        file=stream,
    )
    print(file=stream)
    for entry in audit.leaderboard:
        low, high = entry["score_ci"]
        rank_low, rank_high = entry["rank_ci"]
        span = f"{rank_low}" if rank_low == rank_high else f"{rank_low}-{rank_high}"
        print(
            f"  {entry['rank']:>3}  {entry['model']:<28} {entry['score']:+.3f}  "
            f"[{low:+.3f}, {high:+.3f}]  rank {span}",
            file=stream,
        )
    print(file=stream)
    for probe in audit.probes:
        mark = "FLAG" if probe.triggered else ("skip" if probe.skipped else "  ok")
        print(f"  {mark}  {probe.name:<22} {probe.reason}", file=stream)


def _audit_command(args: argparse.Namespace) -> int:
    arena = Arena(load_battles(args.battles))
    audit = run_audit(
        arena,
        n_resamples=args.resamples,
        seed=args.seed,
        cluster=args.cluster,
        alpha=args.alpha,
    )
    _print_audit(audit)
    if args.json:
        write_json(audit, args.json)
        print(f"\nwrote {args.json}")
    if args.markdown:
        Path(args.markdown).write_text(
            report_markdown(audit, Path(args.battles).stem), encoding="utf-8"
        )
        print(f"wrote {args.markdown}")
    return 1 if audit.triggered and args.strict else 0


def _demo_command(args: argparse.Namespace) -> int:
    rows = run_demo(n_resamples=args.resamples, seed=args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "table.md").write_text(demo_table(rows) + "\n", encoding="utf-8")

    lines = ["# arenakit demo boards", ""]
    failures = []
    for row, demo in zip(rows, DEMOS, strict=True):
        status = "as planted" if row["matches"] else "MISMATCH"
        lines += [
            f"## {row['title']}",
            "",
            f"{row['story']}",
            "",
            f"- planted: `{'`, `'.join(row['expect']) if row['expect'] else 'nothing'}`",
            f"- fired: `{'`, `'.join(row['flags']) if row['flags'] else 'nothing'}` ({status})",
            "",
        ]
        for name, reason in row["reasons"].items():
            lines.append(f"  - **{name}** - {reason}")
        lines.append("")
        if not row["matches"]:
            failures.append(f"{demo.key}: expected {row['expect']}, got {row['flags']}")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")

    summary = {
        row["key"]: {
            "expect": row["expect"],
            "flags": row["flags"],
            "matches": row["matches"],
        }
        for row in rows
    }
    (out / "demo.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # Figures come from the thin board: it is the one with something to show,
    # since a board with enough votes separates every neighbour and the plot
    # says only that.
    thin = next(row for row in rows if row["key"] == "too-few-votes")["audit"]
    board_figure(thin, out / "figures" / "board.png")
    separation_figure(thin, out / "figures" / "separation.png")

    for failure in failures:
        print(f"MISMATCH {failure}", file=sys.stderr)
    print(f"wrote {out}/table.md, report.md, demo.json and 2 figures")
    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="arenakit", description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="audit a file of battles")
    audit.add_argument("battles", help="CSV or JSONL with model_a, model_b, winner")
    audit.add_argument("--resamples", type=int, default=N_RESAMPLES)
    audit.add_argument("--seed", type=int, default=SEED)
    audit.add_argument("--alpha", type=float, default=ALPHA)
    audit.add_argument(
        "--cluster",
        choices=("battle", "item", "voter"),
        default=None,
        help="what to resample; defaults to the most conservative the data allows",
    )
    audit.add_argument("--json", help="write the full audit here")
    audit.add_argument("--markdown", help="write a written report here")
    audit.add_argument("--strict", action="store_true", help="exit non-zero when any probe fires")
    audit.set_defaults(handler=_audit_command)

    demo = subparsers.add_parser("demo", help="regenerate the planted-defect boards")
    demo.add_argument("--out", default="results")
    demo.add_argument("--resamples", type=int, default=N_RESAMPLES)
    demo.add_argument("--seed", type=int, default=SEED)
    demo.set_defaults(handler=_demo_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
