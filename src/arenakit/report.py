"""Tables, figures and README injection.

Everything here reads an :class:`~arenakit.result.AuditResult` and nothing else,
so a report can be regenerated from a stored audit with no model fitting and no
network access.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .result import AuditResult

#: One accent colour, used for every figure.
ACCENT = "#0f766e"
MUTED = "#94a3b8"
WARN = "#b45309"

DEMO_START = "<!-- arenakit:demo -->"
DEMO_END = "<!-- /arenakit:demo -->"


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def demo_table(rows: list[dict[str, Any]]) -> str:
    """Markdown table of the planted-defect diagonal."""
    lines = [
        _row(["Board", "What was planted", "Probes that fired"]),
        _row(["---", "---", "---"]),
    ]
    for row in rows:
        flags = ", ".join(f"`{f}`" for f in row["flags"]) if row["flags"] else "none"
        lines.append(_row([row["title"], row["story"], flags]))
    return "\n".join(lines)


def leaderboard_table(audit: AuditResult, top: int | None = None) -> str:
    """Markdown table of the fitted board."""
    board = audit.leaderboard[:top] if top else audit.leaderboard
    lines = [
        _row(["Rank", "Model", "Score", "95% CI", "Rank interval", "Battles"]),
        _row(["---:", "---", "---:", "---", "---", "---:"]),
    ]
    for entry in board:
        low, high = entry["score_ci"]
        rank_low, rank_high = entry["rank_ci"]
        span = f"{rank_low}" if rank_low == rank_high else f"{rank_low}-{rank_high}"
        lines.append(
            _row(
                [
                    str(entry["rank"]),
                    entry["model"],
                    f"{entry['score']:+.3f}",
                    f"[{low:+.3f}, {high:+.3f}]",
                    span,
                    f"{entry['battles']:,}",
                ]
            )
        )
    return "\n".join(lines)


def probe_table(audit: AuditResult) -> str:
    """Markdown table of every probe and its verdict."""
    lines = [
        _row(["Probe", "Value", "Innocent band", "Verdict"]),
        _row(["---", "---:", "---", "---"]),
    ]
    for probe in audit.probes:
        if probe.skipped:
            value, band, verdict = "-", "-", "skipped"
        else:
            value = "-" if probe.value is None else f"{probe.value:.3f}"
            band = f"{probe.innocent[0]:g} to {probe.innocent[1]:g}" if probe.innocent else "exact"
            verdict = "**flag**" if probe.triggered else "clean"
        lines.append(_row([f"`{probe.name}`", value, band, verdict]))
    return "\n".join(lines)


def report_markdown(audit: AuditResult, title: str) -> str:
    """A full written report for one audited board."""
    parts = [
        f"# {title}",
        "",
        f"{audit.n_battles:,} battles over {audit.n_models} models. "
        f"{len(audit.triggered)} of {len(audit.probes)} probes raised a flag, "
        f"{len(audit.skipped)} did not apply.",
        "",
        "## Leaderboard",
        "",
        leaderboard_table(audit),
        "",
        "## Probes",
        "",
        probe_table(audit),
        "",
        "## What each probe found",
        "",
    ]
    for probe in audit.probes:
        mark = "flag" if probe.triggered else ("skipped" if probe.skipped else "clean")
        parts.append(f"- **{probe.title}** (`{probe.name}`, {mark}) - {probe.reason}")
    return "\n".join(parts) + "\n"


def board_figure(audit: AuditResult, path: str | Path) -> Path:
    """Scores with intervals, shaded by the tiers the data can separate."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    board = audit.leaderboard
    names = [entry["model"] for entry in board]
    scores = [entry["score"] for entry in board]
    low = [entry["score"] - entry["score_ci"][0] for entry in board]
    high = [entry["score_ci"][1] - entry["score"] for entry in board]
    positions = list(range(len(board)))[::-1]

    tiers = audit["separation"].detail.get("tiers", [])
    tier_of = {model: i for i, tier in enumerate(tiers) for model in tier}

    height = max(2.6, 0.32 * len(board) + 1.2)
    figure, axes = plt.subplots(figsize=(7.2, height))
    for index, name in enumerate(names):
        shade = tier_of.get(name, 0) % 2
        if shade:
            axes.axhspan(positions[index] - 0.5, positions[index] + 0.5, color=MUTED, alpha=0.13)
    axes.errorbar(
        scores,
        positions,
        xerr=[low, high],
        fmt="o",
        color=ACCENT,
        ecolor=ACCENT,
        elinewidth=1.4,
        capsize=3,
        markersize=4,
    )
    axes.set_yticks(positions)
    axes.set_yticklabels(names, fontsize=8)
    axes.set_xlabel("Bradley-Terry score (log-odds), 95% interval")
    axes.set_title(
        f"{len(tiers)} tier(s) the votes can separate, out of {len(board)} models", fontsize=10
    )
    axes.grid(axis="x", alpha=0.25)
    for spine in ("top", "right"):
        axes.spines[spine].set_visible(False)
    figure.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def separation_figure(audit: AuditResult, path: str | Path) -> Path:
    """Every adjacent gap against the evidence needed to call it real."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    detail = audit["separation"].detail
    pairs = detail.get("pairs", [])
    if not pairs:
        raise ValueError("no adjacent pairs to plot")
    labels = [f"{p['upper']} / {p['lower']}" for p in pairs]
    values = [p["t"] for p in pairs]
    crit_sim = detail["critical_simultaneous"]
    marginal = [p["critical_marginal"] for p in pairs]
    positions = list(range(len(pairs)))[::-1]
    colours = [ACCENT if p["resolved_simultaneous"] else WARN for p in pairs]

    height = max(2.6, 0.32 * len(pairs) + 1.4)
    figure, axes = plt.subplots(figsize=(7.2, height))
    axes.barh(positions, values, color=colours, height=0.6)
    axes.plot(
        marginal,
        positions,
        linestyle=":",
        color="#334155",
        linewidth=1.2,
        label="needed on its own",
    )
    axes.axvline(crit_sim, color="#334155", linewidth=1.2, label="needed across the whole board")
    axes.set_xlim(0, max(max(values), crit_sim) * 1.35)
    axes.set_yticks(positions)
    axes.set_yticklabels(labels, fontsize=8)
    axes.set_xlabel("gap between neighbours, in standard errors")
    axes.set_title(
        f"{detail['n_resolved_simultaneous']} of {detail['n_pairs']} neighbouring pairs separated",
        fontsize=10,
    )
    axes.legend(fontsize=8, frameon=False, loc="lower right")
    axes.grid(axis="x", alpha=0.25)
    for spine in ("top", "right"):
        axes.spines[spine].set_visible(False)
    figure.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def inject(path: str | Path, body: str, start: str = DEMO_START, end: str = DEMO_END) -> None:
    """Replace the text between two markers in a file."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if start not in text or end not in text:
        raise ValueError(f"{path} has no {start} ... {end} block")
    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    path.write_text(f"{head}{start}\n{body}\n{end}{tail}", encoding="utf-8")


def write_json(audit: AuditResult, path: str | Path) -> Path:
    """Store an audit as JSON, without the objects that cannot be serialised."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(audit.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
