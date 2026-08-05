"""Ten checks on a pairwise leaderboard, and the audit that runs them.

Each probe answers one question a leaderboard invites and rarely answers:

===========================  ==================================================
``separation``               how much of the published order the votes support
``rank_volatility``          how far each model's rank moves under resampling
``connectivity``             whether the scores are comparable at all
``coverage_imbalance``       whether some models were compared far less
``position_bias``            whether being shown first is worth anything
``position_reordering``      whether correcting for position changes the order
``tie_convention``           whether the order survives a different tie rule
``intransitivity``           whether Bradley-Terry fits the preference pattern
``vote_clustering``          whether independent resampling was ever justified
``annotator_influence``      whether one annotator moves the board on their own
===========================  ==================================================

Thresholds live here as named constants so the README can point at the source
rather than restate numbers that might drift away from it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .boot import (
    ALPHA,
    N_RESAMPLES,
    SEED,
    BootstrapResult,
    Cluster,
    Separation,
    battles_needed,
    bootstrap,
    bootstrap_values,
    percentile_ci,
    separation,
)
from .bt import PRIOR_BATTLES, BTFit, fit_bt, fit_rao_kupper, to_elo
from .result import AuditResult, ProbeResult
from .schema import Arena, PairCounts

#: A model compared fewer times than this has a score, but not a rank anyone
#: should quote.
MIN_BATTLES_PER_MODEL = 30
#: Adjacent rank gaps that simultaneous inference cannot separate. Some
#: unresolved neighbours are normal on a crowded board; a quarter of them is a
#: leaderboard reporting more order than it measured.
INNOCENT_UNRESOLVED = (0.0, 0.25)
#: Median width of a model's 95% rank interval, in ranks.
INNOCENT_RANK_WIDTH = (0.0, 2.0)
#: Gini coefficient of battles per model.
INNOCENT_IMBALANCE = (0.0, 0.5)
#: Log-odds advantage of being shown first. 0.05 is a 1.2 point swing in
#: win rate at even strength - below anything a leaderboard reports.
INNOCENT_POSITION = (-0.05, 0.05)
#: Observed cyclic-triad share minus what the fitted model predicts. One-sided
#: on purpose: a board with *fewer* cycles than Bradley-Terry expects is one the
#: model describes comfortably, which is not a finding about the ranking.
INNOCENT_CYCLE_EXCESS = (-1.0, 0.02)
#: Ratio of clustered to battle-level standard errors. Above this, independent
#: resampling was materially too optimistic.
INNOCENT_CLUSTER_INFLATION = (0.0, 1.15)
#: Below this many ties the tie convention cannot matter.
MIN_TIE_SHARE = 0.01
#: Largest score shift from dropping one annotator, in units of the median gap
#: between neighbouring models. At 1.0 one person's absence moves a model a
#: whole rung.
INNOCENT_ANNOTATOR_SHIFT = (0.0, 1.0)
#: Leaving out one annotator is only informative in this range: too few and
#: dropping one is not a perturbation, too many and it is not one either.
LOO_JUDGE_RANGE = (3, 200)
#: Resamples used for the parametric check of the tie/cycle structure.
PARAMETRIC_RESAMPLES = 500
#: Fewest clusters worth resampling. Below this the bootstrap has too few
#: distinct units to say anything, so the audit falls back to a coarser scheme
#: and says so rather than reporting intervals it cannot support.
MIN_CLUSTERS = 20


@dataclass
class _Context:
    arena: Arena
    full: Arena
    fit: BTFit
    boot: BootstrapResult
    sep: Separation
    cluster: Cluster
    cluster_note: str
    n_resamples: int
    seed: int
    alpha: float
    dropped: list[str]


def _gini(values: np.ndarray) -> float:
    x = np.sort(np.asarray(values, dtype=float))
    n = x.size
    total = x.sum()
    if n == 0 or total <= 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2.0 * (index * x).sum()) / (n * total) - (n + 1) / n)


def _majority_graph(wins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Directed majority graph and its undirected support, as integer matrices."""
    beats = (wins > wins.T).astype(np.int64)
    np.fill_diagonal(beats, 0)
    return beats, beats + beats.T


def _cyclic_share(wins: np.ndarray) -> tuple[float, int, int]:
    """Share of fully decided triads that form a preference cycle.

    Counted as ``trace(A^3) / 3`` over the majority graph, against
    ``trace(E^3) / 6`` triangles in its undirected support. Integer matrix
    products are used so the count is exact and platform independent.
    """
    beats, support = _majority_graph(wins)
    cycles = int(np.trace(beats @ beats @ beats)) // 3
    triads = int(np.trace(support @ support @ support)) // 6
    return (cycles / triads if triads else 0.0), cycles, triads


def _reversals(scores: np.ndarray, alternative: np.ndarray, sep: Separation) -> list[str]:
    """Resolved adjacent pairs whose order flips under an alternative fit."""
    flipped = []
    for pair in sep.pairs:
        if not pair["resolved_simultaneous"]:
            continue
        i, j = pair["upper_index"], pair["lower_index"]
        if scores[i] > scores[j] and alternative[i] <= alternative[j]:
            flipped.append(f"{pair['upper']} > {pair['lower']}")
    return flipped


def probe_separation(ctx: _Context) -> ProbeResult:
    """How much of the published order the votes actually support."""
    sep = ctx.sep
    if not sep.pairs:
        return ProbeResult(
            name="separation",
            title="Adjacent ranks the votes separate",
            skipped=True,
            reason="a single model has no neighbours to separate it from",
        )
    unresolved = [p for p in sep.pairs if not p["resolved_simultaneous"]]
    needed = [battles_needed(p, ctx.arena.n_battles, sep.critical_simultaneous) for p in unresolved]
    finite = sorted(n for n in needed if n is not None)
    return ProbeResult(
        name="separation",
        title="Adjacent ranks the votes separate",
        value=sep.unresolved_share,
        innocent=INNOCENT_UNRESOLVED,
        reason=(
            f"{sep.n_simultaneous} of {sep.n_pairs} adjacent pairs are separated with "
            f"family-wise control across the board ({sep.n_marginal} without it); "
            f"the board collapses to {len(sep.tiers)} tiers"
        ),
        detail={
            "n_pairs": sep.n_pairs,
            "n_resolved_simultaneous": sep.n_simultaneous,
            "n_resolved_marginal": sep.n_marginal,
            "n_tiers": len(sep.tiers),
            "critical_simultaneous": sep.critical_simultaneous,
            "tiers": [[ctx.arena.models[i] for i in tier] for tier in sep.tiers],
            "median_battles_for_next_pair": finite[len(finite) // 2] if finite else None,
            "pairs": sep.pairs,
        },
    )


def probe_rank_volatility(ctx: _Context) -> ProbeResult:
    """How far each model's rank moves when the votes are resampled."""
    intervals = ctx.boot.rank_ci(1 - ctx.alpha)
    widths = intervals[:, 1] - intervals[:, 0] + 1
    worst = int(np.argmax(widths))
    return ProbeResult(
        name="rank_volatility",
        title="Width of the rank interval",
        value=float(np.median(widths)),
        innocent=INNOCENT_RANK_WIDTH,
        reason=(
            f"the median model's 95% rank interval spans {np.median(widths):.1f} ranks; "
            f"{ctx.arena.models[worst]} spans {widths[worst]:.0f}"
        ),
        detail={
            "median_width": float(np.median(widths)),
            "max_width": float(widths.max()),
            "widest_model": ctx.arena.models[worst],
            "intervals": {
                ctx.arena.models[i]: [int(intervals[i, 0]), int(intervals[i, 1])]
                for i in range(ctx.arena.n_models)
            },
        },
    )


def probe_connectivity(ctx: _Context) -> ProbeResult:
    """Whether every pair of scores is comparable at all."""
    groups = ctx.full.components()
    counts = ctx.full.battles_per_model()
    thin = [ctx.full.models[i] for i in np.flatnonzero(counts < MIN_BATTLES_PER_MODEL)]
    largest = len(groups[0]) / ctx.full.n_models
    split = len(groups) > 1
    reason = (
        f"the comparison graph has {len(groups)} components, so scores across them are "
        f"not comparable; the audit kept the largest ({len(groups[0])} models) and "
        f"dropped {len(ctx.dropped)}"
        if split
        else "every model is connected to every other through a chain of opponents"
    )
    if thin:
        reason += f"; {len(thin)} model(s) below {MIN_BATTLES_PER_MODEL} battles"
    return ProbeResult(
        name="connectivity",
        title="Comparability of the board",
        value=largest,
        flagged=bool(split or thin),
        reason=reason,
        detail={
            "n_components": len(groups),
            "largest_component_share": largest,
            "components": [[ctx.full.models[i] for i in c] for c in groups],
            "dropped_from_audit": ctx.dropped,
            "under_compared": thin,
            "min_battles": int(counts.min()),
            "median_battles": float(np.median(counts)),
            "failed_resamples": ctx.boot.n_failed,
        },
    )


def probe_coverage_imbalance(ctx: _Context) -> ProbeResult:
    """Whether some models were compared far more often than others."""
    counts = ctx.arena.battles_per_model()
    n = ctx.arena.n_models
    played = int((ctx.arena.counts().comparisons > 0).sum() // 2)
    possible = n * (n - 1) // 2
    return ProbeResult(
        name="coverage_imbalance",
        title="Evenness of the comparison budget",
        value=_gini(counts),
        innocent=INNOCENT_IMBALANCE,
        reason=(
            f"battles per model run from {counts.min()} to {counts.max()}; "
            f"{played} of {possible} possible pairs were ever compared"
        ),
        detail={
            "gini": _gini(counts),
            "min_battles": int(counts.min()),
            "max_battles": int(counts.max()),
            "pairs_played": played,
            "pairs_possible": possible,
        },
    )


def probe_position_bias(ctx: _Context) -> ProbeResult:
    """Whether being shown first is worth anything."""
    randomisation = ctx.arena.order_randomization()
    if not randomisation.looks_positional:
        return ProbeResult(
            name="position_bias",
            title="Advantage of being shown first",
            skipped=True,
            reason=(
                (
                    f"no model pair appears in both orderings "
                    f"({randomisation.both_orders_share:.0%} of "
                    f"{randomisation.n_pairs} pairs do), so which model sits in the "
                    "first slot is fixed per pair and cannot be told apart from the "
                    "models themselves"
                )
                if randomisation.both_orders_share < 0.2
                else (
                    f"{randomisation.forward_share:.0%} of battles list the models in "
                    "name-sorted order, so the a/b assignment records which model "
                    "sorts first, not which was shown first"
                )
            ),
            detail={
                "forward_share": randomisation.forward_share,
                "both_orders_share": randomisation.both_orders_share,
            },
        )
    values = bootstrap_values(
        ctx.arena,
        lambda counts: fit_bt(counts, order=True).order_log or 0.0,
        n_resamples=ctx.n_resamples,
        seed=ctx.seed,
        cluster=ctx.cluster,
    )
    fitted = fit_bt(ctx.arena.counts(), order=True)
    theta = float(fitted.order_log or 0.0)
    decisive = int((ctx.arena.y != 2).sum())
    first_wins = int((ctx.arena.y == 0).sum())
    rate = first_wins / decisive if decisive else 0.5
    return ProbeResult(
        name="position_bias",
        title="Advantage of being shown first",
        value=theta,
        ci=percentile_ci(values, 1 - ctx.alpha),
        innocent=INNOCENT_POSITION,
        reason=(
            f"the model shown first wins {rate:.1%} of decisive battles "
            f"(log-odds advantage {theta:+.3f})"
        ),
        detail={
            "order_log": theta,
            "first_position_win_rate": rate,
            "decisive_battles": decisive,
            "both_orders_share": randomisation.both_orders_share,
        },
    )


def probe_position_reordering(ctx: _Context) -> ProbeResult:
    """Whether removing the position advantage changes the published order."""
    if not ctx.arena.order_randomization().looks_positional:
        return ProbeResult(
            name="position_reordering",
            title="Order changes after correcting for position",
            skipped=True,
            reason="position is not identifiable in this data, so nothing to correct",
        )
    corrected = fit_bt(ctx.arena.counts(), order=True)
    flipped = _reversals(ctx.fit.scores, corrected.scores, ctx.sep)
    at_risk = ctx.sep.n_simultaneous
    moved = int((corrected.ranks != ctx.fit.ranks).sum())
    return ProbeResult(
        name="position_reordering",
        title="Order changes after correcting for position",
        value=float(len(flipped)),
        flagged=bool(flipped),
        reason=(
            f"{len(flipped)} of the {at_risk} separated pair(s) reverse, and {moved} "
            "model(s) change rank, once the first-position advantage is fitted out"
        ),
        detail={
            "reversed_pairs": flipped,
            "models_changing_rank": moved,
            "separated_pairs_at_risk": at_risk,
        },
    )


def probe_tie_convention(ctx: _Context) -> ProbeResult:
    """Whether the order survives a different way of counting ties."""
    tie_share = ctx.arena.n_ties / ctx.arena.n_battles
    if tie_share < MIN_TIE_SHARE:
        return ProbeResult(
            name="tie_convention",
            title="Order changes under another tie rule",
            skipped=True,
            reason=f"ties are {tie_share:.1%} of battles, too few to move anything",
            detail={"tie_share": tie_share},
        )
    counts = ctx.arena.counts()
    alternatives = {
        "drop": fit_bt(counts, ties="drop").scores,
        "rao_kupper": fit_rao_kupper(counts).scores,
    }
    flipped = {
        name: _reversals(ctx.fit.scores, scores, ctx.sep) for name, scores in alternatives.items()
    }
    total = sum(len(v) for v in flipped.values())
    at_risk = ctx.sep.n_simultaneous
    return ProbeResult(
        name="tie_convention",
        title="Order changes under another tie rule",
        value=float(total),
        flagged=total > 0,
        reason=(
            f"ties are {tie_share:.1%} of battles; {total} of the {at_risk} separated "
            "pair(s) reverse when ties are dropped or modelled instead of split"
        ),
        detail={
            "tie_share": tie_share,
            "reversed_under_drop": flipped["drop"],
            "reversed_under_rao_kupper": flipped["rao_kupper"],
            "separated_pairs_at_risk": at_risk,
            "rao_kupper_theta": fit_rao_kupper(counts).tie_theta,
        },
    )


def probe_intransitivity(ctx: _Context) -> ProbeResult:
    """Whether a single strength per model can explain who beats whom.

    Bradley-Terry cannot represent rock-paper-scissors. Some cycles always
    appear by chance, so the observed share is compared against the share the
    fitted model itself produces under the same comparison budget.
    """
    counts = ctx.arena.counts()
    wins = counts.wins
    observed, cycles, triads = _cyclic_share(wins)
    if triads < 10:
        return ProbeResult(
            name="intransitivity",
            title="Preference cycles beyond what the fit predicts",
            skipped=True,
            reason=f"only {triads} fully decided triads, too few to judge cycles",
            detail={"triads": triads},
        )
    strength = np.exp(ctx.fit.scores)
    decisive = wins + wins.T
    prob = strength[:, None] / (strength[:, None] + strength[None, :])
    rng = np.random.Generator(np.random.PCG64(ctx.seed))
    upper = np.triu_indices(ctx.arena.n_models, k=1)
    trials = decisive[upper].astype(np.int64)
    probs = prob[upper]
    simulated = np.empty(PARAMETRIC_RESAMPLES, dtype=float)
    for draw in range(PARAMETRIC_RESAMPLES):
        won = rng.binomial(trials, probs)
        synthetic = np.zeros_like(wins)
        synthetic[upper] = won
        synthetic.T[upper] = trials - won
        simulated[draw] = _cyclic_share(synthetic)[0]
    excess = observed - float(simulated.mean())
    lo, hi = percentile_ci(observed - simulated, 1 - ctx.alpha)
    return ProbeResult(
        name="intransitivity",
        title="Preference cycles beyond what the fit predicts",
        value=excess,
        ci=(lo, hi),
        innocent=INNOCENT_CYCLE_EXCESS,
        reason=(
            f"{cycles} of {triads} decided triads are cyclic ({observed:.1%}); the fitted "
            f"model produces {simulated.mean():.1%} on the same comparison budget"
        ),
        detail={
            "observed_share": observed,
            "expected_share": float(simulated.mean()),
            "cyclic_triads": cycles,
            "decided_triads": triads,
        },
    )


def probe_vote_clustering(ctx: _Context) -> ProbeResult:
    """Whether resampling battles independently was ever justified."""
    if ctx.cluster == "battle":
        return ProbeResult(
            name="vote_clustering",
            title="Cost of pretending votes are independent",
            skipped=True,
            reason=ctx.cluster_note
            or "no voter or item identifiers, so clustering cannot be checked",
            detail={"cluster_note": ctx.cluster_note},
        )
    plain = bootstrap(ctx.arena, n_resamples=ctx.n_resamples, seed=ctx.seed, cluster="battle")
    plain_sep = separation(plain, ctx.arena.models, ctx.alpha)
    ratio = float(
        ctx.boot.scores.std(axis=0, ddof=1).mean() / plain.scores.std(axis=0, ddof=1).mean()
    )
    lost = plain_sep.n_simultaneous - ctx.sep.n_simultaneous
    sizes = np.bincount(
        ctx.arena.item if ctx.cluster == "item" else ctx.arena.voter  # type: ignore[arg-type]
    )
    top = np.sort(sizes)[::-1]
    heavy = float(top[: max(1, sizes.size // 100)].sum() / sizes.sum())
    return ProbeResult(
        name="vote_clustering",
        title="Cost of pretending votes are independent",
        value=ratio,
        innocent=INNOCENT_CLUSTER_INFLATION,
        reason=(
            f"resampling {ctx.cluster}s instead of battles widens standard errors by "
            f"{ratio:.2f}x and costs {lost} separated pair(s); the busiest 1% of "
            f"{ctx.cluster}s account for {heavy:.1%} of votes"
        ),
        detail={
            "cluster": ctx.cluster,
            "cluster_note": ctx.cluster_note,
            "n_clusters": ctx.boot.n_clusters,
            "se_ratio": ratio,
            "pairs_lost": int(lost),
            "separated_battle_level": plain_sep.n_simultaneous,
            "separated_clustered": ctx.sep.n_simultaneous,
            "top_percent_share": heavy,
            "largest_cluster": int(top[0]),
        },
    )


def probe_annotator_influence(ctx: _Context) -> ProbeResult:
    """Whether one annotator moves the board on their own.

    Clustering the bootstrap on voters already widens the intervals when a few
    people carry the votes. What it cannot say is *who*. This drops each
    annotator in turn and measures the largest score shift that follows, in
    units of the median gap between neighbouring models - so a value of 1 means
    one person's absence moves a model by a whole rung of the ladder.
    """
    voters = ctx.arena.voter
    if voters is None:
        return ProbeResult(
            name="annotator_influence",
            title="Largest shift from dropping one annotator",
            skipped=True,
            reason="no voter identifiers in the data",
        )
    n_voters = int(voters.max()) + 1
    low, high = LOO_JUDGE_RANGE
    if not low <= n_voters <= high:
        return ProbeResult(
            name="annotator_influence",
            title="Largest shift from dropping one annotator",
            skipped=True,
            reason=(
                f"{n_voters} annotators: leaving out one of so "
                f"{'few' if n_voters < low else 'many'} is not a perturbation worth "
                f"reading (this runs for {low}-{high})"
            ),
            detail={"n_voters": n_voters},
        )
    gaps = np.abs(np.diff(np.sort(ctx.fit.scores)[::-1]))
    rung = float(np.median(gaps)) if gaps.size else 0.0
    if rung <= 0:
        return ProbeResult(
            name="annotator_influence",
            title="Largest shift from dropping one annotator",
            skipped=True,
            reason="the board has no spread to measure a shift against",
        )
    worst_voter, worst_shift, worst_model = -1, 0.0, ""
    shares = np.bincount(voters, minlength=n_voters) / voters.size
    for voter in range(n_voters):
        keep = voters != voter
        if not keep.any():
            continue
        try:
            alternative = fit_bt(ctx.arena.subset(keep).counts()).scores
        except Exception:  # noqa: BLE001 - dropping a voter can disconnect the board
            continue
        shift = np.abs(alternative - ctx.fit.scores)
        if float(shift.max()) > worst_shift:
            worst_voter = voter
            worst_shift = float(shift.max())
            worst_model = ctx.arena.models[int(np.argmax(shift))]
    return ProbeResult(
        name="annotator_influence",
        title="Largest shift from dropping one annotator",
        value=worst_shift / rung,
        innocent=INNOCENT_ANNOTATOR_SHIFT,
        reason=(
            f"dropping the most influential of {n_voters} annotators "
            f"(who cast {shares[worst_voter]:.1%} of votes) moves {worst_model} by "
            f"{worst_shift / rung:.2f} of the median gap between neighbours"
        ),
        detail={
            "n_annotators": n_voters,
            "worst_annotator_index": worst_voter,
            "worst_annotator_share": float(shares[worst_voter]) if worst_voter >= 0 else 0.0,
            "most_moved_model": worst_model,
            "shift_in_score": worst_shift,
            "median_adjacent_gap": rung,
        },
    )


PROBES = (
    probe_separation,
    probe_rank_volatility,
    probe_connectivity,
    probe_coverage_imbalance,
    probe_position_bias,
    probe_position_reordering,
    probe_tie_convention,
    probe_intransitivity,
    probe_vote_clustering,
    probe_annotator_influence,
)


def _choose_cluster(arena: Arena) -> tuple[Cluster, str]:
    """Pick the most conservative resampling the data can actually support."""
    notes: list[str] = []
    for name, codes in (("voter", arena.voter), ("item", arena.item)):
        if codes is None:
            continue
        count = int(codes.max()) + 1
        if count >= MIN_CLUSTERS:
            return name, "; ".join(notes)  # type: ignore[return-value]
        notes.append(f"only {count} distinct {name}s, too few to resample (needs {MIN_CLUSTERS})")
    return "battle", "; ".join(notes)


def run_audit(
    arena: Arena,
    n_resamples: int = N_RESAMPLES,
    seed: int = SEED,
    cluster: Cluster | None = None,
    alpha: float = ALPHA,
    prior: float = PRIOR_BATTLES,
) -> AuditResult:
    """Fit the leaderboard and run every probe against it.

    Args:
        arena: the battles.
        n_resamples: bootstrap resamples.
        seed: PCG64 seed for every random step.
        cluster: what to resample. The default picks the most conservative
            option the data allows - voters if they are recorded, else items,
            else battles - because independent resampling of correlated votes
            is the more common mistake and the quieter one.
        alpha: two-sided error rate.
        prior: virtual battles per model, see :mod:`arenakit.bt`.
    """
    full = arena
    groups = full.components()
    dropped = [full.models[i] for group in groups[1:] for i in group]
    if dropped:
        arena = full.restrict(groups[0])
    chosen, note = _choose_cluster(arena)
    if cluster is not None:
        chosen, note = cluster, ""
    counts = arena.counts()
    fit = fit_bt(counts, prior=prior)
    boot = bootstrap(
        arena,
        fit=lambda c: fit_bt(c, prior=prior),
        n_resamples=n_resamples,
        seed=seed,
        cluster=chosen,
    )
    sep = separation(boot, arena.models, alpha)
    ctx = _Context(
        arena=arena,
        full=full,
        fit=fit,
        boot=boot,
        sep=sep,
        cluster=chosen,
        cluster_note=note,
        n_resamples=n_resamples,
        seed=seed,
        alpha=alpha,
        dropped=sorted(dropped),
    )
    probes = [probe(ctx) for probe in PROBES]

    score_ci = boot.score_ci(1 - alpha)
    rank_ci = boot.rank_ci(1 - alpha)
    elo = to_elo(fit.scores)
    battles = arena.battles_per_model()
    board: list[dict[str, Any]] = []
    for rank, index in enumerate(fit.ranking.tolist(), start=1):
        board.append(
            {
                "rank": rank,
                "model": arena.models[index],
                "score": float(fit.scores[index]),
                "elo": float(elo[index]),
                "score_ci": [float(score_ci[index, 0]), float(score_ci[index, 1])],
                "rank_ci": [int(rank_ci[index, 0]), int(rank_ci[index, 1])],
                "battles": int(battles[index]),
            }
        )
    return AuditResult(
        probes=probes, leaderboard=board, n_battles=full.n_battles, n_models=full.n_models
    )


def audit_counts(counts: PairCounts) -> BTFit:
    """Convenience: fit a board straight from sufficient statistics."""
    return fit_bt(counts)
