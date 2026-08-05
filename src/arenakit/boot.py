"""Resampling, rank intervals, and the test that decides whether a rank is real.

**The test.** Two models are "separated" when the gap between them is large
compared with how much that gap moves under resampling. This package does *not*
decide separation by asking whether two score intervals overlap: non-overlapping
intervals imply a significant difference, but overlapping ones do not rule it
out, so the overlap rule silently calls real gaps unresolved. Instead the
bootstrap distribution of the *difference* is used directly, studentised, and
compared against a critical value.

**Multiplicity.** A leaderboard of ``m`` models makes ``m - 1`` adjacent
comparisons at once, and the pairs being compared were chosen by the same data
that will judge them. :func:`separation` therefore reports two verdicts per
pair: a marginal one, and a simultaneous one using the max-t critical value
(Westfall and Young's step-down idea in its single-step form), which controls
the family-wise error rate across the whole board. The gap between the two
counts is worth looking at - it is the price of ranking many models at once.

**Clustering.** Real vote streams are not independent. A handful of voters cast
a disproportionate share of the votes, and several votes may concern the same
prompt. Resampling individual battles pretends otherwise and reports intervals
that are too narrow. Pass ``cluster="voter"`` or ``cluster="item"`` to resample
whole clusters instead.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .bt import BTFit, fit_bt
from .schema import A_WINS, B_WINS, TIE, Arena, PairCounts

#: Bootstrap resamples. Large enough that the max-t critical value across a
#: fifty-model board is estimated from a useful number of extreme draws.
N_RESAMPLES = 2000
#: Seed for every resampling routine in this package.
SEED = 0
#: Default two-sided error rate.
ALPHA = 0.05

Cluster = Literal["battle", "item", "voter"]
FitFn = Callable[[PairCounts], BTFit]


class _Sampler:
    """Pre-split battle indices so a resample costs three weighted bincounts."""

    def __init__(self, arena: Arena, cluster: Cluster) -> None:
        self.n_models = arena.n_models
        self.n_battles = arena.n_battles
        self.cluster = cluster
        flat = arena.a.astype(np.int64) * self.n_models + arena.b.astype(np.int64)
        self.parts = [np.flatnonzero(arena.y == code) for code in (A_WINS, B_WINS, TIE)]
        self.flats = [flat[part] for part in self.parts]
        if cluster == "battle":
            self.codes = None
            self.n_clusters = arena.n_battles
        else:
            codes = arena.item if cluster == "item" else arena.voter
            if codes is None:
                raise ValueError(f"arena has no {cluster}_id, cannot cluster on it")
            self.codes = codes
            self.n_clusters = int(codes.max()) + 1

    def weights(self, rng: np.random.Generator) -> np.ndarray:
        draw = rng.integers(0, self.n_clusters, self.n_clusters)
        multiplicity = np.bincount(draw, minlength=self.n_clusters).astype(float)
        return multiplicity if self.codes is None else multiplicity[self.codes]

    def counts(self, weights: np.ndarray) -> PairCounts:
        size = self.n_models * self.n_models
        out = []
        for part, flat in zip(self.parts, self.flats, strict=True):
            counted = np.bincount(flat, weights=weights[part], minlength=size)
            out.append(counted.reshape(self.n_models, self.n_models))
        return PairCounts(wf=out[0], ws=out[1], tt=out[2])


@dataclass(frozen=True)
class BootstrapResult:
    """Score and rank draws from resampling one arena."""

    point: np.ndarray
    scores: np.ndarray
    ranks: np.ndarray
    cluster: Cluster
    n_clusters: int
    n_failed: int = 0

    @property
    def n_resamples(self) -> int:
        return int(self.scores.shape[0])

    def score_ci(self, level: float = 1 - ALPHA) -> np.ndarray:
        lo = (1 - level) / 2 * 100
        return np.percentile(self.scores, [lo, 100 - lo], axis=0).T

    def rank_ci(self, level: float = 1 - ALPHA) -> np.ndarray:
        lo = (1 - level) / 2 * 100
        bounds = np.percentile(self.ranks, [lo, 100 - lo], axis=0)
        return np.stack([np.floor(bounds[0]), np.ceil(bounds[1])], axis=1)


def bootstrap(
    arena: Arena,
    fit: FitFn | None = None,
    n_resamples: int = N_RESAMPLES,
    seed: int = SEED,
    cluster: Cluster = "battle",
) -> BootstrapResult:
    """Resample an arena and refit the leaderboard each time.

    Args:
        arena: the battles.
        fit: how to fit one resample. Defaults to plain Bradley-Terry with
            half-credit ties.
        n_resamples: number of resamples.
        seed: PCG64 seed.
        cluster: resample battles, items or voters.

    Resamples that fail to converge are dropped and counted in
    ``n_failed`` rather than being quietly replaced.
    """
    fit_fn: FitFn = fit if fit is not None else (lambda counts: fit_bt(counts))
    sampler = _Sampler(arena, cluster)
    point = fit_fn(arena.counts()).scores
    rng = np.random.Generator(np.random.PCG64(seed))
    draws: list[np.ndarray] = []
    failed = 0
    for _ in range(n_resamples):
        try:
            draws.append(fit_fn(sampler.counts(sampler.weights(rng))).scores)
        except Exception:  # noqa: BLE001 - a degenerate resample is data, not a bug
            failed += 1
    if not draws:
        raise RuntimeError("every bootstrap resample failed to fit")
    scores = np.vstack(draws)
    order = np.argsort(-scores, axis=1, kind="stable")
    ranks = np.empty_like(order)
    rows = np.arange(scores.shape[0])[:, None]
    ranks[rows, order] = np.arange(1, scores.shape[1] + 1)[None, :]
    return BootstrapResult(
        point=point,
        scores=scores,
        ranks=ranks,
        cluster=cluster,
        n_clusters=sampler.n_clusters,
        n_failed=failed,
    )


def bootstrap_values(
    arena: Arena,
    statistic: Callable[[PairCounts], float],
    n_resamples: int = N_RESAMPLES,
    seed: int = SEED,
    cluster: Cluster = "battle",
) -> np.ndarray:
    """Resample an arena and evaluate a scalar statistic on each resample."""
    sampler = _Sampler(arena, cluster)
    rng = np.random.Generator(np.random.PCG64(seed))
    values: list[float] = []
    for _ in range(n_resamples):
        try:
            values.append(float(statistic(sampler.counts(sampler.weights(rng)))))
        except Exception:  # noqa: BLE001 - a degenerate resample is data, not a bug
            continue
    if not values:
        raise RuntimeError("every bootstrap resample failed")
    return np.array(values, dtype=float)


def percentile_ci(values: np.ndarray, level: float = 1 - ALPHA) -> tuple[float, float]:
    """Percentile interval of a bootstrap sample."""
    lo = (1 - level) / 2 * 100
    return (float(np.percentile(values, lo)), float(np.percentile(values, 100 - lo)))


@dataclass(frozen=True)
class Separation:
    """Which adjacent rank gaps the data actually supports."""

    pairs: list[dict[str, Any]]
    critical_simultaneous: float
    alpha: float

    @property
    def n_pairs(self) -> int:
        return len(self.pairs)

    @property
    def n_marginal(self) -> int:
        return sum(1 for p in self.pairs if p["resolved_marginal"])

    @property
    def n_simultaneous(self) -> int:
        return sum(1 for p in self.pairs if p["resolved_simultaneous"])

    @property
    def unresolved_share(self) -> float:
        """Share of adjacent pairs not separated under simultaneous control."""
        return 1.0 - self.n_simultaneous / self.n_pairs if self.n_pairs else 0.0

    @property
    def tiers(self) -> list[list[int]]:
        """Model indices grouped into blocks that simultaneous inference cannot split."""
        if not self.pairs:
            return []
        groups = [[self.pairs[0]["upper_index"]]]
        for pair in self.pairs:
            if pair["resolved_simultaneous"]:
                groups.append([pair["lower_index"]])
            else:
                groups[-1].append(pair["lower_index"])
        return groups


def separation(boot: BootstrapResult, models: list[str], alpha: float = ALPHA) -> Separation:
    """Test every adjacent pair on the fitted board.

    The direction of each comparison is taken from the point estimate, so the
    test is two-sided: the ordering itself was chosen by the same data.
    """
    order = np.argsort(-boot.point, kind="stable")
    upper, lower = order[:-1], order[1:]
    gap = boot.point[upper] - boot.point[lower]
    draws = boot.scores[:, upper] - boot.scores[:, lower]
    se = draws.std(axis=0, ddof=1)
    usable = se > 0
    t_star = np.zeros_like(draws)
    t_star[:, usable] = (draws[:, usable] - gap[usable]) / se[usable]
    observed = np.divide(gap, se, out=np.zeros_like(gap), where=usable)

    crit_marginal = np.percentile(np.abs(t_star), 100 * (1 - alpha), axis=0)
    crit_sim = float(np.percentile(np.abs(t_star).max(axis=1), 100 * (1 - alpha)))

    pairs = []
    for k in range(order.size - 1):
        ok = bool(usable[k])
        pairs.append(
            {
                "upper": models[upper[k]],
                "lower": models[lower[k]],
                "upper_index": int(upper[k]),
                "lower_index": int(lower[k]),
                "gap": float(gap[k]),
                "se": float(se[k]),
                "t": float(observed[k]),
                "critical_marginal": float(crit_marginal[k]),
                "resolved_marginal": bool(ok and observed[k] > crit_marginal[k]),
                "resolved_simultaneous": bool(ok and observed[k] > crit_sim),
            }
        )
    return Separation(pairs=pairs, critical_simultaneous=crit_sim, alpha=alpha)


def battles_needed(pair: dict[str, Any], n_battles: int, critical: float) -> float | None:
    """How many battles would separate one adjacent pair, at the current design.

    Standard errors shrink like ``1/sqrt(n)``, so resolving a gap needs the
    sample multiplied by ``(critical * se / gap) ** 2``. This assumes the
    observed gap is the true one and that new battles look like the old ones -
    both optimistic, which makes the answer a floor, not a forecast.
    """
    if pair["gap"] <= 0 or pair["se"] <= 0:
        return None
    factor = (critical * pair["se"] / pair["gap"]) ** 2
    return float(n_battles * factor)
