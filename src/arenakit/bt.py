"""Bradley-Terry fits by minorization-maximization.

Three models, all fitted with the MM updates of Hunter (2004):

* :func:`fit_bt` - the plain Bradley-Terry model, optionally with a
  first-position advantage term.
* :func:`fit_rao_kupper` - Rao and Kupper's (1967) tie model, which gives ties
  their own parameter instead of a convention.

Everything is fitted from :class:`~arenakit.schema.PairCounts`, which is a
sufficient statistic. That is what makes a two-thousand-resample bootstrap over
a hundred thousand battles cheap: each refit touches an ``n_models`` square
matrix, never the battle list.

Only differences between scores are identified, so every fit is renormalised to
a geometric mean of 1 on each iteration. That is not cosmetic: without it the
overall level drifts along an almost flat direction of the likelihood and the
iteration spends thousands of steps converging on a quantity nobody reads.

A weak prior keeps the fit finite when a model happens to win every battle it
played, which would otherwise send its score to infinity. Each model plays
``prior`` virtual battles against a phantom opponent of exactly average
strength, split evenly between wins and losses. With even a few dozen real
battles per model its influence on differences is far below the bootstrap noise;
set ``prior=0.0`` for the unpenalised maximum likelihood estimate, which exists
only when every model has both beaten someone and lost to someone along a chain
that connects the whole board.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .schema import PairCounts

#: Convergence threshold on the largest absolute change in log-strength.
TOL = 1e-10
#: Hard iteration cap. Reaching it is reported, never silently accepted.
MAX_ITER = 10_000
#: Virtual battles per model against a phantom opponent of strength 1.
PRIOR_BATTLES = 0.5

TieRule = Literal["half", "drop"]


class ConvergenceError(RuntimeError):
    """Raised when an MM fit hits :data:`MAX_ITER` without converging."""


@dataclass(frozen=True)
class BTFit:
    """A fitted leaderboard.

    Attributes:
        scores: log-strength per model, centred to mean zero.
        order_log: log first-position advantage, or ``None`` when not fitted.
            Positive means being shown first helps.
        tie_theta: Rao-Kupper tie parameter, or ``None``. It is at least 1;
            larger values mean ties are more likely at equal strength.
        n_iter: MM iterations used.
    """

    scores: np.ndarray
    order_log: float | None = None
    tie_theta: float | None = None
    n_iter: int = 0

    @property
    def ranking(self) -> np.ndarray:
        """Model indices from strongest to weakest."""
        return np.argsort(-self.scores, kind="stable")

    @property
    def ranks(self) -> np.ndarray:
        """Rank per model, 1 = strongest."""
        out = np.empty(self.scores.size, dtype=np.int64)
        out[self.ranking] = np.arange(1, self.scores.size + 1)
        return out


def to_elo(scores: np.ndarray, anchor: float = 1000.0, scale: float = 400.0) -> np.ndarray:
    """Rescale log-strengths to the Elo-like units leaderboards usually print.

    A monotone transform: it changes no ordering and no test, only the units.
    """
    return anchor + scale / math.log(10.0) * (scores - float(np.mean(scores)))


def _normalise(pi: np.ndarray) -> np.ndarray:
    """Scale strengths to geometric mean 1. The model is scale free, so this
    changes nothing that is identified, and it removes the flat direction that
    would otherwise dominate the iteration count."""
    return pi / np.exp(np.mean(np.log(pi)))


def fit_bt(
    counts: PairCounts,
    ties: TieRule = "half",
    order: bool = False,
    prior: float = PRIOR_BATTLES,
) -> BTFit:
    """Fit Bradley-Terry strengths.

    Args:
        counts: sufficient statistics for the arena.
        ties: ``"half"`` splits a tie as half a win each; ``"drop"`` discards
            tied battles entirely. Neither is a model of ties - see
            :func:`fit_rao_kupper` for that.
        order: also fit a first-position advantage shared by all models.
        prior: virtual battles against a phantom opponent, see module docs.

    Raises:
        ConvergenceError: if the MM iteration does not converge.
    """
    if ties not in ("half", "drop"):
        raise ValueError(f"unknown tie rule {ties!r}")
    return _fit_ordered(counts, ties, prior) if order else _fit_plain(counts, ties, prior)


def _fit_plain(counts: PairCounts, ties: TieRule, prior: float) -> BTFit:
    wins = counts.wins
    tied = counts.ties
    n = counts.n_models
    if ties == "drop":
        num = wins.sum(axis=1)
        meetings = wins + wins.T
    else:
        num = wins.sum(axis=1) + 0.5 * tied.sum(axis=1)
        meetings = wins + wins.T + tied
    num = num + prior / 2.0

    pi = np.ones(n, dtype=float)
    for step in range(1, MAX_ITER + 1):
        share = meetings / (pi[:, None] + pi[None, :])
        den = share.sum(axis=1) + prior / (pi + 1.0)
        nxt = _normalise(num / np.maximum(den, 1e-300))
        delta = float(np.max(np.abs(np.log(nxt) - np.log(pi))))
        pi = nxt
        if delta < TOL:
            return BTFit(scores=np.log(pi), n_iter=step)
    raise ConvergenceError(f"Bradley-Terry MM did not converge in {MAX_ITER} iterations")


def _fit_ordered(counts: PairCounts, ties: TieRule, prior: float) -> BTFit:
    n = counts.n_models
    if ties == "drop":
        meetings = counts.wf + counts.ws
        won = counts.wf.sum(axis=1) + counts.ws.sum(axis=0)
        first_wins = float(counts.wf.sum())
    else:
        meetings = counts.wf + counts.ws + counts.tt
        won = (
            counts.wf.sum(axis=1)
            + counts.ws.sum(axis=0)
            + 0.5 * (counts.tt.sum(axis=1) + counts.tt.sum(axis=0))
        )
        first_wins = float(counts.wf.sum() + 0.5 * counts.tt.sum())
    num = won + prior / 2.0

    pi = np.ones(n, dtype=float)
    gamma = 1.0
    for step in range(1, MAX_ITER + 1):
        # meetings[i, j] has i first. Denominator gets i's first-position term
        # and, transposed, its second-position term.
        odds = gamma * pi[:, None] + pi[None, :]
        as_first = (meetings / odds).sum(axis=1) * gamma
        as_second = (meetings / odds).sum(axis=0)
        den = as_first + as_second + prior / (pi + 1.0)
        nxt = _normalise(num / np.maximum(den, 1e-300))

        share = gamma * nxt[:, None] + nxt[None, :]
        gamma_den = float((meetings * (nxt[:, None] / share)).sum())
        gamma_next = first_wins / max(gamma_den, 1e-300)

        delta = max(
            float(np.max(np.abs(np.log(nxt) - np.log(pi)))),
            abs(math.log(gamma_next) - math.log(gamma)),
        )
        pi, gamma = nxt, gamma_next
        if delta < TOL:
            return BTFit(scores=np.log(pi), order_log=math.log(gamma), n_iter=step)
    raise ConvergenceError(f"Bradley-Terry MM with order term did not converge in {MAX_ITER}")


def fit_rao_kupper(counts: PairCounts, prior: float = PRIOR_BATTLES) -> BTFit:
    """Fit the Rao-Kupper tie model.

    A tie is its own outcome with probability governed by ``theta >= 1``,
    rather than half a win. When there are no ties this reduces exactly to
    :func:`fit_bt` with ``ties="drop"``.
    """
    wins = counts.wins
    tied = counts.ties
    n = counts.n_models
    num = wins.sum(axis=1) + tied.sum(axis=1) + prior / 2.0
    total_ties = float(tied.sum()) / 2.0

    pi = np.ones(n, dtype=float)
    theta = 2.0 if total_ties > 0 else 1.0
    for step in range(1, MAX_ITER + 1):
        scale = pi[:, None] + theta * pi[None, :]
        forward = ((wins + tied) / scale).sum(axis=1)
        backward = ((wins.T + tied) / scale.T).sum(axis=1) * theta
        den = forward + backward + prior / (pi + 1.0)
        nxt = _normalise(num / np.maximum(den, 1e-300))

        scale = nxt[:, None] + theta * nxt[None, :]
        s = float((((wins + tied) * nxt[None, :]) / scale).sum())
        theta_next = (
            (total_ties + math.sqrt(total_ties**2 + s**2)) / s if s > 0 and total_ties > 0 else 1.0
        )

        delta = max(
            float(np.max(np.abs(np.log(nxt) - np.log(pi)))),
            abs(math.log(theta_next) - math.log(theta)),
        )
        pi, theta = nxt, theta_next
        if delta < TOL:
            return BTFit(scores=np.log(pi), tie_theta=theta, n_iter=step)
    raise ConvergenceError(f"Rao-Kupper MM did not converge in {MAX_ITER} iterations")


def leaderboard(models: list[str], fit: BTFit, elo: bool = True) -> list[dict[str, Any]]:
    """The fitted board as rows, strongest first."""
    elo_scores = to_elo(fit.scores) if elo else None
    rows = []
    for rank, index in enumerate(fit.ranking.tolist(), start=1):
        row: dict[str, Any] = {
            "rank": rank,
            "model": models[index],
            "score": float(fit.scores[index]),
        }
        if elo_scores is not None:
            row["elo"] = float(elo_scores[index])
        rows.append(row)
    return rows
