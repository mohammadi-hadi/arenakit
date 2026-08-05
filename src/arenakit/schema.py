"""Battles in, indexed arena out.

One **battle** is a single pairwise verdict: two models were compared and one of
them won, or the comparison was called a tie.

``model_a`` is the model shown **first**. That convention is load-bearing: the
position probes measure whether being shown first is worth anything, and they
are meaningless if ``model_a`` is really "whichever model sorts first". Arena
checks for that (:meth:`Arena.order_randomization`) instead of assuming it, and
the position probes skip themselves when the assignment looks canonical rather
than positional.

Ties arrive as a single ``"tie"`` outcome on purpose. Real leaderboards ship
richer tie labels - "both bad", "both good", "not sure" - and collapsing them is
a modelling decision with consequences, so the loader refuses to guess. Map them
yourself, and use :func:`arenakit.probes.tie_convention` to see how much the
choice moved the ranking.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, field_validator

Outcome = Literal["a", "b", "tie"]

#: Outcome codes used inside the indexed arrays.
A_WINS, B_WINS, TIE = 0, 1, 2

_WINNER_ALIASES: dict[str, Outcome] = {
    "a": "a",
    "model_a": "a",
    "left": "a",
    "b": "b",
    "model_b": "b",
    "right": "b",
    "tie": "tie",
    "draw": "tie",
}


class Battle(BaseModel):
    """One pairwise verdict.

    Attributes:
        model_a: the model shown first.
        model_b: the model shown second.
        winner: ``"a"``, ``"b"`` or ``"tie"``.
        item_id: the prompt or question compared on, when known. Votes on the
            same item are correlated, so the bootstrap can cluster on it.
        voter_id: who cast the vote, when known. Heavy voters make battles
            correlated in a way battle-level resampling cannot see.
    """

    model_a: str
    model_b: str
    winner: Outcome
    item_id: str | None = None
    voter_id: str | None = None

    @field_validator("winner", mode="before")
    @classmethod
    def _normalise_winner(cls, value: Any) -> Any:
        if isinstance(value, str):
            key = value.strip().lower()
            if key in _WINNER_ALIASES:
                return _WINNER_ALIASES[key]
            raise ValueError(
                f"unrecognised winner {value!r}; accepted values are "
                f"{sorted(set(_WINNER_ALIASES))}. Tie variants such as "
                "'tie (bothbad)' must be mapped explicitly - collapsing them "
                "changes the ranking, so arenakit will not guess for you."
            )
        return value

    @field_validator("model_a", "model_b")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("model names must not be empty")
        return value


@dataclass(frozen=True)
class OrderRandomization:
    """Evidence about whether ``model_a`` means "shown first" or "sorts first"."""

    #: Share of distinct unordered pairs seen in both orderings.
    both_orders_share: float
    #: Share of battles whose (model_a, model_b) is in name-sorted order.
    forward_share: float
    #: Distinct unordered pairs in the data.
    n_pairs: int

    @property
    def looks_positional(self) -> bool:
        """True when the a/b assignment carries position information."""
        return 0.2 <= self.forward_share <= 0.8 and self.both_orders_share >= 0.2


@dataclass(frozen=True)
class PairCounts:
    """Sufficient statistics for every model this package fits.

    All three matrices are indexed ``[first, second]`` by position, so the
    ordered and unordered views are both recoverable from them.
    """

    #: ``wf[i, j]``: i shown first against j, i won.
    wf: np.ndarray
    #: ``ws[i, j]``: i shown first against j, j won.
    ws: np.ndarray
    #: ``tt[i, j]``: i shown first against j, tie.
    tt: np.ndarray

    @property
    def n_models(self) -> int:
        return int(self.wf.shape[0])

    @property
    def wins(self) -> np.ndarray:
        """``W[i, j]``: times i beat j, in either position."""
        return self.wf + self.ws.T

    @property
    def ties(self) -> np.ndarray:
        """``T[i, j]``: times i and j tied, in either position. Symmetric."""
        return self.tt + self.tt.T

    @property
    def comparisons(self) -> np.ndarray:
        """``N[i, j]``: times i and j met, in either position. Symmetric."""
        wins = self.wins
        return wins + wins.T + self.ties

    @property
    def n_battles(self) -> int:
        return int(self.wf.sum() + self.ws.sum() + self.tt.sum())


def _counts(
    a: np.ndarray, b: np.ndarray, y: np.ndarray, n_models: int, weights: np.ndarray | None = None
) -> PairCounts:
    flat = a.astype(np.int64) * n_models + b.astype(np.int64)
    size = n_models * n_models
    out = []
    for code in (A_WINS, B_WINS, TIE):
        mask = y == code
        w = None if weights is None else weights[mask]
        counted = np.bincount(flat[mask], weights=w, minlength=size)
        out.append(counted.reshape(n_models, n_models).astype(float))
    return PairCounts(wf=out[0], ws=out[1], tt=out[2])


class Arena:
    """A set of battles indexed for fitting.

    Args:
        battles: the verdicts to index.
        models: fixes the model order, and keeps models that never appear in a
            battle on the board (they will be flagged, not silently dropped).
    """

    def __init__(self, battles: Sequence[Battle], models: Sequence[str] | None = None) -> None:
        if not battles:
            raise ValueError("an arena needs at least one battle")
        names = (
            list(models)
            if models is not None
            else sorted({b.model_a for b in battles} | {b.model_b for b in battles})
        )
        self.models: list[str] = names
        index = {name: i for i, name in enumerate(names)}
        unknown = {b.model_a for b in battles} | {b.model_b for b in battles}
        missing = unknown - set(index)
        if missing:
            raise ValueError(f"battles mention models missing from `models`: {sorted(missing)}")
        self_play = [b for b in battles if b.model_a == b.model_b]
        if self_play:
            raise ValueError(f"{len(self_play)} battles pit a model against itself")

        self.a = np.array([index[b.model_a] for b in battles], dtype=np.int32)
        self.b = np.array([index[b.model_b] for b in battles], dtype=np.int32)
        codes = {"a": A_WINS, "b": B_WINS, "tie": TIE}
        self.y = np.array([codes[b.winner] for b in battles], dtype=np.int8)
        self.item = _label_codes([b.item_id for b in battles])
        self.voter = _label_codes([b.voter_id for b in battles])

    @property
    def n_battles(self) -> int:
        return int(self.a.size)

    @property
    def n_models(self) -> int:
        return len(self.models)

    @property
    def n_ties(self) -> int:
        return int((self.y == TIE).sum())

    def counts(self, weights: np.ndarray | None = None) -> PairCounts:
        """Sufficient statistics, optionally with a per-battle weight."""
        return _counts(self.a, self.b, self.y, self.n_models, weights)

    def battles_per_model(self) -> np.ndarray:
        return np.bincount(self.a, minlength=self.n_models) + np.bincount(
            self.b, minlength=self.n_models
        )

    def order_randomization(self) -> OrderRandomization:
        """Evidence on whether the a/b assignment is positional."""
        lo = np.minimum(self.a, self.b)
        hi = np.maximum(self.a, self.b)
        key = lo.astype(np.int64) * self.n_models + hi.astype(np.int64)
        forward = self.a < self.b
        uniq, inverse = np.unique(key, return_inverse=True)
        fwd = np.bincount(inverse, weights=forward.astype(float), minlength=uniq.size)
        tot = np.bincount(inverse, minlength=uniq.size).astype(float)
        both = float(np.mean((fwd > 0) & (fwd < tot))) if uniq.size else 0.0
        return OrderRandomization(
            both_orders_share=both,
            forward_share=float(forward.mean()),
            n_pairs=int(uniq.size),
        )

    def components(self) -> list[list[int]]:
        """Connected components of the comparison graph, largest first.

        Bradley-Terry scores are only comparable inside a component: two models
        that never met, directly or through a chain of opponents, have no
        estimable difference at all.
        """
        n = self.n_models
        seen = np.zeros(n, dtype=bool)
        neighbours: list[set[int]] = [set() for _ in range(n)]
        for i, j in zip(self.a.tolist(), self.b.tolist(), strict=True):
            neighbours[i].add(j)
            neighbours[j].add(i)
        found: list[list[int]] = []
        for start in range(n):
            if seen[start]:
                continue
            stack, group = [start], []
            seen[start] = True
            while stack:
                node = stack.pop()
                group.append(node)
                for nxt in neighbours[node]:
                    if not seen[nxt]:
                        seen[nxt] = True
                        stack.append(nxt)
            found.append(sorted(group))
        found.sort(key=len, reverse=True)
        return found

    def subset(self, mask: np.ndarray) -> Arena:
        """A new arena over a boolean battle mask, keeping the model order."""
        keep = np.asarray(mask, dtype=bool)
        if not keep.any():
            raise ValueError("mask keeps no battles")
        clone = object.__new__(Arena)
        clone.models = list(self.models)
        clone.a = self.a[keep]
        clone.b = self.b[keep]
        clone.y = self.y[keep]
        clone.item = None if self.item is None else _recode(self.item[keep])
        clone.voter = None if self.voter is None else _recode(self.voter[keep])
        return clone


def _label_codes(labels: Sequence[str | None]) -> np.ndarray | None:
    if any(label is None for label in labels):
        return None
    uniq = sorted(set(labels))  # type: ignore[arg-type]
    index = {label: i for i, label in enumerate(uniq)}
    return np.array([index[label] for label in labels], dtype=np.int64)


def _recode(codes: np.ndarray) -> np.ndarray:
    _, inverse = np.unique(codes, return_inverse=True)
    return inverse.astype(np.int64).reshape(-1)


def battles_from_records(records: Iterable[dict[str, Any]]) -> list[Battle]:
    """Validate raw dictionaries into battles."""
    return [Battle.model_validate(record) for record in records]


def load_battles(path: str | Path) -> list[Battle]:
    """Read battles from ``.csv``, ``.jsonl`` or ``.json``.

    Required fields are ``model_a``, ``model_b`` and ``winner``; ``item_id`` and
    ``voter_id`` are used when present.
    """
    path = Path(path)
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            records = [
                {k: v for k, v in row.items() if v not in ("", None)}
                for row in csv.DictReader(handle)
            ]
    elif path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
    elif path.suffix.lower() == ".json":
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        records = payload["battles"] if isinstance(payload, dict) else payload
    else:
        raise ValueError(f"unsupported battle file: {path.suffix or path.name}")
    keep = {"model_a", "model_b", "winner", "item_id", "voter_id"}
    return battles_from_records({k: v for k, v in r.items() if k in keep} for r in records)
