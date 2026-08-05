"""Result objects shared by every probe.

A probe reports one of three verdicts:

* ``skipped``  - the data cannot support the check, and ``reason`` says why.
* ``triggered`` - the check found something worth a human look.
* clean       - the check ran and found nothing.

Most probes decide by comparing a bootstrap interval against an *innocent band*:
a range of values that is not worth anyone's attention even if the point estimate
sits outside it. The probe triggers only when the whole interval falls outside
the band, so noise alone does not raise a flag. A few probes answer an exact
question ("is the comparison graph connected?") and set ``flagged`` directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

Band = tuple[float, float]
Interval = tuple[float, float]


@dataclass(frozen=True)
class ProbeResult:
    """One check, its number, and what the number means."""

    name: str
    title: str
    value: float | None = None
    ci: Interval | None = None
    innocent: Band | None = None
    flagged: bool | None = None
    skipped: bool = False
    reason: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def triggered(self) -> bool:
        """Whether this probe is asking for attention."""
        if self.skipped:
            return False
        if self.flagged is not None:
            return self.flagged
        if self.innocent is None or self.value is None:
            return False
        low, high = self.innocent
        if self.ci is None:
            return not (low <= self.value <= high)
        # Trigger only when the whole interval sits outside the innocent band.
        return self.ci[0] > high or self.ci[1] < low

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "value": self.value,
            "ci": list(self.ci) if self.ci else None,
            "innocent": list(self.innocent) if self.innocent else None,
            "triggered": self.triggered,
            "skipped": self.skipped,
            "reason": self.reason,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class AuditResult:
    """Every probe run against one leaderboard, plus the leaderboard itself."""

    probes: list[ProbeResult]
    leaderboard: list[dict[str, Any]]
    n_battles: int
    n_models: int

    def __getitem__(self, name: str) -> ProbeResult:
        for probe in self.probes:
            if probe.name == name:
                return probe
        raise KeyError(name)

    @property
    def triggered(self) -> list[ProbeResult]:
        return [p for p in self.probes if p.triggered]

    @property
    def skipped(self) -> list[ProbeResult]:
        return [p for p in self.probes if p.skipped]

    @property
    def flags(self) -> list[str]:
        return [p.name for p in self.triggered]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_battles": self.n_battles,
            "n_models": self.n_models,
            "flags": self.flags,
            "leaderboard": self.leaderboard,
            "probes": [p.to_dict() for p in self.probes],
        }
