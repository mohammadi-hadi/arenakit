"""Audit a pairwise model leaderboard before you trust its order."""

from .boot import BootstrapResult, Separation, bootstrap, separation
from .bt import BTFit, fit_bt, fit_rao_kupper, leaderboard, to_elo
from .result import AuditResult, ProbeResult
from .schema import Arena, Battle, load_battles

__version__ = "0.1.0"

__all__ = [
    "Arena",
    "AuditResult",
    "BTFit",
    "Battle",
    "BootstrapResult",
    "ProbeResult",
    "Separation",
    "bootstrap",
    "fit_bt",
    "fit_rao_kupper",
    "leaderboard",
    "load_battles",
    "separation",
    "to_elo",
]
