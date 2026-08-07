"""Bradley-Terry leaderboards with simultaneous confidence control."""

from .boot import BootstrapResult, Separation, bootstrap, separation
from .bt import BTFit, DisconnectedBoardError, fit_bt, fit_rao_kupper, leaderboard, to_elo
from .probes import run_audit
from .result import AuditResult, ProbeResult
from .schema import Arena, Battle, load_battles
from .synthetic import simulate

__version__ = "0.1.0"

__all__ = [
    "Arena",
    "AuditResult",
    "BTFit",
    "Battle",
    "BootstrapResult",
    "DisconnectedBoardError",
    "ProbeResult",
    "Separation",
    "bootstrap",
    "fit_bt",
    "fit_rao_kupper",
    "leaderboard",
    "load_battles",
    "run_audit",
    "separation",
    "simulate",
    "to_elo",
]
