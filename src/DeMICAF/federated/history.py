r"""Run-history dataclasses for the FL simulation.

All state produced during a training run is captured in :class:`FLHistory`.
These objects are JSON-serializable and write themselves to disk.

Structure
---------
RoundRecord   metrics captured at the end of one FL round (or epoch for centralized)
FLHistory     complete, JSON-serializable record of one FL run
"""

from __future__ import annotations

import dataclasses
import json
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from pathlib import Path

# ── Per-round record ───────────────────────────────────────────────────────────


@dataclasses.dataclass
class RoundRecord:
    """Metrics captured at the end of one FL round.

    For the centralized baseline ``round_idx`` is the epoch index and
    ``client_train_losses`` contains a single ``"pooled"`` key.

    Attributes
    ----------
    round_idx
        Zero-based round (or epoch) index.
    client_train_losses
        Mean training loss per client over the local SGD steps in this round.
        Logged for monitoring only — the server does not use this value for
        any decision.
    aggregation_weights
        Per-client weight used in the FedAvg aggregation step.
        Empty dict for the centralized baseline.
    val_aucs
        Per-label AUC averaged across clients, evaluated on client validation
        sets using the aggregated global model.  Includes a ``"mean"`` key.
    smoothed_val_auc
        3-round moving average of the ``"mean"`` val AUC.  ``nan`` until the
        moving-average window is full.
    sgd_steps
        Running total of gradient steps executed across all clients up to and
        including this round (``n_clients * local_sgd_steps * (round_idx + 1)``).
    """

    round_idx: int
    client_train_losses: dict[str, float]
    aggregation_weights: dict[str, float]
    val_aucs: dict[str, float] = dataclasses.field(default_factory=dict)
    smoothed_val_auc: float = float("nan")
    sgd_steps: int = 0


# ── Full run history ───────────────────────────────────────────────────────────


@dataclasses.dataclass
class FLHistory:
    """Complete, JSON-serializable record of one FL run.

    Attributes
    ----------
    experiment_name
        Canonical name from the config, e.g. ``"fedavg/seed_0"``.
    strategy
        Aggregation strategy: ``"fedavg"`` or ``"centralized"``.
    seed
        Global RNG seed used for this run.
    rounds_completed
        Actual number of rounds (or epochs) executed.
    round_records
        Per-round metrics in chronological order.
    per_client_test_aucs
        ``{client_name: {label: auc, "mean": float}}`` evaluated on each
        training client's fixed test split using the final global model.
    chexpert_test_aucs
        ``{label: auc, "mean": float}`` evaluated on the CheXpert test set
        using the final global model.
    """

    experiment_name: str
    strategy: str
    seed: int
    rounds_completed: int
    round_records: list[RoundRecord] = dataclasses.field(default_factory=list)
    per_client_test_aucs: dict[str, dict[str, float]] = dataclasses.field(default_factory=dict)
    chexpert_test_aucs: dict[str, float] = dataclasses.field(default_factory=dict)
    best_round: int = -1
    stopped_early: bool = False

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Convert the entire history to a JSON-serializable nested dict."""
        return {
            "experiment_name": self.experiment_name,
            "strategy": self.strategy,
            "seed": self.seed,
            "rounds_completed": self.rounds_completed,
            "stopped_early": self.stopped_early,
            "best_round": self.best_round,
            "round_records": [dataclasses.asdict(r) for r in self.round_records],
            "per_client_test_aucs": self.per_client_test_aucs,
            "chexpert_test_aucs": self.chexpert_test_aucs,
        }

    def save(self, path: Path) -> None:
        """Write the history to a JSON file at ``path``."""
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def save_round_metrics_csv(self, path: Path) -> None:
        """Write one row per round to a wide-format CSV for easy plotting.

        Columns: ``round``, ``strategy``, ``seed``, per-client train-loss
        columns, per-client weight columns.
        """
        rows = []
        for r in self.round_records:
            row: dict[str, Any] = {
                "round": r.round_idx,
                "strategy": self.strategy,
                "seed": self.seed,
                **{f"train_loss_{c}": v for c, v in r.client_train_losses.items()},
                **{f"weight_{c}": v for c, v in r.aggregation_weights.items()},
                **{f"val_auc_{lbl}": v for lbl, v in r.val_aucs.items()},
                "smoothed_val_auc": r.smoothed_val_auc,
                "sgd_steps": r.sgd_steps,
            }
            rows.append(row)
        pd.DataFrame(rows).to_csv(path, index=False)

    def save_test_aucs_csv(self, path: Path) -> None:
        """Write per-dataset per-label test AUC to CSV.

        Each row corresponds to one dataset (client or CheXpert).  The
        ``dataset`` column distinguishes the training-client test sets from
        the external CheXpert evaluation set.
        """
        rows = []
        for client_name, aucs in self.per_client_test_aucs.items():
            rows.append(
                {
                    "dataset": client_name,
                    "strategy": self.strategy,
                    "seed": self.seed,
                    **aucs,
                }
            )
        rows.append(
            {
                "dataset": "CheXpert",
                "strategy": self.strategy,
                "seed": self.seed,
                **self.chexpert_test_aucs,
            }
        )
        pd.DataFrame(rows).to_csv(path, index=False)
