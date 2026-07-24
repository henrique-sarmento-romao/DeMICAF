r"""FL client: receives global weights, trains locally, returns updates.

The client is the only component that touches local training data.  It never
shares raw data or validation metrics with the server — only model parameters
leave this object.

Structure
---------
FLClient   stateful FL participant with SGD-step-based local training
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from DeMICAF.classifier import build_model, compute_pos_weights, evaluate, masked_bce_loss

from .data import ClientSplit, make_data_loader

if TYPE_CHECKING:
    import pandas as pd
    from torch.utils.data import DataLoader

# ── FL Client ──────────────────────────────────────────────────────────────────


class FLClient:
    """Stateful FL participant: receives global weights, trains locally, returns updates.

    Each client holds its own DataLoaders and a local copy of the model.  The
    server only ever receives model state dicts from this object; raw data and
    validation metrics never leave the client.

    Local training uses a **fixed number of SGD steps** per round rather than
    full epochs.  This normalises the training budget across clients with
    different dataset sizes: a client with 200 samples and one with 800 samples
    both perform exactly ``local_sgd_steps`` gradient updates per round.
    """

    def __init__(
        self,
        split: ClientSplit,
        label_cols: list[str],
        *,
        batch_size: int,
        num_workers: int,
        lr: float,
        uncertain_policy: str,
        device: str,
        architecture: str,
        dropout_rate: float,
        pretrained: bool,
        image_column: str,
        seed: int,
    ) -> None:
        self.name: str = split.name
        self.split: ClientSplit = split
        self.label_cols: list[str] = label_cols
        self.device: str = device
        self.lr: float = lr

        def _loader(df: pd.DataFrame, *, is_train: bool) -> DataLoader[Any]:
            return make_data_loader(
                df,
                split.root_dir,
                label_cols=label_cols,
                is_train=is_train,
                batch_size=batch_size,
                num_workers=num_workers,
                image_column=image_column,
                uncertain_policy=uncertain_policy,
                seed=seed,
            )

        self.train_loader: DataLoader[Any] = _loader(split.train_df, is_train=True)
        self.val_loader: DataLoader[Any] = _loader(split.val_df, is_train=False)
        self.test_loader: DataLoader[Any] = _loader(split.test_df, is_train=False)

        # Pre-compute class-imbalance weights once; reused every round.
        self._pos_weight: torch.Tensor = compute_pos_weights(split.train_df, label_cols).to(device)

        # Local model — weights are overwritten by set_global_weights() each round.
        self.model: nn.Module = build_model(
            pretrained=pretrained,
            num_labels=len(label_cols),
            architecture=architecture,
            dropout_rate=dropout_rate,
        ).to(device)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def n_train(self) -> int:
        """Training set size; used as the FedAvg aggregation weight."""
        return len(self.split.train_df)

    # ── Weight exchange ────────────────────────────────────────────────────────

    def set_global_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Overwrite local model parameters with a deep copy of ``state_dict``."""
        self.model.load_state_dict(copy.deepcopy(state_dict))

    def get_weights(self) -> dict[str, torch.Tensor]:
        """Return the current local model parameters as a CPU state dict."""
        return {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

    # ── Local training ────────────────────────────────────────────────────────

    def local_train(self, n_steps: int) -> float:
        """Run exactly ``n_steps`` gradient updates; return the mean train loss.

        A fresh Adam optimiser is created at the start of each call (stateless
        local optimiser, standard FedAvg behaviour).  If the training DataLoader
        is exhausted before ``n_steps`` steps are completed, the iterator wraps
        around so the step count is always exactly ``n_steps`` regardless of
        dataset size.
        """
        optimiser = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.train()

        train_iter = iter(self.train_loader)
        total_loss = 0.0

        for _ in range(n_steps):
            try:
                _, imgs, targets = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                _, imgs, targets = next(train_iter)

            labels, masks = targets
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)
            masks = masks.to(self.device)

            optimiser.zero_grad()
            logits = self.model(imgs)
            loss = masked_bce_loss(logits, labels, masks, self._pos_weight)
            loss.backward()
            optimiser.step()
            total_loss += loss.item()

        return total_loss / max(1, n_steps)

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(self, split: str = "test") -> dict[str, float]:
        """Evaluate the local model on the val or test split.

        Called only after training completes — never during the round loop.
        """
        loader = self.test_loader if split == "test" else self.val_loader
        return evaluate(self.model, loader, self.device, self.label_cols)
