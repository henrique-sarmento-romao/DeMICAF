r"""FL server: aggregation logic and round loops.

The server owns the global model, distributes weights to clients, and collects
aggregated updates.  It has **no access to training or validation data** — the
only data it ever touches is the CheXpert test set, which is used exclusively
for the final evaluation *after* all rounds complete.

Both :meth:`FLServer.run_federated` and :meth:`FLServer.run_centralized` use
early stopping based on a smoothed (moving-average) validation AUC computed
on client validation sets.  The best checkpoint is saved and restored before
final evaluation.

Structure
---------
_moving_average    moving-average helper for smoothing val-AUC curves
aggregate_fedavg   pure FedAvg aggregation function (no side effects)
FLServer           server class — round loops for FL and centralized training
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from DeMICAF.classifier import build_model, compute_pos_weights, evaluate, masked_bce_loss

from .history import FLHistory, RoundRecord

if TYPE_CHECKING:
    from pathlib import Path

    from .client import FLClient

# ── Helpers ───────────────────────────────────────────────────────────────────


def _moving_average(values: list[float], window: int) -> float:
    """Return the mean of the last ``window`` values, or NaN if fewer exist."""
    if len(values) < window:
        return float("nan")
    return float(np.mean(values[-window:]))


# ── Aggregation ────────────────────────────────────────────────────────────────


def aggregate_fedavg(
    state_dicts: list[dict[str, torch.Tensor]],
    client_names: list[str],
    client_sizes: list[int],
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """FedAvg: weighted average of parameters by client training-set size.

    Returns
    -------
    (aggregated_state_dict, weights_used)
        ``aggregated_state_dict`` contains the weighted-average parameters.
        ``weights_used`` maps each client name to its normalised weight
        (fraction of total training samples).
    """
    total = sum(client_sizes)
    weights = [s / total for s in client_sizes]

    avg: dict[str, torch.Tensor] = {}
    for key in state_dicts[0]:
        avg[key] = sum(  # type: ignore[assignment]
            w * sd[key].float() for w, sd in zip(weights, state_dicts, strict=False)
        )

    return avg, dict(zip(client_names, weights, strict=False))


# ── FL Server ──────────────────────────────────────────────────────────────────


class FLServer:
    """FL coordinator: owns the global model, distributes weights, aggregates updates.

    The server has **no access to training or validation data**.  It only
    receives model state dicts from clients.  The CheXpert test set is passed
    as a DataLoader argument to the run methods and is evaluated only at the
    end of training.
    """

    def __init__(
        self,
        clients: list[FLClient],
        label_cols: list[str],
        *,
        architecture: str = "resnet18",
        dropout_rate: float = 0.1,
        pretrained: bool = True,
        device: str = "cuda",
    ) -> None:
        self.clients = clients
        self.label_cols = label_cols
        self.device = device

        self.global_model: nn.Module = build_model(
            pretrained=pretrained,
            num_labels=len(label_cols),
            architecture=architecture,
            dropout_rate=dropout_rate,
        ).to(device)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _distribute(self) -> None:
        """Push the current global model state dict to every client."""
        global_state = {k: v.cpu() for k, v in self.global_model.state_dict().items()}
        for client in self.clients:
            client.set_global_weights(global_state)

    def _aggregate(
        self,
        local_states: list[dict[str, torch.Tensor]],
        client_names: list[str],
        client_sizes: list[int],
    ) -> dict[str, float]:
        """Apply FedAvg, update the global model, return per-client weights used."""
        new_state, weights_used = aggregate_fedavg(local_states, client_names, client_sizes)
        self.global_model.load_state_dict({k: v.to(self.device) for k, v in new_state.items()})
        return weights_used

    def evaluate_chexpert(self, chexpert_loader: DataLoader[Any]) -> dict[str, float]:
        """Evaluate the final global model on the CheXpert test set.

        Called **once** at the end of training. Never called during the round loop.
        """
        return evaluate(self.global_model, chexpert_loader, self.device, self.label_cols)

    # ── Checkpoint helpers ─────────────────────────────────────────────────────

    def save_checkpoint(self, path: Path) -> None:
        """Save the current global model state dict to ``path``."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.global_model.state_dict(), path)

    def load_checkpoint(self, path: Path) -> None:
        """Restore the global model state dict from ``path``."""
        self.global_model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))

    # ── FL round loop ──────────────────────────────────────────────────────────

    def run_federated(
        self,
        rounds: int,
        local_sgd_steps: int,
        chexpert_loader: DataLoader[Any],
        output_dir: Path,
        experiment_name: str,
        seed: int,
        *,
        patience: int = 10,
        ma_window: int = 3,
    ) -> FLHistory:
        """Execute the FL training loop with early stopping.

        Each round:

        1. Distribute the global model to all clients.
        2. Each client trains for exactly ``local_sgd_steps`` gradient steps.
        3. Collect updated weights and aggregate via FedAvg.
        4. Evaluate the aggregated model on each client's validation set.
        5. Apply a ``ma_window``-round moving average to the mean val AUC.
        6. Save a checkpoint whenever the smoothed AUC improves; stop early
           after ``patience`` consecutive rounds without improvement.

        After the loop, the best checkpoint is restored before evaluating the
        final global model on each client's test set and on CheXpert.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        history = FLHistory(
            experiment_name=experiment_name,
            strategy="fedavg",
            seed=seed,
            rounds_completed=0,
        )

        client_names = [c.name for c in self.clients]
        client_sizes = [c.n_train for c in self.clients]
        n_clients = len(self.clients)

        # ── Early stopping state ───────────────────────────────────────────────
        raw_val_aucs: list[float] = []
        best_smoothed_auc = float("-inf")
        best_round_idx = -1
        rounds_without_improvement = 0
        stopped_early = False
        best_ckpt_path = output_dir / "best_model.pt"

        pbar = tqdm(range(rounds), desc=f"[{experiment_name}]", unit="round")
        for round_idx in pbar:
            self._distribute()

            local_states: list[dict[str, torch.Tensor]] = []
            train_losses: dict[str, float] = {}

            for client in self.clients:
                pbar.set_description(f"[{experiment_name}] R{round_idx + 1}/{rounds} | {client.name[:14]}")
                t_loss = client.local_train(local_sgd_steps)
                local_states.append(client.get_weights())
                train_losses[client.name] = t_loss

            weights_used = self._aggregate(local_states, client_names, client_sizes)

            # ── Validation AUC on global model ─────────────────────────────────
            per_client_aucs = [
                evaluate(self.global_model, c.val_loader, self.device, self.label_cols) for c in self.clients
            ]
            mean_val_auc = float(np.mean([a.get("mean", 0.0) for a in per_client_aucs]))
            agg_val_aucs: dict[str, float] = {
                lbl: float(np.nanmean([a.get(lbl, float("nan")) for a in per_client_aucs])) for lbl in self.label_cols
            }
            agg_val_aucs["mean"] = mean_val_auc

            raw_val_aucs.append(mean_val_auc)
            smoothed = _moving_average(raw_val_aucs, ma_window)
            sgd_steps = (round_idx + 1) * n_clients * local_sgd_steps

            record = RoundRecord(
                round_idx=round_idx,
                client_train_losses=train_losses,
                aggregation_weights=weights_used,
                val_aucs=agg_val_aucs,
                smoothed_val_auc=smoothed,
                sgd_steps=sgd_steps,
            )
            history.round_records.append(record)

            mean_loss = float(np.mean(list(train_losses.values())))
            pbar.set_postfix(
                {
                    "mean_train_loss": f"{mean_loss:.4f}",
                    "val_auc": f"{mean_val_auc:.4f}",
                    "smoothed": f"{smoothed:.4f}" if not np.isnan(smoothed) else "warmup",
                }
            )

            # ── Early stopping check ───────────────────────────────────────────
            if not np.isnan(smoothed):
                if smoothed > best_smoothed_auc:
                    best_smoothed_auc = smoothed
                    best_round_idx = round_idx
                    rounds_without_improvement = 0
                    self.save_checkpoint(best_ckpt_path)
                    tqdm.write(f"  [ES] R{round_idx + 1}: new best smoothed AUC = {smoothed:.4f} — checkpoint saved.")
                else:
                    rounds_without_improvement += 1
                    if rounds_without_improvement >= patience:
                        tqdm.write(
                            f"  [ES] Stopping early at round {round_idx + 1} "
                            f"(no improvement for {patience} rounds). Best: R{best_round_idx + 1}."
                        )
                        stopped_early = True
                        break

        history.rounds_completed = len(history.round_records)
        history.stopped_early = stopped_early
        history.best_round = best_round_idx
        tqdm.write(
            f"  [{experiment_name}] Training complete — "
            f"{history.rounds_completed}/{rounds} rounds" + (" (early stop)" if stopped_early else "") + "."
        )

        # Restore the best checkpoint before final evaluation.
        if best_ckpt_path.exists():
            self.load_checkpoint(best_ckpt_path)
            tqdm.write(f"  [ES] Restored best checkpoint from round {best_round_idx + 1}.")
        else:
            tqdm.write("  [ES] Warning: no best checkpoint found — using final model weights.")

        # ── Final evaluation ───────────────────────────────────────────────────
        # Distribute the restored global model to all clients before evaluating.
        self._distribute()

        print("  Evaluating on client test sets ...", flush=True)
        for client in self.clients:
            print(f"\r    [{client.name}] ...", end="", flush=True)
            history.per_client_test_aucs[client.name] = client.evaluate("test")
            mean_auc = history.per_client_test_aucs[client.name].get("mean", float("nan"))
            print(f"\r    [{client.name}] mean AUC = {mean_auc:.4f}")

        print("  Evaluating on CheXpert test set ...", flush=True)
        history.chexpert_test_aucs = self.evaluate_chexpert(chexpert_loader)
        chexpert_mean = history.chexpert_test_aucs.get("mean", float("nan"))
        print(f"    [CheXpert] mean AUC = {chexpert_mean:.4f}")

        # Save final global model.
        self.save_checkpoint(output_dir / "final_model.pt")

        return history

    # ── Centralized baseline ───────────────────────────────────────────────────

    def run_centralized(
        self,
        epochs: int,
        chexpert_loader: DataLoader[Any],
        output_dir: Path,
        experiment_name: str,
        seed: int,
        batch_size: int,
        num_workers: int,
        image_column: str,
        uncertain_policy: str,
        *,
        patience: int = 10,
        ma_window: int = 3,
    ) -> FLHistory:
        """Train one model on all client data pooled; evaluate on all test sets.

        All client training datasets are concatenated into one
        :class:`torch.utils.data.ConcatDataset`.  The pooled model is trained
        for up to ``epochs`` epochs with early stopping based on a smoothed
        (moving-average) validation AUC, matching the FL early-stopping policy.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        history = FLHistory(
            experiment_name=experiment_name,
            strategy="centralized",
            seed=seed,
            rounds_completed=0,
        )

        # ── Build pooled DataLoaders ───────────────────────────────────────────
        # Use ConcatDataset so each client's DatasetCXR handles its own root_dir.
        pooled_train_dataset: ConcatDataset[Any] = ConcatDataset([c.train_loader.dataset for c in self.clients])
        pooled_val_dataset: ConcatDataset[Any] = ConcatDataset([c.val_loader.dataset for c in self.clients])

        generator = torch.Generator()
        generator.manual_seed(seed)

        pooled_train_loader = DataLoader(
            pooled_train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            generator=generator,
            pin_memory=True,
        )
        pooled_val_loader = DataLoader(
            pooled_val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        # Compute positive weights from the union of all training DataFrames.
        all_train_df = pd.concat([c.split.train_df for c in self.clients], ignore_index=True)
        pos_weight = compute_pos_weights(all_train_df, self.label_cols).to(self.device)

        # Use the first client's learning rate for the pooled optimiser.
        lr = self.clients[0].lr
        optimiser = torch.optim.Adam(self.global_model.parameters(), lr=lr)

        # ── Early stopping state ───────────────────────────────────────────────
        raw_val_aucs: list[float] = []
        best_smoothed_auc = float("-inf")
        best_epoch_idx = -1
        rounds_without_improvement = 0
        stopped_early = False
        best_ckpt_path = output_dir / "best_model.pt"

        pbar = tqdm(range(epochs), desc=f"[{experiment_name}]", unit="epoch")
        for epoch_idx in pbar:
            pbar.set_description(f"[{experiment_name}] E{epoch_idx + 1}/{epochs} | training")
            self.global_model.train()
            total_loss = 0.0
            for _, imgs, targets in pooled_train_loader:
                labels, masks = targets
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                masks = masks.to(self.device)
                optimiser.zero_grad()
                logits = self.global_model(imgs)
                loss = masked_bce_loss(logits, labels, masks, pos_weight)
                loss.backward()
                optimiser.step()
                total_loss += loss.item()

            train_loss = total_loss / max(1, len(pooled_train_loader))

            # Validation AUC on pooled val set.
            val_metrics = evaluate(self.global_model, pooled_val_loader, self.device, self.label_cols)
            mean_val_auc = val_metrics.get("mean", 0.0)

            raw_val_aucs.append(mean_val_auc)
            smoothed = _moving_average(raw_val_aucs, ma_window)
            sgd_steps = (epoch_idx + 1) * len(pooled_train_loader)

            record = RoundRecord(
                round_idx=epoch_idx,
                client_train_losses={"pooled_train": train_loss, "pooled_val_auc": mean_val_auc},
                aggregation_weights={},
                val_aucs=val_metrics,
                smoothed_val_auc=smoothed,
                sgd_steps=sgd_steps,
            )
            history.round_records.append(record)
            pbar.set_postfix(
                {
                    "train_loss": f"{train_loss:.4f}",
                    "val_auc": f"{mean_val_auc:.4f}",
                    "smoothed": f"{smoothed:.4f}" if not np.isnan(smoothed) else "warmup",
                }
            )

            # ── Early stopping check ───────────────────────────────────────────
            if not np.isnan(smoothed):
                if smoothed > best_smoothed_auc:
                    best_smoothed_auc = smoothed
                    best_epoch_idx = epoch_idx
                    rounds_without_improvement = 0
                    self.save_checkpoint(best_ckpt_path)
                    tqdm.write(f"  [ES] E{epoch_idx + 1}: new best smoothed AUC = {smoothed:.4f} — checkpoint saved.")
                else:
                    rounds_without_improvement += 1
                    if rounds_without_improvement >= patience:
                        tqdm.write(
                            f"  [ES] Stopping early at epoch {epoch_idx + 1} "
                            f"(no improvement for {patience} epochs). Best: E{best_epoch_idx + 1}."
                        )
                        stopped_early = True
                        break

        history.rounds_completed = len(history.round_records)
        history.stopped_early = stopped_early
        history.best_round = best_epoch_idx
        tqdm.write(
            f"  [{experiment_name}] Training complete — "
            f"{history.rounds_completed}/{epochs} epochs" + (" (early stop)" if stopped_early else "") + "."
        )

        # Restore the best checkpoint before final evaluation.
        if best_ckpt_path.exists():
            self.load_checkpoint(best_ckpt_path)
            tqdm.write(f"  [ES] Restored best checkpoint from epoch {best_epoch_idx + 1}.")
        else:
            tqdm.write("  [ES] Warning: no best checkpoint found — using final model weights.")

        # ── Final evaluation ───────────────────────────────────────────────────
        print("  Evaluating on client test sets ...", flush=True)
        for client in self.clients:
            # Load restored global weights into each client's model for evaluation.
            client.set_global_weights({k: v.cpu() for k, v in self.global_model.state_dict().items()})
            print(f"\r    [{client.name}] ...", end="", flush=True)
            history.per_client_test_aucs[client.name] = client.evaluate("test")
            mean_auc = history.per_client_test_aucs[client.name].get("mean", float("nan"))
            print(f"\r    [{client.name}] mean AUC = {mean_auc:.4f}")

        print("  Evaluating on CheXpert test set ...", flush=True)
        history.chexpert_test_aucs = self.evaluate_chexpert(chexpert_loader)
        chexpert_mean = history.chexpert_test_aucs.get("mean", float("nan"))
        print(f"    [CheXpert] mean AUC = {chexpert_mean:.4f}")

        # Save final global model.
        self.save_checkpoint(output_dir / "final_model.pt")

        return history
