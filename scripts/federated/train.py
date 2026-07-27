r"""Train the DeMICAF pathology classifier via federated averaging (FedAvg).

This is the FedAvg baseline pathology classifier used by the per-cause
benchmark's ``knn``/``maha_pp``/``gen`` OOD-detection baselines: a ResNet-18
multi-label classifier trained on the 3 client datasets (ChestX-ray8,
MIMIC-CXR, PadChest); CheXpert is held out and used only for the final
external-evaluation AUC — never seen during training. Its penultimate-layer
features feed the ``knn``/``maha_pp`` per-cause benchmark scorers (see
``scripts/extract_classifier_features.py``).

Typical usage::

    # Verify data loading only (fast).
    uv run python -m scripts.federated.train --dry-run

    # Time a handful of rounds before committing to the full run.
    uv run python -m scripts.federated.train --rounds 3

    # Full run (resumes are not supported — early stopping keeps this bounded;
    # re-run from scratch if interrupted).
    uv run python -m scripts.federated.train
"""

from __future__ import annotations

import argparse
from pathlib import Path

from DeMICAF.federated.client import FLClient
from DeMICAF.federated.data import ClientConfig, load_client_splits, load_evaluation_set, make_data_loader
from DeMICAF.federated.server import FLServer
from DeMICAF.utils.paths import get_cxr_root, get_results_root, resolve_path
from DeMICAF.utils.seeding import set_global_determinism

# CheXpert is excluded from training and used only as the held-out evaluation dataset.
CLIENTS = ["ChestX-ray8", "PadChest", "MIMIC-CXR"]
PRIOR_DATASET = "CheXpert"
LABEL_COLS = ["Cardiomegaly", "Atelectasis", "Nodule", "Alveolar Pattern", "Pleural Effusion", "Pneumothorax"]


def parse_args() -> argparse.Namespace:
    """Define and parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train the DeMICAF pathology classifier via FedAvg.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", type=str, default=str(get_cxr_root()))
    parser.add_argument("--out-dir", type=Path, default=get_results_root() / "per_cause" / "classifier")
    parser.add_argument("--architecture", type=str, default="resnet18", choices=["resnet18", "resnet152"])
    parser.add_argument("--dropout-rate", type=float, default=0.1)
    parser.add_argument("--rounds", type=int, default=100, help="Maximum FL communication rounds.")
    parser.add_argument("--local-sgd-steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--n-val", type=int, default=200)
    parser.add_argument("--patience", type=int, default=5, help="Rounds without val-AUC improvement before stopping.")
    parser.add_argument("--ma-window", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--experiment-name", type=str, default="fedavg_resnet18/seed_0")
    parser.add_argument("--dry-run", action="store_true", help="Verify data loading without training.")
    return parser.parse_args()


def main() -> None:
    """Load the 3 client splits, run FedAvg training, and save checkpoints/history."""
    args = parse_args()
    set_global_determinism(args.seed)
    datasets_root = resolve_path(args.data_root)

    print(f"\n{'═' * 72}")
    print(f"  Experiment : {args.experiment_name}")
    print("  Strategy   : fedavg")
    print(f"  Seed       : {args.seed}")
    print(f"  Output     : {args.out_dir}")
    print(f"{'═' * 72}\n")

    clients: list[FLClient] = []
    for ci, name in enumerate(CLIENTS, 1):
        client_cfg = ClientConfig(
            name=name,
            harmonized_csv="harmonized.csv",
            subsets_csv="subsets.csv",
            training_col="subset1k",
            val_pool_col="subset100k",
            test_col="test",
        )
        print(f"\r  [{ci}/{len(CLIENTS)}] Loading {name} ...", end="", flush=True)
        split = load_client_splits(
            datasets_root=datasets_root,
            client_cfg=client_cfg,
            label_cols=LABEL_COLS,
            n_val=args.n_val,
            image_column="image_path",
            seed=args.seed,
        )
        print(
            f"\r  [{ci}/{len(CLIENTS)}] {name:20s} | "
            f"train={len(split.train_df):4d}  val={len(split.val_df):4d}  test={len(split.test_df):4d}"
        )

        if args.dry_run:
            for split_name, df in [("train", split.train_df), ("val", split.val_df), ("test", split.test_df)]:
                loader = make_data_loader(
                    df,
                    split.root_dir,
                    label_cols=LABEL_COLS,
                    is_train=(split_name == "train"),
                    batch_size=args.batch_size,
                    num_workers=0,
                    image_column="image_path",
                    uncertain_policy="ignore",
                    seed=args.seed,
                )
                print(f"    [{name}] {split_name}: {len(loader.dataset)} samples, {len(loader)} batches")
            continue

        clients.append(
            FLClient(
                split=split,
                label_cols=LABEL_COLS,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                lr=args.lr,
                uncertain_policy="ignore",
                device=args.device,
                architecture=args.architecture,
                dropout_rate=args.dropout_rate,
                pretrained=True,
                image_column="image_path",
                seed=args.seed,
            )
        )

    if args.dry_run:
        print(f"\n  [DRY-RUN] {args.experiment_name}: all data splits verified — no training performed.")
        return

    print(f"  Loading {PRIOR_DATASET} evaluation set ...", end="", flush=True)
    chexpert_test_df, chexpert_root = load_evaluation_set(
        datasets_root=datasets_root,
        dataset_name=PRIOR_DATASET,
        label_cols=LABEL_COLS,
        subsets_csv="subsets.csv",
        test_col="test",
        image_column="image_path",
    )
    chexpert_loader = make_data_loader(
        chexpert_test_df,
        chexpert_root,
        label_cols=LABEL_COLS,
        is_train=False,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        image_column="image_path",
        uncertain_policy="ignore",
        seed=args.seed,
    )
    print(f"\r  {PRIOR_DATASET} eval set: {len(chexpert_test_df)} samples.            ")

    print("  Building server ...", end="", flush=True)
    server = FLServer(
        clients=clients,
        label_cols=LABEL_COLS,
        architecture=args.architecture,
        dropout_rate=args.dropout_rate,
        pretrained=True,
        device=args.device,
    )
    print("\r  Server ready.                   ")

    output_dir = args.out_dir / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    history = server.run_federated(
        rounds=args.rounds,
        local_sgd_steps=args.local_sgd_steps,
        chexpert_loader=chexpert_loader,
        output_dir=output_dir,
        experiment_name=args.experiment_name,
        seed=args.seed,
        patience=args.patience,
        ma_window=args.ma_window,
    )

    history.save(output_dir / "history.json")
    history.save_round_metrics_csv(output_dir / "round_metrics.csv")
    history.save_test_aucs_csv(output_dir / "test_aucs.csv")

    print(f"\n  Rounds completed : {history.rounds_completed}")
    print(f"  Stopped early    : {history.stopped_early}")
    if history.stopped_early:
        print(f"  Best round       : {history.best_round + 1}")
    for client_name, aucs in history.per_client_test_aucs.items():
        print(f"  Test AUC [{client_name:20s}] mean = {aucs.get('mean', float('nan')):.4f}")
    chexpert_mean = history.chexpert_test_aucs.get("mean", float("nan"))
    print(f"  Test AUC [{PRIOR_DATASET:20s}] mean = {chexpert_mean:.4f}")
    print(f"\n  Results saved → {output_dir}\n")


if __name__ == "__main__":
    main()
