"""Loss/scorer ablation: contrastive framework and ranking-function comparison.

Backs Sec. 5's claim that "we use SupCon loss ... and unclustered MD for
Scoring, as they brought the best performance on our annotations": trains
encoders under each of SimCLR/CSI/UniCon/UniConSA (SupCon is trained
separately, see ``scripts/scoring/generalization.py``), then sweeps clustered
Mahalanobis and kNN-cosine scorers to compare against unclustered MD.
"""

import argparse
import copy
import datetime

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp

matplotlib.use("Agg")

import matplotlib.colors
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

matplotlib.rcParams["font.family"] = "Times New Roman"

from DeMICAF.caf.encoder import encode, train
from DeMICAF.caf.scorer import cosine_sim, mahalanobis
from DeMICAF.data.base import AugmentedDataset, DatasetBase, multi_view_collate
from DeMICAF.data.datasets import ChestXray, ChestXrayDataset
from DeMICAF.evaluation.auc import compute_auc
from DeMICAF.utils.io import load_features
from DeMICAF.utils.io import upsert_scores_csv as save_scores_csv
from DeMICAF.utils.paths import get_cxr_root, get_results_root
from DeMICAF.utils.seeding import seed_worker, set_seed

CONTRASTIVE_LOSSES = ["SimCLR", "CSI", "UniCon", "UniConSA"]  # , "SupCon" already computed
TARGET_DATASETS = ["CheXpert", "All"]
BASE_DATASETS = ["CheXpert", "ChestX-ray8", "MIMIC-CXR", "PadChest"]
MAHALA_CLUSTERS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
KNN_CS_NEIGHBORS = [1, 5, 10, 50, 100, 500, 1000, 5000, 10000]
MAHALA_CLUSTERS = []
KNN_CS_NEIGHBORS = [2, 3, 4]
PERCENTAGE = True

BATCH_SIZE = 256
EPOCHS = 100
LEARNING_RATE = 1e-4
NUM_WORKERS = 8
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

DEFAULT_COLORS = {
    "SimCLR": "#0033A0",
    "CSI": "#7BDA19",
    "UniCon": "#A6A0F5",
    "UniConSA": "#888888",
    "SupCon": "#64C7EA",
}

METRIC_STYLES = {
    "mahalanobis": {"short": "MD", "linestyle": "-", "marker": "o", "filled": True},
    "knncs": {"short": "CS", "linestyle": "--", "marker": "x", "filled": False},
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train encoders, extract features, compute clustered Mahalanobis and kNN cosine "
            "scores for CheXpert and All across frameworks."
        )
    )
    parser.add_argument(
        "--subset",
        choices=["1k", "10k", "100k"],
        default=None,
        help="Run only the given subset. Runs all subsets if omitted.",
    )
    parser.add_argument(
        "--frameworks", nargs="+", default=CONTRASTIVE_LOSSES, choices=CONTRASTIVE_LOSSES, help="Frameworks to include."
    )

    parser.add_argument("--train", action="store_true", help="Run only the training section.")
    parser.add_argument("--features", action="store_true", help="Run only the feature extraction section.")
    parser.add_argument("--scoring", action="store_true", help="Run only the score section.")
    parser.add_argument("--auc", action="store_true", help="Run only the AUC section.")
    parser.add_argument("--report", action="store_true", help="Run only the report section.")
    args = parser.parse_args()

    subsets = [args.subset] if args.subset else ["1k", "10k", "100k"]

    selected_any = args.train or args.features or args.scoring or args.auc or args.report
    run_train = args.train or not selected_any
    run_features = args.features or not selected_any
    run_scoring = args.scoring or not selected_any
    run_auc = args.auc or not selected_any
    run_report = args.report or not selected_any

    set_seed(0)
    mp.set_start_method("spawn", force=True)

    assets_root = get_results_root()
    results_root = get_results_root() / "loss_scorer_ablation"
    data_root = get_cxr_root()

    for subset in subsets:
        subset_name = "annotations.csv" if subset == "1k" else f"subset{subset}.csv"

        # BUILD DATASETS
        dataframes: dict[str, pd.DataFrame] = {}
        datasets: dict[str, DatasetBase] = {}
        if run_train or run_features:
            print(f"Retrieving {subset} dataframes...")
            # Build all metadata tables once.
            for dataset_name in BASE_DATASETS:
                subset_csv = data_root / dataset_name / subset_name
                dataframe = pd.read_csv(subset_csv)
                dataframes[dataset_name] = dataframe

            # Enforce identical dataset lengths for training parity.
            dataset_lengths = {name: len(df) for name, df in dataframes.items()}
            assert len(set(dataset_lengths.values())) == 1, f"Datasets must have the same length: {dataset_lengths}"
            subset_samples = len(dataframes["CheXpert"])

            if run_train:
                mixed_df_parts = []
                for dataset_name in BASE_DATASETS:
                    current_df = dataframes[dataset_name]
                    n_samples = subset_samples // 4
                    mixed_df_parts.append(current_df.iloc[:n_samples])
                dataframes["All"] = pd.concat(mixed_df_parts, ignore_index=True)

            print("Building datasets...")
            for dataset_name, dataframe in dataframes.items():
                datasets[dataset_name] = ChestXrayDataset(
                    root_dir=str(data_root),
                    dataframe=dataframe,
                )
                print(f"{len(datasets[dataset_name]) / 1000:.0f}k samples in {dataset_name}")

        model_names: dict[str, str] = {}
        if run_train:
            print(f"Training {subset} encoders, using device: {DEVICE}.")
            for dataset_name, dataset in datasets.items():
                if dataset_name not in TARGET_DATASETS:
                    continue

                for loss in tqdm(
                    CONTRASTIVE_LOSSES, desc=f"Training {dataset_name.replace('/', ' ').replace('_', ' ')} losses"
                ):
                    date = datetime.datetime.now().strftime("%b_%d_%H_%M")
                    modelname = loss + "_" + date
                    saveto_path = assets_root / "encoders" / subset / dataset_name / modelname
                    saveto_path.mkdir(parents=True, exist_ok=True)

                    encoder = dataset.get_network(bn_head=True, normalize_f=True).to(DEVICE)
                    optimizer = torch.optim.Adam(encoder.parameters(), lr=LEARNING_RATE)

                    view_dataset = AugmentedDataset(dataset, transforms=copy.deepcopy(ChestXray))
                    if loss == "SimCLR":
                        view_dataset.transforms.OOD = None

                    dataloader = DataLoader(
                        view_dataset,
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        num_workers=NUM_WORKERS,
                        worker_init_fn=seed_worker,
                        collate_fn=multi_view_collate,
                        drop_last=False,
                        pin_memory=True,
                        persistent_workers=NUM_WORKERS > 0,
                        prefetch_factor=2 if NUM_WORKERS > 0 else None,
                    )

                    train(
                        dataloader,
                        encoder,
                        optimizer,
                        saveto_path,
                        epochs=EPOCHS,
                        framework=loss,
                    )

                    del dataloader
                    torch.cuda.empty_cache()
                    model_names[dataset_name + "_" + loss] = modelname

        if run_features:
            print("\nCOMPUTING FEATURES...")
            # load all encoders
            encoders = {}
            for dataset_name in TARGET_DATASETS:
                if dataset_name != "All":
                    continue

                for loss in CONTRASTIVE_LOSSES:
                    encoder_name = dataset_name + "_" + loss
                    modelname = model_names.get(encoder_name, loss)
                    model_path = assets_root / "encoders" / subset / dataset_name / modelname / "model.pt"

                    encoders[encoder_name] = datasets["CheXpert"].get_network(bn_head=True, normalize_f=True).to(DEVICE)
                    encoders[encoder_name].load_state_dict(torch.load(model_path, map_location=DEVICE))

            datasets.pop("All", None)
            for dataset_name, dataset in datasets.items():
                print(f"\n--- {dataset_name.replace('/', ' ')} ---------------")
                dataloader = DataLoader(
                    dataset,
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                    num_workers=NUM_WORKERS,
                    worker_init_fn=seed_worker,
                    collate_fn=None,
                    drop_last=False,
                    pin_memory=True,
                    persistent_workers=NUM_WORKERS > 0,
                    prefetch_factor=2 if NUM_WORKERS > 0 else None,
                )

                for encoder_name, encoder in encoders.items():
                    dataset = encoder_name.split("_")[0]
                    loss = encoder_name.split("_")[-1]

                    features_filename = f"{dataset_name}_using_{dataset}"
                    if loss != "SupCon":
                        features_filename += "_" + loss
                    features_filename += ".npz"
                    features_path = assets_root / "features" / subset / features_filename

                    encode(dataloader, encoder, features_path, desc=f"Using {encoder_name}")

                del dataloader
                torch.cuda.empty_cache()

        if run_scoring:
            print("\nSCORING...")
            for encoder_dataset in TARGET_DATASETS:
                print(f"\n--- Encoder trained on {encoder_dataset} ---")
                for loss in [*CONTRASTIVE_LOSSES, "SupCon"]:
                    print(f"Using {loss} loss...")
                    score_file = assets_root / "scores" / "loss_scorer_ablation" / f"{subset}_{encoder_dataset}_{loss}.csv"

                    all_feats: dict[str, np.ndarray] = {}
                    all_names: dict[str, np.ndarray] = {}
                    for dataset_name in BASE_DATASETS:
                        encoder_name = encoder_dataset + "_" + loss if loss != "SupCon" else encoder_dataset
                        all_names[dataset_name], all_feats[dataset_name] = load_features(
                            assets_root / "features" / subset / f"{dataset_name}_using_{encoder_name}.npz"
                        )
                    # Concatenate features and names from all datasets to create "All"
                    all_feats["All"] = np.concatenate([all_feats[ds] for ds in BASE_DATASETS], axis=0)
                    all_names["All"] = np.concatenate([all_names[ds] for ds in BASE_DATASETS], axis=0)

                    # Iterate over evaluation sets and their features (no tqdm here)
                    for eval_set, feats in all_feats.items():
                        names = all_names[eval_set]
                        # tqdm for all k in MAHALA_CLUSTERS and KNN_CS_NEIGHBORS
                        len(MAHALA_CLUSTERS) + len(KNN_CS_NEIGHBORS)
                        k_iter = [("MD", k) for k in MAHALA_CLUSTERS] + [("CS", k) for k in KNN_CS_NEIGHBORS]
                        for method, k in tqdm(k_iter, desc=f"{eval_set} scoring", unit="k", leave=True):
                            if method == "MD":
                                scores, _ = mahalanobis(feats, feats, k=k)
                                save_scores_csv(
                                    csv_path=score_file,
                                    tag=f"MD_k{k}",
                                    image_names=names,
                                    scores=np.asarray(scores),
                                    dataset_name=eval_set,
                                )
                            else:
                                scores = cosine_sim(feats, feats, k=k)
                                save_scores_csv(
                                    csv_path=score_file,
                                    tag=f"CS_k{k}",
                                    image_names=names,
                                    scores=np.asarray(scores),
                                    dataset_name=eval_set,
                                )

        if run_auc:
            results_root.mkdir(parents=True, exist_ok=True)
            annotation_dfs = []
            for dataset_name in BASE_DATASETS:
                ann_path = data_root / dataset_name / "annotations.csv"
                df = pd.read_csv(ann_path)
                df["dataset"] = dataset_name  # Optionally add a column to track source
                annotation_dfs.append(df)
            annotations_df = pd.concat(annotation_dfs, ignore_index=True)

            auc_rows = []
            for encoder_dataset in TARGET_DATASETS:
                auc_row = {"Encoder Dataset": encoder_dataset}
                for loss in [*CONTRASTIVE_LOSSES, "SupCon"]:
                    auc_row["Loss"] = loss
                    scores_df = pd.read_csv(assets_root / "scores" / "loss_scorer_ablation" / f"{subset}_{encoder_dataset}_{loss}.csv")

                    for scoring in tqdm(
                        [score for score in scores_df.columns if score not in ["image_path", "dataset"]],
                        desc=f"AUC for {encoder_dataset} {loss}",
                        unit="score",
                    ):
                        auc_row["Scoring"] = scoring

                        focused_scores = scores_df.loc[:, ["image_path", "dataset", scoring]].rename(
                            columns={scoring: "score"}
                        )

                        for dataset_name in [*BASE_DATASETS, "All"]:
                            filtered_scores = focused_scores[focused_scores["dataset"] == dataset_name]
                            if dataset_name == "All":
                                filtered_labels = annotations_df
                            else:
                                filtered_labels = annotations_df[annotations_df["dataset"] == dataset_name]

                            # compute_auc expects annotations first, then scores (column "score").
                            auc_row[dataset_name] = compute_auc(
                                filtered_labels,
                                filtered_scores,
                                percentage=PERCENTAGE,
                                invert_labels=True,
                            )
                        auc_rows.append(auc_row.copy())

                        auc_df = pd.DataFrame(
                            auc_rows,
                            columns=[
                                "Encoder Dataset",
                                "Loss",
                                "Scoring",
                                "CheXpert",
                                "ChestX-ray8",
                                "MIMIC-CXR",
                                "PadChest",
                                "All",
                            ],
                        )
                        auc_df.to_csv(results_root / f"{subset}_AUCs.csv", index=False)

        if run_report:
            plots = [
                {"Eval Set": "CheXpert", "Encoder": "CheXpert"},
                {"Eval Set": "All", "Encoder": "CheXpert"},
                {"Eval Set": "All", "Encoder": "All"},
                {"Eval Set": "CheXpert", "Encoder": "All"},
                {"Eval Set": "ChestX-ray8", "Encoder": "All"},
                {"Eval Set": "MIMIC-CXR", "Encoder": "All"},
                {"Eval Set": "PadChest", "Encoder": "All"},
            ]
            percentage = PERCENTAGE
            auc_df = pd.read_csv(results_root / f"{subset}_AUCs.csv")
            auc_df["Function"] = auc_df["Scoring"].apply(lambda x: "mahalanobis" if x.startswith("MD") else "knncs")
            auc_df["k"] = auc_df["Scoring"].apply(lambda x: int(x.split("_k")[1]))

            y_min, y_max = (
                auc_df[["CheXpert", "ChestX-ray8", "MIMIC-CXR", "PadChest", "All"]].min().min(),
                auc_df[["CheXpert", "ChestX-ray8", "MIMIC-CXR", "PadChest", "All"]].max().max(),
            )
            y_min = y_min - (y_max - y_min) * 0.02
            y_max = y_max + (y_max - y_min) * 0.02
            if percentage:
                y_min *= 100
                y_max *= 100

            # plots = [{"Eval Set": "CheXpert", "Encoder": "CheXpert"}]
            # percentage = False; y_min = 0.550; y_max = 0.755

            for plot in plots:
                filtered_df = auc_df.loc[auc_df["Encoder Dataset"] == plot["Encoder"]]
                filtered_df.rename(columns={plot["Eval Set"]: "AUC"}, inplace=True)
                k_values = filtered_df["k"].unique()

                k_to_position = {k: idx for idx, k in enumerate(k_values)}

                fig, ax = plt.subplots(figsize=(5, 4))
                # fig, ax = plt.subplots(figsize=(6, 3.8))
                plotted = False

                for loss in [*CONTRASTIVE_LOSSES, "SupCon"]:
                    fw_df = filtered_df[filtered_df["Loss"] == loss]
                    if fw_df.empty:
                        continue

                    for metric_key in ["mahalanobis", "knncs"]:
                        style = METRIC_STYLES[metric_key]
                        metric_df = fw_df[fw_df["Function"] == metric_key].sort_values("k")
                        if metric_df.empty:
                            continue

                        xs = [k_to_position[k] for k in metric_df["k"].tolist() if k in k_to_position]
                        ys = metric_df["AUC"].tolist()
                        if percentage:
                            ys = [y * 100 for y in ys]
                        if not xs:
                            continue

                        marker_face = DEFAULT_COLORS[loss] if style["filled"] else "none"
                        ax.plot(
                            xs,
                            ys,
                            color=DEFAULT_COLORS[loss],
                            linestyle=style["linestyle"],
                            marker=style["marker"],
                            markerfacecolor=marker_face,
                            markeredgecolor=DEFAULT_COLORS[loss],
                            linewidth=1.8,
                            markersize=6,
                            label=f"{loss} - {style['short']}",
                        )
                        plotted = True

                    if not plotted:
                        plt.close(fig)
                        continue

                    # ax.set_title(f"Encoder train: {plot['Encoder']} | Evaluation: {plot['Eval Set']}", fontsize=14)
                    ax.set_xlabel("k (clusters / neighbors)")
                    ax.set_ylabel("AUC")
                    ax.set_xticks(list(range(len(k_values))))
                    ax.set_xticklabels([str(k) for k in k_values], rotation=45, ha="right")
                    ax.set_ylim(y_min, y_max)
                    ax.grid(axis="y", alpha=0.3)
                    # ax.legend(
                    #     loc="upper left",
                    #     bbox_to_anchor=(1.02, 1.0),
                    #     borderaxespad=0,
                    #     ncol=1,
                    #     frameon=True,
                    #     fontsize=11,
                    #     title=None,
                    #     alignment="left"
                    # )

                    fig.tight_layout()
                    results_root.mkdir(parents=True, exist_ok=True)
                    fig.savefig(
                        results_root / f"{plot['Eval Set']}_using_{plot['Encoder']}.pdf",
                        format="pdf",
                        bbox_inches="tight",
                    )
                    plt.close(fig)


if __name__ == "__main__":
    main()
