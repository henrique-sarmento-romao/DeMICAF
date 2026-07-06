import argparse
import copy
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18
from tqdm import tqdm

from data_compliance.caf.encoder import encode, train
from data_compliance.caf.scorer import mahalanobis_distance, reference_vector
from data_compliance.data.base import AugmentedDataset, DatasetBase, RandomPatchDataset, multi_view_collate
from data_compliance.data.datasets import ChestXray, ChestXrayDataset
from data_compliance.evaluation.auc import compute_auc
from data_compliance.utils.io import load_features
from data_compliance.utils.io import upsert_scores_csv as save_scores_csv
from data_compliance.utils.paths import get_repo_root
from data_compliance.utils.seeding import seed_worker, set_seed

study = "CA3_TDD"
base_datasets = ["CheXpert", "ChestX-ray8", "MIMIC-CXR", "PadChest"]
framework = "SupCon"
batch_size = 256
epochs = 100
l_r = 1e-4
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
num_workers = 8
train_encoder = True
get_features = True
specific_models: list[str] = []
percentage = True


class ImageNetResNet18Features(nn.Module):
    """ResNet-18 feature extractor compatible with encode(..., head=False)."""

    def __init__(self) -> None:
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Adapt RGB pretrained stem to grayscale by averaging channel weights.
        conv1_rgb = backbone.conv1
        conv1_gray = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            conv1_gray.weight.copy_(conv1_rgb.weight.mean(dim=1, keepdim=True))
        backbone.conv1 = conv1_gray

        self.feats = nn.Sequential(*list(backbone.children())[:-1])

    def forward(self, x: torch.Tensor, head: bool = False) -> torch.Tensor:
        del head
        return self.feats(x).flatten(1)


def export_methods_table(csv_path: Path, caption: str, label: str, percentage=True) -> None:
    """Export a LaTeX table from a CSV file, matching the provided example's structure and formatting."""
    df = pd.read_csv(csv_path)
    # Map 'MIMIC-CXR' to 'MIMIC' in columns and Encoder values
    df = df.rename(columns={"MIMIC-CXR": "MIMIC"})
    if "Encoder" in df.columns:
        df["Encoder"] = df["Encoder"].replace("MIMIC-CXR", "MIMIC")

    col_order = ["Reference", "Encoder", "CheXpert", "ChestX-ray8", "MIMIC", "PadChest", "All", "Holdout"]
    ENCODER_ORDER = ["CheXpert", "ChestX-ray8", "MIMIC", "PadChest", "All"]
    SELF_ORDER = ["CheXpert", "ChestX-ray8", "MIMIC", "PadChest", "All", "ImageNet", "Aggregated"]

    lines = [
        r"\begin{table}[ht]",
        r"    \centering",
        f"    \\caption{{{caption}}} \\label{{{label}}}",
        r"    \begin{tabular}{>{\centering\arraybackslash}m{0.6cm}|c|c|c|c|c|c|c}",
        r"        \hline",
        r"        \multicolumn{2}{c|}{Scoring Method} & \multicolumn{6}{c}{Evaluation Dataset} \\",
        r"        \hline",
        r"        {Ref.} & {Encoder} & CheXpert & ChestX-ray8 & MIMIC & PadChest & All & Hold-Out \\",
        r"        \hhline{========}",
    ]

    for ref_type, multirow, row_order in [
        ("Self", r"\multirow[c]{7}{*}{\shortstack{S\\e\\l\\f}}", SELF_ORDER),
        ("Encoder", r"\multirow[c]{5}{*}{\shortstack{E\\n\\c\\o\\d\\e\\r}}", ENCODER_ORDER),
    ]:
        ref_rows = df[df["Reference"].str.lower() == ref_type.lower()]
        if ref_rows.empty:
            continue

        if ref_type == "Encoder":
            lines.append(r"        \hhline{========}")

        # Hard-enforce vertical order
        ref_rows = ref_rows.set_index("Encoder").reindex(row_order).reset_index()
        total = len(ref_rows)

        for idx, (_, row) in enumerate(ref_rows.iterrows()):
            # Hard-coded separator between "All" (idx=4) and "ImageNet" (idx=5) in Self group
            if ref_type == "Self" and idx == 5:
                lines.append(r"        \hhline{~|=======}")

            row_dict = row.to_dict()
            row_list = []
            row_list.append(multirow if idx == 0 else "")

            encoder = str(row_dict["Encoder"])
            for col in col_order[1:]:
                if col == "Reference":
                    continue
                val = row_dict.get(col, None)
                cell = ""
                if val is None or (isinstance(val, float) and pd.isna(val)):
                    cell = ""
                elif isinstance(val, int | float):
                    v = val * 100 if percentage else val
                    if encoder == col and encoder in ["CheXpert", "ChestX-ray8", "MIMIC", "PadChest"]:
                        cell = r"\cellcolor{gray!10} " + f"{v:.2f}"
                    else:
                        cell = f"{v:.2f}"
                else:
                    cell = str(val)
                row_list.append(cell)

            lines.append("        " + " & ".join(row_list) + r" \\")

            is_last = idx == total - 1
            is_before_hhline = ref_type == "Self" and idx == 4  # suppress \cline before \hhline
            if not is_last and not is_before_hhline:
                lines.append(r"        \cline{2-8}")

    lines.append(r"        \hline")
    lines.append(r"    \end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    out_path = csv_path.with_suffix(".tex")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def export_scaling_table(subset_tables: dict, caption: str, label: str, outdir: Path, percentage=True) -> None:
    """Export a LaTeX table from a dict of DataFrames (keys: '100k', '10k', '1k').

    Focuses on the All and Holdout columns for each subset.
    """
    subset_names = ["100k", "10k", "1k"]
    dfs = []
    for s in subset_names:
        df = subset_tables[s].copy()
        if "Encoder" in df.columns:
            df["Encoder"] = df["Encoder"].replace("MIMIC-CXR", "MIMIC")
        dfs.append(df)

    ENCODER_ORDER = ["CheXpert", "ChestX-ray8", "MIMIC", "PadChest", "All"]
    SELF_ORDER = ["CheXpert", "ChestX-ray8", "MIMIC", "PadChest", "All", "ImageNet", "Aggregated"]

    lines = [
        r"\begin{table}[ht]",
        r"    \centering",
        f"    \\caption{{{caption}}} \\label{{{label}}}",
        r"    \begin{tabular}{>{\centering\arraybackslash}m{0.6cm}|c|>{\centering\arraybackslash}m{1.2cm}"
        r"|c|c|>{\centering\arraybackslash}m{1.2cm}|c|c}",
        r"        \hline",
        r"        \multicolumn{2}{c|}{Scoring Method} & \multicolumn{3}{c|}{All} & \multicolumn{3}{c}{Hold-Out} \\",
        r"        \hline",
        r"        {Ref.} & {Encoder} & 100k & 10k & 1k & 100k & 10k & 1k \\",
        r"        \hhline{========}",
    ]

    for ref_type, multirow, row_order in [
        ("Self", r"\multirow[c]{7}{*}{\shortstack{S\\e\\l\\f}}", SELF_ORDER),
        ("Encoder", r"\multirow[c]{5}{*}{\shortstack{E\\n\\c\\o\\d\\e\\r}}", ENCODER_ORDER),
    ]:
        ref_rows = [df[df["Reference"].str.lower() == ref_type.lower()] for df in dfs]
        if all(r.empty for r in ref_rows):
            continue
        if ref_type == "Encoder":
            lines.append(r"        \hhline{========}")
        ref_rows = [r.set_index("Encoder").reindex(row_order).reset_index() for r in ref_rows]
        total = len(row_order)
        for idx in range(total):
            if ref_type == "Self" and idx == 5:
                lines.append(r"        \hhline{~|=======}")
            row_list = []
            row_list.append(multirow if idx == 0 else "")
            encoder = str(row_order[idx])
            row_list.append(encoder)
            for col in ["All", "Holdout"]:
                # Get 100k value for diff
                val_100k = None
                if idx < len(ref_rows[0]):
                    row_100k = ref_rows[0].iloc[idx]
                    val_100k = row_100k.get(col, None)
                    if isinstance(val_100k, int | float) and percentage:
                        val_100k = val_100k * 100
                for dfi in range(3):
                    val = None
                    if idx < len(ref_rows[dfi]):
                        row = ref_rows[dfi].iloc[idx]
                        val = row.get(col, None)
                    cell = ""
                    if val is None or (isinstance(val, float) and pd.isna(val)):
                        cell = ""
                    elif isinstance(val, int | float):
                        v = val * 100 if percentage else val
                        cell = f"{v:.2f}"
                        # Add difference for 10k/1k
                        if dfi > 0 and isinstance(val_100k, int | float) and val_100k is not None:
                            diff = v - val_100k
                            if abs(diff) > 1e-6:
                                cell += r" {\scriptsize \textit" + ("+" if diff >= 0 else "") + f"{diff:.2f}" + "}"
                    else:
                        cell = str(val)
                    row_list.append(cell)
            lines.append("        " + " & ".join(row_list) + r" \\")
            is_last = idx == total - 1
            is_before_hhline = ref_type == "Self" and idx == 4
            if not is_last and not is_before_hhline:
                lines.append(r"        \cline{2-8}")
    lines.append(r"        \hline")
    lines.append(r"    \end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")
    # Output to file (optional: could return as string)
    out_path = outdir / "auc_scaling.tex"
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train encoders, extract features, and compute scores.")
    parser.add_argument(
        "--subset",
        choices=["1k", "10k", "22k", "100k"],
        default=None,
        help="Run only the given subset (1k, 10k, or 100k). Runs all subsets if omitted.",
    )
    parser.add_argument(
        "--score", choices=["mahala", "chi"], default="mahala", help="Run only the given score type (mahala or chi)."
    )
    parser.add_argument("--train", action="store_true", help="Run only the training section.")
    parser.add_argument("--features", action="store_true", help="Run only the feature extraction section.")
    parser.add_argument("--scoring", action="store_true", help="Run only the score section.")
    parser.add_argument("--auc", action="store_true", help="Run only the AUC section.")
    parser.add_argument("--report", action="store_true", help="Run only the report section.")
    args = parser.parse_args()

    subsets = [args.subset] if args.subset else ["1k", "10k", "100k"]
    score_name = "mahalanobis" if args.score == "mahala" else "chi"

    selected_any = args.train or args.features or args.scoring or args.auc or args.report
    run_train = args.train or not selected_any
    run_features = args.features or not selected_any
    run_score = args.scoring or not selected_any
    run_auc = args.auc or not selected_any
    run_report = args.report or not selected_any

    set_seed(0)
    mp.set_start_method("spawn", force=True)

    repo_root = get_repo_root()
    assets_root = repo_root / "Assets"
    results_root = repo_root / "Results" / study
    data_root = Path("dvc/chest_xray")

    encoder_names = [*base_datasets, "ImageNet", "Aggregated", "All"]

    for subset in subsets:
        subset_samples: int | None = None

        if subset == "1k":
            subset_name = "annotations.csv"
        elif subset == "22k":
            subset_name = "subset.csv"
        else:
            subset_name = f"subset{subset}.csv"

        dataframes: dict[str, pd.DataFrame] = {}
        datasets: dict[str, DatasetBase] = {}
        models_by_combo: dict[str, str] = {}

        scores_filename = assets_root / "Scores" / study / f"{subset}_{score_name}.csv"

        # GET DATASETS
        if run_train or run_features or run_score:
            print(f"Retrieving {subset} dataframes...")
            # Build all metadata tables once.
            for dataset_name in base_datasets:
                subset_csv = data_root / dataset_name / subset_name
                dataframe = pd.read_csv(subset_csv)
                dataframes[dataset_name] = dataframe

            # Enforce identical dataset lengths for training parity.
            dataset_lengths = {name: len(df) for name, df in dataframes.items()}
            assert len(set(dataset_lengths.values())) == 1, f"Datasets must have the same length: {dataset_lengths}"
            subset_samples = len(dataframes["CheXpert"])

            # make All dataset
            mixed_df_parts = []
            for dataset_name in base_datasets:
                current_df = dataframes[dataset_name]
                n_samples = subset_samples // 4
                mixed_df_parts.append(current_df.iloc[:n_samples])
            dataframes["All"] = pd.concat(mixed_df_parts, ignore_index=True)

            if not run_score:
                print("Building datasets...")
                for dataset_name, dataframe in dataframes.items():
                    datasets[dataset_name] = ChestXrayDataset(
                        root_dir=str(data_root),
                        dataframe=dataframe,
                    )
                    print(f"{len(datasets[dataset_name]) / 1000:.0f}k samples in {dataset_name}")

        """
        TRAIN ENCODERS
        --------------------------------------------------------
        Train ResNet-18 encoders with the following data combinations:
        - N CheXpert
        - N ChestX-ray8
        - N MIMIC-CXR
        - N PadChest
        - N/4 from each dataset

        Then compute the average of the model weights, leading to a new "aggregated" encoder.
        """
        if run_train:
            print(f"Training {subset} encoders, using device: {device}.")
            for dataset_name, dataset in datasets.items():
                print(f"\n--- {dataset_name.replace('/', ' ').replace('_', ' ')} ---------------")

                date = datetime.datetime.now().strftime("%b_%d_%H_%M")
                modelname = date
                saveto_path = assets_root / "Encoders" / dataset_name / subset / modelname
                saveto_path.mkdir(parents=True, exist_ok=True)

                encoder = dataset.get_network(bn_head=True, normalize_f=True).to(device)
                optimizer = torch.optim.Adam(encoder.parameters(), lr=l_r)

                view_dataset = AugmentedDataset(dataset, transforms=copy.deepcopy(ChestXray))
                train_ds = RandomPatchDataset(view_dataset) if hasattr(view_dataset, "patch_size") else view_dataset

                dataloader = DataLoader(
                    train_ds,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    worker_init_fn=seed_worker,
                    collate_fn=multi_view_collate,
                    drop_last=False,
                    pin_memory=True,
                    persistent_workers=num_workers > 0,
                    prefetch_factor=2 if num_workers > 0 else None,
                )

                train(
                    dataloader,
                    encoder,
                    optimizer,
                    saveto_path,
                    epochs=epochs,
                    framework=framework,
                )

                del dataloader, train_ds
                torch.cuda.empty_cache()

                models_by_combo[dataset_name] = modelname

            # Aggregated encoder
            timestamp = datetime.datetime.now().strftime("%b_%d_%H_%M")

            state_dicts = []
            for dataset_name in datasets:
                modelname = models_by_combo.get(dataset_name, "SupCon")
                model_path = assets_root / "Encoders" / subset / dataset_name / modelname / "model.pt"
                if not model_path.exists():
                    raise FileNotFoundError(f"Expected model checkpoint not found at: {model_path}")
                state_dicts.append(torch.load(model_path, map_location="cpu"))

            ref_keys = set(state_dicts[0].keys())
            for idx, state_dict in enumerate(state_dicts[1:], start=1):
                if set(state_dict.keys()) != ref_keys:
                    raise ValueError(f"Model at index {idx} has incompatible state_dict keys.")

            aggregated_state_dict = {}
            for key in state_dicts[0]:
                tensors = [state_dict[key] for state_dict in state_dicts]
                if torch.is_floating_point(tensors[0]):
                    stacked = torch.stack([tensor.float() for tensor in tensors], dim=0)
                    aggregated_state_dict[key] = stacked.mean(dim=0).to(dtype=tensors[0].dtype)
                else:
                    aggregated_state_dict[key] = tensors[0]

            aggregate_model_path = assets_root / "Encoders" / subset / "Aggregated" / timestamp / "model.pt"
            aggregate_model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(aggregated_state_dict, aggregate_model_path)
            models_by_combo["Aggregated"] = timestamp

            print("\n--- Aggregating trained checkpoints ---------------")
            print(f"Saved aggregated model to: {aggregate_model_path}")

        """
        GET FEATURES
        --------------------------------------------------------
        Get the features for the previous encoders, and also for a ResNet-18 ImageNet,
        representing a not trained encoder.
        The features are computed for the N images of each dataset which are fixed
        and for the 1000 annotated of each dataset.
        """
        if run_features:
            # Load encoders trained on this subset
            encoders = {}
            encoder_list = [encoder_name for encoder_name in encoder_names if encoder_name != "ImageNet"]
            for encoder in encoder_list:
                modelname = models_by_combo.get(encoder, "SupCon")
                model_path = assets_root / "Encoders" / subset / encoder / modelname / "model.pt"
                encoders[encoder] = datasets["CheXpert"].get_network(bn_head=True, normalize_f=True).to(device)
                encoders[encoder].load_state_dict(torch.load(model_path, map_location=device))
            encoders["ImageNet"] = ImageNetResNet18Features().to(device)

            for dataset_name, dataset in list(datasets.items()):
                # skip All dataset for feature extraction, as it corresponds to the same samples in the other datasets
                if dataset_name == "All":
                    continue

                print(f"\n--- {dataset_name.replace('/', ' ')} ---------------")
                dataloader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    worker_init_fn=seed_worker,
                    collate_fn=None,
                    drop_last=False,
                    pin_memory=True,
                    persistent_workers=num_workers > 0,
                    prefetch_factor=2 if num_workers > 0 else None,
                )

                for encoder_name, encoder in encoders.items():
                    features_path = assets_root / "Features" / subset / f"{dataset_name}_using_{encoder_name}.npz"
                    encode(dataloader, encoder, features_path, desc=encoder_name)
                    print(f"Saved to {features_path}")

                del dataloader
                torch.cuda.empty_cache()

        """
        COMPUTE SCORES
        --------------------------------------------------------
        """
        if run_score:
            if subset_samples is None:
                raise RuntimeError("subset_samples must be set before scoring.")

            for encoder_name in encoder_names:
                # GET FEATURES
                refer_feats: dict[str, np.ndarray] = {}
                refer_names: dict[str, np.ndarray] = {}

                score_feats: dict[str, np.ndarray] = {}
                score_names: dict[str, np.ndarray] = {}

                refer_names_parts, refer_feats_parts = [], []
                score_names_parts, score_feats_parts = [], []
                for dataset_name in base_datasets:
                    refer_names[dataset_name], refer_feats[dataset_name] = load_features(
                        assets_root / "Features" / subset / f"{dataset_name}_using_{encoder_name}.npz"
                    )
                    refer_names_parts.append(refer_names[dataset_name][: subset_samples // 4])
                    refer_feats_parts.append(refer_feats[dataset_name][: subset_samples // 4])

                    score_names[dataset_name], score_feats[dataset_name] = (
                        refer_names[dataset_name],
                        refer_feats[dataset_name],
                    )
                    if subset == "1k":
                        score_names_parts.append(refer_names[dataset_name])
                        score_feats_parts.append(refer_feats[dataset_name])

                refer_names["All"] = np.concatenate(refer_names_parts, axis=0)
                refer_feats["All"] = np.concatenate(refer_feats_parts, axis=0)
                if subset == "1k":
                    score_names["All"] = np.concatenate(score_names_parts, axis=0)
                    score_feats["All"] = np.concatenate(score_feats_parts, axis=0)
                else:
                    score_names["All"] = refer_names["All"]
                    score_feats["All"] = refer_feats["All"]

                # Build encoder-specific holdout sets from all other datasets.
                if encoder_name in base_datasets:
                    holdout_names_parts: list[np.ndarray] = []
                    holdout_feats_parts: list[np.ndarray] = []
                    score_names_parts, score_feats_parts = [], []

                    for dataset_name in base_datasets:
                        if dataset_name == encoder_name:
                            continue
                        holdout_names_parts.append(refer_names[dataset_name][: subset_samples // 3])
                        holdout_feats_parts.append(refer_feats[dataset_name][: subset_samples // 3])
                        if subset == "1k":
                            score_names_parts.append(refer_names[dataset_name])
                            score_feats_parts.append(refer_feats[dataset_name])

                    refer_names["Holdout"] = np.concatenate(holdout_names_parts, axis=0)
                    refer_feats["Holdout"] = np.concatenate(holdout_feats_parts, axis=0)
                    if subset == "1k":
                        score_names["Holdout"] = np.concatenate(score_names_parts, axis=0)
                        score_feats["Holdout"] = np.concatenate(score_feats_parts, axis=0)
                    else:
                        score_names["Holdout"] = refer_names["Holdout"]
                        score_feats["Holdout"] = refer_feats["Holdout"]

                for reference_mode in ["encoder", "self"]:
                    if reference_mode == "encoder" and encoder_name not in [*base_datasets, "All"]:
                        continue
                    scoring_method = reference_mode.capitalize() + "_" + encoder_name
                    print(f"{encoder_name} {subset} [{reference_mode.capitalize()} Reference]")

                    mean, cov = None, None
                    if reference_mode == "encoder":
                        mean, cov = reference_vector(refer_feats[encoder_name])

                    # Iterate over evaluation sets and their features
                    eval_set_iterator = tqdm(
                        score_feats.keys(),
                        desc="scoring",
                        unit="set",
                        leave=False,
                    )
                    for eval_set in eval_set_iterator:
                        image_names = score_names[eval_set]
                        if reference_mode == "self":
                            ref_mean, ref_cov = reference_vector(refer_feats[eval_set])
                        else:
                            if mean is None or cov is None:
                                raise RuntimeError("mean and cov must be set for encoder reference mode.")
                            ref_mean, ref_cov = mean, cov

                        scores = mahalanobis_distance(score_feats[eval_set], [ref_mean], [ref_cov])
                        dataset_name = eval_set if eval_set != "Holdout" else f"{encoder_name}_Holdout"
                        save_scores_csv(
                            csv_path=scores_filename,
                            tag=scoring_method,
                            image_names=image_names,
                            dataset_name=dataset_name,
                            scores=np.asarray(scores),
                        )

        """
        COMPUTE AUC FROM SCORES CSV
        --------------------------------------------------------
        """
        if run_auc:
            scores_df = pd.read_csv(scores_filename)

            annotation_dfs = []
            for dataset_name in base_datasets:
                ann_path = data_root / dataset_name / "annotations.csv"
                df = pd.read_csv(ann_path)
                annotation_dfs.append(df)
            annotations_df = pd.concat(annotation_dfs, ignore_index=True)
            annotations_df["dataset"] = annotations_df["image_path"].str.split("/").str[0]

            auc_rows = []
            for encoder_name in encoder_names:
                for reference_mode in ["Encoder", "Self"]:
                    if reference_mode == "Encoder" and encoder_name not in [*base_datasets, "All"]:
                        continue
                    scoring_method = reference_mode.capitalize() + "_" + encoder_name

                    focused_scores = scores_df.rename(columns={scoring_method: "score"})

                    auc_row = {"Reference": reference_mode, "Encoder": encoder_name}
                    for test_data in [*base_datasets, "All"]:
                        filtered_scores = focused_scores.loc[focused_scores["dataset"] == test_data]

                        if test_data == "All":
                            filtered_annotations = annotations_df
                        else:
                            filtered_annotations = annotations_df.loc[annotations_df["dataset"] == test_data]

                        # invert_labels=False keeps the historical CA2 convention (positive = non-compliant).
                        auc_row[test_data] = compute_auc(
                            filtered_annotations, filtered_scores, percentage=False, invert_labels=False
                        )

                    if encoder_name in base_datasets:
                        filtered_scores = focused_scores.loc[focused_scores["dataset"] == f"{encoder_name}_Holdout"]
                        filtered_annotations = annotations_df.loc[annotations_df["dataset"] != encoder_name]
                        auc_row["Holdout"] = compute_auc(
                            filtered_annotations, filtered_scores, percentage=False, invert_labels=False
                        )
                    else:
                        auc_row["Holdout"] = None
                    auc_rows.append(auc_row)
            auc = pd.DataFrame(auc_rows, columns=["Reference", "Encoder", *base_datasets, "All", "Holdout"])
            auc.to_csv(results_root / f"auc_{score_name}_{subset}.csv", index=False)

        if run_report:
            caption = "AUC percentage scores of encoders trained and evaluated across different datasets"
            if subset == "100k":
                caption += "."
            else:
                caption += f", using {subset} samples."
            export_methods_table(
                results_root / f"auc_{score_name}_{subset}.csv", caption=caption, label=f"tab:datagen_{subset}"
            )

    # After all processing, check if auc_mahala_1k, 10k, 100k exist and plot comparison if so
    if run_report:
        auc_1k, auc_10k, auc_100k = (
            results_root / "auc_mahalanobis_1k.csv",
            results_root / "auc_mahalanobis_10k.csv",
            results_root / "auc_mahalanobis_100k.csv",
        )

        if auc_1k.exists() and auc_10k.exists() and auc_100k.exists():
            # Load tables
            t1k = pd.read_csv(auc_1k)
            t10k = pd.read_csv(auc_10k)
            t100k = pd.read_csv(auc_100k)
            subset_mahala_tables = {"1k": t1k, "10k": t10k, "100k": t100k}
            export_scaling_table(
                subset_mahala_tables,
                caption=(
                    "AUC percentage scores of encoders trained and evaluated across different datasets, "
                    "on varying number of samples"
                ),
                label="tab:data_scaling",
                outdir=results_root,
            )


if __name__ == "__main__":
    main()
