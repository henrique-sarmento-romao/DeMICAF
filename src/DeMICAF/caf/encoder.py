"""Contrastive encoder training and feature extraction for the CAF pipeline."""

import copy
import datetime
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from data_compliance.caf.losses import NTXentLoss, SupCon, UniCon, compute_mmd, generate_uniform_samples
from data_compliance.utils.seeding import set_seed

warnings.filterwarnings("ignore")


def train(
    dataloader: torch.utils.data.DataLoader[Any],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    saveto_path: str | Path,
    epochs: int,
    framework: str = "SupCon",
    T: float = 0.07,
) -> None:
    """Train a contrastive encoder and save checkpoints plus a config.txt to ``saveto_path``."""
    # Save training configuration
    config_path = Path(saveto_path) / "config.txt"

    date = datetime.datetime.now().strftime("%b %d at %H:%M")
    loader_batch = dataloader.batch_size or 0
    batch_size = loader_batch * 2 if framework == "SimCLR" else loader_batch * 4
    l_r = optimizer.param_groups[0]["lr"]

    with config_path.open("w") as f:
        f.write(f"Trained on {date}.\n\n")

        f.write("HYPER-PARAMETERS\n===============================\n")
        f.write(f"framework: {framework}\n")
        f.write(f"batch_size: {batch_size}\n")
        f.write(f"epochs: {epochs}\n")
        f.write(f"learning_rate: {l_r}\n")

        f.write("\nTRANSFORMS\n===============================\n")
        transforms = getattr(dataloader.dataset, "transforms", None)
        if transforms is None or getattr(transforms, "ID", None) is None:
            f.write("No transforms configured.\n")
        else:
            f.write("In Distribuition\n")
            for t in transforms.ID.transforms:
                name = t.func.__name__ if hasattr(t, "func") else str(t)
                params = ", ".join(f"{k}={v}" for k, v in t.keywords.items()) if hasattr(t, "keywords") else ""
                f.write(f"  - {name.replace('_', ' ').title()}: {params}\n")

            if framework != "SimCLR" and getattr(transforms, "OOD", None):
                f.write("\nOut of Distribuition\n")
                for t in transforms.OOD.transforms:
                    name = t.func.__name__ if hasattr(t, "func") else str(t)
                    params = ", ".join(f"{k}={v}" for k, v in t.keywords.items()) if hasattr(t, "keywords") else ""
                    f.write(f"  - {name.replace('_', ' ').title()}: {params}\n")

    device = next(model.parameters()).device
    model.to(device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    best_loss = 1e20
    loss_list = []
    mmd_list = []
    steps_per_epoch = len(dataloader)
    best_epoch = 1

    scaler = torch.amp.GradScaler()  # pyright: ignore[reportPrivateImportUsage]
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    losses = {
        "CSI": NTXentLoss(temperature=T),
        "SimCLR": NTXentLoss(temperature=T),
        "SupCon": SupCon(temperature=T),
        "UniCon": UniCon(temperature=T, soft_aggregation=False),
        "UniConSA": UniCon(temperature=T, soft_aggregation=True),
    }
    loss = losses[framework]

    for e in range(epochs):
        running_loss = 0.0
        running_mmd = 0.0
        d_s = 0
        loop = tqdm(
            enumerate(dataloader), total=len(dataloader), position=0, leave=False, desc=f"Epoch {e + 1:03}/{epochs:03}"
        )

        for _i, (fnames, inputs) in loop:
            inputs = inputs.to(device, non_blocking=True)

            if inputs.dtype == torch.uint8:
                inputs = inputs.float()

            device_type = device.type
            with torch.autocast(device_type=device_type, dtype=torch.float16):
                outputs = model(inputs, head=True)
                outputs = outputs.to(device)

                # Build labels in linear time instead of repeated list.index lookups
                label_map: dict[str, int] = {}
                labels_list = []
                for view_name in fnames:
                    if view_name not in label_map:
                        label_map[view_name] = len(label_map)
                    labels_list.append(label_map[view_name])
                labels = torch.tensor(labels_list, device=device)
                batch_loss = loss(outputs, labels, fnames)

                uniform_samples = generate_uniform_samples(outputs.size(0), outputs.size(1)).to(device)
                mmd_value = compute_mmd(F.normalize(outputs.detach()), uniform_samples.detach())
                running_mmd += mmd_value.item()

            loss_value = float(batch_loss.item())

            if loss_value != 0.0:
                scaler.scale(batch_loss).backward()
                scaler.step(optimizer)
                scaler.update()

                optimizer.zero_grad(set_to_none=True)

                running_loss += loss_value * inputs.size(0)  # over entire batch

            loop.set_postfix(loss=f"{loss_value:.3f}", mmd=f"{mmd_value.item():.3f}")
            d_s += inputs.size(0)

            # Explicit cleanup of intermediate tensors
            del batch_loss, mmd_value, outputs, uniform_samples, inputs

        scheduler.step()

        epoch_loss = running_loss / d_s
        loss_list.append(epoch_loss)
        mmd_list.append(running_mmd / steps_per_epoch)

        running_loss = 0.0
        running_mmd = 0.0

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_epoch = e
            # torch.save(copy.deepcopy(model.state_dict()), os.path.join(saveto_path, "model_best.pt"))

        torch.save(copy.deepcopy(model.state_dict()), Path(saveto_path) / "model.pt")
        # torch.save(copy.deepcopy(optimizer.state_dict()), os.path.join(saveto_path, "optimizer.pt"))

        plt.clf()
        plt.plot(loss_list)
        plt.axvline(x=best_epoch, color="red", linestyle="--")  # vertical red dotted line
        plt.ylim(top=min(loss_list[0] * 1.25, max(loss_list) * 1.05))
        plt.savefig(Path(saveto_path) / "loss.png")

        plt.clf()
        plt.plot(mmd_list)
        plt.axvline(x=best_epoch, color="red", linestyle="--")
        plt.savefig(Path(saveto_path) / "mmd.png")

    torch.cuda.empty_cache()


def encode(
    dataloader: torch.utils.data.DataLoader[Any],
    encoder: torch.nn.Module,
    save_path: str | Path,
    agg_function: str | Callable[..., np.ndarray] | None = None,
    desc: str = "Features",
) -> None:
    """Extract features for every image and save ``(name, features)`` rows to a compressed npz."""
    set_seed(0)
    encoder.eval()
    device = next(encoder.parameters()).device
    device_type = device.type

    fnames_list = []
    feats_list = []

    with torch.no_grad(), torch.autocast(device_type=device_type, dtype=torch.float16):
        loop = tqdm(dataloader, position=0, leave=False, desc=desc)
        for batch in loop:
            if not isinstance(batch, list | tuple):
                raise TypeError("Expected DataLoader batch to be a tuple")
            if len(batch) == 2:
                fnames, inputs = batch
            elif len(batch) == 3:
                fnames, inputs, _ = batch
            else:
                raise ValueError("Expected DataLoader batch to have 2 or 3 items")

            inputs = inputs.to(device).float()
            outputs = encoder(inputs, head=False)
            # outputs is a torch tensor (B, feat_dim). Move to CPU and convert to numpy for stacking
            feats_list.append(outputs.detach().cpu().numpy())
            fnames_list.extend(fnames)

    # Convert lists to numpy arrays
    fnames_array = np.array(fnames_list, dtype=object)
    features_array = np.vstack(feats_list)

    # If agg_function is provided, aggregate per base study id
    if agg_function is not None:
        # Group rows by filename and aggregate feature vectors inside each group.
        groups: dict[str, list[np.ndarray]] = {}
        for fname, feat in zip(fnames_array, features_array, strict=False):
            base = str(fname)
            if base not in groups:
                groups[base] = []
            groups[base].append(feat)

        agg_fnames = []
        agg_feats = []
        for base, feats_list_group in groups.items():
            arr = np.vstack(feats_list_group)
            if callable(agg_function):
                agg = agg_function(arr, axis=0)
            else:
                if agg_function in ("mean", "avg"):
                    agg = np.mean(arr, axis=0)
                elif agg_function == "median":
                    agg = np.median(arr, axis=0)
                elif agg_function == "max":
                    agg = np.max(arr, axis=0)
                elif agg_function == "min":
                    agg = np.min(arr, axis=0)
                elif agg_function == "sum":
                    agg = np.sum(arr, axis=0)
                else:
                    raise ValueError(f"Unsupported agg_function: {agg_function}")

            agg_fnames.append(base)
            agg_feats.append(agg)

        fnames_array = np.array(agg_fnames, dtype=object)
        features_array = np.vstack(agg_feats)

    # Combine the arrays into final format: first column is filename, remaining columns are features
    combined_array = np.hstack((fnames_array.reshape(-1, 1), features_array))

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(save_path, combined_array)
