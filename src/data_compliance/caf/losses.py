"""Contrastive losses (NT-Xent, SupCon, UniCon) and uniformity regularizer helpers."""

import torch
import torch.nn.functional as F
from scipy.stats import uniform
from torch import nn


def gaussian_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """Pairwise Gaussian (RBF) kernel matrix between two sample batches."""
    xx = x.unsqueeze(1).expand(x.size(0), y.size(0), x.size(1))
    yy = y.unsqueeze(0).expand(x.size(0), y.size(0), y.size(1))
    return torch.exp(-torch.sum((xx - yy) ** 2, dim=2) / (2 * sigma**2))


def compute_mmd(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """Maximum mean discrepancy between two sample batches using a Gaussian kernel."""
    xx_kernel = gaussian_kernel(x, x, sigma)
    yy_kernel = gaussian_kernel(y, y, sigma)
    xy_kernel = gaussian_kernel(x, y, sigma)
    return torch.mean(xx_kernel) + torch.mean(yy_kernel) - 2 * torch.mean(xy_kernel)


def generate_uniform_samples(batch_size: int, feature_dim: int) -> torch.Tensor:
    """Draw L2-normalized uniform samples used as the MMD reference distribution."""
    uniform_samples = uniform.rvs(size=(batch_size, feature_dim))
    samples = torch.tensor(uniform_samples, dtype=torch.float32)
    return F.normalize(samples, p=2, dim=1)


class NTXentLoss(nn.Module):
    """Normalized temperature-scaled cross-entropy loss (SimCLR / CSI)."""

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor, fnames: list[str]) -> torch.Tensor:
        """Compute the contrastive loss for a batch of embeddings.

        Parameters
        ----------
        features : torch.Tensor, shape (batch_size, feature_dim)
            Embeddings of the samples.
        labels : torch.Tensor, shape (batch_size,)
            Labels identifying positive pairs.
        fnames : list[str]
            Sample filenames (unused; kept for API parity with the other losses).

        Returns
        -------
        torch.Tensor
            The scalar contrastive loss.
        """
        device = features.device

        # Normalize the features
        features = F.normalize(features, p=2, dim=1)

        # Compute pairwise cosine similarity
        similarity_matrix = torch.matmul(features, features.t()).to(device) / self.temperature

        # Create mask for positive pairs based on labels
        labels = labels.unsqueeze(1)
        positive_mask = torch.eq(labels, labels.t()).float().to(device)

        # Zero out diagonal (self-similarity)
        mask = torch.eye(similarity_matrix.size(0), dtype=torch.bool, device=device).to(device)
        similarity_matrix = (
            similarity_matrix * (~mask).float()
        )  # Set diagonal to 0 by multiplying by the inverse of the mask

        # Separate positive and negative pairs
        pos_pairs = (similarity_matrix * positive_mask).sum(dim=1)
        neg_pairs = similarity_matrix * (1 - positive_mask)

        # Handle max value for numerical stability
        max_val = torch.max(similarity_matrix, dim=1, keepdim=True)[0].detach()
        # Numerator for positive pairs
        numerator = torch.exp(pos_pairs - max_val)

        # Denominator for negative pairs
        denominator = torch.sum(torch.exp(neg_pairs - max_val), dim=1) + numerator

        # Compute the final loss
        log_exp = torch.log(numerator / (denominator + 1e-8))
        return -log_exp.mean()


class UniCon(nn.Module):
    """Unified contrastive loss with shifted-instance handling and optional soft aggregation."""

    def __init__(
        self,
        temperature: float = 0.07,
        soft_aggregation: bool = False,
        soft_agg_temperature: float | None = None,
        debug: bool = False,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.soft_aggregation = soft_aggregation
        self.soft_agg_temperature = soft_agg_temperature if soft_agg_temperature is not None else temperature
        self.debug = debug

    def compute_soft_weights(
        self,
        logits: torch.Tensor,
        positive_mask: torch.Tensor,
        soft_agg_temperature: float,
        is_shifted: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-instance soft aggregation weights for non-shifted samples."""
        if self.debug:
            print("Computing soft weights...")

        device = logits.device

        # Initialize the weight mask with ones (for all instances)
        soft_weights = torch.ones(logits.size(0), device=device)

        # Select non-shifted instances
        non_shifted_indices = (is_shifted == 0).squeeze()

        if non_shifted_indices.any():
            # Slice the logits and positive_mask for non-shifted instances
            logits_non_shifted = logits[non_shifted_indices][:, non_shifted_indices]
            positive_mask_non_shifted = positive_mask[non_shifted_indices][:, non_shifted_indices]

            # Compute logits divided by tau_w (temperature scaling)
            scaled_logits = logits_non_shifted / soft_agg_temperature

            # Numerical stability: Subtract the maximum value from each row after scaling
            max_logits = torch.max(scaled_logits, dim=1, keepdim=True)[0].detach()
            stabilized_logits = scaled_logits - max_logits  # Subtract max for stability

            exp_logits = torch.exp(stabilized_logits)
            exp_logits_no_diag = exp_logits * (1 - torch.eye(exp_logits.size(0), device=device))  # Exclude diagonal

            # Numerator: Mean of exponentials for each anchor to its positives
            numerator = torch.sum(
                exp_logits_no_diag * positive_mask_non_shifted, dim=1
            ) / positive_mask_non_shifted.sum(dim=1)

            # Denominator: Mean of the numerators across the batch (global denominator)
            denominator = numerator.mean()

            # Soft weights: normalized sum of positive pairs
            non_shifted_weights = numerator / (denominator + 1e-8)  # Adding epsilon for numerical stability

            # Update the corresponding indices in the full soft_weights array
            soft_weights[non_shifted_indices] = non_shifted_weights

        return soft_weights.unsqueeze(1)

    def forward(self, features: torch.Tensor, labels: torch.Tensor, fnames: list[str]) -> torch.Tensor:
        """Compute the loss; ``fnames`` containing ``_shifted`` mark OOD-augmented views."""
        if self.debug:
            print("Starting forward pass...")

        device = features.device

        features = F.normalize(features, p=2, dim=1)
        logits = torch.matmul(features, features.t()).to(device)  # Logits without temperature scaling
        logits_scaled = logits / self.temperature
        max_val = torch.max(logits_scaled, dim=1, keepdim=True)[0].detach()

        if self.debug:
            print(f"Normalized Features:\n{features}")
            print(f"Logits:\n{logits_scaled}")

        labels = labels.unsqueeze(1)
        positive_mask = torch.eq(labels, labels.t()).float().to(device)

        mask = torch.eye(logits.size(0), dtype=torch.bool, device=device)
        logits_scaled = logits_scaled * (~mask).float()

        if self.debug:
            print(f"Positive Mask:\n{positive_mask}")

        # Identify shifted instances
        is_shifted = (
            torch.tensor([1 if "_shifted" in fname else 0 for fname in fnames], device=device).unsqueeze(1).float()
        )

        # Create the adjusted positive mask based on shifted status
        non_shifted_mask = (is_shifted == 0).float()
        non_shifted_mask = non_shifted_mask @ non_shifted_mask.t()

        adjusted_positive_mask = non_shifted_mask + positive_mask * (is_shifted == is_shifted.t()).float()
        adjusted_positive_mask = adjusted_positive_mask.clamp(
            max=1
        )  # Cap the values at 1 to avoid having any 2s in the mask

        if self.debug:
            print(f"Adjusted Positive Mask:\n{adjusted_positive_mask}")

        if self.soft_aggregation:
            # Calculate soft weights for all instances, including setting weights for shifted instances to 1
            soft_weights = self.compute_soft_weights(
                logits_scaled.detach(), adjusted_positive_mask, self.soft_agg_temperature, is_shifted
            )

        else:
            soft_weights = torch.ones(labels.size(0), device=device)

        scaling_factor = (soft_weights * soft_weights.t() * adjusted_positive_mask).sum(dim=1)

        if self.debug:
            print(f"Soft Weights:\n{soft_weights}")
            print(f"Scaling Factor:\n{scaling_factor}")

        # Exponentiate the logits with numerical stability
        exp_logits = torch.exp(logits_scaled - max_val)

        # Calculate the weighted numerator
        weighted_pos = (exp_logits * adjusted_positive_mask) * (soft_weights * soft_weights.t())
        numerator = weighted_pos.sum(dim=1)

        # Calculate the denominator
        denominator = exp_logits.sum(dim=1)

        if self.debug:
            print(f"Numerator:\n{numerator}")
            print(f"Denominator:\n{denominator}")

        # Final loss computation
        log_exp = torch.log(numerator / (denominator + 1e-8))

        loss = -((1 / scaling_factor) * log_exp).mean()

        if self.debug:
            print(f"Log Exp:\n{log_exp}")
            print(f"Loss:\n{loss}")

        # Check for NaN values in the loss
        if torch.isnan(loss):
            print("Encountered NaN in loss calculation.")
            loss = torch.tensor(0.0, device=device)

        return loss


class SupCon(nn.Module):
    """Supervised contrastive loss where positives share the shifted / non-shifted status."""

    def __init__(self, temperature: float = 0.07, debug: bool = False) -> None:
        super().__init__()
        self.temperature = temperature
        self.debug = debug

    def forward(self, features: torch.Tensor, labels: torch.Tensor, fnames: list[str]) -> torch.Tensor:
        """Compute the loss; ``fnames`` containing ``_shifted`` mark OOD-augmented views."""
        if self.debug:
            print("Starting forward pass...")

        device = features.device

        features = F.normalize(features, p=2, dim=1)
        logits = torch.matmul(features, features.t()).to(device)  # Logits without temperature scaling
        logits_scaled = logits / self.temperature
        max_val = torch.max(logits_scaled, dim=1, keepdim=True)[0].detach()

        if self.debug:
            print(f"Normalized Features:\n{features}")
            print(f"Logits:\n{logits_scaled}")

        labels = labels.unsqueeze(1)

        mask = torch.eye(logits.size(0), dtype=torch.bool, device=device)
        logits_scaled = logits_scaled * (~mask).float()

        # Identify shifted instances
        is_shifted = (
            torch.tensor([1 if "_shifted" in fname else 0 for fname in fnames], device=device).unsqueeze(1).float()
        )
        positive_mask = (is_shifted == is_shifted.t()).float().to(device)

        if self.debug:
            print(f"Positive Mask:\n{positive_mask}")

        # Exponentiate the logits with numerical stability
        exp_logits = torch.exp(logits_scaled - max_val)

        # Calculate the weighted numerator
        weighted_pos = exp_logits * positive_mask
        numerator = weighted_pos.sum(dim=1)

        # Calculate the denominator
        denominator = exp_logits.sum(dim=1)

        if self.debug:
            print(f"Numerator:\n{numerator}")
            print(f"Denominator:\n{denominator}")

        # Final loss computation
        log_exp = -torch.log(numerator / (denominator + 1e-8))

        loss = log_exp.mean()

        if self.debug:
            print(f"Log Exp:\n{log_exp}")
            print(f"Loss:\n{loss}")

        # Check for NaN values in the loss
        if torch.isnan(loss):
            print("Encountered NaN in loss calculation.")
            loss = torch.tensor(0.0, device=device)

        return loss


# # Example usage to test losses:
# if __name__ == "__main__":
#     # Dummy data
#     embeddings = torch.rand(8, 128).cuda()  # 4 samples with 128-dimensional embeddings
#     fnames = ["a_shifted", "b_shifted","a_shifted", "b_shifted","a", "b", "a", "b"]
#     labels = torch.tensor([fnames.index(f) for f in fnames]).cuda()
#     # print(labels)
#     # Loss function
#     # loss_fn = UniCon(temperature=0.07, soft_aggregation=False, debug=True)
#     # loss_fn = NTXentLoss()
#     loss_fn = SupCon(temperature=0.07, debug=True)


#     # Case 2: Calculate loss without fnames (using all negatives)
#     loss_without_fnames = loss_fn(embeddings, labels,fnames)
#     print("Loss  ", loss_without_fnames.item())
