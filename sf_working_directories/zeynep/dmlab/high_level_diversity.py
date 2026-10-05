"""Optional trial-end mode discriminator for Y-maze controller rewards."""

import torch
from torch import nn
from torch.nn import functional as F


class TrialEndModeClassifier(nn.Module):
    """Predict the previous mode from the observation reached at trial end.

    Only the visual observation is used by default. The one-hot chosen arm can
    be included for levels that return to an identical center view at outcome.
    Neither the selected mode nor the reward is an input to the classifier.
    """

    def __init__(self, image_channels: int, num_modes: int, include_chosen_arm: bool = False):
        super().__init__()
        if num_modes < 2:
            raise ValueError("The diversity reward requires at least two modes")
        self.num_modes = num_modes
        self.include_chosen_arm = include_chosen_arm
        self.visual = nn.Sequential(
            nn.Conv2d(image_channels, 16, kernel_size=5, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(32 * 4 * 4 + (3 if include_chosen_arm else 0), 64),
            nn.ReLU(),
            nn.Linear(64, num_modes),
        )

    def forward(self, image: torch.Tensor, chosen_arm: torch.Tensor | None = None) -> torch.Tensor:
        features = self.visual(image.float())
        if self.include_chosen_arm:
            if chosen_arm is None:
                raise ValueError("chosen_arm is required when include_chosen_arm=True")
            arm = F.one_hot(chosen_arm.long().reshape(-1).clamp(0, 2), num_classes=3)
            features = torch.cat((features, arm.to(features.dtype)), dim=-1)
        return self.head(features)


def predictability_bonus(logits: torch.Tensor, modes: torch.Tensor) -> torch.Tensor:
    """Bounded, nonnegative evidence for the true mode above its batch prior.

    A classifier that emits the same probabilities for every reached state
    receives zero bonus, even if one mode is selected much more often.
    """
    if logits.size(0) < 2 or modes.unique().numel() < 2:
        return logits.new_zeros(modes.shape)
    probabilities = logits.softmax(dim=-1)
    true_probability = probabilities.gather(1, modes[:, None]).squeeze(1)
    marginal_probability = probabilities.mean(dim=0)[modes]
    evidence = true_probability.clamp_min(1e-8).log() - marginal_probability.clamp_min(1e-8).log()
    return (evidence / torch.log(logits.new_tensor(logits.size(-1)))).clamp(0.0, 1.0)


@torch.no_grad()
def trial_end_bonus(
    classifier: TrialEndModeClassifier,
    normalized_obs,
    selected_modes: torch.Tensor,
    dones: torch.Tensor,
    valids: torch.Tensor,
    coefficient: float,
) -> torch.Tensor:
    """Return a bonus for action t using observation t+1 at trial end."""
    next_outcome = normalized_obs["outcome_event"][:, 1:].reshape_as(dones) > 0.5
    mask = next_outcome & ~dones & valids & (selected_modes.sum(dim=-1) > 0.5)
    bonus = dones.new_zeros(dones.shape, dtype=torch.float32)
    if mask.any():
        image = normalized_obs["obs"][:, 1:][mask]
        arm = normalized_obs["chosen_arm"][:, 1:][mask] if classifier.include_chosen_arm else None
        labels = selected_modes.argmax(dim=-1)[mask]
        bonus[mask] = coefficient * predictability_bonus(classifier(image, arm), labels)
    return bonus


def trial_end_classifier_loss(
    classifier: TrialEndModeClassifier,
    normalized_obs,
    pre_event_modes: torch.Tensor,
    valids: torch.Tensor,
):
    """Fit on valid outcome observations and the mode held before that event."""
    outcome = normalized_obs["outcome_event"].reshape(-1) > 0.5
    mask = outcome & valids & (pre_event_modes.sum(dim=-1) > 0.5)
    count = int(mask.sum().item())
    if count == 0:
        return pre_event_modes.new_zeros(()), 0, 0.0
    image = normalized_obs["obs"][mask]
    arm = normalized_obs["chosen_arm"][mask] if classifier.include_chosen_arm else None
    labels = pre_event_modes.argmax(dim=-1)[mask]
    logits = classifier(image, arm)
    # Equalize represented modes so a majority-only predictor is not enough.
    counts = torch.bincount(labels, minlength=classifier.num_modes).clamp_min(1)
    weights = counts[labels].float().reciprocal()
    per_example_loss = F.cross_entropy(logits, labels, reduction="none")
    loss = (per_example_loss * weights).sum() / weights.sum()
    accuracy = (logits.argmax(dim=-1) == labels).float().mean().item()
    return loss, count, accuracy
