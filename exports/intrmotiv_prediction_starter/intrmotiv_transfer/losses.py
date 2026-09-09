"""Head-only form of the existing shadow predictor objective."""

import torch
from torch.nn import functional as F


def shadow_prediction_loss(head, ca3, target_onehot, hit, delay, usable, horizon):
    """Flattened batch. Detach representation and labels; optimize head only.

    BCE uses IntrMotiv's clipped positive weighting. Its probabilities are NOT
    guaranteed calibrated. Timing loss/MAE apply only to observed positives.
    `delay` and reported MAE use policy decisions, not engine frames.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    hit, delay, usable = hit.detach(), delay.detach(), usable.detach().bool()
    prediction = head(ca3.detach(), target_onehot.detach())
    if prediction.shape != (hit.numel(), 2) or delay.shape != hit.shape or usable.shape != hit.shape or hit.ndim != 1:
        raise ValueError("Expected flat labels [N] and predictions [N,2]")
    if ((hit[usable] != 0) & (hit[usable] != 1)).any():
        raise ValueError("usable hit labels must be binary")
    positive = usable & hit.gt(0)
    if ((delay[positive] < 1) | (delay[positive] > horizon)).any():
        raise ValueError("positive delays must lie in [1,horizon]")
    zero = prediction.sum() * 0.0
    count = usable.sum()
    if usable.any():
        selected_hit = hit[usable].to(prediction)
        positives = selected_hit.sum()
        weight = ((count - positives) / positives.clamp_min(1)).clamp(1, 20)
        bce = F.binary_cross_entropy_with_logits(prediction[usable, 0], selected_hit, pos_weight=weight)
        accuracy = ((prediction[usable, 0] >= 0) == selected_hit.bool()).float().mean()
        positive_fraction = selected_hit.mean()
    else:
        bce, accuracy, positive_fraction = zero, zero.detach(), zero.detach()
    predicted_delay = prediction[:, 1].sigmoid() * horizon
    if positive.any():
        time_loss = F.smooth_l1_loss(predicted_delay[positive] / horizon, delay[positive] / horizon)
        time_mae = (predicted_delay[positive] - delay[positive]).abs().mean()
    else:
        time_loss, time_mae = zero, zero.detach()
    return bce + time_loss, {
        "usable_count": count.detach(),
        "positive_count": positive.sum().detach(),
        "positive_fraction": positive_fraction.detach(),
        "hit_accuracy": accuracy.detach(),
        "time_mae_decisions": time_mae.detach(),
    }
