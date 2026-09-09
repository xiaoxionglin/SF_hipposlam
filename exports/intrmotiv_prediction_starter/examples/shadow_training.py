"""Synthetic wiring check, NOT evidence of navigation or transfer learning.

Run from the kit directory: python -m examples.shadow_training
"""

import torch

from intrmotiv_transfer import CA3TargetPredictor, future_target_labels, shadow_prediction_loss


def main():
    torch.manual_seed(4)
    torch.set_num_threads(1)
    batch, steps, features, ca3_size, horizon = 16, 8, 3, 12, 3
    # Replace with real sampled/replayed tensors using INTEGRATION.md conventions.
    ca3 = torch.randn(batch, steps, ca3_size, requires_grad=True)
    goals = torch.nn.functional.one_hot(torch.randint(features, (batch, steps)), features).float()
    dg = (torch.rand(batch, steps, features) < .15).float()
    dones_after = torch.zeros(batch, steps, dtype=torch.bool)
    hit, delay, usable = future_target_labels(goals, dg, dones_after, horizon)
    head = CA3TargetPredictor(ca3_size, features, 32)
    actor = torch.nn.Linear(ca3_size, 5)
    original = actor(ca3).detach().clone()
    optimizer = torch.optim.Adam(head.parameters(), lr=0.01)
    losses = []
    for _ in range(100):
        loss, stats = shadow_prediction_loss(
            head, ca3.reshape(-1, ca3_size), goals.reshape(-1, features),
            hit.flatten(), delay.flatten(), usable.flatten(), horizon,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert ca3.grad is None
    assert torch.equal(original, actor(ca3).detach())
    assert losses[-1] < losses[0]
    print(f"Synthetic training loss: {losses[0]:.4f} -> {losses[-1]:.4f}")
    print({key: value.item() for key, value in stats.items()})
    print("PASS: head learns; original actor output and real representation are unchanged.")
    print("This deliberately overfits a synthetic batch; it does not measure generalization.")


if __name__ == "__main__":
    main()
