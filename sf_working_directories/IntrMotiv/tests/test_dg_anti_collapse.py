import torch

from sf_working_directories.IntrMotiv.dmlab.custom_encoder import DGProjection_batchnorm_relu
from sf_working_directories.IntrMotiv.dmlab.custom_learner import dg_global_punishment_loss, dg_row_repulsion_loss


def test_batchnorm_projection_retains_pre_threshold_logits():
    projection = DGProjection_batchnorm_relu(in_features=4, out_features=3, intercept=2.0)
    output = projection(torch.randn(5, 4))

    assert projection.last_pre_threshold_logits is not None
    assert projection.last_pre_threshold_logits.shape == output.shape
    assert torch.equal(output, torch.relu(projection.last_pre_threshold_logits - 2.0))


def test_global_punishment_reaches_subthreshold_valid_logits():
    logits = torch.tensor([[-2.0, -1.0], [3.0, 4.0]], requires_grad=True)
    loss = dg_global_punishment_loss(
        logits, intercept=2.0, temperature=0.5, coefficient=0.03, valids=torch.tensor([True, False]), num_invalids=1
    )
    loss.backward()
    assert loss.item() > 0.0
    assert torch.all(logits.grad[0] > 0.0)
    assert torch.all(logits.grad[1] == 0.0)


def test_global_punishment_zero_coefficient_is_gradient_neutral():
    logits = torch.randn(3, 2, requires_grad=True)
    loss = dg_global_punishment_loss(
        logits, intercept=2.0, temperature=0.5, coefficient=0.0, valids=torch.ones(3, dtype=torch.bool), num_invalids=0
    )
    loss.backward()
    assert loss.item() == 0.0
    assert torch.all(logits.grad == 0.0)


def test_row_repulsion_excludes_diagonal_and_uses_angles():
    orthogonal = torch.eye(3, requires_grad=True)
    orthogonal_loss = dg_row_repulsion_loss(orthogonal, coefficient=0.01)
    assert orthogonal_loss.item() == 0.0

    repeated = torch.tensor([[1.0, 0.0], [1.0, 0.0]], requires_grad=True)
    repeated_loss = dg_row_repulsion_loss(repeated, coefficient=0.01)
    repeated_loss.backward()
    assert repeated_loss.item() > 0.0
    assert torch.isfinite(repeated.grad).all()
