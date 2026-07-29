import torch

from gene_nn.models import MLP


def test_mlp_forward_shape():
    batch = 4
    input_dim = 10
    output_dim = 3

    model = MLP(input_dim=input_dim, output_dim=output_dim, hidden_dims=(16, 8), dropout=0.0)
    x = torch.randn(batch, input_dim)
    y = model(x)

    assert y.shape == (batch, output_dim)
    