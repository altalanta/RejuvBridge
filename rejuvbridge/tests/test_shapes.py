import torch

from rejuvbridge.models.image_encoder import ImageEncoder, ImageEncoderConfig
from rejuvbridge.models.omics_mlp import OmicsEncoder, OmicsEncoderConfig


def test_image_encoder_outputs_unit_norm() -> None:
    model = ImageEncoder(ImageEncoderConfig(embed_dim=32))
    inputs = torch.randn(4, 3, 64, 64)
    with torch.no_grad():
        embeddings = model(inputs)
    assert embeddings.shape == (4, 32)
    norms = embeddings.norm(p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_omics_encoder_outputs_unit_norm() -> None:
    model = OmicsEncoder(OmicsEncoderConfig(input_dim=50, hidden_dim=32, embed_dim=16))
    vectors = torch.randn(4, 50)
    with torch.no_grad():
        embeddings = model(vectors)
    assert embeddings.shape == (4, 16)
    norms = embeddings.norm(p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
