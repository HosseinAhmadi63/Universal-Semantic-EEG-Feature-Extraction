import torch
import torch.nn.functional as F
from torch import nn

from useeg.models import (
    BinaryFocalLoss,
    HierarchicalEEGAutoencoder,
    PostNormSelfAttentionBlock,
    SemanticFeatureClassifier,
    SinusoidalPositionalEncoding,
    build_autoencoder,
    build_classifier,
)


def test_positional_encoding_is_deterministic() -> None:
    layer = SinusoidalPositionalEncoding(d_model=7, max_len=16)
    inputs = torch.zeros(2, 8, 7)
    first = layer(inputs)
    second = layer(inputs)
    assert first.shape == inputs.shape
    assert torch.equal(first, second)
    assert not torch.equal(first[:, 0], first[:, 1])


def test_autoencoder_shapes_and_self_attention_only_decoder() -> None:
    model = HierarchicalEEGAutoencoder(
        input_channels=3,
        window_size=16,
        d_model=16,
        num_heads=4,
        encoder_layers=1,
        decoder_layers=1,
        ffn_dim=32,
        input_layout="batch_time_channels",
    )
    inputs = torch.randn(2, 16, 3, requires_grad=True)
    reconstructed, latent = model(inputs, return_latent=True)
    features = model.semantic_features(inputs, normalize=True)
    assert reconstructed.shape == inputs.shape
    assert latent.shape == (2, 16, 16)
    assert features.shape == (2, 16)
    assert torch.allclose(features.norm(dim=1), torch.ones(2), atol=1e-5)
    reconstructed.mean().backward()
    assert inputs.grad is not None
    assert all(isinstance(block, PostNormSelfAttentionBlock) for block in model.transformer_decoder)
    assert not any(isinstance(module, nn.TransformerDecoderLayer) for module in model.modules())


def test_autoencoder_rejects_channels_time_layout() -> None:
    model = HierarchicalEEGAutoencoder(
        input_channels=3,
        window_size=12,
        d_model=12,
        num_heads=3,
        encoder_layers=1,
        decoder_layers=1,
        ffn_dim=24,
        input_layout="batch_time_channels",
    )
    try:
        model(torch.randn(2, 3, 12))
    except ValueError as error:
        assert "channel count" in str(error)
    else:
        raise AssertionError("channels-first input was accepted")


def test_model_builders_read_repository_config_keys() -> None:
    autoencoder = build_autoencoder(
        {
            "autoencoder": {
                "input_samples": 16,
                "d_model": 16,
                "nhead": 4,
                "num_encoder_layers": 1,
                "num_decoder_layers": 1,
                "dim_feedforward": 32,
                "input_layout": "batch_time_channels",
            }
        },
        input_channels=2,
    )
    classifier = build_classifier(
        {"classifier": {"filters": [4, 8, 12, 16], "dropout": 0.25}},
        num_classes=4,
        binary=False,
    )
    assert autoencoder(torch.randn(2, 16, 2)).shape == (2, 16, 2)
    assert classifier(torch.randn(2, 1, 16, 16)).shape == (2, 4)


def test_classifier_outputs_and_spatial_attention() -> None:
    multiclass = SemanticFeatureClassifier(
        num_classes=4,
        filters=(4, 8, 12, 16),
        dropout=0.0,
    )
    binary = SemanticFeatureClassifier(
        num_classes=2,
        binary=True,
        filters=(4, 8, 12, 16),
        dropout=0.0,
    )
    inputs = torch.randn(4, 16, 16)
    multiclass_logits = multiclass(inputs)
    binary_logits = binary(inputs)
    assert multiclass_logits.shape == (4, 4)
    assert binary_logits.shape == (4, 1)
    assert binary.probabilities(inputs).shape == (4,)
    assert binary.spatial_attention.convolution.kernel_size == (7, 7)


def test_binary_focal_loss_matches_bce_when_gamma_zero() -> None:
    logits = torch.tensor([[-1.0], [0.5], [2.0]])
    targets = torch.tensor([[0.0], [1.0], [1.0]])
    focal = BinaryFocalLoss(alpha=None, gamma=0.0)(logits, targets)
    expected = F.binary_cross_entropy_with_logits(logits, targets)
    assert torch.allclose(focal, expected)
