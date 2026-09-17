from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _section(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    merged = dict(config)
    nested = config.get(name)
    if isinstance(nested, Mapping):
        merged.update(nested)
    return merged


def _first(config: Mapping[str, Any], names: Sequence[str], default: Any) -> Any:
    for name in names:
        if name in config:
            return config[name]
    return default


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 128, max_len: int = 128) -> None:
        super().__init__()
        if d_model < 1 or max_len < 1:
            raise ValueError("d_model and max_len must be positive")
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        encoding = torch.zeros(max_len, d_model, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(position * div_term)
        odd_width = encoding[:, 1::2].shape[1]
        encoding[:, 1::2] = torch.cos(position * div_term[:odd_width])
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=True)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError("positional encoding expects a B x T x D tensor")
        if x.shape[1] > self.encoding.shape[1]:
            raise ValueError("sequence length exceeds configured maximum")
        if x.shape[2] != self.encoding.shape[2]:
            raise ValueError("embedding dimension does not match positional encoding")
        return x + self.encoding[:, : x.shape[1]].to(dtype=x.dtype, device=x.device)


class PostNormSelfAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 8,
        ffn_dim: int = 512,
        dropout: float = 0.0,
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.linear1 = nn.Linear(d_model, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, d_model)
        self.attention_dropout = nn.Dropout(dropout)
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        attended, _ = self.attention(
            x,
            x,
            x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = self.norm1(x + self.attention_dropout(attended))
        feed_forward = self.linear2(self.ffn_dropout(F.relu(self.linear1(x))))
        return self.norm2(x + self.ffn_dropout(feed_forward))


class HierarchicalEEGAutoencoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        window_size: int = 128,
        d_model: int = 128,
        num_heads: int = 8,
        encoder_layers: int = 2,
        decoder_layers: int = 2,
        ffn_dim: int = 512,
        dropout: float = 0.0,
        layer_norm_eps: float = 1e-5,
        input_layout: str = "batch_time_channels",
    ) -> None:
        super().__init__()
        if input_channels < 1:
            raise ValueError("input_channels must be positive")
        if encoder_layers < 1 or decoder_layers < 1:
            raise ValueError("encoder_layers and decoder_layers must be positive")
        if input_layout != "batch_time_channels":
            raise ValueError("input_layout must be batch_time_channels")
        self.input_channels = input_channels
        self.window_size = window_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.encoder_layer_count = encoder_layers
        self.decoder_layer_count = decoder_layers
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.input_layout = input_layout
        self.cnn_encoder = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.encoder_norm = nn.LayerNorm(128, eps=layer_norm_eps)
        self.encoder_projection = nn.Linear(128, d_model)
        self.positional_encoding = SinusoidalPositionalEncoding(d_model, window_size)
        self.transformer_encoder = nn.ModuleList(
            [
                PostNormSelfAttentionBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                    layer_norm_eps=layer_norm_eps,
                )
                for _ in range(encoder_layers)
            ]
        )
        self.transformer_decoder = nn.ModuleList(
            [
                PostNormSelfAttentionBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                    layer_norm_eps=layer_norm_eps,
                )
                for _ in range(decoder_layers)
            ]
        )
        self.decoder_projection = nn.Linear(d_model, 128)
        self.cnn_decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.decoder_norm = nn.LayerNorm(64, eps=layer_norm_eps)
        self.output_layer = nn.Conv1d(64, input_channels, kernel_size=1, stride=1)

    def _validate_input(self, x: Tensor) -> None:
        if x.ndim != 3:
            raise ValueError("autoencoder expects a B x T x C tensor")
        if x.shape[2] != self.input_channels:
            raise ValueError("input channel count does not match the model")
        if x.shape[1] > self.window_size:
            raise ValueError("input sequence exceeds configured window_size")

    def encode(
        self,
        x: Tensor,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        self._validate_input(x)
        encoded = self.cnn_encoder(x.transpose(1, 2)).transpose(1, 2)
        encoded = self.encoder_projection(self.encoder_norm(encoded))
        encoded = self.positional_encoding(encoded)
        for block in self.transformer_encoder:
            encoded = block(encoded, key_padding_mask=key_padding_mask)
        return encoded

    def decode(
        self,
        latent: Tensor,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        if latent.ndim != 3 or latent.shape[2] != self.d_model:
            raise ValueError("latent input must have shape B x T x d_model")
        decoded = latent
        for block in self.transformer_decoder:
            decoded = block(decoded, key_padding_mask=key_padding_mask)
        decoded = self.decoder_projection(decoded).transpose(1, 2)
        decoded = self.cnn_decoder(decoded).transpose(1, 2)
        decoded = self.decoder_norm(decoded).transpose(1, 2)
        return self.output_layer(decoded).transpose(1, 2)

    def forward(
        self,
        x: Tensor,
        key_padding_mask: Tensor | None = None,
        return_latent: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        latent = self.encode(x, key_padding_mask=key_padding_mask)
        reconstructed = self.decode(latent, key_padding_mask=key_padding_mask)
        if return_latent:
            return reconstructed, latent
        return reconstructed

    def semantic_features(
        self,
        x: Tensor,
        key_padding_mask: Tensor | None = None,
        normalize: bool = False,
    ) -> Tensor:
        latent = self.encode(x, key_padding_mask=key_padding_mask)
        if key_padding_mask is None:
            features = latent.mean(dim=1)
        else:
            valid = (~key_padding_mask).to(dtype=latent.dtype).unsqueeze(-1)
            features = (latent * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        if normalize:
            features = F.normalize(features, p=2, dim=1)
        return features

    def configuration(self) -> dict[str, Any]:
        return {
            "input_channels": self.input_channels,
            "window_size": self.window_size,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "encoder_layers": self.encoder_layer_count,
            "decoder_layers": self.decoder_layer_count,
            "ffn_dim": self.ffn_dim,
            "dropout": self.dropout,
            "input_layout": self.input_layout,
        }


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        self.convolution = nn.Conv2d(
            2,
            1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        average = x.mean(dim=1, keepdim=True)
        maximum = x.amax(dim=1, keepdim=True)
        weights = torch.sigmoid(self.convolution(torch.cat((average, maximum), dim=1)))
        return x * weights


class SemanticFeatureClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        binary: bool = False,
        input_channels: int = 1,
        filters: Sequence[int] = (64, 128, 256, 512),
        dropout: float = 0.5,
        attention_kernel_size: int = 7,
    ) -> None:
        super().__init__()
        if len(filters) != 4 or any(value < 1 for value in filters):
            raise ValueError("filters must contain four positive values")
        if binary and num_classes != 2:
            raise ValueError("binary classifiers require num_classes=2")
        if not binary and num_classes < 2:
            raise ValueError("multiclass classifiers require at least two classes")
        self.num_classes = num_classes
        self.binary = binary
        self.input_channels = input_channels
        self.filters = tuple(int(value) for value in filters)
        blocks: list[nn.Module] = []
        channels = input_channels
        for index, output_channels in enumerate(self.filters):
            layers: list[nn.Module] = [
                nn.Conv2d(
                    channels,
                    output_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(),
            ]
            if index < 3:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            blocks.append(nn.Sequential(*layers))
            channels = output_channels
        self.blocks = nn.ModuleList(blocks)
        self.spatial_attention = SpatialAttention(attention_kernel_size)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout_layer = nn.Dropout(dropout)
        self.output_layer = nn.Linear(self.filters[-1], 1 if binary else num_classes)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if x.ndim != 4:
            raise ValueError("classifier expects B x T x D or B x C x T x D")
        if x.shape[1] != self.input_channels:
            raise ValueError("classifier input channel count does not match the model")
        for block in self.blocks:
            x = block(x)
        x = self.spatial_attention(x)
        x = self.global_pool(x).flatten(1)
        return self.output_layer(self.dropout_layer(x))

    def probabilities(self, x: Tensor) -> Tensor:
        logits = self.forward(x)
        if self.binary:
            return torch.sigmoid(logits).squeeze(-1)
        return torch.softmax(logits, dim=1)

    def configuration(self) -> dict[str, Any]:
        return {
            "num_classes": self.num_classes,
            "binary": self.binary,
            "input_channels": self.input_channels,
            "filters": list(self.filters),
            "dropout": self.dropout_layer.p,
            "attention_kernel_size": self.spatial_attention.convolution.kernel_size[0],
        }


class BinaryFocalLoss(nn.Module):
    def __init__(
        self,
        alpha: float | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if alpha is not None and not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if gamma < 0.0:
            raise ValueError("gamma must be non-negative")
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be none, mean, or sum")
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        targets = targets.to(dtype=logits.dtype, device=logits.device)
        if targets.shape != logits.shape:
            targets = targets.reshape(logits.shape)
        cross_entropy = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        )
        probabilities = torch.sigmoid(logits)
        probability_true = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
        loss = (1.0 - probability_true).pow(self.gamma) * cross_entropy
        if self.alpha is not None:
            alpha_true = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            loss = alpha_true * loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def build_autoencoder(
    config: Mapping[str, Any],
    input_channels: int | None = None,
) -> HierarchicalEEGAutoencoder:
    values = _section(config, "autoencoder")
    channels = input_channels
    if channels is None:
        channels = int(_first(values, ("input_channels", "n_channels", "channels"), 0))
    total_layers = int(_first(values, ("transformer_layers", "num_layers"), 4))
    encoder_layers = int(values.get("encoder_layers", total_layers // 2))
    decoder_layers = int(values.get("decoder_layers", total_layers - encoder_layers))
    return HierarchicalEEGAutoencoder(
        input_channels=channels,
        window_size=int(_first(values, ("window_size", "input_samples"), 128)),
        d_model=int(values.get("d_model", 128)),
        num_heads=int(_first(values, ("num_heads", "attention_heads", "nhead"), 8)),
        encoder_layers=int(_first(values, ("num_encoder_layers",), encoder_layers)),
        decoder_layers=int(_first(values, ("num_decoder_layers",), decoder_layers)),
        ffn_dim=int(_first(values, ("ffn_dim", "feedforward_dim", "dim_feedforward"), 512)),
        dropout=float(_first(values, ("dropout", "attention_dropout"), 0.0)),
        layer_norm_eps=float(
            _first(values, ("layer_norm_eps", "encoder_layer_norm_epsilon"), 1e-5)
        ),
        input_layout=str(values.get("input_layout", "batch_time_channels")),
    )


def build_classifier(
    config: Mapping[str, Any],
    num_classes: int | None = None,
    binary: bool | None = None,
) -> SemanticFeatureClassifier:
    values = _section(config, "classifier")
    resolved_classes = num_classes
    if resolved_classes is None:
        resolved_classes = int(values.get("num_classes", 2))
    resolved_binary = binary
    if resolved_binary is None:
        resolved_binary = bool(values.get("binary", values.get("task") == "erp"))
    return SemanticFeatureClassifier(
        num_classes=resolved_classes,
        binary=resolved_binary,
        input_channels=int(values.get("input_channels", 1)),
        filters=tuple(values.get("filters", (64, 128, 256, 512))),
        dropout=float(values.get("dropout", 0.5)),
        attention_kernel_size=int(values.get("attention_kernel_size", 7)),
    )


HierarchicalDualAutoencoder = HierarchicalEEGAutoencoder
EEG2DClassifier = SemanticFeatureClassifier
