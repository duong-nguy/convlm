import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """Left-padded Conv1d that never peeks at future tokens."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
    ):
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.left_padding, 0))
        return self.conv(x)


class ConvBlock(nn.Module):
    """
    Gated causal conv (GLU) followed by a position-wise FFN,
    each with a residual connection and LayerNorm.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.conv = CausalConv1d(channels, channels * 2, kernel_size, dilation=dilation)
        self.norm1 = nn.LayerNorm(channels)
        self.ff = nn.Sequential(
            nn.Linear(channels, 4 * channels),
            nn.GELU(),
            nn.Linear(4 * channels, channels),
        )
        self.norm2 = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.conv(x.transpose(1, 2))  # (B,C,T)
        a, b = y.chunk(2, dim=1)
        y = (a * torch.sigmoid(b)).transpose(1, 2)  # (B,T,C)
        x = self.norm1(residual + self.dropout(y))

        residual = x
        x = self.norm2(residual + self.dropout(self.ff(x)))
        return x


class ConvGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_layers: int = 8,
        kernel_size: int = 3,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        tie_weights: bool = True,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        dilations = [2**i for i in range(num_layers)]
        self.kernel_size = kernel_size
        self.dilations = dilations

        self.blocks = nn.ModuleList(
            [
                ConvBlock(
                    channels=d_model,
                    kernel_size=kernel_size,
                    dilation=d,
                    dropout=dropout,
                )
                for d in dilations
            ]
        )

        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        if tie_weights:
            self.lm_head.weight = self.token_embedding.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict:
        _, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)

        x = self.dropout(
            self.token_embedding(input_ids) + self.position_embedding(positions)
        )

        for block in self.blocks:
            x = block(x)

        logits = self.lm_head(self.norm(x))

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )

        return {"loss": loss, "logits": logits}


def receptive_field(kernel_size: int, dilations: list[int]) -> int:
    """Total left context (in tokens) seen by the final layer."""
    return 1 + sum((kernel_size - 1) * d for d in dilations)
