"""A compact nanoGPT-style decoder-only Transformer for PokerGPT.

The architecture follows Andrej Karpathy's nanoGPT model.py, while the forward
pass additionally accepts the per-token loss mask emitted by poker_pipeline.
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    """LayerNorm with optional bias, matching nanoGPT."""

    def __init__(self, ndim: int, bias: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(inputs, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: "GPTConfig") -> None:
        super().__init__()
        if config.n_embd % config.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(F, "scaled_dot_product_attention")
        if not self.flash:
            mask = torch.tril(torch.ones(config.block_size, config.block_size))
            self.register_buffer("causal_mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, channels = x.shape
        query, key, value = self.c_attn(x).split(self.n_embd, dim=2)
        head_size = channels // self.n_head
        query = query.view(batch, steps, self.n_head, head_size).transpose(1, 2)
        key = key.view(batch, steps, self.n_head, head_size).transpose(1, 2)
        value = value.view(batch, steps, self.n_head, head_size).transpose(1, 2)
        if self.flash:
            output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            attention = (query @ key.transpose(-2, -1)) / math.sqrt(head_size)
            attention = attention.masked_fill(
                self.causal_mask[:, :, :steps, :steps] == 0, float("-inf")
            )
            attention = self.attn_dropout(F.softmax(attention, dim=-1))
            output = attention @ value
        output = output.transpose(1, 2).contiguous().view(batch, steps, channels)
        return self.resid_dropout(self.c_proj(output))


class MLP(nn.Module):
    def __init__(self, config: "GPTConfig") -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, config: "GPTConfig") -> None:
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


@dataclass(frozen=True)
class GPTConfig:
    """Small-from-scratch defaults suitable for the structured poker vocabulary."""

    block_size: int = 256
    vocab_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    bias: bool = False


class GPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        if config.block_size <= 0 or config.vocab_size <= 0:
            raise ValueError("block_size and vocab_size must be positive")
        self.config = config
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "wpe": nn.Embedding(config.block_size, config.n_embd),
                "drop": nn.Dropout(config.dropout),
                "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                "ln_f": LayerNorm(config.n_embd, bias=config.bias),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)
        for name, parameter in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(parameter, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding: bool = True) -> int:
        count = sum(parameter.numel() for parameter in self.parameters())
        return count - self.transformer.wpe.weight.numel() if non_embedding else count

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return logits and optionally masked next-token cross entropy.

        ``loss_mask`` has the same shape as ``targets``; a nonzero entry means
        that target contributes to the loss. Padding targets may also be -1.
        """
        _, steps = idx.shape
        if steps > self.config.block_size:
            raise ValueError(f"sequence length {steps} exceeds block_size {self.config.block_size}")
        positions = torch.arange(steps, device=idx.device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(positions))
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is None:
            return self.lm_head(x[:, [-1], :]), None
        if targets.shape != idx.shape:
            raise ValueError("targets must have the same shape as idx")
        logits = self.lm_head(x)
        token_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-1,
            reduction="none",
        ).view_as(targets)
        valid = targets.ne(-1)
        if loss_mask is not None:
            if loss_mask.shape != targets.shape:
                raise ValueError("loss_mask must have the same shape as targets")
            valid = valid & loss_mask.bool()
        if not torch.any(valid):
            raise ValueError("no target tokens selected for loss")
        return logits, token_loss[valid].mean()

    def configure_optimizer(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: tuple[float, float] = (0.9, 0.95),
        device_type: str = "cpu",
    ) -> torch.optim.AdamW:
        parameters = [parameter for parameter in self.parameters() if parameter.requires_grad]
        groups = [
            {"params": [p for p in parameters if p.dim() >= 2], "weight_decay": weight_decay},
            {"params": [p for p in parameters if p.dim() < 2], "weight_decay": 0.0},
        ]
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        fused = fused_available and device_type == "cuda"
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas, fused=fused)

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Autoregressively sample raw model predictions for later engine evaluation."""
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        for _ in range(max_new_tokens):
            context = idx[:, -self.config.block_size :]
            logits, _ = self(context)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))
            next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx
