"""第 3 章：只用 PyTorch 写 Qwen3 前向与生成；CPU、随机权重、无 KV Cache。"""

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Config:
    vocab_size: int = 64
    hidden_size: int = 32
    intermediate_size: int = 64
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    head_dim: int = 8
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    max_position_embeddings: int = 256
    tie_word_embeddings: bool = False

    def __post_init__(self):
        if self.head_dim <= 0 or self.head_dim % 2:
            raise ValueError("RoPE requires a positive, even head_dim")
        if self.num_key_value_heads <= 0 or self.num_attention_heads <= 0:
            raise ValueError("head counts must be positive")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("query heads must be divisible by KV heads")


class RMSNorm(nn.Module):
    def __init__(self, width, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        values = x.float()
        normalized = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + self.eps)
        return normalized.to(x.dtype) * self.weight


def rope(x, theta):
    # x: [B, H, T, D]；Qwen3 将最后一维分成两半做旋转。
    length, dim = x.shape[-2:]
    frequencies = theta ** (-torch.arange(0, dim, 2, device=x.device).float() / dim)
    angles = torch.arange(length, device=x.device)[:, None] * frequencies
    angles = torch.cat((angles, angles), dim=-1)
    left, right = x.chunk(2, dim=-1)
    rotated = torch.cat((-right, left), dim=-1)
    return x * angles.cos().to(x.dtype) + rotated * angles.sin().to(x.dtype)


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.head_dim = c.head_dim
        self.groups = c.num_attention_heads // c.num_key_value_heads
        self.theta = c.rope_theta
        self.q_proj = nn.Linear(c.hidden_size, c.num_attention_heads * c.head_dim, bias=False)
        self.k_proj = nn.Linear(c.hidden_size, c.num_key_value_heads * c.head_dim, bias=False)
        self.v_proj = nn.Linear(c.hidden_size, c.num_key_value_heads * c.head_dim, bias=False)
        self.o_proj = nn.Linear(c.num_attention_heads * c.head_dim, c.hidden_size, bias=False)
        self.q_norm = RMSNorm(c.head_dim, c.rms_norm_eps)
        self.k_norm = RMSNorm(c.head_dim, c.rms_norm_eps)

    def forward(self, x):
        batch, length, _ = x.shape

        def split_heads(projection):
            return projection(x).view(batch, length, -1, self.head_dim).transpose(1, 2)

        q = rope(self.q_norm(split_heads(self.q_proj)), self.theta)
        k = rope(self.k_norm(split_heads(self.k_proj)), self.theta)
        v = split_heads(self.v_proj)
        k = k.repeat_interleave(self.groups, dim=1)  # GQA：多个 Q 头共享一组 K/V。
        v = v.repeat_interleave(self.groups, dim=1)
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)
        allowed = torch.ones(length, length, dtype=torch.bool, device=x.device).tril()
        scores = scores.masked_fill(~allowed, float("-inf"))
        weights = scores.float().softmax(-1).to(x.dtype)
        output = (weights @ v).transpose(1, 2).reshape(batch, length, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.gate_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.up_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.down_proj = nn.Linear(c.intermediate_size, c.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.self_attn = Attention(c)
        self.mlp = MLP(c)
        self.input_layernorm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(c.hidden_size, c.rms_norm_eps)

    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        return x + self.mlp(self.post_attention_layernorm(x))


class Qwen3(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = c = config or Config()
        # 保留参考实现的权重名，方便用同一份权重比较 logits。
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(c.vocab_size, c.hidden_size)
        self.model.layers = nn.ModuleList(Block(c) for _ in range(c.num_hidden_layers))
        self.model.norm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.lm_head = nn.Linear(c.hidden_size, c.vocab_size, bias=False)
        if c.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids):
        if not 0 < input_ids.shape[1] <= self.config.max_position_embeddings:
            raise ValueError("input length must be within the context limit")
        x = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            x = layer(x)
        return self.lm_head(self.model.norm(x))  # [B, T, vocab_size]


@torch.inference_mode()
def generate(model, prompt, max_new_tokens=8, eos_id=None, temperature=0.0):
    """单请求生成；返回新增 token（含 EOS）。model 应先调用 eval()。"""
    if not prompt or max_new_tokens < 0:
        raise ValueError("nonempty prompt and nonnegative output length required")
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and nonnegative")
    if len(prompt) + max_new_tokens > model.config.max_position_embeddings:
        raise ValueError("prompt + output exceeds context limit")
    device = next(model.parameters()).device
    tokens = torch.tensor([prompt], dtype=torch.long, device=device)
    output = []
    for _ in range(max_new_tokens):
        logits = model(tokens)[0, -1]  # 最后一行预测下一个 token。
        if temperature == 0:
            token = int(logits.argmax())
        else:
            probs = (logits.float() / temperature).softmax(-1)
            token = int(torch.multinomial(probs, 1).item())
        output.append(token)
        if token == eos_id:
            break
        tokens = torch.cat((tokens, tokens.new_tensor([[token]])), dim=1)
    return output


def main():
    torch.manual_seed(7)
    torch.set_num_threads(1)
    model = Qwen3().eval()
    with torch.inference_mode():
        logits = model(torch.tensor([[1, 2, 3]]))
    print("logits shape:", tuple(logits.shape))
    print("new token IDs:", generate(model, [1, 2, 3]))
    print("随机权重只演示计算流程，输出不具有语言含义。")


if __name__ == "__main__":
    main()
