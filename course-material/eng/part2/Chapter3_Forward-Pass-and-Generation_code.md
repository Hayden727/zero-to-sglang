# Chapter 3 Forward Pass and Generation (Code Implementation)

Given a sequence of token IDs, how does a model compute the next token and keep generating? This chapter turns the [concept article](./Chapter3_Forward-Pass-and-Generation.md) into a small Qwen3 model you can run on a CPU: embedding → Transformer Block → logits → new token.

We use random weights and implement dense Qwen3 with full causal attention. Each generation step recomputes the entire prefix so the relationship between a forward pass and generation stays visible. [Chapter 4](./Chapter4_KV-Cache-Optimization.md) covers incremental computation with a KV cache. This example runs independently and needs no model download.

## 1 Learning Objectives

1. Follow tensor shapes through Attention, the MLP, and residual connections.
2. Generate tokens from the last row of logits and handle EOS and output limits.
3. Check numerical results against a reference with identical weights, beyond checking shapes.

## 2 Run the Example First

Run all commands from the repository root. Use Python 3.12 and a CPU. The program requires PyTorch; Transformers and pytest are used for verification.

```bash
python3.12 -m venv examples/part2/chapter03/.venv
source examples/part2/chapter03/.venv/bin/activate
python -m pip install -r examples/part2/chapter03/requirements.txt
python examples/part2/chapter03/forward.py
```

The first output line is:

```text
logits shape: (1, 3, 64)
```

This means one request, three input positions, and 64 vocabulary entries. The second line prints eight new token IDs. Random weights have no learned language ability; focus on the shapes, computation order, and output length.

The complete program is `examples/part2/chapter03/forward.py`. Read it in this order: `Qwen3.forward → Block.forward → Attention.forward → generate`.

## 3 From Input IDs to Logits

### 3.1 Start with the Shapes

Let `B` be the batch size, `T` the sequence length, `C` the hidden size, `Hq` the number of query heads, `Hkv` the number of KV heads, and `D` the dimension of each head.

| Computation | Output shape | Default single-request example |
| --- | --- | --- |
| Token IDs | `[B,T]` | `[1,3]` |
| Embedding | `[B,T,C]` | `[1,3,32]` |
| Q projection and head split | `[B,Hq,T,D]` | `[1,4,3,8]` |
| K/V projections and head split | `[B,Hkv,T,D]` | `[1,2,3,8]` |
| Attention scores | `[B,Hq,T,T]` | `[1,4,3,3]` |
| Head merge and output projection | `[B,T,C]` | `[1,3,32]` |
| Final normalization and LM Head | `[B,T,V]` | `[1,3,64]` |

`head_dim` is configured separately: do not assume every Qwen3 configuration satisfies `C = Hq * D`. The output projection maps the concatenated heads back to dimension `C`.

### 3.2 Four Steps Inside Attention

Three linear layers produce Q, K, and V, then each token's vector is split into heads. Qwen3 applies RMSNorm to each Q and K head before using RoPE to encode position. RMSNorm normalizes along the last dimension. RoPE rotates the two halves of that dimension; V is not rotated.

GQA lets several query heads share fewer KV heads. This example uses `repeat_interleave` to expand K/V to `Hq` heads for ordinary matrix multiplication. This makes the operation easy to read; an inference kernel can access shared KV heads directly.

The core attention computation takes three steps:

```python
scores = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)
scores = scores.masked_fill(~allowed, float("-inf"))
weights = scores.float().softmax(-1).to(x.dtype)
```

`allowed` is a lower-triangular boolean matrix: row `i` can read only input positions `0..i`. Filling future positions with negative infinity makes their softmax weights zero. Finally, compute `weights @ v`, merge the heads, and apply the output projection.

### 3.3 Blocks and the LM Head

Each Block applies Attention followed by a gated MLP. Both operations receive normalized inputs and use residual connections:

```python
x = x + self.self_attn(self.input_layernorm(x))
return x + self.mlp(self.post_attention_layernorm(x))
```

The MLP computes `down_proj(silu(gate_proj(x)) * up_proj(x))`. It transforms features at each position independently; Attention exchanges information between tokens.

After the Blocks, a final RMSNorm and LM Head produce logits of shape `[B,T,V]`. This forward pass is implemented here; the Transformers model is used only as a reference in tests.

## 4 Turn a Forward Pass into a Generation Loop

### 4.1 Why Read the Last Row?

For input `[1,2,3]`, the last row of logits gives scores for the token after 3. Choose a token, append it to the input, and repeat:

```python
logits = model(tokens)[0, -1]
token = int(logits.argmax())
tokens = torch.cat((tokens, tokens.new_tensor([[token]])), dim=1)
```

The complete `generate` function also appends the token to `output` and checks for EOS. Its return value contains only new tokens, including EOS if generated. With `max_new_tokens=0`, it returns an empty list without calling the model.

### 4.2 Greedy Decoding and Temperature Sampling

With `temperature=0`, choose the highest-scoring token. With a positive temperature, compute `softmax(logits / temperature)` and sample from that distribution. Start Python from the repository root and try:

```python
from examples.part2.chapter03.forward import Qwen3, generate

model = Qwen3().eval()
print(generate(model, [1, 2, 3], max_new_tokens=5, temperature=0.8))
```

The concept article explains Top-k and Top-p; this example keeps the two minimal generation paths above. `eval()` selects evaluation mode, while `inference_mode()` disables gradient recording. They serve different purposes.

### 4.3 What Still Gets Recomputed?

The first generation step receives three tokens, the second four, and the third five. Every step recomputes the historical tokens' Q/K/V and Block outputs. This straightforward but inefficient implementation provides a correctness baseline for the KV cache in the next chapter.

## 5 Verify Results and Compare the Source

```bash
python -m pytest examples/part2/chapter03 -q
```

The tests copy the weights of a random Qwen3 model from Transformers `4.57.1` into this implementation, compare CPU FP32 logits with `atol=2e-6, rtol=2e-5`, and compare eight steps of greedy generation. Another test changes only the final two input tokens and checks that logits at the first two positions stay unchanged, verifying the causal mask.

The tests also cover tied weights, EOS, zero output, and the context limit. Passing these checks establishes agreement for the supported configurations and paths. The example does not load pretrained weights or implement padding, sliding attention, RoPE scaling, or a KV cache.

In the pinned Transformers source linked below, find `Qwen3Attention`, `Qwen3MLP`, and `Qwen3DecoderLayer` and map their computations to this chapter. Compare the mathematical steps first, then read the cache handling and backend selection code.

The course site includes the runnable file directly so there is only one complete implementation to maintain. On GitHub, open the source under `examples/part2/chapter03/`.

::: details Complete program: forward.py

<<< @/examples/part2/chapter03/forward.py

:::

## 6 Summary and Exercises

### 6.1 Summary

A forward pass turns token IDs into logits at each position. The generation loop repeatedly reads the last row, chooses a new token, and appends it to the prefix. Once this flow is clear, use identical-weight comparisons and causality tests to check the Attention details.

### 6.2 Exercises

1. If the input shape is `[2,5]` and the vocabulary has 64 entries, what is the logits shape? Which row continues generation?
2. Which test should fail if you remove the causal mask, and why?
3. Change `num_key_value_heads` from 2 to 1. Which tensor shapes change and which stay the same?
4. What does the generation loop recompute? Which computations can a KV cache avoid?

## References

- [Transformers v4.57.1: Qwen3 forward implementation](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3/modeling_qwen3.py)
- [Chapter 3: Forward Pass and Generation concepts](./Chapter3_Forward-Pass-and-Generation.md)
- [Chapter 4: KV Cache Optimization](./Chapter4_KV-Cache-Optimization.md)
