# Chapter 10 Speculative Decoding (Code Implementation)

Ordinary generation confirms one new token per target-model call. Speculative decoding first asks a draft model for several candidates, then verifies them in one target-model call. A consecutive accepted prefix can be committed together; at the first mismatch, correct the token and draft again.

This article accompanies the [concept article](./Chapter10_Speculative-Decoding.md). A conditional probability table with only a few tokens replaces the neural network, so acceptance, rejection, and position alignment can all be checked by hand. The code runs independently of earlier chapters and needs no model weights. Start with greedy decoding, then read the sampling path that preserves the target distribution.

## 1 Learning Objectives

1. Write the draft → verify → commit loop.
2. Match each draft token to its prediction row and handle the first rejection and bonus token.
3. Implement speculative sampling with acceptance probabilities and the residual distribution.

## 2 Run an Example You Can Check by Hand

Only the Python 3.12 standard library is required. Run from the repository root:

```bash
python3 examples/part2/chapter10/speculative.py
```

The greedy path prints:

```text
greedy: [1, 2, 0, 1, 2, 0, 1, 2]
matches target: True
target_calls=3, accepted/proposed=5/9
```

Target-only generation and speculative generation produce the same tokens. The program also prints a sampled sequence. Verify the sampling path by its distribution: the same seed need not produce identical tokens because the two algorithms consume random numbers in different orders.

### 2.1 What Does the Probability Table Represent?

`TableModel` uses `table[i][j]` for the probability that the next token is j given that the current token is i. The target table is:

| Current token | Next token is 0 | Next token is 1 | Next token is 2 |
| --- | --- | --- | --- |
| 0 | 0.1 | 0.8 | 0.1 |
| 1 | 0.1 | 0.1 | 0.8 |
| 2 | 0.8 | 0.1 | 0.1 |

Starting from 0, greedy generation therefore cycles through 1, 2, 0. The draft table changes only the second row to `[0.1,0.7,0.2]`, making it deliberately choose the wrong token after 1.

`model(tokens)` returns a conditional distribution after each input position, like applying softmax to each row of a language model's logits. The table model depends only on the last token; a real causal model usually depends on the full prefix. Both follow the same convention: row i predicts token i+1. Target and draft must assign the same meanings to token IDs; matching vocabulary sizes is necessary but not sufficient.

## 3 Start with the Greedy Path

### 3.1 Draft: Extend a Temporary Prefix

Let `tokens` be the committed prefix. Each round proposes at most `draft_steps` candidates. Every draft step appends earlier candidates to the prefix, so later candidates depend on earlier ones.

The candidates live in a temporary `proposals` list. Only accepted tokens or target corrections enter the committed prefix. A draft EOS stops further drafting, but that EOS still needs verification.

### 3.2 Verify: Get k+1 Rows in One Call

Let the prefix length be `L` and the draft length be `k`. Feed the prefix and draft to the target together. Causality ensures each prediction row sees only its conditioning prefix:

| Target row, indexed from 0 | Purpose |
| --- | --- |
| `L-1` | Verify the first draft token |
| `L` | Verify the second draft token |
| `L+k-2` | Verify the final draft token |
| `L+k-1` | Choose an extra token if all candidates are accepted |

The program selects the verification rows with:

```python
verification = target(tokens + proposals)[len(tokens) - 1:]
```

Starting the slice at `L` would be one position too late: that row has already read the first candidate as input.

### 3.3 Commit: Stop at the First Rejection

In the example's first round, the draft proposes `[1,1,1]`. The target accepts the first 1, but its second choice is 2. The round therefore commits `[1,2]` and discards the remaining draft. The third candidate's conditioning prefix still contains the incorrect 1, so its verification row cannot be reused.

If every draft token is accepted, the target's final row already predicts the next token. Commit it as a bonus token. The code uses Python's `for ... else`: the bonus branch runs only if the verification loop did not `break` on rejection.

Every commit checks EOS and the remaining output budget. An accepted EOS, correction EOS, or bonus EOS ends generation immediately. Reaching `max_new_tokens` also prevents an extra bonus token from being emitted.

## 4 Read the Sampling Path

### 4.1 Acceptance Probability

Let `p` be the target distribution and `q` the draft distribution. A candidate `y` drawn from `q` is accepted with probability:

$$
\min\left(1,\frac{p(y)}{q(y)}\right).
$$

Because the candidate was drawn from `q`, it has `q(y)>0`. The program saves the actual `q` used at each draft step and retrieves `p` for the same conditioning prefix during verification. With neural networks, both must be the distributions actually used for sampling. If temperature or Top-k/Top-p filtering is applied, use the resulting probabilities.

### 4.2 Correction After Rejection

After rejecting a candidate, sample from the normalized positive residual:

$$
r(v)=\frac{\max(p(v)-q(v),0)}{\sum_u\max(p(u)-q(u),0)}.
$$

For example, let `p=[0.4,0.6]` and `q=[0.8,0.2]`. A proposed 0 is accepted with probability 0.5, while a proposed 1 is always accepted. After rejection, the correction must be 1. The final probability of emitting 0 is `0.8×0.5=0.4`; the probability of emitting 1 is `0.2+0.8×0.5=0.6`. This recovers the target distribution.

In general, acceptance contributes probability mass `min(p(v),q(v))` to token v, and rejection correction supplies the remaining positive difference. Simply drawing again from p after rejection would not provide this correction. Discard all candidates after the first rejection and start the next draft from the corrected prefix.

## 5 Verify Correctness Before Measuring Speed

```bash
python3 -m unittest examples.part2.chapter10.test_speculative -v
```

Tests first compare greedy outputs for different draft lengths, full acceptance, and full rejection, then check EOS, bonus tokens, and zero output. A separate model depends on the full prefix to check conditioning positions beyond the first-order table model.

The sampling test starts at token 0 and generates two tokens with target table `[[0.8,0.2],[0.3,0.7]]`. The exact joint probabilities are `P(00)=0.64`, `P(01)=0.16`, `P(10)=0.06`, and `P(11)=0.14`. With a fixed seed, 10000 trials check that each observed frequency is within 0.02 of its exact value. Another test checks the residual identity directly.

This example recomputes the full prefix each round without a persistent KV cache; only committed tokens enter the next round. With real caches, remove KV entries for unaccepted draft tokens. A correction or bonus token has valid KV only after it has passed through the model.

Table lookups do not model neural-network latency, and `target_calls` is not a speedup ratio. Actual gains depend on draft cost, accepted length, target verification, cache handling, and sampling overhead. This example implements linear speculative decoding. Tree verification, EAGLE, and GPU kernels are further engineering topics.

The complete program is `examples/part2/chapter10/speculative.py`. The course site includes the runnable file directly:

::: details Complete program: speculative.py

<<< @/examples/part2/chapter10/speculative.py

:::

## 6 Summary and Exercises

### 6.1 Summary

Speculative decoding drafts candidates and lets the target confirm them. Greedy decoding compares each candidate with the target's choice; sampling uses acceptance probabilities and positive residuals to preserve the target distribution. Both commit only a consecutive accepted prefix and draft again after the first rejection.

### 6.2 Exercises

1. With a prefix of length 5 and a draft of length 3, which rows verify the candidates and produce the bonus token?
2. Set `draft = target` and run again. How do acceptance and the number of target calls change?
3. Replace residual sampling after rejection with a direct draw from p. What happens to the joint-distribution test?
4. If the draft proposes EOS but the target rejects it, what should be committed, and should generation stop?

## References

- [Leviathan et al.: Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [Chapter 10: Speculative Decoding concepts](./Chapter10_Speculative-Decoding.md)
