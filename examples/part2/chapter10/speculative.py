"""Chapter 10: linear speculative decoding with small conditional probability tables."""

from dataclasses import dataclass
import math
import random


class TableModel:
    """table[i][j] = P(next token is j | current token is i)."""

    def __init__(self, table):
        self.table = table
        self.vocab_size = len(table)
        if not table or any(len(row) != self.vocab_size or
                            any(not math.isfinite(p) or p < 0 for p in row) or
                            not math.isclose(sum(row), 1.0) for row in table):
            raise ValueError("each row must be a probability distribution over the vocabulary")

    def __call__(self, tokens):
        # Row i reads only tokens[i], so it cannot see later candidates.
        return [self.table[token] for token in tokens]


def pick(probs, sample, rng):
    if sample:
        return rng.choices(range(len(probs)), weights=probs, k=1)[0]
    return max(range(len(probs)), key=probs.__getitem__)


def residual_distribution(p, q):
    residual = [max(a - b, 0.0) for a, b in zip(p, q)]
    mass = sum(residual)
    if mass <= 0:
        raise ValueError("a rejected proposal must leave positive residual mass")
    return [value / mass for value in residual]


@dataclass
class Stats:
    target_calls: int = 0
    draft_calls: int = 0
    proposed: int = 0
    accepted: int = 0


def speculative_generate(target, draft, prompt, max_new_tokens=8, draft_steps=3,
                         eos_id=None, sample=False, rng=None):
    """Return (new tokens, stats); both models must use the same token ID meanings."""
    if not prompt or max_new_tokens < 0 or draft_steps < 1:
        raise ValueError("nonempty prompt, nonnegative length and positive draft_steps required")
    if target.vocab_size != draft.vocab_size:
        raise ValueError("target and draft must share a vocabulary")
    rng = rng if rng is not None else random.Random()
    tokens, output, stats = list(prompt), [], Stats()

    def commit(token):
        tokens.append(token)
        output.append(token)
        return token == eos_id or len(output) == max_new_tokens

    while len(output) < max_new_tokens:
        proposals, draft_probs = [], []
        remaining = max_new_tokens - len(output)
        for _ in range(min(draft_steps, remaining)):
            q = draft(tokens + proposals)[-1]
            stats.draft_calls += 1
            token = pick(q, sample, rng)
            proposals.append(token)
            draft_probs.append(q)
            if token == eos_id:
                break  # A draft EOS still needs verification before ending the request.

        # Prefix length L, draft length k: starting at L-1 gives exactly k+1 rows.
        verification = target(tokens + proposals)[len(tokens) - 1:]
        stats.target_calls += 1
        stats.proposed += len(proposals)
        for i, token in enumerate(proposals):
            p, q = verification[i], draft_probs[i]
            if sample:
                accepted = rng.random() < min(1.0, p[token] / q[token])
                replacement = None if accepted else pick(residual_distribution(p, q), True, rng)
            else:
                replacement = pick(p, False, rng)
                accepted = token == replacement
            if accepted:
                stats.accepted += 1
                if commit(token):
                    return output, stats
            else:
                if commit(replacement):
                    return output, stats
                break  # Later draft conditionals are invalid; restart from the corrected prefix.
        else:
            # No break: all proposals were accepted; use the final row for a bonus token.
            if commit(pick(verification[-1], sample, rng)):
                return output, stats
    return output, stats


def greedy_generate(model, prompt, max_new_tokens, eos_id=None):
    """Target-only, one-token-at-a-time baseline for checking speculative greedy output."""
    tokens, output = list(prompt), []
    for _ in range(max_new_tokens):
        token = pick(model(tokens)[-1], False, None)
        tokens.append(token)
        output.append(token)
        if token == eos_id:
            break
    return output


def main():
    target = TableModel([[0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.8, 0.1, 0.1]])
    draft = TableModel([[0.1, 0.8, 0.1], [0.1, 0.7, 0.2], [0.8, 0.1, 0.1]])
    output, stats = speculative_generate(target, draft, [0])
    print("greedy:", output)
    print("matches target:", output == greedy_generate(target, [0], 8))
    print(f"target_calls={stats.target_calls}, accepted/proposed={stats.accepted}/{stats.proposed}")
    sampled, _ = speculative_generate(target, draft, [0], sample=True, rng=random.Random(7))
    print("sampled:", sampled)
    print("Call counts illustrate the algorithm; they are not a measured speedup.")


if __name__ == "__main__":
    main()
