"""Check greedy outputs, conditioning positions, EOS, and the sampled sequence distribution."""

from collections import Counter
import random
import unittest

from .speculative import TableModel, greedy_generate, residual_distribution, speculative_generate


class SpeculativeTests(unittest.TestCase):
    def test_greedy_with_correct_and_wrong_drafts(self):
        target = TableModel([[0.1, 0.9], [0.8, 0.2]])
        wrong = TableModel([[0.9, 0.1], [0.1, 0.9]])
        for draft in (target, wrong):
            for steps in (1, 3, 8):
                with self.subTest(draft=draft, steps=steps):
                    result, stats = speculative_generate(target, draft, [0], 13, steps)
                    self.assertEqual(result, greedy_generate(target, [0], 13))
                    if draft is target:
                        self.assertEqual(stats.accepted, stats.proposed)
                        self.assertEqual(stats.target_calls, (13 + steps) // (steps + 1))

    def test_verification_uses_full_conditioning_prefix(self):
        class PrefixModel:
            vocab_size = 3

            def __init__(self, offset=0):
                self.offset = offset

            def __call__(self, tokens):
                # Use the full prefix to catch conditioning errors a first-order model might miss.
                return [[float(v == (sum(tokens[:i + 1]) + i + self.offset) % 3)
                         for v in range(3)] for i in range(len(tokens))]

        target, draft = PrefixModel(), PrefixModel(1)
        for sample in (False, True):
            result, _ = speculative_generate(target, draft, [1, 2], 11, 4,
                                             sample=sample, rng=random.Random(7))
            self.assertEqual(result, greedy_generate(target, [1, 2], 11))

    def test_eos_and_output_budget(self):
        target = TableModel([[0, 1], [0, 1]])
        wrong = TableModel([[1, 0], [1, 0]])
        flip = TableModel([[0, 1], [1, 0]])
        for sample in (False, True):
            def run(model, draft, prompt, limit, steps=3, eos=None):
                return speculative_generate(model, draft, prompt, limit, steps, eos,
                                            sample, random.Random(7))[0]
            # Rejected draft EOS; correction EOS; accepted EOS; bonus EOS.
            self.assertEqual(run(target, wrong, [1], 3, eos=0), [1, 1, 1])
            self.assertEqual(run(target, wrong, [0], 5, eos=1), [1])
            self.assertEqual(run(target, target, [0], 5, eos=1), [1])
            self.assertEqual(run(flip, flip, [0], 5, 1, eos=0), [1, 0])
            self.assertEqual(run(target, target, [0], 1, 8), [1])
        result, stats = speculative_generate(target, wrong, [0], 0)
        self.assertEqual(result, [])
        self.assertEqual((stats.target_calls, stats.draft_calls), (0, 0))

    def test_residual_recovers_missing_target_mass(self):
        p, q = [0.0, 0.3, 0.7], [0.5, 0.5, 0.0]
        residual = residual_distribution(p, q)
        accepted_mass = [min(a, b) for a, b in zip(p, q)]
        for wanted, accepted, correction in zip(p, accepted_mass, residual):
            self.assertAlmostEqual(accepted + (1 - sum(accepted_mass)) * correction, wanted)

    def test_sampled_joint_distribution(self):
        target = TableModel([[0.8, 0.2], [0.3, 0.7]])
        draft = TableModel([[0.1, 0.9], [0.9, 0.1]])
        rng = random.Random(2026)
        counts = Counter()
        for _ in range(10000):
            result, _ = speculative_generate(target, draft, [0], 2, sample=True, rng=rng)
            counts[tuple(result)] += 1
        # P(y1,y2 | x=0) = P(y1 | 0) * P(y2 | y1)。
        exact = {(0, 0): 0.64, (0, 1): 0.16, (1, 0): 0.06, (1, 1): 0.14}
        for pair, probability in exact.items():
            self.assertAlmostEqual(counts[pair] / 10000, probability, delta=0.02)

    def test_invalid_inputs(self):
        model = TableModel([[0.5, 0.5], [0.5, 0.5]])
        for kwargs in ({"draft_steps": 0}, {"max_new_tokens": -1}, {"prompt": []}):
            arguments = dict(prompt=[0]) | kwargs
            with self.assertRaises(ValueError):
                speculative_generate(model, model, **arguments)
        with self.assertRaises(ValueError):
            speculative_generate(model, TableModel([[1.0]]), [0])
        with self.assertRaises(ValueError):
            TableModel([[0.2, 0.2], [0.5, 0.5]])


if __name__ == "__main__":
    unittest.main()
