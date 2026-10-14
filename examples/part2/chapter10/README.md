# Chapter 10: Speculative Decoding

Draft verification, greedy correction, and sampling with small probability tables. This example runs independently of the other chapters.

[English tutorial](../../../course-material/eng/part2/Chapter10_Speculative-Decoding_code.md) · [中文教程](../../../course-material/ch/part2/第10章_投机解码_代码.md)

Run from the repository root with Python 3.12:

```bash
python examples/part2/chapter10/speculative.py
python -m unittest examples.part2.chapter10.test_speculative -v
```

Uses only the Python standard library. Conditional probability tables replace neural networks while preserving the linear draft verification and rejection-correction algorithm. No persistent KV cache, tree verification, or EAGLE; call counts are not a speedup measurement.

The default example and tests need no model download or GPU.
