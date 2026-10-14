# Chapter 3: Forward Pass and Generation

Hand-written Qwen3 forward pass, generation, and identical-weight checks. This example runs independently of the other chapters.

[English tutorial](../../../course-material/eng/part2/Chapter3_Forward-Pass-and-Generation_code.md) · [中文教程](../../../course-material/ch/part2/第3章_前向与生成_代码.md)

Run from the repository root with Python 3.12:

```bash
python3.12 -m venv examples/part2/chapter03/.venv
source examples/part2/chapter03/.venv/bin/activate
python -m pip install -r examples/part2/chapter03/requirements.txt
python examples/part2/chapter03/forward.py
python -m pytest examples/part2/chapter03 -q
```

Uses random weights and CPU FP32. Supports dense Qwen3 with full causal attention, greedy decoding, and temperature sampling. No KV cache, padding, or pretrained weight loader. Transformers is used only as a test reference.

The default example and tests need no model download or GPU. Installing dependencies for the first time needs network access.
