# 第 3 章：前向与生成

手写 Qwen3 前向、生成循环与同权重验证。本目录独立运行，不依赖其他章节示例。

[阅读中文教程](../../../course-material/ch/part2/第3章_前向与生成_代码.md)

在仓库根目录运行，使用 Python 3.12：

```bash
python3.12 -m venv examples/part2/chapter03/.venv
source examples/part2/chapter03/.venv/bin/activate
python -m pip install -r examples/part2/chapter03/requirements.txt
python examples/part2/chapter03/forward.py
python -m pytest examples/part2/chapter03 -q
```

CPU FP32、随机权重，支持 dense Qwen3 全量因果注意力与贪心/温度采样。没有 KV Cache、padding 或预训练权重加载。Transformers 仅用于测试参照。

默认运行和测试均不下载模型、不需要 GPU。首次安装依赖需要网络。
