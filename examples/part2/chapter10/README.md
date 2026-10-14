# 第 10 章：投机解码

可手算的草稿验证、贪心纠正与随机采样。本目录独立运行，不依赖其他章节示例。

[阅读中文教程](../../../course-material/ch/part2/第10章_投机解码_代码.md)

在仓库根目录运行，使用 Python 3.12：

```bash
python examples/part2/chapter10/speculative.py
python -m unittest examples.part2.chapter10.test_speculative -v
```

只使用 Python 标准库。条件概率表代替神经网络，保留线性草稿验证与随机拒绝校正算法。没有持久化 KV、树状验证或 EAGLE；调用次数不代表加速比。

默认运行和测试均不下载模型、不需要 GPU。
