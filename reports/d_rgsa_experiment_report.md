# D-RGSA 实验阶段报告

## 1. 研究问题

Direct-RGSA 把 teacher 生成的 rationale tokens 直接设为 global attention anchors，
但测试阶段仍需调用大模型。D-RGSA 的目标不是把 Direct-RGSA 称作 oracle，也不假设
teacher 给出的 anchors 是性能上限；它学习 teacher 的 token-selection 行为，并在测试
阶段用轻量 selector 独立预测 anchors。

因此本文的中心问题是：

> teacher rationale 中的 token-selection knowledge 能否被蒸馏进一个轻量 selector，
> 在移除测试阶段 teacher 调用后，仍保留 rationale-guided sparse attention 的主要收益？

## 2. 已实现方法

- Direct-RGSA：直接使用 teacher rationale anchors。
- D-RGSA：BiGRU token selector 预测 anchors，student 使用 local + predicted global
  connectivity。
- MLP selector：检验上下文编码是否必要。
- Shuffled anchors：保持 anchor 数量，随机打乱位置，检验收益是否来自语义位置。
- Anchor-budget 消融：`k = 1, 2, 3, 4`。
- Window-radius 消融：`r = 2, 4, 8`。
- Qwen teacher：已完成 SiliconFlow 兼容、并发、重试、失败记录和断点续跑代码，并
  完成 SST-2 全量 6,000 条标注与五随机种子 teacher-sensitivity 实验。

所有 classifier 共享相同的 2-layer、hidden size 128、4-head Transformer student。
当前 sparsity 是 attention connectivity 的逻辑稀疏率；实现仍使用 dense mask，不能
把训练时间写成 sparse-kernel 加速结果。

## 3. 五随机种子主结果

随机种子为 13、42、87、101、123。数值均为 test mean ± sample standard deviation。

| Dataset | Method | Macro-F1 (%) | Accuracy (%) | Logical sparsity (%) |
|---|---|---:|---:|---:|
| SST-2 | Full | 79.64 ± 1.39 | 79.97 ± 1.30 | 0.00 |
| SST-2 | TF-IDF | 79.91 ± 0.64 | 80.20 ± 0.57 | 47.08 |
| SST-2 | Direct-RGSA | 81.24 ± 0.97 | 81.43 ± 0.96 | 46.39 |
| SST-2 | D-RGSA | 80.87 ± 0.47 | 81.12 ± 0.51 | 46.14 |
| TREC | Full | 80.39 ± 1.29 | 80.80 ± 2.60 | 0.00 |
| TREC | TF-IDF | 81.35 ± 1.56 | 82.17 ± 2.50 | 23.46 |
| TREC | Direct-RGSA | 83.46 ± 2.30 | 85.10 ± 1.46 | 21.11 |
| TREC | D-RGSA | 82.64 ± 0.94 | 82.33 ± 0.38 | 18.54 |
| AG News | Full | 86.79 ± 0.86 | 86.70 ± 0.87 | 0.00 |
| AG News | TF-IDF | 81.69 ± 1.43 | 81.69 ± 1.44 | 74.44 |
| AG News | Direct-RGSA | 84.14 ± 1.59 | 84.13 ± 1.58 | 74.24 |
| AG News | D-RGSA | 83.37 ± 0.72 | 83.32 ± 0.74 | 74.16 |

完整六方法表在 `outputs/distilled-altair-figures/main_results.md`。

### 主要观察

1. D-RGSA 在三个数据集上均优于 TF-IDF，Macro-F1 分别提高 0.96、1.29、1.68
   points。
2. D-RGSA 相比 Direct-RGSA 分别下降 0.37、0.82、0.77 points，三个数据集平均只下降
   0.65 points，但移除了测试阶段 teacher 调用。
3. 以 TF-IDF 为非 teacher sparse baseline，D-RGSA 保留了 Direct-RGSA 增益的大约
   72%（SST-2）、61%（TREC）和 69%（AG News）。
4. D-RGSA 的五种子标准差在三个数据集上都小于 Direct-RGSA，说明蒸馏 selector
   可能起到平滑 teacher anchor noise 的作用。
5. AG News 上 Full 仍比 D-RGSA 高 3.42 points。这应被明确写作方法边界：长文本中
   dense context 仍有优势，D-RGSA 的价值是高逻辑稀疏率下优于其他 sparse baselines，
   不是在所有数据集上击败 Full。

## 4. Selector 蒸馏质量

| Dataset | Test anchor F1 | Precision | Recall | Exact match |
|---|---:|---:|---:|---:|
| SST-2 | 67.62 | 64.12 | 71.52 | 32.43 |
| TREC | 74.51 | 68.32 | 81.94 | 29.32 |
| AG News | 40.03 | 39.97 | 40.08 | 11.68 |

AG News 的 token-level imitation 明显更难，可能来自文本更长、主题证据更分散和
teacher rationale 表达更多样。尽管如此，AG News D-RGSA 仍优于 Random 与 TF-IDF，
说明分类收益不要求逐 token 完全复制 teacher。

## 5. Teacher sensitivity：DeepSeek 与 Qwen

Qwen 使用 `Qwen/Qwen3.6-27B`、temperature 0、non-thinking 模式，并与 DeepSeek
使用相同 split、prompt、JSON schema 和无 gold-label 输入。SST-2 的 6,000 条请求
全部返回，Qwen anchor validity 为 99.93%，teacher/gold agreement 为 93.38%。

在 5,985 个 DeepSeek/Qwen 共同样本上：

- teacher label agreement：94.47%；
- anchor-position Jaccard：74.60%；
- DeepSeek teacher accuracy：91.13%；
- Qwen teacher accuracy：93.42%。

| Teacher | Method | SST-2 Macro-F1 (%) | Runs |
|---|---|---:|---:|
| DeepSeek | Direct-RGSA | 81.24 ± 0.97 | 5 |
| DeepSeek | D-RGSA | 80.87 ± 0.47 | 5 |
| Qwen | Direct-RGSA | 81.31 ± 1.05 | 5 |
| Qwen | D-RGSA | 80.78 ± 1.22 | 5 |

换 teacher 后 Direct-RGSA 均值变化 0.07 point，D-RGSA 变化 -0.09 point。这个结果
说明论文主线不依赖某个特定 teacher；Qwen 是 robustness/sensitivity 实验，而不是
另一条主方法。

Qwen→selector 的 test anchor F1 为 65.52%，DeepSeek→selector 为 67.62%。尽管两个
teacher 的 rationale 位置并不完全相同，D-RGSA 下游结果仍非常接近。

### Teacher-correct / teacher-wrong 分组

| Group | Direct-RGSA | D-RGSA | Samples |
|---|---:|---:|---:|
| Qwen teacher correct | 82.62 ± 1.13 | 81.63 ± 1.18 | 558 |
| Qwen teacher wrong | 62.31 ± 8.36 | 69.04 ± 4.01 | 42 |

在 teacher 正确样本上 Direct-RGSA 更高约 0.99 point；在 teacher 错误的小样本组上，
D-RGSA 高约 6.73 points 且方差更低。这支持 selector 可能平滑错误 rationale 的解释，
但 teacher-wrong 组只有 42 条，应作为机制性观察而不是强统计结论。

## 6. 三随机种子消融

### Selector architecture 与语义位置

| Variant | SST-2 Macro-F1 (%) |
|---|---:|
| Direct-RGSA | 81.54 ± 1.24 |
| D-RGSA (BiGRU) | 80.99 ± 0.61 |
| D-RGSA (MLP) | 80.57 ± 0.90 |
| Shuffled anchors | 78.77 ± 1.14 |

BiGRU 比 MLP 高 0.42 points，说明上下文 token 表示有用。D-RGSA 比 shuffled anchors
高 2.22 points，说明提升来自预测出的语义位置，而不只是相同数量的 global edges。

### Anchor budget

| k | Macro-F1 (%) | Logical sparsity (%) |
|---:|---:|---:|
| 1 | 80.10 ± 1.09 | 59.44 |
| 2 | 80.52 ± 0.76 | 54.93 |
| 3 | 81.01 ± 1.41 | 50.48 |
| 4 | 80.99 ± 0.61 | 46.14 |

从 1 到 3 个 anchors 效果提升，4 个基本持平；增加 connectivity 并非单调提高性能。

### Local window radius

| Radius | Macro-F1 (%) | Logical sparsity (%) |
|---:|---:|---:|
| 2 | 80.59 ± 0.67 | 54.15 |
| 4 | 80.99 ± 0.61 | 46.14 |
| 8 | 80.29 ± 1.61 | 31.98 |

半径 4 在当前配置下最稳健。半径 2 的局部上下文不足，半径 8 则降低稀疏性且没有带来
更高 Macro-F1。

## 7. 当前可以支持的论文结论

可以支持：

- teacher rationale 的 token-selection behavior 可被轻量 selector 学习；
- 测试阶段无需 teacher，也能保留 Direct-RGSA 相对非 teacher sparse baseline 的大部分
  收益；
- 语义 anchor 位置比等数量随机位置更重要；
- D-RGSA 的优势主要体现在 controlled sparse baselines，而不是无条件超过 Full。
- 更换 DeepSeek/Qwen 后 SST-2 结果基本稳定，方法不依赖单一 teacher。
- teacher-wrong 子集结果提示蒸馏 selector 可能具有一定的 rationale-noise smoothing
  效果。

暂时不能支持：

- teacher anchors 是 oracle 或性能上限；
- dense mask 的 wall-clock 加速；
- 所有数据集上都对不同 teacher 完全稳健（当前 Qwen 正式实验覆盖 SST-2）；
- 任意长序列或任意数据集都优于 Full Attention。

## 8. 尚待完成

1. 如果篇幅和预算允许，再将 Qwen teacher sensitivity 从 SST-2 扩展到 TREC 与
   AG News；当前 SST-2 五种子结果已足以作为代表性 robustness 实验。
2. 对 teacher-wrong 的 42 个样本做人工错误类型抽检。
3. 将最终数字、Altair 图和限制陈述写入新论文版本。
