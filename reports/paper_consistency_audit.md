# D-RGSA 论文数据、图表与文字一致性审计

审计日期：2026-07-26

## 审计范围

- 五随机种子主实验：13、42、87、101、123。
- 三随机种子消融：13、42、87。
- 原始 `metrics.json`、Altair `main_results.csv`、实验报告、论文表格、图注与正文结论。
- DeepSeek/Qwen teacher sensitivity、selector 指标和 teacher-correct/wrong 分组。

## 结论

1. 18 个“数据集 × attention method”主结果均从原始五种子记录重新计算；与
   `outputs/distilled-altair-figures/main_results.csv` 的最大绝对差为 0。
2. 论文嵌入的五张 Altair PNG 与输出目录中的对应 PNG SHA-256 完全一致，不存在插入
   旧图或错图的问题。
3. 主表中的 Macro-F1、Accuracy 和 logical sparsity 与 Altair 图、实验报告一致。
4. Selector quality、teacher sensitivity、teacher-correct/wrong 和三种子消融均可从原始
   文件复现。
5. 原文有两处需要修正：
   - TREC 上 Local 为 82.88，D-RGSA 为 82.64，因此不能写成 D-RGSA 在三个数据集上
     超过所有 non-teacher sparse baselines。图注已改为：D-RGSA 在三个数据集上均超过
     Random 和 TF-IDF，并在 TREC 上与 Local 保持竞争力。
   - Window radius = 2 的样本标准差为 0.674647，保留两位小数应为 0.67；报告中的
     0.68 已修正。

## 主结果复核

| Dataset | Method | Macro-F1 (%) | Accuracy (%) | Sparsity (%) |
|---|---|---:|---:|---:|
| SST-2 | Full | 79.64 ± 1.39 | 79.97 ± 1.30 | 0.00 |
| SST-2 | Local | 77.05 ± 1.94 | 77.39 ± 1.86 | 63.93 |
| SST-2 | Random | 79.56 ± 1.22 | 79.86 ± 1.17 | 46.57 |
| SST-2 | TF-IDF | 79.91 ± 0.64 | 80.20 ± 0.57 | 47.08 |
| SST-2 | Direct-RGSA | 81.24 ± 0.97 | 81.43 ± 0.96 | 46.39 |
| SST-2 | D-RGSA | 80.87 ± 0.47 | 81.12 ± 0.51 | 46.14 |
| TREC | Full | 80.39 ± 1.29 | 80.80 ± 2.60 | 0.00 |
| TREC | Local | 82.88 ± 2.25 | 82.93 ± 1.35 | 36.64 |
| TREC | Random | 80.90 ± 2.05 | 81.24 ± 1.55 | 18.41 |
| TREC | TF-IDF | 81.35 ± 1.56 | 82.17 ± 2.50 | 23.46 |
| TREC | Direct-RGSA | 83.46 ± 2.30 | 85.10 ± 1.46 | 21.11 |
| TREC | D-RGSA | 82.64 ± 0.94 | 82.33 ± 0.38 | 18.54 |
| AG News | Full | 86.79 ± 0.86 | 86.70 ± 0.87 | 0.00 |
| AG News | Local | 69.28 ± 0.93 | 69.37 ± 0.91 | 85.48 |
| AG News | Random | 82.78 ± 1.00 | 82.77 ± 0.95 | 74.54 |
| AG News | TF-IDF | 81.69 ± 1.43 | 81.69 ± 1.44 | 74.44 |
| AG News | Direct-RGSA | 84.14 ± 1.59 | 84.13 ± 1.58 | 74.24 |
| AG News | D-RGSA | 83.37 ± 0.72 | 83.32 ± 0.74 | 74.16 |

## 引用与篇幅

- 参考文献共 30 条，正文引用 30 条；无缺失引用、无未使用条目。
- 2023–2025 年首发文献为 21/30（70.0%）；删除的 10 条均为与主实验关系较弱的
  通用蒸馏、综述或额外解码器稀疏注意力文献。
- 实验实际使用的 Qwen3.6-27B 已从错误的 2025 Qwen3 技术报告改为 Qwen Team 2026
  年模型卡。
- `texcount` 的总字符统计为 14,815（空格不计）；其中正文 13,806、标题 362、
  图表题注 616。
- 最终 PDF 共 7 页，正文与结论延伸至第 5 页，参考文献位于第 5–7 页。
- 已逐页渲染复核：Table 1 的 18 行主结果在第 4 页完整显示；Macro-F1、RGSA gain、
  消融和 teacher sensitivity 图的坐标轴、图例与面板均未被裁切；Figure 4 完整
  显示后才开始 REFERENCES。
