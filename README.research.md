# Distilling Teacher Rationales into Global Anchors

本项目是在冻结的 RGSA 实验之上建立的独立研究版本，主方法为 **D-RGSA
(Distilled Rationale-Guided Sparse Attention)**。原始 Direct-RGSA 在训练和测试时都
需要大模型生成 rationale anchors；D-RGSA 先用 teacher rationales 监督一个轻量
token selector，测试阶段只运行 selector 与 student classifier，不再调用 teacher。

核心比较保持同一个自定义 Lightweight Transformer student，只改变 attention
connectivity：

1. `full`：全连接 self-attention；
2. `local`：固定局部窗口；
3. `random`：local + 随机 global anchors；
4. `tfidf`：local + TF-IDF global anchors；
5. `rationale`：Direct-RGSA，直接使用 teacher rationale anchors；
6. `distilled`：D-RGSA，使用轻量 selector 预测的 global anchors；
7. `shuffled`：与 rationale 数量相同但位置被打乱的语义消融。

当前 DeepSeek teacher 的五随机种子主结果、三随机种子消融和 Altair 图表在：

- `outputs/distilled-altair-figures/main_results.md`
- `outputs/distilled-altair-figures/main_macro_f1.{html,svg,png}`
- `outputs/distilled-altair-figures/d_rgsa_gain.{html,svg,png}`
- `outputs/distilled-altair-figures/selector_quality.{html,svg,png}`
- `outputs/distilled-altair-figures/sst2_ablation.{html,svg,png}`
- `outputs/distilled-altair-figures/d_rgsa_hyperparameters.{html,svg,png}`
- `outputs/distilled-altair-figures/teacher_sensitivity.{html,svg,png}`
- `outputs/distilled-altair-figures/teacher_correctness_ablation.{html,svg,png}`

Qwen teacher sensitivity 使用与 DeepSeek 完全相同的 prompt、split、temperature=0 和
JSON schema：

```bash
export SILICONFLOW_API_KEY="..."
PYTHON_BIN="$PWD/.venv/bin/python" scripts/run_qwen_teacher.sh
PYTHON_BIN="$PWD/.venv/bin/python" scripts/run_qwen_experiments.sh
```

两段脚本均支持断点续跑：已有 teacher 标注和已有 `metrics.json` 不会重复计算。

## Direct-RGSA 基础实现

这是“方案 A”的完整实验代码：teacher model 为每个文本样本生成 `label`、
`short_rationale` 和可在原文中定位的 `rationale_tokens`；student tokenizer 再把这些
文本片段对齐到 WordPiece token，并将其作为 global attention anchors。

主 student 是从零初始化、由本项目自行实现的 `SmallTransformerClassifier`，不是
BERT 或 DistilBERT 模型。Hugging Face 的 DistilBERT tokenizer 只负责分词和提供
词表，不加载任何 DistilBERT 预训练权重。默认结构为：

| Component | Default |
|---|---:|
| Encoder layers | 2 |
| Hidden size | 128 |
| Attention heads | 4 |
| FFN size | 512 |
| Maximum length | 128 |
| Dropout | 0.1 |
| Pooling | CLS |

代码显式实现了 multi-head self-attention、FFN、residual connection、LayerNorm、
token embedding 和 positional embedding。所有实验变体共用完全相同的结构与初始化
方式，唯一变化是 attention connectivity。

实现的五组公平对照：

1. `full`：全连接 self-attention；
2. `local`：固定局部窗口；
3. `random`：local + 随机 global anchors；
4. `tfidf`：local + TF-IDF global anchors；
5. `rationale`：local + teacher-rationale global anchors（论文主方法）。

> 本项目使用 dense tensor 表示逻辑稀疏 mask，适合比较分类质量与 attention
> connectivity。它不会自动获得稀疏 kernel 的真实加速；论文中应把
> `sparsity/connectivity_ratio` 写成理论连接率，不能把普通 dense mask 的耗时宣称为
> sparse-kernel speedup。

## 1. 安装

建议 Python 3.10–3.13：

```bash
cd "/Users/starsfalling/Desktop/Project/Distilling-Teacher-Rationales-into-Global-Anchors-for-Sparse-Text-Classification"
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
cp .env.example .env
```

项目固定使用已完成兼容性验证的 Transformers 4.x。Student attention 本身使用
PyTorch `MultiheadAttention` 实现，不依赖 Hugging Face 模型内部 attention。

API key 只放在 `.env` 或 shell 环境变量中，不要写入代码或提交 Git。

## 2. 先做 20 条 teacher 标注

论文实验建议先从 SST-2 有标签训练集固定抽取 6000 条，并做分层划分：

```bash
rgsa-prepare-sst2 \
  --output-dir data/sst2-6k \
  --train-size 4800 --validation-size 600 --test-size 600 \
  --seed 42
```

生成的 `manifest.json` 记录每个 split 的大小、标签分布和 SHA-256；官方 SST-2
test 没有公开 gold label，因此这里从原始 train split 中划出独立测试集。

6000 条 teacher 标注建议使用 SiliconFlow Batch API（每个 split 一个 batch）。Batch
模型固定为其官方支持列表中的 `deepseek-ai/DeepSeek-V3`；V3.2 当前不支持 batch：

```bash
for split in train validation test; do
  rgsa-batch prepare \
    --input "data/sst2-6k/${split}.jsonl" \
    --output "data/sst2-6k/batch/${split}.requests.jsonl" \
    --split "$split"
done

export SILICONFLOW_API_KEY="..."
for split in train validation test; do
  rgsa-batch submit \
    --requests "data/sst2-6k/batch/${split}.requests.jsonl" \
    --job "data/sst2-6k/batch/${split}.job.json" \
    --description "RGSA SST-2 6k ${split}"
done
```

任务完成后先用 `rgsa-batch status --job ...` 查询，再用：

```bash
rgsa-batch collect \
  --job data/sst2-6k/batch/train.job.json \
  --source data/sst2-6k/train.jsonl \
  --output data/sst2-6k/train.teacher.jsonl \
  --split train
```

收集完成后检查 teacher label agreement、无效 anchor 比例和 rationale 数量：

```bash
rgsa-audit \
  --input data/sst2-6k/train.teacher.jsonl \
          data/sst2-6k/validation.teacher.jsonl \
          data/sst2-6k/test.teacher.jsonl \
  --output data/sst2-6k/teacher-audit.json
```

正式主实验不要向 teacher 提供 gold label，否则 validation/test 会发生标签泄漏。
`rgsa-batch prepare --provide-gold-label` 仅用于单独报告的 oracle-rationale ablation。

统一输出 JSONL schema：

```json
{
  "sample_id": "sst2:train:0",
  "text": "Although the acting was excellent, the plot was painfully boring.",
  "gold_label": "negative",
  "teacher_label": "negative",
  "short_rationale": "The negative plot judgment dominates the sentiment.",
  "rationale_tokens": ["Although", "excellent", "painfully boring"],
  "annotation_valid": true
}
```

Teacher 被要求返回“原文中的精确连续片段”，代码还会二次校验并丢弃无法定位的
幻觉 token。重复执行同一命令会跳过已写入的 `sample_id`，因此可以断点续跑。

### GPT / OpenAI

```bash
export OPENAI_API_KEY="..."
rgsa-annotate \
  --dataset sst2 --split train \
  --provider openai --model YOUR_OPENAI_MODEL \
  --output data/sst2/train.teacher.jsonl \
  --limit 20 --provide-gold-label
```

### DeepSeek

```bash
export DEEPSEEK_API_KEY="..."
rgsa-annotate \
  --dataset sst2 --split train \
  --provider deepseek --model deepseek-chat \
  --output data/sst2/train.teacher.jsonl \
  --limit 20 --provide-gold-label
```

### Qwen / DashScope

```bash
export DASHSCOPE_API_KEY="..."
rgsa-annotate \
  --dataset sst2 --split train \
  --provider qwen --model qwen-plus \
  --output data/sst2/train.teacher.jsonl \
  --limit 20 --provide-gold-label
```

### DeepSeek / 硅基流动 SiliconFlow

```bash
export SILICONFLOW_API_KEY="..."
rgsa-annotate \
  --dataset sst2 --split train \
  --provider siliconflow --model deepseek-ai/DeepSeek-V3.2 \
  --output data/sst2/train.teacher.jsonl \
  --limit 20 --provide-gold-label
```

该 provider 使用硅基流动官方 OpenAI-compatible endpoint
`https://api.siliconflow.cn/v1` 和 JSON mode。API key 只通过环境变量传入。

如果 Hugging Face 数据下载不稳定，也可以直接标注本地 JSONL。输入至少需要
`text` 和 `gold_label`：

```bash
rgsa-annotate \
  --input-file tests/fixtures/toy.validation.jsonl \
  --dataset-name toy --split validation \
  --provider siliconflow --model deepseek-ai/DeepSeek-V3.2 \
  --output data/toy/siliconflow.teacher.jsonl \
  --provide-gold-label
```

### 本地 Llama / Qwen

```bash
rgsa-annotate \
  --dataset sst2 --split train \
  --provider local-hf --model YOUR_HF_INSTRUCT_MODEL \
  --output data/sst2/train.teacher.jsonl \
  --limit 20 --provide-gold-label
```

也支持任意 OpenAI-compatible 服务：

```bash
export TEACHER_API_KEY="..."
rgsa-annotate \
  --dataset sst2 --split train \
  --provider compatible --model YOUR_MODEL \
  --base-url http://localhost:8000/v1 \
  --output data/sst2/train.teacher.jsonl
```

`--provide-gold-label` 推荐用于主实验：它固定正确类别，让实验只研究
“rationale anchors 是否有效”，避免把 teacher 分类错误混进注意力方法的效果。
去掉该参数即可研究 teacher 同时预测 label 的纯蒸馏设置。两种设置不要混报。

## 3. 创建 TF-IDF baseline

训练集：

```bash
rgsa-tfidf \
  --input data/sst2/train.teacher.jsonl \
  --output data/sst2/train.ready.jsonl \
  --top-k 4
```

验证集必须复用训练集拟合的词表，避免数据泄漏：

```bash
rgsa-tfidf \
  --input data/sst2/validation.teacher.jsonl \
  --fit-input data/sst2/train.teacher.jsonl \
  --output data/sst2/validation.ready.jsonl \
  --top-k 4
```

## 4. 训练五组模型

```bash
for strategy in full local random tfidf rationale; do
  rgsa-train \
    --train-file data/sst2/train.ready.jsonl \
    --validation-file data/sst2/validation.ready.jsonl \
    --strategy "$strategy" \
    --tokenizer-name distilbert/distilbert-base-uncased \
    --hidden-size 128 --num-layers 2 \
    --num-heads 4 --ffn-size 512 \
    --window-radius 4 \
    --anchor-budget 4 \
    --seed 42 \
    --output-dir "outputs/sst2/$strategy/seed-42"
done
```

所有方法必须使用相同 student architecture、tokenizer、窗口、anchor budget、
训练参数和 seed。Student 从随机初始化训练，因此默认使用 `3e-4` learning rate 和
10 epochs，而不是预训练 BERT 常见的微调超参数。
默认会为所有 attention 变体统一删除 `annotation_valid=false` 的记录，保证五组模型
看到完全相同的 train/validation/test 样本。
推荐至少运行 `13/42/87` 三个 seed。完整 SST-2 脚本：

```bash
TEACHER_PROVIDER=deepseek TEACHER_MODEL=deepseek-chat \
  bash scripts/run_sst2.sh
```

先廉价跑通：

```bash
LIMIT=100 TEACHER_PROVIDER=deepseek TEACHER_MODEL=deepseek-chat \
  bash scripts/run_sst2.sh
```

6000 条正式数据在 batch teacher 标注收集完成后运行：

```bash
bash scripts/run_sst2_6k.sh
```

该脚本使用 validation 选择最佳 checkpoint，最终另外报告从未用于调参的 test 指标。

TREC 数据准备：

```bash
rgsa-prepare-trec --output-dir data/trec --validation-size 500 --seed 42
```

通用五 seeds 实验：

```bash
DATA_DIR=data/trec OUTPUT_ROOT=outputs/trec \
  bash scripts/run_dataset.sh
```

SST-2 anchor-budget 和 window-radius 消融：

```bash
bash scripts/run_sst2_ablations.sh

rgsa-ablation-summary \
  --ablation-root outputs/sst2-6k-ablations \
  --main-root outputs/sst2-6k-preliminary \
  --output outputs/sst2-6k-ablations/summary.json
```

脚本新跑 anchor budget `1/2/3/5` 和 window radius `2/8`；默认设置
budget `4`、radius `4` 直接复用主实验的 `13/42/87` 三个 seed，避免重复训练。

## 5. 汇总结果和 Recovery Rate

```bash
rgsa-summarize \
  --input-root outputs/sst2 \
  --output outputs/sst2/results.csv
```

`results.csv` 包含 Accuracy、Macro-F1、连接率、稀疏率、eval runtime 和：

```text
Recovery Rate = (F1_RGSA - mean(F1_Local)) /
                (mean(F1_Full) - mean(F1_Local))
```

## 6. Attention pattern 可视化

```bash
rgsa-visualize \
  --input data/examples/sample.teacher.jsonl \
  --strategy rationale \
  --output outputs/figures/rationale-mask.png
```

热力图只是结构辅助分析，不应被表述为“attention 本身就是解释”。

## 7. Ablation 设置

- anchor budget：`1, 2, 3, 4, 5`；
- window radius：`2, 4, 8`（完整窗口宽度为 `2r+1`）；
- 数据集：先 SST-2，再 AG News / TREC；
- teacher label：gold-constrained 与 free prediction 分开比较；
- 无效 rationale：默认 `drop`，并报告无效率；也可用
  `--invalid-rationale keep|error` 做敏感性分析。

## 8. 测试

核心逻辑测试（未安装 PyTorch 时会自动跳过模型层测试）：

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

安装依赖后再运行：

```bash
pytest
rgsa-visualize --help
rgsa-train --help
```

支持的数据集定义在 `src/rgsa/data.py`。API 可能更新，正式实验时应在论文中记录
provider、完整 model ID、prompt version、生成日期、无效标注比例和人工抽检结果。

## 9. AG News 6K 与三数据集图表

AG News 使用官方 `train` split 抽取训练/验证集，使用官方 `test` split 抽取测试集，
并和 SST-2 保持相同实验规模：

| Split | Size | Source |
|---|---:|---|
| train | 4800 | official train |
| validation | 600 | official train |
| test | 600 | official test |

四个类别 `world / sports / business / sci/tech` 采用分层均衡抽样。准备数据和 teacher
batch 请求文件：

```bash
cd "/Users/starsfalling/Desktop/Project/Teacher-Rationale-Sparse-Attention"
source .venv/bin/activate
PROJECT_ROOT="$PWD" scripts/prepare_agnews_6k.sh
```

提交 SiliconFlow batch 前，先在本机 shell 设置 key；不要把 key 写入代码：

```bash
export SILICONFLOW_API_KEY="..."
PROJECT_ROOT="$PWD" scripts/submit_agnews_batches.sh
```

batch 完成后：

```bash
PROJECT_ROOT="$PWD" scripts/status_agnews_batches.sh
PROJECT_ROOT="$PWD" scripts/collect_agnews_batches.sh
PROJECT_ROOT="$PWD" scripts/run_agnews_6k.sh
```

三数据集结果都完成后，生成论文图表与 attention pattern 热力图：

```bash
PROJECT_ROOT="$PWD" scripts/generate_paper_figures.sh
```

输出位置：

- `outputs/paper-figures/cross_dataset_summary.csv`
- `outputs/paper-figures/accuracy_by_dataset.png`
- `outputs/paper-figures/macro_f1_by_dataset.png`
- `outputs/paper-figures/sparsity_by_dataset.png`
- `outputs/paper-figures/rgsa_f1_gain.png`
- `outputs/paper-figures/heatmap_<dataset>_<strategy>.png`
