# Conference-Paper Supplement Experiments

## Scope

This supplement now covers the expanded conference-paper evidence:

1. three datasets: SST-2 6K, TREC, and AG News 6K;
2. five-seed main experiments with matched student architecture;
3. SST-2 anchor-budget and local-window-radius ablations;
4. cross-dataset result tables and paper-style figures.

All attention variants use the same randomly initialized lightweight student:
2 Transformer encoder layers, hidden size 128, 4 heads, FFN size 512, maximum
length 128, batch size 16, learning rate `3e-4`, and 10 epochs. Only attention
connectivity changes. DistilBERT is used as a tokenizer only; no BERT weights
are loaded into the student.

## Teacher annotation

The teacher is `deepseek-ai/DeepSeek-V3`, accessed through the SiliconFlow
Batch API. Gold labels are not included in teacher prompts.

| Dataset | Parsed usable records | Teacher/gold agreement | Anchor validity |
|---|---:|---:|---:|
| SST-2 6K | 5,935 | 91.13% | 99.16% |
| TREC | 5,902 | 81.80% | 99.98% |
| AG News 6K | 6,000 | 82.78% | 98.85% |

For TREC, a first generic prompt achieved only about 63% teacher/gold
agreement. Adding definitions of the six answer-type labels raised agreement
to 81.80%. The generic annotations are archived but are not used in the main
experiment.

For AG News, the guided prompt uses explicit definitions for `world`, `sports`,
`business`, and `sci/tech`. Teacher agreement is lower than SST-2 and close to
TREC because `business` and `sci/tech` news are often semantically adjacent.
This makes AG News a useful stress test for rationale-guided anchors under
noisier teacher labels.

## Five-seed main results

Test results are mean ± sample standard deviation over seeds
`13/42/87/101/123`.

### SST-2 6K

| Attention strategy | Accuracy (%) | Macro-F1 (%) | Sparsity (%) |
|---|---:|---:|---:|
| Full | 79.97 ± 1.30 | 79.64 ± 1.39 | 0.00 |
| Local | 77.39 ± 1.86 | 77.05 ± 1.94 | 63.93 |
| Random global | 79.86 ± 1.17 | 79.56 ± 1.22 | 46.57 |
| TF-IDF global | 80.20 ± 0.57 | 79.91 ± 0.64 | 47.08 |
| **RGSA** | **81.43 ± 0.96** | **81.24 ± 0.97** | **46.39** |

| Comparison | RGSA gain (F1 points) | Paired t-test p | Exact Wilcoxon p |
|---|---:|---:|---:|
| vs Full | +1.60 | 0.0218 | 0.0625 |
| vs Local | +4.19 | 0.0056 | 0.0625 |
| vs Random | +1.68 | 0.0434 | 0.0625 |
| vs TF-IDF | +1.33 | 0.0222 | 0.0625 |

RGSA has nearly the same connectivity budget as Random and TF-IDF, while
improving Macro-F1 by 1.68 and 1.33 points, respectively. It is higher than
every baseline on all five matched seeds.

### TREC

The official 500-example TREC test set is used. After uniform validity
filtering, every variant sees 4,907 train, 496 validation, and 498 test
examples.

| Attention strategy | Accuracy (%) | Macro-F1 (%) | Sparsity (%) |
|---|---:|---:|---:|
| Full | 80.80 ± 2.60 | 80.39 ± 1.29 | 0.00 |
| Local | 82.93 ± 1.35 | 82.88 ± 2.25 | 36.64 |
| Random global | 81.24 ± 1.55 | 80.90 ± 2.05 | 18.41 |
| TF-IDF global | 82.17 ± 2.50 | 81.35 ± 1.56 | 23.46 |
| **RGSA** | **85.10 ± 1.46** | **83.46 ± 2.30** | **21.11** |

| Comparison | RGSA gain (F1 points) | Paired t-test p | Exact Wilcoxon p |
|---|---:|---:|---:|
| vs Full | +3.07 | 0.0114 | 0.0625 |
| vs Local | +0.57 | 0.6355 | 0.6250 |
| vs Random | +2.56 | 0.0036 | 0.0625 |
| vs TF-IDF | +2.11 | 0.1335 | 0.3125 |

RGSA again has the highest mean result, but the advantage over Local and
TF-IDF is not statistically significant. TREC questions are short, so a
radius-4 local window already preserves much of each sequence. This dataset
supports cross-task robustness of the mean improvement, not a claim of
universal statistically significant superiority.

### AG News 6K

AG News uses a stratified 6K subset to match the SST-2 experimental scale:
4,800 examples from the official train split for training, 600 from the
official train split for validation, and 600 from the official test split for
testing. The four labels are perfectly balanced in all splits. After uniform
validity filtering, every variant sees 4,747 train, 593 validation, and 591
test examples.

| Attention strategy | Accuracy (%) | Macro-F1 (%) | Sparsity (%) |
|---|---:|---:|---:|
| Full | 86.70 ± 0.87 | 86.79 ± 0.86 | 0.00 |
| Local | 69.37 ± 0.91 | 69.28 ± 0.93 | 85.48 |
| Random global | 82.77 ± 0.95 | 82.78 ± 1.00 | 74.54 |
| TF-IDF global | 81.69 ± 1.44 | 81.69 ± 1.43 | 74.44 |
| **RGSA** | **84.13 ± 1.58** | **84.14 ± 1.59** | **74.24** |

| Comparison | RGSA gain (F1 points) | Paired t-test p | Exact Wilcoxon p |
|---|---:|---:|---:|
| vs Full | -2.65 | 0.0418 | 0.0625 |
| vs Local | +14.87 | <0.0001 | 0.0625 |
| vs Random | +1.36 | 0.0404 | 0.0625 |
| vs TF-IDF | +2.46 | 0.0022 | 0.0625 |

AG News gives a complementary picture. Full attention remains the upper-bound
baseline, but Local collapses because the news topic signal often depends on
non-local entities and event terms. Adding global anchors recovers most of the
lost performance. Under a comparable sparse connectivity budget, RGSA improves
over Random by 1.36 F1 points and over TF-IDF by 2.46 F1 points. The result
therefore supports the mechanism claim that teacher-selected anchors are more
informative than generic sparse anchors, while also showing the limitation that
noisy teacher rationales do not fully replace full attention.

With five non-zero pairs, 0.0625 is the smallest attainable two-sided exact
Wilcoxon p-value. We therefore report both paired t-tests and Wilcoxon tests
and describe the evidence as consistent but small-sample.

## SST-2 ablations

The ablations report mean ± sample standard deviation over seeds `13/42/87`.
The scripts are resumable and reuse the matching k=4/r=4 main runs.

### Anchor budget

| Rationale anchor budget | Macro-F1 (%) | Sparsity (%) |
|---:|---:|---:|
| 1 | 79.74 ± 1.27 | 59.22 |
| 2 | 80.84 ± 0.70 | 54.73 |
| 3 | 81.30 ± 0.95 | 50.46 |
| 4 | 81.54 ± 1.24 | 46.39 |
| **5** | **81.60 ± 0.44** | 42.56 |

Moving from one to two anchors yields the largest mean quality gain (+1.11 F1
points). Improvements then diminish: k=4 and k=5 differ by only 0.07 points,
while k=5 uses more connections. The practical elbow is therefore around
k=3–4, and “more anchors are always better” is not supported.

### Local window radius

| Radius | Local F1 | RGSA F1 | RGSA gain | Local sparsity | RGSA sparsity |
|---:|---:|---:|---:|---:|---:|
| 2 | 70.57 ± 1.62 | **81.85 ± 0.34** | **+11.28** | 78.19 | 54.76 |
| 4 | 76.53 ± 2.43 | **81.54 ± 1.24** | **+5.01** | 63.93 | 46.39 |
| 8 | 79.43 ± 0.33 | **81.72 ± 1.62** | **+2.29** | 42.50 | 32.41 |

The mechanism prediction is supported: rationale anchors help most when the
local window is smallest. At radius 2, RGSA recovers 11.28 F1 points over Local
while still removing 54.76% of logical attention connections. Larger windows
improve Local because more context is directly reachable, reducing the amount
of missing information that global anchors need to restore.

## Claims supported by the expanded evidence

- Teacher rationale tokens can be used as structural global-attention anchors,
  rather than only as output supervision.
- RGSA achieves the highest mean Macro-F1 on SST-2 and TREC, and the highest
  mean Macro-F1 among sparse methods on AG News.
- On SST-2 and AG News, RGSA outperforms Random and TF-IDF under closely
  matched logical connectivity budgets.
- On AG News, RGSA recovers most of the large Local-attention drop while
  keeping roughly 74% logical sparsity.
- The implementation establishes logical connectivity reduction, not
  sparse-kernel wall-clock acceleration, because masks are stored in dense
  tensors.

## Artifacts

- `data/sst2-6k/teacher-audit.json`
- `data/trec/teacher-audit.json`
- `data/agnews-6k/teacher-audit.json`
- `outputs/sst2-6k-preliminary/results.csv`
- `outputs/sst2-6k-preliminary/results.summary.json`
- `outputs/trec/results.csv`
- `outputs/trec/results.summary.json`
- `outputs/agnews-6k/results.csv`
- `outputs/agnews-6k/results.summary.json`
- `outputs/sst2-6k-ablations/summary.json`
- `outputs/paper-figures/cross_dataset_summary.csv`
- `outputs/paper-figures/accuracy_by_dataset.png`
- `outputs/paper-figures/macro_f1_by_dataset.png`
- `outputs/paper-figures/sparsity_by_dataset.png`
- `outputs/paper-figures/rgsa_f1_gain.png`
- `outputs/paper-figures/heatmap_<dataset>_<strategy>.png`
