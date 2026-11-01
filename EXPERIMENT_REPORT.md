# SST-2 6K Main Experiment Report

## Experimental setup

- Source: labeled `stanfordnlp/sst2` train split from Hugging Face.
- Fixed stratified sample: 6,000 examples, seed 42.
- Original split: train 4,800 / validation 600 / test 600.
- Main teacher: `deepseek-ai/DeepSeek-V3` through SiliconFlow Batch API.
- Main teacher setting: free label prediction; no gold label was included in the prompt.
- Parsed teacher records: 5,985/6,000.
- Exact-substring anchor-valid records used by every model: train 4,750 /
  validation 596 / test 589.
- Student: randomly initialized 2-layer Small Transformer, hidden size 128,
  4 heads, FFN size 512, maximum length 128.
- Training: 10 epochs, learning rate `3e-4`, batch size 16.
- Seeds: 13, 42, 87, 101, 123.
- All variants use the same data, architecture, initialization seed and
  optimization settings. Only attention connectivity changes.

Teacher quality audit:

- Teacher/gold label agreement: 91.13%.
- Exact-substring anchor validity among parsed outputs: 99.16%.
- Mean rationale phrases per example: 1.57.
- The earlier gold-constrained batch is retained only as an oracle ablation and
  is not used in the main results below.

## Main test results (mean ± sample standard deviation, 5 seeds)

| Attention strategy | Accuracy (%) | Macro-F1 (%) | Sparsity (%) |
|---|---:|---:|---:|
| Full | 79.97 ± 1.30 | 79.64 ± 1.39 | 0.00 |
| Local | 77.39 ± 1.86 | 77.05 ± 1.94 | 63.93 |
| Random global | 79.86 ± 1.17 | 79.56 ± 1.22 | 46.57 |
| TF-IDF global | 80.20 ± 0.57 | 79.91 ± 0.64 | 47.08 |
| **Teacher-rationale global (RGSA)** | **81.43 ± 0.96** | **81.24 ± 0.97** | **46.39** |

Per-seed test Macro-F1:

| Seed | Full | Local | Random | TF-IDF | RGSA |
|---:|---:|---:|---:|---:|---:|
| 13 | 78.46 | 74.47 | 77.71 | 78.99 | **80.12** |
| 42 | 81.74 | 79.21 | 78.96 | 80.29 | **82.39** |
| 87 | 79.60 | 75.90 | 80.47 | 80.16 | **82.11** |
| 101 | 78.34 | 77.05 | 80.61 | 79.54 | **80.96** |
| 123 | 80.06 | 78.60 | 80.06 | 80.59 | **80.63** |

## Derived results

- RGSA versus Local: +4.19 Macro-F1 points.
- RGSA versus Random: +1.68 Macro-F1 points.
- RGSA versus TF-IDF: +1.33 Macro-F1 points.
- RGSA versus Full: +1.60 Macro-F1 points.
- Recovery rate from five-seed means:

  `(F1_RGSA - F1_Local) / (F1_Full - F1_Local) = 161.8%`

The recovery rate exceeds 100% because the mean RGSA score is above the mean
Full score in this experiment. It should be described as complete recovery plus
an additional gain, not as a guaranteed general property.

Exploratory paired tests over five matched seeds:

| Comparison | Mean RGSA gain | Paired t-test p | Wilcoxon p |
|---|---:|---:|---:|
| RGSA vs Local | +4.19 | 0.0056 | 0.0625 |
| RGSA vs Random | +1.68 | 0.0434 | 0.0625 |
| RGSA vs TF-IDF | +1.33 | 0.0222 | 0.0625 |
| RGSA vs Full | +1.60 | 0.0218 | 0.0625 |

RGSA is higher on all five paired seeds for every comparison. The parametric
paired t-tests pass 0.05, while the exact two-sided Wilcoxon tests do not: with
only five non-zero pairs, their smallest attainable p-value is 0.0625. The paper
should report both and describe the evidence as consistent but still
small-sample, rather than relying on only the more favorable test.

## Interpretation

The experiment supports the core hypothesis. Local attention removes 63.93% of
logical connections but loses 2.59 Macro-F1 points relative to Full attention.
Adding random global anchors does not reliably recover that loss. TF-IDF anchors
are stronger, but teacher-rationale anchors achieve the best mean quality while
using a comparable connection budget. This suggests that task-aware semantic
anchor selection matters, rather than the gain coming only from adding global
tokens.

## Important limitations and next experiments

1. The attention graph is represented by a dense PyTorch mask. Report logical
   connectivity and sparsity, not sparse-kernel wall-clock speedup.
2. SST-2 sentences are short. Add AG News, TREC, or a longer-text dataset before
   making a broad efficiency claim.
3. Run window-radius and anchor-budget ablations.
4. Report the oracle-rationale batch separately to measure the cost of teacher
   label errors, without mixing it into the main result.
5. Repeat with more seeds if formal significance testing is central to the
   paper.

## Artifacts

- Split manifest: `data/sst2-6k/manifest.json`
- Teacher audit: `data/sst2-6k/teacher-audit.json`
- Per-run CSV: `outputs/sst2-6k-preliminary/results.csv`
- Aggregate JSON: `outputs/sst2-6k-preliminary/results.summary.json`
- Individual checkpoints and metrics:
  `outputs/sst2-6k-preliminary/<strategy>/seed-<seed>/`
