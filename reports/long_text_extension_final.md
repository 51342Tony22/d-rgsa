# RQ4–RQ5 Medium/Long-Text and Cost Extension: Final Experiment Report

## Scope and status

This report is the final record for the medium/long-text and measured-cost
extension. It preserves the original RQ1–RQ3 experiments and adds RQ4–RQ5. All
main quality results use the same five seeds (`13, 42, 87, 101, 123`). The
collection is complete: 120 main quality runs (four datasets, six methods, five
seeds), 20 selector-architecture ablation runs, 25 controlled-length cost
measurements, and eight real-input end-to-end cost measurements.

The artifact label `Direct-RGSA` denotes RGSA with teacher-provided anchors at
prediction time. The paper may call this method simply **RGSA** as long as it
clearly contrasts RGSA with teacher-free **D-RGSA**.

## Research questions

1. **RQ1:** Do distilled anchors retain the classification gains of RGSA?
2. **RQ2:** Does semantic anchor position matter beyond the number of sparse
   connections?
3. **RQ3:** Are the conclusions stable across teacher models?
4. **RQ4:** Under real medium- and long-text input distributions, how does
   D-RGSA trade classification quality against practical inference cost?
5. **RQ5:** As sequence length grows, how does D-RGSA trade classification
   quality against measured latency, memory, and attention-score computation?

RQ1–RQ3 retain the conclusions and limitations of the frozen paper. AG News
and IMDb real-input measurements answer RQ4; the controlled-length sweep
answers RQ5.

## Data and annotation protocol

IMDb was sampled once with seed 42 into 4,800 training, 600 validation, and 600
test examples, with a 50/50 positive/negative balance in every split. After the
same teacher-annotation validity filter was applied to all six methods, the
shared classifier set contained 4,796/600/600 examples. No method received a
different subset.

DeepSeek produced 5,998 parsed annotations across the three splits. The anchor
validity rate was 99.97%, teacher/gold label agreement was 94.51%, and the mean
number of rationale phrases was 4.76. Two invalid annotations were excluded by
the shared filter.

Natural DistilBERT-tokenizer lengths establish the intended dataset roles:

| Dataset/test split | Mean | Median | P90 | >128 | >256 | >512 |
|---|---:|---:|---:|---:|---:|---:|
| SST-2 | 13.57 | 10.0 | 28.1 | 0.00% | 0.00% | 0.00% |
| TREC | 10.32 | 10.0 | 14.0 | 0.00% | 0.00% | 0.00% |
| AG News | 53.69 | 51.0 | 71.0 | 0.83% | 0.00% | 0.00% |
| IMDb | 305.10 | 232.5 | 570.4 | 88.67% | 43.67% | 12.83% |

AG News is therefore a medium-text bridge, not a genuinely long-text dataset.
IMDb is the long-text quality benchmark. AG News follows the frozen protocol
(maximum length 128 and local radius 4), while IMDb uses maximum length 512 and
local radius 16. The separate controlled cost sweep uses radius 16 and fixed
lengths through 2,048. The 12.83% of IMDb test examples longer than 512 are
truncated in the quality and real-input experiments, which must be stated as a
limitation.

## Main quality results

All variants use the same lightweight Transformer, four global anchors, ten
training epochs, validation-selected checkpoints, and five seeds. AG News uses
radius 4; IMDb uses radius 16. Values are test Macro-F1 mean ± sample standard
deviation in percentage points.

| Method | AG News | IMDb | IMDb Accuracy | IMDb Sparsity |
|---|---:|---:|---:|---:|
| Full | 86.79 ± 0.86 | **84.12 ± 0.85** | 84.13 | 0.00 |
| Local | 69.28 ± 0.93 | 67.21 ± 0.66 | 67.30 | 90.57 |
| Random | 82.78 ± 1.00 | 75.93 ± 1.82 | 75.97 | 88.50 |
| TF-IDF | 81.69 ± 1.43 | 71.58 ± 1.28 | 71.67 | 88.50 |
| RGSA (teacher anchors) | 84.14 ± 1.59 | **81.69 ± 0.81** | 81.70 | 88.51 |
| D-RGSA (BiGRU) | 83.37 ± 0.72 | **78.86 ± 1.45** | 79.00 | 88.46 |

On IMDb, RGSA trails Full by 2.43 points but exceeds Random by 5.76 points
(paired 95% CI [3.06, 8.46], p=0.0041). BiGRU D-RGSA exceeds Random by 2.93
points (CI [1.67, 4.19], p=0.0030) and Local by 11.65 points (CI [9.60,
13.70], p<0.001). These comparisons support the semantic-anchor mechanism.

However, BiGRU D-RGSA trails RGSA by 2.82 points (CI [-4.75, -0.90],
p=0.0151) and Full by 5.26 points (CI [-6.63, -3.89], p<0.001). It retains
93.75% of Full Macro-F1. The preregistered target—within two points of Full or
at least 98% retention—is **not met** on IMDb. RGSA itself retains 97.11%, also
slightly below that target. The long-text result must therefore be framed as
partial recovery of the information lost by local sparsity, not as matching
Full Attention.

Across all four datasets, the unweighted Macro-F1 means are Full 82.73, RGSA
82.63, D-RGSA 81.44, Random 79.79, TF-IDF 78.63, and Local 74.11. This aggregate
is useful as context, but it must not hide the larger IMDb gap.

## Selector-architecture ablation

The MLP selector was evaluated under the same dataset-specific local radius as
the corresponding BiGRU model. It reduces sequential selector overhead without
significantly changing downstream five-seed quality:

| Dataset | Selector | Macro-F1 | Accuracy | Sparsity | Test anchor-F1 |
|---|---|---:|---:|---:|---:|
| AG News | BiGRU | 83.37 ± 0.72 | 83.32 | 74.16 | 40.03 |
| AG News | MLP | **84.03 ± 0.61** | 84.03 | 74.15 | 37.06 |
| IMDb | BiGRU | 78.86 ± 1.45 | 79.00 | 88.46 | 28.70 |
| IMDb | MLP | **79.49 ± 1.50** | 79.53 | 88.44 | 16.75 |

The paired MLP−BiGRU difference is +0.66 points on AG News (95% CI
[-0.89, 2.21], p=0.304) and +0.63 points on IMDb (CI [-2.32, 3.58],
p=0.587). Neither difference is statistically significant. The defensible
conclusion is comparable downstream quality with a more parallel selector, not
significantly higher MLP quality. The disconnect between token anchor-F1 and
classification quality also shows that exact teacher-position imitation is
stricter than preserving useful global connectivity.

## Controlled sequence-length cost

The structured sparse kernel computes local windows, global-key access, and
global-query access without materializing an n-by-n score tensor. Numerical
tests match the dense logical-mask implementation within 2e-5 absolute and
relative tolerance.

Benchmark setting: Apple M5 MPS, float32, batch 8, two student layers, hidden
size 128, four heads, FFN size 512, radius 16, four anchors, 10 warm-up and 50
measured iterations. Latency is median end-to-end milliseconds. BiGRU and MLP
rows include selector, top-k, sparse student, and classifier; `student only` is
diagnostic.

| Length | Full | Local | D-RGSA BiGRU | D-RGSA MLP | D-RGSA student only |
|---:|---:|---:|---:|---:|---:|
| 128 | 1.31 | 6.16 | 18.77 | 7.88 | 7.36 |
| 256 | 2.90 | 9.24 | 33.24 | 11.52 | 10.69 |
| 512 | 10.04 | 17.03 | 65.35 | 19.55 | 19.10 |
| 1,024 | 33.92 | 32.80 | 126.69 | 36.61 | 36.22 |
| 2,048 | 123.67 | 64.31 | 256.67 | 72.45 | 70.36 |

At length 2,048, Local is 1.92× faster than Full, MLP D-RGSA is 1.71× faster,
and the sparse student alone is 1.76× faster. BiGRU D-RGSA is only 0.48× as
fast as Full because sequential recurrent selection dominates the saved
attention work. Optimized Full remains fastest through length 512; the MLP
variant is still about 7% slower at 1,024 and crosses over only by 2,048.

The D-RGSA structured kernel removes 92.0%, 96.0%, and 98.0% of attention-score
elements at lengths 512, 1,024, and 2,048. Observed MPS driver-memory reduction
is only 5.9% at 512 because fixed model/runtime memory dominates, but grows to
20.1% at 1,024 and 49.4% at 2,048. Logical sparsity is therefore retained as a
structural metric, while speed and memory claims are based only on measured
structured execution.

## Real-input end-to-end cost

The real-input benchmark loads the trained seed-42 checkpoints and uses the
same teacher-valid test subset as the quality evaluation. Examples are sorted
by natural tokenized length and dynamically padded in batches of eight. Every
observed batch shape is warmed before five measured passes, preventing the MPS
first-execution cost of a new tensor shape from inflating the tail latency.
Latency excludes host tokenization but includes host-to-device transfer,
selector inference, top-k anchor selection, sparse student inference, and the
classifier. AG News contains 591 shared valid examples and uses radius 4;
IMDb contains 600 examples and uses radius 16 with truncation at 512 tokens.

| Dataset | Method | Median (ms) | P95 (ms) | Throughput (samples/s) | Score reduction |
|---|---|---:|---:|---:|---:|
| AG News | Full | **1.45** | **2.62** | **5,067.68** | 0.00% |
| AG News | Local | 4.42 | 10.63 | 1,503.56 | 85.10% |
| AG News | D-RGSA (BiGRU) | 15.85 | 26.92 | 482.73 | 71.86% |
| AG News | D-RGSA (MLP) | 8.66 | 18.96 | 811.59 | 71.86% |
| IMDb | Full | **2.76** | **5.38** | **2,574.43** | 0.00% |
| IMDb | Local | 11.96 | 20.09 | 639.97 | 90.33% |
| IMDb | D-RGSA (BiGRU) | 40.03 | 67.29 | 182.15 | 87.98% |
| IMDb | D-RGSA (MLP) | 18.58 | 25.94 | 431.68 | 87.98% |

Full Attention is the fastest method on both real datasets. Relative to the
BiGRU selector, the MLP reduces D-RGSA median latency by 1.83x on AG News and
2.15x on IMDb, but it remains 5.96x and 6.72x slower than Full, respectively.
This negative result is important: reducing the number of score elements does
not guarantee lower wall-clock latency when structured sparse kernels and
selection overhead are included.

Memory metrics must also be kept separate. On IMDb, sampled active tensor
allocation falls from 62.59 MB for Full to 31.86 MB for MLP D-RGSA, whereas MPS
driver memory rises from 1,122.78 to 3,164.78 MB and process RSS rises from
774.98 to 894.17 MB. Dynamic sparse shapes cause graph/runtime caching on this
backend. On AG News, active allocation falls from 44.49 to 31.48 MB and driver
memory falls slightly from 1,102.05 to 1,060.03 MB, but process RSS rises from
693.58 to 791.56 MB. The real-input evidence therefore does not support a
general claim of lower total memory on Apple MPS.

## Answers to RQ4 and RQ5

**RQ4 does not support practical cost reduction under the tested real-input
setting.** Semantic anchors recover much of the severe Local-Attention quality
loss, but neither RGSA nor D-RGSA meets the declared IMDb closeness target
relative to Full. Full also has the lowest median and P95 latency on both AG
News and IMDb. Active tensor allocation can be lower, but driver memory and RSS
do not establish a general total-memory saving. This is a negative deployment
result for the current sequence lengths, sparse implementation, and MPS
backend, rather than evidence against the semantic-anchor mechanism.

**RQ5 is supported as a trade-off result, not an unconditional speedup claim.**
In the controlled fixed-shape sweep, the proportion of eliminated attention
scores and measured driver-memory savings grow strongly with sequence length.
Whether this becomes latency speedup depends on selector architecture: BiGRU
overhead dominates at every tested length, while the MLP variant crosses
optimized Full at 2,048. Thus global-anchor semantics and selector efficiency
are separate design axes. Future work should train and evaluate the classifier
at 1,024–2,048 tokens and test parallel selectors on a discrete GPU before
claiming general production speedups.

## Recommended paper framing

1. Keep SST-2 and TREC as the principal capability tests for RQ1–RQ3.
2. Present AG News as a medium-length bridge and IMDb as the long-text quality
   stress test; do not call AG News long text.
3. Retain the sparsity table. It is central structural evidence, but accompany
   it with measured latency and memory.
4. State that RGSA is the teacher-available reference and D-RGSA is the
   teacher-free deployment method. The IMDb gap quantifies the cost of
   distilling semantic connectivity.
5. Use the MLP result as a selector-architecture ablation. Say “comparable
   five-seed quality with substantially lower selector overhead,” not
   “significantly higher quality.”
6. Avoid the claims “D-RGSA matches Full on long text” and “D-RGSA reduces
   practical cost on AG News or IMDb.” A supportable claim is: “Semantic global
   anchors outperform equal-budget sparse baselines on IMDb; in a controlled
   fixed-length benchmark, latency gains require a parallel selector and
   sequences longer than those used for quality training.”

## Final artifacts

- Main raw/summary: `outputs/long-text-extension/tables/quality_raw.csv` and
  `quality_summary.csv`
- Selector ablation: `selector_ablation_raw.csv` and
  `selector_ablation_summary.csv`
- Paired statistics: `paired_comparisons.csv`
- Controlled cost: `cost_raw.csv` and `cost_comparison_vs_full.csv`
- Real-input cost: `real_input_cost_raw.csv` and
  `real_input_cost_comparison_vs_full.csv`
- Dataset lengths: `dataset_stats.csv`
- Completion status: `collection_status.json`
- Altair figures: `long_text_macro_f1`, `cost_scaling`, `score_reduction`,
  `real_input_cost`, `performance_cost_pareto`, and
  `selector_architecture_medium_long` in PNG, SVG, and HTML formats under
  `outputs/long-text-extension/figures/`
