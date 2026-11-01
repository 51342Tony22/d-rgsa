# RQ4–RQ5 Long-Text Extension: Interim Experiment Record

This file records results produced before IMDb teacher annotation and five-seed
quality training are complete. It must not be cited as the final experiment
report.

## Fixed data protocol

IMDb was sampled with seed 42 into 4,800 training, 600 validation, and 600 test
examples, balanced 50/50 between positive and negative labels. This matches the
approximately 6K scale of SST-2, TREC, and AG News.

Natural DistilBERT-tokenizer lengths confirm the intended division:

| Dataset/test split | Mean | Median | P90 | >128 | >256 | >512 |
|---|---:|---:|---:|---:|---:|---:|
| SST-2 | 13.57 | 10.0 | 28.1 | 0.0% | 0.0% | 0.0% |
| TREC | 10.32 | 10.0 | 14.0 | 0.0% | 0.0% | 0.0% |
| AG News | 53.69 | 51.0 | 71.0 | 0.83% | 0.0% | 0.0% |
| IMDb | 305.10 | 232.5 | 570.4 | 88.67% | 43.67% | 12.83% |

AG News is therefore a medium-text bridge, not a genuinely long-text dataset.
IMDb provides the long-text quality benchmark.

## Kernel validation

The original project used a dense tensor plus a logical sparse mask. The new
structured implementation computes local windows, global-key access, and
global-query access without constructing an n-by-n graph. Unit tests compare
its logits against the original dense-mask graph using identical weights and
pass at 2e-5 absolute and relative tolerance.

## Controlled sequence-length cost benchmark

Hardware: Apple M5 GPU through PyTorch MPS. Float32, batch size 8, two student
layers, hidden size 128, four heads, FFN size 512, local radius 16, four global
anchors, 10 warm-up iterations, and 50 measured iterations per point.

Median end-to-end latency (ms):

| Length | Full | Local | D-RGSA BiGRU | D-RGSA MLP | D-RGSA student only |
|---:|---:|---:|---:|---:|---:|
| 128 | 1.31 | 6.16 | 18.77 | 7.88 | 7.36 |
| 256 | 2.90 | 9.24 | 33.24 | 11.52 | 10.69 |
| 512 | 10.04 | 17.03 | 65.35 | 19.55 | 19.10 |
| 1024 | 33.92 | 32.80 | 126.69 | 36.61 | 36.22 |
| 2048 | 123.67 | 64.31 | 256.67 | 72.45 | 70.36 |

At length 2048, the MLP end-to-end variant is 1.71x faster than optimized Full,
the sparse student alone is 1.76x faster, and Local is 1.92x faster. The
structured D-RGSA kernel removes about 98.0% of score elements and reduces
observed MPS driver memory by about 49.4%. The optimized Full kernel remains
faster through length 512; the MLP variant is approximately tied but still 7%
slower at 1024.

The main BiGRU selector does **not** provide end-to-end latency savings in this
implementation: its sequential recurrent computation dominates sparse-student
savings. This is a real boundary result, not to be hidden. The MLP curve is a
diagnostic deployment variant already motivated by the paper's selector
architecture ablation; it does not replace the BiGRU quality result unless the
IMDb quality experiment independently supports that choice.

## Remaining work

1. Submit the prepared 4,800/600/600 DeepSeek batch requests.
2. Audit exact-substring alignment and define the common usable sample set.
3. Train Full, Local, Random, TF-IDF, Direct-RGSA, and D-RGSA with all five
   seeds at maximum length 512.
4. Optionally train the MLP-selector D-RGSA deployment variant if its long-text
   quality/cost trade-off is to be reported.
5. Rebuild all CSV files and Altair charts, then assess RQ1–RQ5 without changing
   the predeclared success criteria.
