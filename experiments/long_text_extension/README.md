# Long-text and cost extension

This directory adds RQ4 and RQ5 without modifying the frozen seven-page paper or
the original SST-2, TREC, and AG News experiment outputs.

## Fixed protocol

- Main seeds: `13, 42, 87, 101, 123`.
- Comparable raw subset sizes: 4,800 train, 600 validation, and 600 test.
- Part I (method capability): SST-2 and TREC at maximum length 128.
- Part II (performance and cost): AG News as the medium-text bridge and IMDb as
  the long-text benchmark.
- Main IMDb length: 512; controlled cost sweep: 128, 256, 512, 1024, and 2048.
- Sparse setting for the long-text study: local radius 16 and four global
  anchors.
- D-RGSA end-to-end timing includes the BiGRU selector, top-k selection, sparse
  student, and classifier head.
- Two diagnostic cost curves are also retained: `D-RGSA student only` isolates
  the sparse classifier, while `D-RGSA (MLP)` tests whether a parallel selector
  removes BiGRU sequential overhead. They do not replace the main BiGRU result.
- `process_peak_rss_mb` is operating-system process RSS, not accelerator-only
  memory. Hardware, device, dtype, batch size, warm-up, and repeat counts are
  stored in every raw benchmark record.

The success criteria are declared before observing IMDb results:

1. D-RGSA should outperform controlled sparse baselines on Macro-F1.
2. D-RGSA should remain close to Direct-RGSA.
3. The Full-Attention gap should be reported rather than hidden; retaining at
   least 98% of Full Macro-F1 or remaining within two percentage points is the
   target, not a guaranteed outcome.
4. Actual latency, throughput, and process memory must be measured with the
   structured sparse kernel. Logical sparsity alone is not called speedup.
5. Cost advantages should become clearer as sequence length increases.

## Completion

The experiment is complete. The final machine-readable status records 120 main
quality runs, 10 IMDb selector-architecture ablation runs, 25 cost records, and
29 paired comparisons. See `reports/long_text_extension_final.md` for the
evidence-backed RQ4–RQ5 conclusions. CSV files are under
`outputs/long-text-extension/tables/`, and the final Altair PNG/SVG/HTML figures
are under `outputs/long-text-extension/figures/`.
