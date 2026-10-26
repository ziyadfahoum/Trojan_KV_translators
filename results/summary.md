| condition | ARC-C | MMLU-Pro | CoQA F1 | RULER | mean |
|---|---|---|---|---|---|
| native_small | 0.534 | 0.249 | 0.379 | 0.973 | 0.534 |
| native_large | 0.683 | 0.346 | 0.611 | 0.997 | 0.659 |
| closed_headmix_L2S | 0.411 | 0.244 | 0.334 | 0.723 | 0.428 |
| closed_headwise_L2S | 0.493 | 0.133 | 0.258 | 0.787 | 0.418 |
| kvlingo_headmix_L2S | 0.717 | 0.282 | 0.424 | 0.983 | 0.601 |
| kvlingo_headwise_L2S | 0.717 | 0.278 | 0.360 | 0.987 | 0.585 |
| kvlingo_mlp_L2S | 0.704 | 0.289 | 0.436 | 0.957 | 0.596 |
| closed_headmix_S2L | 0.448 | 0.169 | 0.139 | 0.870 | 0.406 |
| closed_headwise_S2L | 0.291 | 0.167 | 0.071 | 0.750 | 0.320 |
| kvlingo_headmix_S2L | 0.544 | 0.248 | 0.440 | 0.977 | 0.552 |
| kvlingo_headwise_S2L | 0.536 | 0.216 | 0.353 | 0.993 | 0.525 |
| kvlingo_mlp_S2L | 0.547 | 0.249 | 0.377 | 0.970 | 0.536 |

| condition | val KL | PPL translated | PPL native | RULER 4k | 8k | 16k |
|---|---|---|---|---|---|---|
| native_small | 0.000 | nan | 7.18 | 0.98 | 0.98 | 0.96 |
| native_large | 0.000 | nan | 12.22 | 0.99 | 1.00 | 1.00 |
| closed_headmix_L2S | 0.862 | 15.08 | 7.18 | 0.77 | 0.76 | 0.64 |
| closed_headwise_L2S | 1.153 | 20.49 | 7.18 | 0.83 | 0.77 | 0.76 |
| kvlingo_headmix_L2S | 0.066 | 6.71 | 7.18 | 1.00 | 1.00 | 0.95 |
| kvlingo_headwise_L2S | 0.087 | 6.68 | 7.18 | 1.00 | 1.00 | 0.96 |
| kvlingo_mlp_L2S | 0.076 | 6.85 | 7.18 | 1.00 | 0.97 | 0.90 |
| closed_headmix_S2L | 0.938 | 21.89 | 12.22 | 0.87 | 0.90 | 0.84 |
| closed_headwise_S2L | 1.032 | 19.02 | 12.22 | 0.74 | 0.76 | 0.75 |
| kvlingo_headmix_S2L | 0.139 | 6.85 | 12.22 | 0.99 | 0.99 | 0.95 |
| kvlingo_headwise_S2L | 0.161 | 7.07 | 12.22 | 1.00 | 0.99 | 0.99 |
| kvlingo_mlp_S2L | 0.138 | 7.17 | 12.22 | 0.98 | 0.99 | 0.94 |