# prompt

baseline 在 v7 基础上跑的，即加 --eval-baseline 参数。
v6 的 system prompt 与 baseline 不一致，可能不太合适，于是进而有 v7。结果（v6 v7都没调参）：

| Method | MathVista | MathVision | MM-Math | HallusionBench | MMVP | MMStar | ScienceQA |
|--- |---|---|---|---|---|---|---|
| **baseline (original_system_prompt)** | 52.67 | 21.00 | 28.33 | 62.67 | 63.00 | 51.33 | 53.67 |
| **baseline (short_system_prompt)** | 47.00 | 16.67 | 25.33 | 65.00 | 59.00 | 45.33 | 49.33 |
| **v7 (original_system_prompt)** | 41.67 | 16.67 | 29.00 | 56.67 | 58.00 | 46.33 | 50.67 |
| **v7 (short_system_prompt)** | 45.00 | 19.67 | 27.00 | 65.67 | 60.00 | 47.33 | 50.33 |
| **v6** | 48.67 | 21.00 | 26.67 | 64.00 | 60.00 | 46.00 | 52.67 |

两个baseline之间唯一区别就是system prompt。
