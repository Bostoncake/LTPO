# prompt

## 自己跑的

下面实验采用 2.5-3B 模型。

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

进一步地，将 v7 在 3-4B 和 3-8B 上面跑。结果如下：

**3-4B**

其中：v7 (short_system_prompt) 未调参，参数为：**tokens2-lr0.01-sigma25.0-sigdecay0.95-steps15-topk10-conf-dmlr**

| Method | MathVista | MathVision | MM-Math | HallusionBench | MMVP | MMStar | ScienceQA |
|---|---|---|---|---|---|---|---|
| **v7 (short_system_prompt)** | 45.00 | 22.33 | 54.67 | 64.67 | 74.67 | 44.00 | 57.33 |
| **baseline (short_system_prompt)** | 61.00 | 40.00 | 57.67 | 69.00 | 77.67 | 57.67 | 59.67 |

**3-8B**

其中：v7 (short_system_prompt) 未调参，参数为：**tokens2-lr0.01-sigma25.0-sigdecay0.95-steps15-topk10-conf-dmlr**

| Method | MathVista | MathVision | MM-Math | HallusionBench | MMVP | MMStar | ScienceQA |
|---|---|---|---|---|---|---|---|
| **v7 (short_system_prompt)** | 45.67 | 28.67 | 55.67 | 66.67 | 74.33 | 47.67 | 57.67 |
| **baseline (short_system_prompt)** | 63.00 | 39.00 | 57.67 | 74.67 | 74.33 | 54.33 | 61.00 |

2.5-3B:

| Method | MathVista | MathVision | MM-Math | HallusionBench | MMVP | MMStar | ScienceQA |
|---|---|---|---|---|---|---|---|
| **v8_full (sota_system_prompts)** | 40.20 | - | - | 56.26 | 61.33 | - | - |
| **baseline_v8_full (modified_system_prompts)** | 53.80 | - | - | 67.19 | 62.67 | 52.13 | - |
| **v7_full (sota_system_prompts)** | 43.80 | 19.90 | 28.25 | 63.62 | 60.33 | 46.00 | 46.55 |
| **baseline_v7_full (modified_system_prompts)** | 47.10 | 19.41 | 27.79 | 64.98 | 59.00 | 44.73 | 47.94 |

3-4B:

| Method | MathVista | MathVision | MM-Math | HallusionBench | MMVP | MMStar | ScienceQA |
|---|---|---|---|---|---|---|---|
| **v8 (sota_system_prompts)** | 47.33 | 25.67 | 59.33 | 68.33 | 75.33 | 52.33 | 58.00 |
| **baseline (modified_system_prompts)** | 56.33 | 17.00 | 54.67 | 68.00 | 70.67 | 51.67 | 42.00 |
| **baseline (original_short_system_prompt)** | 61.00 | 40.00 | 57.67 | 69.00 | 77.67 | 57.67 | 59.67 |
| **v8_full (sota_system_prompts)** | 44.00 | 24.97 | - | 67.51 | 75.33 | 54.07 | 52.95 |
| **baseline_v8_full (modified_system_prompts)** | 54.90 | 18.16 | 57.18 | 65.72 (71.82) | 70.67 | 52.20 | 36.94 |

- **baseline (modified_system_prompts)**、**v8 (sota_system_prompts)** 里面每个bench的system prompt可能不同。

- **baseline (original_short_system_prompt)** 里面 system prompt 相同，就是简单的短system prompt，用来对照一下。

3-8B：

| Method | MathVista | MathVision | MM-Math | HallusionBench | MMVP | MMStar | ScienceQA |
|---|---|---|---|---|---|---|---|
|**baseline_sota**| 57.33 | 24.00 | 54.33 | 73.00 | 73.00 | 52.33 | 55.67 |
|**ltpo_sota**| 50.67 | 31 | 57.33 | 68 | 74.67 | 54.67 | 59 |
|**baseline_v6**| 69.33 | 24.00 | 58.00 | 75.33 | 76.33 | 63.00 | 63.00 |
|**baseline_v10**| 57.33 | 38.00 | 54.33 | 73.00 | 73.00 | 52.33 | 55.67 |
|**最开始调参的结果**| 47.67 | 31 | 57.33 | 68 | 74.67 | 54.67 | 59 |
|**v8**| 50.67 | 29.33 | 56.33 | 66.33 | 74.33 | 51.67 | 57.67 |
|**v8_full (sota_system_prompts)**| 48.90 | - | - | 70.56 | 74.33 | 55.53 | 53.50 |
|**baseline_v8_full (sota_system_prompts)**| 58.70 | - | - | 66.56 (74.34) | 73.00 | 54.27 | 51.96 |


## 汇报

2.5-3B:

| Method | MathVista | MathVision | MM-Math | HallusionBench | MMVP | MMStar | ScienceQA |
|---|---|---|---|---|---|---|---|
| **v7_full (sota_system_prompts)** | 43.80 | 19.90 | 28.25 | 63.62 | 60.33 | 46.00 | 46.55 |
| **baseline_v7_full (modified_system_prompts)** | 47.10 | 19.41 | 27.79 | 64.98 | 59.00 | 44.73 | 47.94 |

3-4B:

| Method | MathVista | MathVision | MM-Math | HallusionBench | MMVP | MMStar | ScienceQA |
|---|---|---|---|---|---|---|---|
| **v8_full (sota_system_prompts)** | 44.00 | - | - | 67.51 | 75.33 | 54.07 | 52.95 |
| **baseline_v8_full (modified_system_prompts)** | 54.90 | - | - | 65.72 | 70.67 | 52.20 | 36.94 |

3-8B：

| Method | MathVista | MathVision | MM-Math | HallusionBench | MMVP | MMStar | ScienceQA |
|---|---|---|---|---|---|---|---|
|**v8_full (sota_system_prompts)**| 48.90 | - | - | 70.56 | 74.33 | 55.53 | 53.50 |
|**baseline_v8_full (sota_system_prompts)**| 58.70 | - | - | 66.56 | 73.00 | 54.27 | 51.96 |
