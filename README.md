We introduce a novel framework that enables the efficient and effective adaptation of LLMs to recommendation tasks.

## Main results
|                                 |  |movie |  ||   book |  |
| ------------------------------- | ----- | ----- | ----- | ----- | ----- | ----- |
| sample                          | 1     | 8     | 64 | 1 | 8 | 64 |
| GRU                             | 50.39 | 49.64 | 50.18 | 49 | 48.78 | 49.54 |
| Caser                           | 50.24 | 49.64 | 50.34 | 49.63 | 50.29 | 50.07 |
| SASRec                          | **50.7**  | 52.04 | 50.81 | 50.15 | 49.51 | 49.96 |
| DROS                            | 50    | 50.2  | 51.69 | 49.3 | 49.17 | 48.74 |
| rec-tuning (ours)               | 49.75 | 56.12 | 64.65 | 48.5 | 53.09 | 54.93 |
| instruction-tuning + rec-tuning (ours) | 48.18 | **59.04** | **65.83** | **53.02** | **54.89** | **60.35** |

Table 1. we shown the AUC results of the baseline models and our frameworks on movie and book scenarios.


## Cross-domain results

|       | movie | book  |
| ----- | ----- | ----- |
| movie | 65.83 | 64.2  |
| book  | 59.99 | 60.35 |

Table 2. we show the strong generalization ability of our frameworks, the column represents the training domain while the row represents the test domain.image.png