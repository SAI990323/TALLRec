We introduce a novel framework (TALLRec) that enables the efficient and effective adaptation of LLMs to recommendation tasks.
Our project is based on Alpaca_lora (https://github.com/tloen/alpaca-lora) and the python environment is the same as Alpaca_lora.

## Main results
|                                 |  |movie |  ||   book |  |
|                                 | ----- | ----- | ----- | ----- | ----- | ----- |
| sample                          | 16     | 64     | 256 | 16 | 64 | 256 |
| ------------------------------- | ----- | ----- | ----- | ----- | ----- | ----- |
| GRU                             | 49.07 | 49.87 | 52.89 | 48.95 | 49.64 | 49.86 |
| Caser                           | 49.68 | 51.06 | 54.20 | 49.84 | 49.72 | 49.57 |
| SASRec                          | 50.43  | 50.48 | 52.25 | 49.48 | 50.06 | 50.20 |
| DROS                            | 50.76    | 51.54  | 54.07 | 49.28 | 49.13 | 49.13 |
| GRU-BERT                         | 50.85  | 51.65 | 53.44 | 50.07 | 49.64 | 49.79 |
| DROS-BERT                         | 50.21  | 51.71 | 53.94 | 50.07 | 48.98 | 50.20 |
| ------------------------------- | ----- | ----- | ----- | ----- | ----- | ----- |
| TALLRec (ours)               | **67.24** | **67.48** | **71.98** | **56.36** | **60.39** | **64.38** |

Table 1. we shown the AUC results of the baseline models and our frameworks on movie and book scenarios.

Train TALLRec base on LLAMA7B:
```
bash ./shell/instruct_7B.sh  gpu_id  random_seed
```
The gpu_id stands for the id of GPU you want to run the code on.

If you want to run it under your environment, you need to make changes to the sh file:

output_dir=XXX  Save result path
base_model=XXX  LLAMA model weight position, hugginface format
train_data=XXX  Training data set path such as "./data/movie/train.json" for movie dataset.
val_data=XXX  Validation data set path such as "./data/movie/valid.json" for movie dataset.
instruction_model=XXX The lora weight after alpaca-lora instruction tuning.

After training, you need to evluate the test result on the best model evaluated by the validation set.

```
bash ./shell/evaluate.sh  gpu_id  model_type
```
If you want to run it under your environment, you need to make changes to the sh file:

base_model=XXX LLAMA model weight position, hugginface format
test_data=XXX Test data set path such as "./data/movie/test.json" for movie dataset.