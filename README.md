Our weights for the instruction tuning model is uploading [here](https://drive.google.com/file/d/1teUwLm4BOqhngfCKKXE1tiMhJPf_FvRJ/view?usp=sharing)

**TALLRec: An Effective and Efficient Tuning Framework to Align Large Language Model with Recommendation** is available at https://arxiv.org/abs/2305.00447.

**Wrongly delete the line in evaluate.py by mistake, now it has been updated**

We introduce a novel framework (TALLRec) that enables the efficient and effective adaptation of LLMs to recommendation tasks.

# Main results
|                                 |  |movie |  ||   book |  |
|-------------------------------                  | ----- | ----- | ----- | ----- | ----- | ----- |
| Few-shot                          | 16     | 64     | 256 | 16 | 64 | 256 |
| GRU                             | 49.07 | 49.87 | 52.89 | 48.95 | 49.64 | 49.86 |
| Caser                           | 49.68 | 51.06 | 54.20 | 49.84 | 49.72 | 49.57 |
| SASRec                          | 50.43  | 50.48 | 52.25 | 49.48 | 50.06 | 50.20 |
| DROS                            | 50.76    | 51.54  | 54.07 | 49.28 | 49.13 | 49.13 |
| GRU-BERT                         | 50.85  | 51.65 | 53.44 | 50.07 | 49.64 | 49.79 |
| DROS-BERT                         | 50.21  | 51.71 | 53.94 | 50.07 | 48.98 | 50.20 |
| TALLRec (ours)               | **67.24** | **67.48** | **71.98** | **56.36** | **60.39** | **64.38** |

Table 1. we shown the AUC results of the baseline models and our frameworks on movie and book scenarios.

Train TALLRec base on LLaMA7B:
```
bash ./shell/instruct_7B.sh  gpu_id random_seed
```
If you want to run it under your environment, you need to make changes to the sh file:
- output_dir: Model save path，we will automatically add the seed and the sample to the end of the path for each experiments.
- base_model: LLaMA parameter weight path in Hugginface format
- train_data:  Training data path such as "./data/movie/train.json" for movie dataset.
- val_data: Validation data set path such as "./data/movie/valid.json" for movie dataset.
- instruction_model: The LoRA weights after the instruction tuning, for example lora weight from alpaca-lora.

After training, you need to evluate the test result on the best model evaluated by the validation set.
```
bash ./shell/evaluate.sh  gpu_id  output_dir
```
If you want to run it under your environment, you need to make changes to the sh file:
- base_model: LLaMA parameter weight path in Hugginface format
- test_data: Test data set path such as "./data/movie/test.json" for movie dataset.

Note that we will automatically detect all the different seed and sample files in the output_dir directory, and then integrate these results into the output_dir.json file.

Our project is developed based on the Alpaca_lora [repo](https://github.com/tloen/alpaca-lora), thanks for their contributions.

For "Environment setting sharing for CUDA 12.0", please see [here](https://github.com/SAI990323/TALLRec/issues/46).
