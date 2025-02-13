

## Environments
- Ubuntu-18.0.4
- Python (3.7)
- Cuda (11.1)

## Installation
Install [Pytorch](https://pytorch.org/) 1.9.0, then run the following in the terminal:
```shell
cd Persona-DB
conda create -n Persona-DB python=3.7 -y  # create a new conda environment
conda activate Persona-DB

chmod +x scripts/setup.sh
./scripts/setup.sh
```
Create a directories `data/CNN/` and `data/OpinionQA/`. Then put content from data.txt into them respectively. For user data in response forecasting, please refer to https://github.com/chenkaisun/response_forecasting.

## Note
The running of the system might require [wandb](wandb.ai) account login

## Run API
To run the system, edit the parameters with $ appended in the following and run in the terminal.
Note: --prompt_types: can choose from prompt folder. --neighbor_topk: amount of neighbors. --item_topk_collab:  number of items from collaboration, --item_topk: number of items from self. --num_hist: recency. --cases is a string representing the case study including lurker_hist_100_test and longest_hist_300_test. --dataset chosen from "34", "82", or "41".

```shell
python llm_predict.py \
    --prompt_types $prompt_types \
    --cases $cases \
    --neighbor_topk $neighbor_topk \
    --item_topk_collab $item_topk_collab \
    --item_topk $item_topk \
    --num_hist $num_hist
    
python llm_predict_opinionqa.py \
     --prompt_types $prompt_types \
     --cases $cases \
     --neighbor_topk $neighbor_topk \
     --dataset $dataset \
     --item_topk_collab $item_topk_collab \
     --item_topk $item_topk
```