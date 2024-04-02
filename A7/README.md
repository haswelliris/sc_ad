
# SC24's Artifact Description and Artifact Evaluation
If you would like to evaluate on the Fire-Flyer 2 AIHPC Platform, please contact us via email at `ly.zhang at high-flyer dot cn` to request a platform account. We can provide the necessary compute resources and nodes for your evaluations.

# GPT-2

You can get more information about these codes in [https://github.com/HFAiLab/hfai-models/blob/main/gpt/README_en.md](https://github.com/HFAiLab/hfai-models/blob/main/gpt/README_en.md)

This is an example of haiscale to train a GPT-2 model on Wikitext-103 dataset.

## Data Preparation

0. Install huggingface transformers:

    ```
    pip install transformers
    ```

1. Download Wikitext-103 and decompress it to `data/wikitext-103`. The directory structure is as follows:

    ```
    data/wikitext-103
    ├── wiki.test.tokens
    ├── wiki.train.tokens
    └── wiki.valid.tokens
    ```

2. Run the preprocess script to tokenize the raw text with BPE:

    ```
    python preprocess.py
    ```

    Now the directory structure is as follows:

    ```
    data/wikitext-103
    ├── test.npy
    ├── train.npy
    ├── valid.npy
    ├── wiki.test.tokens
    ├── wiki.train.tokens
    └── wiki.valid.tokens
    ```

## Training

This example includes multiple training scripts. Each script uses a different parallelism method:

1. Haiscale

    ```
    python train_haiscale.py
    ```

2. Torch: FSDP: fully sharded data parallel

    ```
    python train_torch_fsdp.py
    ```
## Evaluation
Submit tasks in Fire-Flyer Platform by `./submit.sh`