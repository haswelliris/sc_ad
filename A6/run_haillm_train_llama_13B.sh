source haienv moe4.8.1c
cd /weka-jd/prod/jupyter/sc24/notebooks/hai-llm
python train_gpt3.py --train_config configs/a100/llama_13b_seq2k_bs4096s.py --log_dir logs/llama_13b_seq2k_bs4096 --dummy