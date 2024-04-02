source haienv 202207
# submit for PCIe A100
# 8 node 64 GPUs
hfai bash run_haillm_train_llama_13B.sh -- -n 8 -i ubuntu2004-cu113 -g jd_a100#B --name run_haillm_train_llama_13B_nodes_8
# 16 node 128 GPUs
hfai bash run_haillm_train_llama_13B.sh -- -n 16 -i ubuntu2004-cu113 -g jd_a100#B --name run_haillm_train_llama_13B_nodes_16
# 32 node 256 GPUs
hfai bash run_haillm_train_llama_13B.sh -- -n 32 -i ubuntu2004-cu113 -g jd_a100#B --name run_haillm_train_llama_13B_nodes_32
# 64 node 512 GPUs
hfai bash run_haillm_train_llama_13B.sh -- -n 64 -i ubuntu2004-cu113 -g jd_a100#B --name run_haillm_train_llama_13B_nodes_64