source haienv 202207
# submit for PCIe A100
# 2 node 16 GPUs
hfai python bench_allreduce.py -- -n 2 -i ubuntu2004-cu113 -g jd_a100#A --name bench_allreduce_nodes_2
# 4 node 32 GPUs
hfai python bench_allreduce.py -- -n 4 -i ubuntu2004-cu113 -g jd_a100#A --name bench_allreduce_nodes_4
# 8 node 64 GPUs
hfai python bench_allreduce.py -- -n 8 -i ubuntu2004-cu113 -g jd_a100#A --name bench_allreduce_nodes_8
# 16 node 128 GPUs
hfai python bench_allreduce.py -- -n 16 -i ubuntu2004-cu113 -g jd_a100#A --name bench_allreduce_nodes_16
# 32 node 256 GPUs
hfai python bench_allreduce.py -- -n 32 -i ubuntu2004-cu113 -g jd_a100#A --name bench_allreduce_nodes_32
# 64 node 512 GPUs
hfai python bench_allreduce.py -- -n 64 -i ubuntu2004-cu113 -g jd_a100#A --name bench_allreduce_nodes_64
# 128 node 1024 GPUs
hfai python bench_allreduce.py -- -n 128 -i ubuntu2004-cu113 -g jd_a100#A --name bench_allreduce_nodes_128
# 160 node 1440 GPUs
hfai python bench_allreduce.py -- -n 160 -i ubuntu2004-cu113 -g jd_a100#A --name bench_allreduce_nodes_160