source haienv 202207
# submit for PCIe A100
# 2 node 16 GPUs
hfai bash bench_hfreduce_with_nvlink.sh -- -n 2 -i ubuntu2004-cu113 -g jd_a100#B --name bench_hfreduce_with_nvlink_nodes_2
# 4 node 32 GPUs
hfai bash bench_hfreduce_with_nvlink.sh -- -n 4 -i ubuntu2004-cu113 -g jd_a100#A --name bench_hfreduce_with_nvlink_nodes_4
# 8 node 64 GPUs
hfai bash bench_hfreduce_with_nvlink.sh -- -n 8 -i ubuntu2004-cu113 -g jd_a100#A --name bench_hfreduce_with_nvlink_nodes_8
# 16 node 128 GPUs
hfai bash bench_hfreduce_with_nvlink.sh -- -n 16 -i ubuntu2004-cu113 -g jd_a100#A --name bench_hfreduce_with_nvlink_nodes_16
# 32 node 256 GPUs
hfai bash bench_hfreduce_with_nvlink.sh -- -n 32 -i ubuntu2004-cu113 -g jd_a100#A --name bench_hfreduce_with_nvlink_nodes_32
# 64 node 512 GPUs
hfai bash bench_hfreduce_with_nvlink.sh -- -n 64 -i ubuntu2004-cu113 -g jd_a100#A --name bench_hfreduce_with_nvlink_nodes_64
# 128 node 1024 GPUs
hfai bash bench_hfreduce_with_nvlink.sh -- -n 128 -i ubuntu2004-cu113 -g jd_a100#A --name bench_hfreduce_with_nvlink_nodes_128

# 32 node 256 GPUs cross fat-tree Zone
hfai bash bench_hfreduce_with_nvlink.sh -- -n 32 -i ubuntu2004-cu113 -g jd_a100#AB --name bench_hfreduce_with_nvlink_nodes_32
# 64 node 512 GPUs cross fat-tree Zone
hfai bash bench_hfreduce_with_nvlink.sh -- -n 64 -i ubuntu2004-cu113 -g jd_a100#AB --name bench_hfreduce_with_nvlink_nodes_64
# 128 node 1024 GPUs cross fat-tree Zone
hfai bash bench_hfreduce_with_nvlink.sh -- -n 128 -i ubuntu2004-cu113 -g jd_a100#AB --name bench_hfreduce_with_nvlink_nodes_128