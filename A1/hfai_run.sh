source haienv 202207
# submit for PCIe A100
hfai python bench_gemm.py -- -n 1 -i ubuntu2004-cu113 -g jd_a100 --name gemm_test
# submit for DGX A100
hfai python bench_gemm.py -- -n 1 -i ubuntu2004-cu113 -g jd_a100#DGX --name gemm_test