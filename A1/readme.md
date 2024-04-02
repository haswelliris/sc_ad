# SC24's Artifact Description and Artifact Evaluation
If you would like to evaluate on the Fire-Flyer 2 AIHPC Platform, please contact us via email at `ly.zhang at high-flyer dot cn` to request a platform account. We can provide the necessary compute resources and nodes for your evaluations.

# Computational Artifact A1
Relation To Contributions:
```
Our PCIe A100 architecture managed to attain approximately 80% of NVIDIA DGX A100 performance.
TF32 GEMM performance: A100 PCIe 107TFLOPS vs DGX-A100 131TFLOPS.
FP16 GEMM performance: A100 PCIe 220TFLOPS vs DGX-A100 263TFLOPS 
```
Tested GPU GEMM computation performance by running GEMM calculation code through Torch.
The GEMM of the PCIe card can achieve about 80% of the  performance of DGX A100.
# Expected Results
TF32 GEMM performance:
1) A100 PCIe: 107TFLOPS;
2) DGX A100: 131TFLOPS.
FP16 GEMM performance:
1) A100 PCIe: 220TFLOPS;
2) DGX A100: 263TFLOPS.
# Artifact Setup (incl. Inputs)
## Hardware: 
  1) DGX A100: epyc 7742 * 2, DDR-3200*16,SXM4 40GB; 
  2) PCIe A100: epyc roma 7502 *2, DDR4-3200*16, A100 PCIe 40GB * 8, IB HDR 200Gbps * 1.
## Software: Ubuntu20.04 / Python3.8 / PyTorch 1.12 / CUDA 11.3.
Datasets / Inputs: N/A.
## Installation and Deployment: 
Used HaiPlatform
ubuntu2004-cu113 image, source haienv 202111.
## Artifact Execution
Refer to the files at https://github.com/haswelliris/sc ad/A1.
Obtained results by executing `python bench gemm.py` on A100 and DGX A100 separately
OR use HaiPlatfrom, in dev pod, `bash hfai_run.sh`