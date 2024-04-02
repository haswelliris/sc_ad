import torch
import argparse
from common import *
import os
torch.backends.cuda.matmul.allow_tf32 = True
# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True
# os.environ["ASCEND_LAUNCH_BLOCKING"]="1"
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--device_index', default=0, help="device index")

def gemm(a, b):
    return a @ b

def test_main(device_index: int, test_type, N = 13824):
    set_device(device_index)

    a = torch.zeros((N, N), device=get_device_type(), dtype=test_type).fill_(0.5)
    b = torch.zeros((N, N), device=get_device_type(), dtype=test_type).fill_(2)
    c = torch.zeros((N, N), device=get_device_type(), dtype=test_type).fill_(N)
    c_ = a @ b 
    assert c.allclose(c_)

    # torch.backends.cuda.matmul.allow_tf32 = True
    a = torch.rand((N, N), device=get_device_type(), dtype=test_type)
    b = torch.rand((N, N), device=get_device_type(), dtype=test_type)
    t_per_iter = benchmark(lambda: gemm(a, b))
    print(f"Type:{test_type} N={N} time_per_iter: {t_per_iter * 1e6:.2f}usecs, average performance: {2 * N * N * N / t_per_iter / 1e12:.2f}tflops")

if __name__ == '__main__':
    args = parser.parse_args()
    device_index = int(args.device_index)

    print(f"test gemm on {get_device_name(device_index)}")
    # let matrix shape fit num of sms to get the peak performance.
    base_size = 64 # get_gpu_num_sms(device_index)
    # ! tf32/hf32 is not tested.
    test_types = [torch.float32, torch.float16, torch.bfloat16]
    test_Ns = [(1 << i) * base_size for i in range(5, 9)]
    for test_type in test_types:
        for N in test_Ns:
            test_main(device_index, test_type, N)