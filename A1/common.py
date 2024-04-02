import torch
import time
import numpy as np
import os
import torch.distributed as dist
import psutil
from typing import List

device_type: str = ""
device_mod = None

if torch.cuda.is_available():
    device_type = "cuda"
    device_mod = torch.cuda
else:
    assert torch.npu.is_available()
    device_type = "npu"
    device_mod = torch.npu

def bind_process_to_cpu(cpus: List[int]):
    p = psutil.Process()
    p.cpu_affinity(cpus)

def get_device_type():
    return device_type

def get_device_mod():
    return device_mod

def get_device_name(device_index: int):
    return device_mod.get_device_name(device_index)

def sync_device():
    device_mod.synchronize()

def set_device(device_index: int):
    device_mod.set_device(device_index)

def to_bandwidth_str(bw: float, IEC = False) -> str:
    if IEC:
        return f"{bw / (1 << 30):.3f} GiB/s"
    return f"{bw / 1e9:.3f} GB/s"


# repeat fn and return the average time elapsed per run.
def benchmark(fn, warmup_iters: int = 30, run_iters = 100, sync_once = True) -> float:
    for _ in range(warmup_iters):
        fn()
    sync_device()

    start = time.perf_counter()
    for _ in range(run_iters):
        fn()
        if not sync_once:
            sync_device()
    if sync_once:
        sync_device()

    end = time.perf_counter()

    return (end - start) / run_iters

def get_gpu_num_sms(device_index: int):
    return 64 if device_type == "npu" else torch.cuda.get_device_properties(device_index).multi_processor_count

def init_dist(local_rank, num_gpus = -1):
    ip = os.getenv('MASTER_ADDR', "127.0.0.1")
    port = int(os.getenv('MASTER_PORT', '4130'))
    nodes = int(os.getenv('WORLD_SIZE', 1))  # number of nodes
    node_rank = int(os.getenv('RANK', 0))    # node rank
    if num_gpus == -1:
        num_gpus = device_mod.device_count()         # number of gpus per node

    backend = "nccl" if device_type == "cuda" else "hccl"

    dist.init_process_group(
        backend=backend,
        init_method=f'tcp://{ip}:{port}',
        world_size=nodes * num_gpus,
        rank=node_rank * num_gpus + local_rank
    )
    device_map = torch.arange(0, num_gpus).tolist()
    # print(device_map, flush=True)
    device_mod.set_device(device_map[local_rank])

    return dist.get_rank(), dist.get_world_size()

def sync_gap():
    x = torch.zeros(1, device=device_type)
    dist.all_reduce(x)