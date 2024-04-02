# -*- coding: UTF-8 -*-
import os
import sys
import gc
import uuid
import torch.nn as nn
import time
import hfai.distributed as dist
import hfai
import torch
import hfreduce._hfreduce_impl as hfr_impl
import hfreduce
import hfreduce.torch as hfr
from collections import defaultdict
print(hfreduce.__version__, hfr, hfr_impl)
import numpy as np

def init_dist(local_rank):
    # 多机通信
    ip = os.getenv("MASTER_ADDR", "127.0.0.1")
    port = int(os.getenv('MASTER_PORT', '33111'))
    hosts = int(os.getenv('WORLD_SIZE', '1'))  # 机器个数
    node_rank = int(os.getenv('RANK', '0'))  # 当前机器编号
    gpus = torch.cuda.device_count()  # 每台机器的 GPU 个数

    # world_size 是全局 GPU 个数，rank 是当前 GPU 全局编号
    dist.init_process_group(backend='nccl',
                            init_method=f'tcp://{ip}:{port}',
                            world_size=hosts * gpus,
                            rank=node_rank * gpus + local_rank)
    torch.cuda.set_device(local_rank)
    # torch.cuda.set_device(4)

    return dist.get_rank(), dist.get_world_size()


class Timer():

    def __init__(self):
        self.t = 0
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

    def time(self, f):
        self.start.record()
        r = f()
        self.end.record()
        self.end.synchronize()
        self.t += self.start.elapsed_time(self.end)

        return r


class HFReducer():

    def __init__(self, model) -> None:
        self.model = model

        host = os.getenv("MASTER_ADDR", "127.0.0.1")
        port = int(os.getenv('MASTER_PORT', '33145')) + 5
        node_rank = int(os.getenv('RANK', '0'))
        nnodes = int(os.getenv('WORLD_SIZE', '1'))

        device = torch.cuda.current_device()
        proc_rank = torch.cuda.current_device()
        nprocs = torch.cuda.device_count()
        print(device, proc_rank, nprocs)

        params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        for p in model.parameters():
            p.grad = torch.ones_like(p.data)
            # print('p.numel()', p.numel())
            # print('p.view(dtype=torch.float).numel()', p.view(dtype=torch.float).numel())
            
        reducer_id = uuid.uuid4().hex

        self.recip = 1.0 / (nprocs * nnodes)

        self.reducer = hfr_impl.AsyncReduceFloat(
            host, port, device, proc_rank, nprocs, node_rank, nnodes, len(params), reducer_id)

        for n, p in params.items():
            # print('p.numel()', p.numel())
            # print('p.view(dtype=torch.float).numel()', p.view(dtype=torch.float).numel())
            self.reducer.register_float_tensor(n, p.view(dtype=torch.float).numel())

        self.fused_grads = torch.empty(
            self.reducer.total_tensor_floats(), device=device, dtype=torch.float)
        self.fused_grads.detach_()
        self.reducer.set_fused_dest(self.fused_grads.data_ptr())

        self.params = params

    def get_params(self):
        return self.params

    def all_reduce(self):
        str_p = torch.cuda.current_stream().cuda_stream

        for n, p in self.params.items():
            g = p.grad.contiguous().view(dtype=torch.float)
            # print('g.numel()', g.numel())
            self.reducer.async_reduce(str_p, n, g.data_ptr(), g.numel())

    def synchronize(self):
        str_p = torch.cuda.current_stream().cuda_stream
        names = self.reducer.synchronize(str_p)


def do_hfreduce(local_rank):
    rank, ws = init_dist(local_rank)

    timers = defaultdict(Timer)

    # 46M * 4B parameters = 184MiB
    model = nn.Sequential(*[nn.Linear(1024, 1024, bias=False)
                          for _ in range(46)])
    model.cuda()
    nparams = sum(p.numel() for p in model.parameters()) / (1 << 20)
    for p in model.parameters():
        element_size = p.element_size()
        break
    print(f"nparams {nparams} M, size {nparams * element_size}MiB", flush=True)
    reducer = HFReducer(model)
    print("init hfreduce", flush=True)

    for p in model.parameters():
        p.grad = torch.zeros_like(p.data)

    warmup_iters = 200
    iters = 100

    dist.barrier()

    for _ in range(warmup_iters):
        reducer.all_reduce()
        reducer.synchronize()

    dist.barrier()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        timers['reduce'].time(reducer.all_reduce)
        timers['synchronize'].time(reducer.synchronize)
        torch.cuda.synchronize()
    torch.cuda.synchronize()
    t1 = time.time()

    size = nparams * element_size  # MiB
    algoBW = nparams * element_size * iters * 1.024 * 1.024 / ( 1000 ) / (t1 - t0)
    busBW = algoBW * 2 * (ws - 1) / ws

    print(f'[RANK]: {rank} , [hfreduce] size: {size:.3} MiB, algoBW {algoBW:.3f}GB/s , Network Bandwidth {busBW}GB/s', flush=True)



def do_allreduce_base(local_rank):
    rank, ws = init_dist(local_rank)
    print("init distributed nccl", flush=True)
    n = 46 * (1 << 20)  # 46M * 4B = 184MiB
    device = torch.cuda.current_device()
    x = torch.empty(n, dtype=torch.float32, device=device)

    iters = 100
    warmup_iters = 200

    # warmup
    for _ in range(warmup_iters):
        dist.all_reduce(x)

    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.time()
    for _ in range(iters):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    t1 = time.time()

    size = x.numel() * x.element_size() / (1 << 20)
    t = (t1 - t0) / iters
    algoBW = size * 1.024 * 1.024 / ( 1000 ) / t
    busBW = algoBW * 2 * (ws - 1) / ws
    dist.barrier()
    print(f'[RANK]: {rank} , [NCCL allreduce] size: {size:.3} MiB, algoBW {algoBW:.3f}GB/s , Network Bandwidth {busBW}GB/s', flush=True)

def bench_nccl_allreduce():
    ngpus = hfai.utils.num_gpus()
    torch.multiprocessing.spawn(do_allreduce_base, args=(), nprocs=ngpus)

def bench_hfreduce():
    ngpus = hfai.utils.num_gpus()
    torch.multiprocessing.spawn(do_hfreduce, args=(), nprocs=ngpus)

if __name__ == '__main__':
    print('bench_nccl_allreduce')
    bench_nccl_allreduce()
    print('bench_hfreduce')
    bench_hfreduce()