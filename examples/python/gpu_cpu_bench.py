#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
GPU↔CPU transfer benchmark using the NIXL Python API.

Two modes:
  two_agents   - GPU agent (VRAM) + CPU agent (DRAM), in-process metadata exchange
  single_agent - One agent with both VRAM and DRAM registered, loopback transfer

Usage:
  python gpu_cpu_bench.py --mode two_agents
  python gpu_cpu_bench.py --mode single_agent
  python gpu_cpu_bench.py --mode two_agents --num_buffers 16 --buf_sizes 65536,1048576 --verify
  python gpu_cpu_bench.py --mode two_agents --backend UCCL
"""

import argparse
import time

import torch

from nixl._api import nixl_agent, nixl_agent_config
from nixl.logging import get_logger

logger = get_logger(__name__)

DEFAULT_BUF_SIZES = "4096,65536,1048576,16777216,268435456"


def parse_args():
    parser = argparse.ArgumentParser(
        description="NIXL GPU→CPU transfer benchmark (two_agents vs single_agent)"
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["two_agents", "single_agent"],
        help="Benchmark mode",
    )
    parser.add_argument(
        "--num_buffers",
        type=int,
        default=8,
        help="Number of buffers in the transfer vector (default: 8)",
    )
    parser.add_argument(
        "--buf_sizes",
        type=str,
        default=DEFAULT_BUF_SIZES,
        help="Comma-separated buffer sizes in bytes (default: %(default)s)",
    )
    parser.add_argument(
        "--warmup_iters",
        type=int,
        default=5,
        help="Number of warmup iterations per buffer size (default: 5)",
    )
    parser.add_argument(
        "--bench_iters",
        type=int,
        default=20,
        help="Number of timed iterations per buffer size (default: 20)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="UCX",
        help="NIXL backend to use (default: UCX, e.g. UCX, UCCL)",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="CUDA GPU device ID (default: 0)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify data correctness after the first transfer iteration",
    )
    return parser.parse_args()


def _fmt_size(n_bytes):
    """Format byte count as a human-readable string."""
    if n_bytes >= 1 << 30:
        return f"{n_bytes / (1 << 30):.1f} GB"
    if n_bytes >= 1 << 20:
        return f"{n_bytes / (1 << 20):.1f} MB"
    if n_bytes >= 1 << 10:
        return f"{n_bytes / (1 << 10):.1f} KB"
    return f"{n_bytes} B"


def make_buffers(num_bufs, buf_size, gpu_id):
    """Allocate GPU (source) and CPU (destination) tensors.

    Returns:
        gpu_tensor: 2-D uint8 tensor on CUDA, shape (num_bufs, buf_size), filled with 1
        cpu_tensor: 2-D uint8 tensor on CPU,  shape (num_bufs, buf_size), filled with 0
        gpu_bufs:   list of row-views into gpu_tensor
        cpu_bufs:   list of row-views into cpu_tensor
    """
    gpu_tensor = torch.ones(
        (num_bufs, buf_size), dtype=torch.uint8, device=f"cuda:{gpu_id}"
    )
    cpu_tensor = torch.zeros((num_bufs, buf_size), dtype=torch.uint8)
    gpu_bufs = [gpu_tensor[i] for i in range(num_bufs)]
    cpu_bufs = [cpu_tensor[i] for i in range(num_bufs)]
    return gpu_tensor, cpu_tensor, gpu_bufs, cpu_bufs


def run_transfers(agent, xfer_handle, total_bytes, warmup_iters, bench_iters):
    """Run warmup then timed benchmark transfers.

    Returns:
        throughput in GB/s
    """
    def do_one_transfer():
        state = agent.transfer(xfer_handle)
        if state == "ERR":
            raise RuntimeError("transfer() returned ERR")
        while True:
            state = agent.check_xfer_state(xfer_handle)
            if state == "DONE":
                break
            if state == "ERR":
                raise RuntimeError("check_xfer_state() returned ERR")

    # Warmup
    for _ in range(warmup_iters):
        do_one_transfer()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(bench_iters):
        do_one_transfer()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    return (bench_iters * total_bytes) / elapsed / 1e9


def bench_two_agents(args, buf_sizes):
    """Benchmark with two separate agents (gpu_agent and cpu_agent)."""
    results = []

    for buf_size in buf_sizes:
        total_bytes = args.num_buffers * buf_size
        gpu_tensor, cpu_tensor, gpu_bufs, cpu_bufs = make_buffers(
            args.num_buffers, buf_size, args.gpu_id
        )

        # Create two agents — no listen thread needed for in-process exchange
        config = nixl_agent_config(
            enable_prog_thread=True,
            enable_listen_thread=False,
            listen_port=0,
            backends=[args.backend],
        )
        gpu_agent = nixl_agent("gpu_agent", config)
        cpu_agent = nixl_agent("cpu_agent", config)

        # Register memory
        gpu_reg = gpu_agent.register_memory(gpu_tensor)
        if not gpu_reg:
            raise RuntimeError("GPU memory registration failed")
        cpu_reg = cpu_agent.register_memory(cpu_tensor)
        if not cpu_reg:
            raise RuntimeError("CPU memory registration failed")

        # In-process metadata exchange (no sockets)
        gpu_agent.add_remote_agent(cpu_agent.get_agent_metadata())
        cpu_agent.add_remote_agent(gpu_agent.get_agent_metadata())

        # Build transfer descriptors
        gpu_descs = gpu_agent.get_xfer_descs(gpu_bufs)
        if not gpu_descs:
            raise RuntimeError("Failed to build GPU transfer descriptors")

        # CPU descs from cpu_agent's perspective, then hand to gpu_agent
        cpu_descs_local = cpu_agent.get_xfer_descs(cpu_bufs)
        if not cpu_descs_local:
            raise RuntimeError("Failed to build CPU transfer descriptors")
        cpu_desc_bytes = cpu_agent.get_serialized_descs(cpu_descs_local)
        remote_cpu_descs = gpu_agent.deserialize_descs(cpu_desc_bytes)

        # Create transfer handle: gpu_agent WRITEs (pushes) GPU→CPU
        xfer_handle = gpu_agent.initialize_xfer(
            "WRITE", gpu_descs, remote_cpu_descs, "cpu_agent"
        )
        if not xfer_handle:
            raise RuntimeError("initialize_xfer failed")

        # Optionally verify correctness on the first transfer
        if args.verify and not results:
            state = gpu_agent.transfer(xfer_handle)
            if state == "ERR":
                raise RuntimeError("Verification transfer failed to post")
            while gpu_agent.check_xfer_state(xfer_handle) != "DONE":
                pass
            expected = torch.ones((args.num_buffers, buf_size), dtype=torch.uint8)
            if not torch.all(cpu_tensor == expected):
                raise RuntimeError("Data verification FAILED for two_agents mode")
            logger.info("Data verification passed (two_agents, buf_size=%d)", buf_size)

        gbps = run_transfers(
            gpu_agent, xfer_handle, total_bytes, args.warmup_iters, args.bench_iters
        )
        results.append((buf_size, total_bytes, gbps))

        # Teardown
        gpu_agent.release_xfer_handle(xfer_handle)
        gpu_agent.deregister_memory(gpu_reg)
        cpu_agent.deregister_memory(cpu_reg)
        gpu_agent.remove_remote_agent("cpu_agent")
        cpu_agent.remove_remote_agent("gpu_agent")

    return results


def bench_single_agent(args, buf_sizes):
    """Benchmark with a single agent holding both VRAM and DRAM (loopback)."""
    results = []

    for buf_size in buf_sizes:
        total_bytes = args.num_buffers * buf_size
        gpu_tensor, cpu_tensor, gpu_bufs, cpu_bufs = make_buffers(
            args.num_buffers, buf_size, args.gpu_id
        )

        config = nixl_agent_config(
            enable_prog_thread=True,
            enable_listen_thread=False,
            listen_port=0,
            backends=[args.backend],
        )
        agent = nixl_agent("nixl_agent", config)

        # Register both GPU and CPU memory with the same agent
        gpu_reg = agent.register_memory(gpu_tensor)
        if not gpu_reg:
            raise RuntimeError("GPU memory registration failed")
        cpu_reg = agent.register_memory(cpu_tensor)
        if not cpu_reg:
            raise RuntimeError("CPU memory registration failed")

        # Build descriptors — both from this agent's perspective
        gpu_descs = agent.get_xfer_descs(gpu_bufs)
        if not gpu_descs:
            raise RuntimeError("Failed to build GPU transfer descriptors")
        cpu_descs = agent.get_xfer_descs(cpu_bufs)
        if not cpu_descs:
            raise RuntimeError("Failed to build CPU transfer descriptors")

        # Loopback transfer: remote_agent == own name
        xfer_handle = agent.initialize_xfer(
            "WRITE", gpu_descs, cpu_descs, "nixl_agent"
        )
        if not xfer_handle:
            raise RuntimeError("initialize_xfer failed")

        # Optionally verify correctness on the first transfer
        if args.verify and not results:
            state = agent.transfer(xfer_handle)
            if state == "ERR":
                raise RuntimeError("Verification transfer failed to post")
            while agent.check_xfer_state(xfer_handle) != "DONE":
                pass
            expected = torch.ones((args.num_buffers, buf_size), dtype=torch.uint8)
            if not torch.all(cpu_tensor == expected):
                raise RuntimeError("Data verification FAILED for single_agent mode")
            logger.info("Data verification passed (single_agent, buf_size=%d)", buf_size)

        gbps = run_transfers(
            agent, xfer_handle, total_bytes, args.warmup_iters, args.bench_iters
        )
        results.append((buf_size, total_bytes, gbps))

        # Teardown
        agent.release_xfer_handle(xfer_handle)
        agent.deregister_memory(gpu_reg)
        agent.deregister_memory(cpu_reg)

    return results


def print_results(mode, backend, num_buffers, bench_iters, results):
    header = (
        f"\nMode: {mode} | backend={backend} | num_buffers={num_buffers} | bench_iters={bench_iters}"
    )
    sep = "-" * 52
    col = f"{'Buffer Size':>12}  {'Buf/Total':>14}  {'Throughput':>14}"
    print(header)
    print(sep)
    print(col)
    print(sep)
    for buf_size, total_bytes, gbps in results:
        bs_str = _fmt_size(buf_size)
        tot_str = _fmt_size(total_bytes)
        print(f"{bs_str:>12}  {tot_str:>14}  {gbps:>12.3f} GB/s")
    print(sep)


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available — GPU required for this benchmark")
    if args.gpu_id >= torch.cuda.device_count():
        raise RuntimeError(
            f"GPU {args.gpu_id} not found ({torch.cuda.device_count()} GPU(s) available)"
        )

    buf_sizes = [int(s.strip()) for s in args.buf_sizes.split(",")]
    if any(s <= 0 for s in buf_sizes):
        raise ValueError("All buffer sizes must be positive")

    logger.info(
        "Starting benchmark: mode=%s, backend=%s, num_buffers=%d, buf_sizes=%s, "
        "warmup=%d, bench=%d, gpu=%d",
        args.mode,
        args.backend,
        args.num_buffers,
        [_fmt_size(s) for s in buf_sizes],
        args.warmup_iters,
        args.bench_iters,
        args.gpu_id,
    )

    if args.mode == "two_agents":
        results = bench_two_agents(args, buf_sizes)
    else:
        results = bench_single_agent(args, buf_sizes)

    print_results(args.mode, args.backend, args.num_buffers, args.bench_iters, results)


if __name__ == "__main__":
    main()
