import argparse
import logging
import random
import statistics
import time
from typing import List
from venv import logger
import torch
from nixl._api import nixl_agent, nixl_agent_config

def register_mem(agent, t):
    mem_addr = [(t.data_ptr(), t.numel() * t.element_size(), 0, "a")]
    mem_type = "VRAM" if t.is_cuda else "DRAM"
    reg_desc = agent.register_memory(mem_addr, mem_type, ["UCX"])
    assert reg_desc is not None

    blocks = []

    block = t[0,0,0]
    block_len = block.numel() * block.element_size()
    logger.info(f"Tensor uses {block_len} bytes")

    block_id = 0
    for idx in range(t.shape[0]):          # layers
        for jdx in range(t.shape[1]):     # kv
            for kdx in range(t.shape[2]): # blocks                                                                                                                                                                                         block = t[idx, jdx, kdx]  # shape = [block_size]
                block = t[idx, jdx, kdx]  # shape = [block_size]

                last_ptr = block.data_ptr()
                blocks.append((last_ptr, block_len, 0))

    xfer_descs = agent.get_xfer_descs(blocks, mem_type)

    return xfer_descs

def parse_pattern(arg: str):
    valid = {"seq", "rand"}
    parts = arg.split(",")
    if len(parts) != 2 or any(p not in valid for p in parts):
        raise argparse.ArgumentTypeError(
            "Pattern must be two values separated by a comma, each either 'seq' or 'rand'"
        )
    return tuple(parts)

def get_block_indices(tot_blocks: int, blocks: int, pattern: str) -> List[int]:
    block_indices = None
    match pattern:
        case "rand":
            block_indices = random.sample(range(tot_blocks), blocks)
        case "seq":
            start = random.randint(0, tot_blocks - blocks)
            block_indices = [(start + i) % tot_blocks for i in range(blocks)]

    logger.debug(f"blocks={block_indices}")

    return block_indices

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Run with a number of iterations")
    parser.add_argument("--iterations", "-i", type=int, required=True,
                    help="Number of iterations to run")
    parser.add_argument("--blocks", "-b", type=int, default=4,
                    help="Number of blocks")
    parser.add_argument("--direction", "-d", choices=["h2h", "h2d", "d2h", "d2d"],
                    default="h2h",
                    help="Transfer direction: h2h=host2host, h2d=host2device, d2h=device2host, d2d=device2device")
    parser.add_argument("--size_gb", "-s", type=int, default=8,
                    help="Total transfer size in GB (default: 8)")
    parser.add_argument("--pattern", "-p",
                    type=parse_pattern,
                    default=("seq", "seq"),
                    help="Access pattern as tuple: seq,seq | seq,rand | rand,seq | rand,rand (default: seq,seq)")

    args = parser.parse_args()

    # tensor dimensions
    cache_mem_size = args.size_gb*1024*1024*1024
    layers = 32
    kv = 2
    block_size = 128*1024
    blocks_dim = cache_mem_size // (block_size*kv*layers*2)

    # initailize nixl agents
    agent_config = nixl_agent_config(backends=["UCX"])
    nixl_agent1 = nixl_agent("source", agent_config)
    nixl_agent2 = nixl_agent("target", agent_config)

    # allocate two tensors
    src_dev = "cpu"
    if args.direction == "d2h" or args.direction == "d2d":
        src_dev = "cuda"
    dst_dev = "cpu"
    if args.direction == "h2d" or args.direction == "d2d":
        dst_dev = "cuda"

    src = torch.empty((layers, kv, blocks_dim, block_size), dtype=torch.float16, device=src_dev, pin_memory=True if src_dev=="cpu" else False)
    dst = torch.empty_like(src, device=dst_dev, pin_memory=True if dst_dev=="cpu" else False)

    block = src[0,0,0]
    block_len = block.numel() * block.element_size()

    total_size = src.numel() * src.element_size()
    logger.info(f"Total size={total_size} numel={src.numel()} element_size={src.element_size()}")

    src_descs_ids = register_mem(nixl_agent1, src)
    dst_descs_ids = register_mem(nixl_agent2, dst)

    meta = nixl_agent2.get_agent_metadata()
    remote_name = nixl_agent1.add_remote_agent(meta)

    local_prep_handle = nixl_agent1.prep_xfer_dlist(
        "NIXL_INIT_AGENT", src_descs_ids, "VRAM" if src.is_cuda else "DRAM"
    )
    remote_prep_handle = nixl_agent1.prep_xfer_dlist(
        remote_name, dst_descs_ids, "VRAM" if dst.is_cuda else "DRAM"
    )

    # start transfer
    xfer_size = args.blocks*block_len

    logger.info(f"Starting transfer with NIXL: direction={args.direction} msg_size={xfer_size} " \
                f"blocks={args.blocks} iterations={args.iterations} pattern={args.pattern[0]},{args.pattern[1]}")

    bw_list = []
    for i in range(args.iterations):
        start = time.perf_counter()
        msg_id = f"UUID{i}"

        src_block_indices = get_block_indices(blocks_dim*kv*layers, args.blocks, args.pattern[0])
        dst_block_indices = get_block_indices(blocks_dim*kv*layers, args.blocks, args.pattern[1])

        xfer_handle = nixl_agent1.make_prepped_xfer(
            "WRITE", local_prep_handle, src_block_indices, remote_prep_handle, dst_block_indices, msg_id.encode("utf-8")
        )

        if not local_prep_handle or not remote_prep_handle:
            exit()

        if not xfer_handle:
            exit()

        state = nixl_agent1.transfer(xfer_handle)
        assert state != "ERR"

        target_done = False
        init_done = False

        logger.info("Transfer started")

        while (not init_done):
            if not init_done:
                state = nixl_agent1.check_xfer_state(xfer_handle)
                if state == "ERR":
                    logger.error("Transfer got to Error state.")
                    exit()
                elif state == "DONE":
                    init_done = True
                    end = time.perf_counter()
                    elapsed = end - start
                    bw = (xfer_size) / (elapsed * (1024**3))
                    bw_list.append(bw)
                    logger.info(f"Initiator done: Size: {xfer_size/ (1024**2)} MB Bandwidth: {bw:.3f} GB/s")
                    nixl_agent1.release_xfer_handle(xfer_handle)

    # Calculate mean and median
    mean_bw = statistics.mean(bw_list[1:])
    median_bw = statistics.median(bw_list[1:])

    print(f"\nMean BW: {mean_bw:.3f} GB/s")
    print(f"Median BW: {median_bw:.3f} GB/s")
