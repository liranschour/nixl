import argparse
import logging
import random
import statistics
import time
from venv import logger
import torch
from nixl._api import nixl_agent, nixl_agent_config

def register_mem(agent, t):
    mem_addr = [(t.data_ptr(), t.numel() * t.element_size(), 0, "a")]

    reg_desc = agent.register_memory(mem_addr, "DRAM", ["UCX"])
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

    xfer_descs = agent.get_xfer_descs(blocks, "DRAM")

    return xfer_descs

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Run with a number of iterations")
    parser.add_argument("--iterations", "-i", type=int, required=True,
                    help="Number of iterations to run")
    parser.add_argument("--blocks", "-b", type=int, default=4,
                    help="Number of blocks")

    args = parser.parse_args()

    # tensor dimensions
    cache_mem_size = 16*1024*1024*1024
    layers = 32
    kv = 2
    block_size = 128*1024
    blocks = cache_mem_size // (block_size*kv*layers*2)

    # initailize nixl agents XXX
    agent_config = nixl_agent_config(backends=["UCX"])
    nixl_agent1 = nixl_agent("source", agent_config)
    nixl_agent2 = nixl_agent("target", agent_config)

    # allocate two tensors
    src = torch.empty((layers, kv, blocks, block_size), dtype=torch.float16, device="cpu")
    dst = torch.empty_like(src)

    block = src[0,0,0]
    block_len = block.numel() * block.element_size()

    total_size = src.numel() * src.element_size()
    logger.info(f"Total size={total_size} numel={src.numel()} element_size={src.element_size()}")

    src_descs_ids = register_mem(nixl_agent1, src)
    dst_descs_ids = register_mem(nixl_agent2, dst)

    meta = nixl_agent2.get_agent_metadata()
    remote_name = nixl_agent1.add_remote_agent(meta)

    local_prep_handle = nixl_agent1.prep_xfer_dlist(
        "NIXL_INIT_AGENT", src_descs_ids, "DRAM"
    )
    remote_prep_handle = nixl_agent1.prep_xfer_dlist(
        remote_name, dst_descs_ids, "DRAM"
    )

    # start transfer
    logger.info(f"Starting transfer with NIXL... {args.blocks}")

    src_block_indices = random.sample(range(args.blocks), args.blocks)
    dst_block_indices = random.sample(range(args.blocks), args.blocks)
    xfer_size = args.blocks*block_len

    bw_list = []
    for i in range(args.iterations):
        start = time.perf_counter()
        xfer_handle = nixl_agent1.make_prepped_xfer(
            "WRITE", local_prep_handle, src_block_indices, remote_prep_handle, dst_block_indices, b"UUID2"
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
    mean_bw = statistics.mean(bw_list)
    median_bw = statistics.median(bw_list)

    print(f"\nMean BW: {mean_bw:.3f} GB/s")
    print(f"Median BW: {median_bw:.3f} GB/s")
