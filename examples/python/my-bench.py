import logging
import random
import time
from venv import logger
import torch
from nixl._api import nixl_agent, nixl_agent_config

def register_mem(agent, t):
    mem_addr = [(t.data_ptr(), t.numel() * t.element_size(), 0, "a")]
    print(f"{mem_addr}")

    reg_desc = agent.register_memory(mem_addr, "DRAM", ["UCX"])
    assert reg_desc is not None
    print(f"register mem={reg_desc}")

    last_ptr = 0
    blocks = []

    block = t[0,0,0]
    block_len = block.numel() * block.element_size()
    basic_offset = t.data_ptr()
    print(f"Tensor uses {block_len} bytes")

    block_id = 0
    for idx in range(t.shape[0]):          # layers
        for jdx in range(t.shape[1]):     # kv
            for kdx in range(t.shape[2]): # blocks                                                                                                                                                                                         block = t[idx, jdx, kdx]  # shape = [block_size]
                block = t[idx, jdx, kdx]  # shape = [block_size]
                
                #print(f"data_ptr={block.data_ptr()} offset={basic_offset+(block_id*block_len)}")
                #print(f"block shape: {block.shape}, values: {block} ptr={block.data_ptr()} len={block.data_ptr()-last_ptr}")
                last_ptr = block.data_ptr()
                blocks.append((basic_offset+(block_id*block_len), block_len, 0))
                block_id += 1

    xfer_descs = agent.get_xfer_descs(blocks, "DRAM")
    print(f"len blocks={len(blocks)} xfer_descs={xfer_descs}")

    return xfer_descs

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # tensor dimensions
    cache_size = 16*1024*1024*1024
    layers = 32
    kv = 2
    block_size = 256*1024
    blocks = cache_size // (block_size*kv*layers)

    # initailize nixl agents XXX
    agent_config = nixl_agent_config(backends=["UCX"])
    nixl_agent1 = nixl_agent("source", agent_config)
    
    nixl_agent2 = nixl_agent("target", agent_config)

    # allocate two tensors
    src = torch.empty((layers, kv, blocks, block_size), dtype=torch.float16, device="cpu")
    dst = torch.empty_like(src)

    total_size = src.numel() * src.element_size()
    print(f"Total size={total_size}")

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
    print("Starting transfer with NIXL...")

    src_block_indices = random.sample(range(blocks), blocks)
    dst_block_indices = random.sample(range(blocks), blocks)

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
                bw = total_size / (elapsed * 1e9)
                logger.info(f"Initiator done: Bandwidth: {bw:.3f} GB/s")
