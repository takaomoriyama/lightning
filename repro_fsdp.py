import os

import torch.cuda
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.nn.parallel import DistributedDataParallel


def work(rank):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "1234"
    dist.init_process_group("nccl", world_size=2, rank=rank)

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    model = nn.Linear(100, 50).to(device)
    model = FullyShardedDataParallel(model)

    # no error with DDP
    # model = DistributedDataParallel(model, device_ids=[rank])

    # with torch.inference_mode():
    model(torch.rand(2, 100))

    torch.save(model.state_dict(), "fsdp_model.pt")

    with torch.inference_mode():
        model.load_state_dict(torch.load("fsdp_model.pt"))
        model(torch.rand(2, 100))


def run():
    mp.spawn(work, nprocs=2)


if __name__ == "__main__":
    run()