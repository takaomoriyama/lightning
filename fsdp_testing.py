import torch
from lightning import Fabric
from lightning.fabric.strategies import FSDPStrategy
from torch._subclasses import FakeTensor
from torch.distributed.fsdp.wrap import always_wrap_policy

strategy = FSDPStrategy(auto_wrap_policy=always_wrap_policy)
fabric = Fabric(accelerator="cuda", devices=2, strategy=strategy)
fabric.launch()

with fabric.sharded_model():
    large_number = 1_000_000_000
    # try to fit this in your memory, and the universe will collapse
    large_model = torch.nn.Linear(large_number, large_number, bias=False)

fabric_model = fabric.setup_module(large_model)

# the linear layer got sharded and each part is on the expected device
assert isinstance(next(fabric_model.parameters()), FakeTensor)
assert next(fabric_model.parameters()).device == torch.device("cuda", fabric.local_rank)

print(torch.cuda.max_memory_allocated())
print(torch.cuda.max_memory_reserved())
