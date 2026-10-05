import torch
import inspect
from torch.distributed.fsdp import FSDPModule

print(torch.__version__)
print([name for name in dir(FSDPModule) if "sync" in name or "reshard" in name])
print(inspect.getsource(FSDPModule.set_requires_gradient_sync))
print(inspect.getsource(FSDPModule.set_reshard_after_backward))
