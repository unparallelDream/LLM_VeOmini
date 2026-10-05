import os
import shutil

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed._composable.fsdp import fully_shard

from utils.trainable_checkpointer import TrainableDistributedCheckpointer


class ToyFSDP2FrozenBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(16, 16, bias=False)
        self.anchor = nn.Parameter(torch.linspace(-0.8, 0.7, 16))

    def forward(self, value):
        return self.backbone(value) + self.anchor


def local_tensor(value):
    return value.to_local() if hasattr(value, "to_local") else value


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    base = "/9950backfile/zhangem/outputs/.codex_anchor_ckpt_fsdp2_validation_20260814"
    checkpoint = os.path.join(base, "global_step_9")
    if rank == 0 and os.path.exists(base):
        resolved = os.path.realpath(base)
        expected = os.path.realpath("/9950backfile/zhangem/outputs/.codex_anchor_ckpt_fsdp2_validation_20260814")
        if resolved != expected:
            raise RuntimeError(f"Refusing to clean unexpected validation path: {resolved}")
        shutil.rmtree(resolved)
    dist.barrier()

    torch.manual_seed(4321)
    model = ToyFSDP2FrozenBackbone().to(device)
    fully_shard(model)
    model.backbone.weight.requires_grad_(False)
    optimizer = torch.optim.AdamW([model.anchor], lr=1e-2)
    loss = model(torch.ones(4, 16, device=device)).square().mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    saved_anchor = local_tensor(model.anchor).detach().clone()
    saved_exp_avg = local_tensor(optimizer.state[model.anchor]["exp_avg"]).detach().clone()
    state = {"model": model, "optimizer": optimizer, "extra_state": {"global_step": 9, "rank": rank}}
    TrainableDistributedCheckpointer.save(base, state, global_steps=9)

    with torch.no_grad():
        model.anchor.fill_(23.0)
        model.backbone.weight.fill_(17.0)
    optimizer.state[model.anchor]["exp_avg"].zero_()
    loaded = {"model": model, "optimizer": optimizer, "extra_state": {}}
    TrainableDistributedCheckpointer.load(checkpoint, loaded)

    assert torch.equal(local_tensor(model.anchor), saved_anchor)
    assert torch.equal(local_tensor(optimizer.state[model.anchor]["exp_avg"]), saved_exp_avg)
    assert torch.all(local_tensor(model.backbone.weight) == 17.0)
    assert loaded["extra_state"]["global_step"] == 9

    dist.barrier()
    if rank == 0:
        size = sum(item.stat().st_size for item in os.scandir(checkpoint) if item.is_file())
        print(f"FSDP2_TRAINABLE_CHECKPOINT_TEST_OK checkpoint={checkpoint} top_level_bytes={size}")
        print("files=", sorted(os.listdir(checkpoint)))
        shutil.rmtree(base)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
