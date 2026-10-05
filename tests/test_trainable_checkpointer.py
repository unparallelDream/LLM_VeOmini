import os
import shutil

import torch
import torch.distributed as dist
from torch import nn

from utils.trainable_checkpointer import TrainableDistributedCheckpointer


class ToyFrozenBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(8, 8, bias=False)
        self.anchor = nn.Parameter(torch.linspace(-0.4, 0.3, 8))
        self.backbone.weight.requires_grad_(False)

    def forward(self, value):
        return self.backbone(value) + self.anchor


def main():
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    base = "/9950backfile/zhangem/outputs/.codex_anchor_ckpt_validation_20260814"
    checkpoint = os.path.join(base, "global_step_7")
    if rank == 0 and os.path.exists(base):
        resolved = os.path.realpath(base)
        expected = os.path.realpath("/9950backfile/zhangem/outputs/.codex_anchor_ckpt_validation_20260814")
        if resolved != expected:
            raise RuntimeError(f"Refusing to clean unexpected validation path: {resolved}")
        shutil.rmtree(resolved)
    dist.barrier()

    torch.manual_seed(1234)
    model = ToyFrozenBackbone()
    optimizer = torch.optim.AdamW([model.anchor], lr=1e-2)
    loss = model(torch.ones(2, 8)).square().mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    saved_anchor = model.anchor.detach().clone()
    saved_exp_avg = optimizer.state[model.anchor]["exp_avg"].detach().clone()
    state = {"model": model, "optimizer": optimizer, "extra_state": {"global_step": 7, "rank": rank}}
    TrainableDistributedCheckpointer.save(base, state, global_steps=7)

    with torch.no_grad():
        model.anchor.fill_(23.0)
        model.backbone.weight.fill_(17.0)
    optimizer.state[model.anchor]["exp_avg"].zero_()
    loaded = {"model": model, "optimizer": optimizer, "extra_state": {}}
    TrainableDistributedCheckpointer.load(checkpoint, loaded)

    assert torch.equal(model.anchor, saved_anchor), (model.anchor, saved_anchor)
    assert torch.equal(optimizer.state[model.anchor]["exp_avg"], saved_exp_avg)
    assert torch.all(model.backbone.weight == 17.0), "Frozen backbone should not be present in checkpoint"
    assert loaded["extra_state"]["global_step"] == 7

    dist.barrier()
    if rank == 0:
        size = sum(item.stat().st_size for item in os.scandir(checkpoint) if item.is_file())
        print(f"TRAINABLE_CHECKPOINT_TEST_OK checkpoint={checkpoint} top_level_bytes={size}")
        print("files=", sorted(os.listdir(checkpoint)))
        shutil.rmtree(base)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
