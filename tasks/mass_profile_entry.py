"""Profiling entry: per-rank TileLang cache dir + single-GPU multi-rank pinning.

单卡双 rank 剖析用:
1. tilelang 缓存目录无文件锁,按 LOCAL_RANK 拆分,避免两 rank 并发 JIT 竞争;
2. CUDA_VISIBLE_DEVICES 只含一张卡时,把 local_rank>0 的 set_device 钳制到
   唯一可见卡(NCCL 支持多 rank 共享一卡),随后进入原 pretrain_with_ruler。
"""

import os

_rank = os.environ.get("LOCAL_RANK", "0")
os.environ.setdefault("TILELANG_CACHE_DIR", f"/dev/shm/tilelang_cache_r{_rank}")
os.makedirs(os.environ["TILELANG_CACHE_DIR"], exist_ok=True)

if os.environ.get("MASS_PIN_VISIBLE_GPUS", "0") == "1":
    import torch

    _orig_set_device = torch.cuda.set_device

    def _clamped_set_device(device):
        n = torch.cuda.device_count()
        if n <= 0:
            return _orig_set_device(device)
        if isinstance(device, torch.device):
            idx = device.index or 0
        else:
            try:
                idx = int(str(device).split(":")[-1])
            except ValueError:
                idx = 0
        return _orig_set_device(min(idx, n - 1))

    torch.cuda.set_device = _clamped_set_device

import pretrain_mass_with_ruler  # noqa: F401,E402  (MASS 注册与 env 钩子)
import pretrain_with_ruler  # noqa: E402

if __name__ == "__main__":
    pretrain_with_ruler.main()
