# RoPE TileLang FP32 实现详解

> 本文档逐行讲解 [ops/rope_tilelang_fp32.py](file:///home/zem/HiLS-Attention-main/ops/rope_tilelang_fp32.py) 的实现。
>
> 该文件用 **TileLang** DSL 实现了 **RoPE（Rotary Position Embedding，旋转位置编码）** kernel，计算在 FP32 下完成、输入输出为 BF16，并提供了 PyTorch autograd 封装、正确性测试与延迟基准。
>
> 虽然本文件实现的是 RoPE 而非 FlashAttention，但二者共用同一套 **TileLang 编程模型**（分块、fragment、并行循环、kernel 调度）。因此本文在讲解 RoPE 之前，会先用一节系统讲清 **FlashAttention 的核心思想**，再讲 **TileLang 的语法**，最后对代码逐行剖析——这样你可以把“算法思想 → DSL 语法 → 具体代码”三者对应起来。

---

## 目录

1. [背景：RoPE 与 FlashAttention 的思想](#1-背景rope-与-flashattention-的思想)
2. [TileLang 语法速查](#2-tilelang-语法速查)
3. [逐行讲解](#3-逐行讲解)
   - 3.1 [导入与模块头](#31-导入与模块头)
   - 3.2 [`rope_fwd_kernel`：双 tensor RoPE kernel](#32-rope_fwd_kernel双-tensor-rope-kernel)
   - 3.3 [`rope_rotary_pos_emb`：Python wrapper](#33-rope_rotary_pos_emb-python-wrapper)
   - 3.4 [`single_tensor_rope_kernel`：单 tensor 通用 kernel](#34-single_tensor_rope_kernel单-tensor-通用-kernel)
   - 3.5 [`_single_tensor_rope`：wrapper 与 dtype 适配](#35-_single_tensor_ropewrapper-与-dtype-适配)
   - 3.6 [`SingleTensorRoPEFunction`：autograd 前向/反向](#36-singletensorropefunctionautograd-前向反向)
   - 3.7 [`rotate_half_torch`：PyTorch 参考实现](#37-rotate_half_torchpytorch-参考实现)
   - 3.8 [正确性测试 `test_single_tensor_rope_correctness`](#38-正确性测试-test_single_tensor_rope_correctness)
   - 3.9 [性能基准 `test_single_tensor_rope_performance`](#39-性能基准-test_single_tensor_rope_performance)
   - 3.10 [Liger 对比测试（已禁用）](#310-liger-对比测试已禁用)
   - 3.11 [`__main__` 入口](#311-__main__-入口)
4. [运行结果](#4-运行结果)
5. [总结与延伸](#5-总结与延伸)

---

## 1. 背景：RoPE 与 FlashAttention 的思想

### 1.1 RoPE：旋转位置编码

Transformer 原始位置编码是“加”上去的（绝对位置）。RoPE（Su et al., 2021）的核心想法是：**把位置信息编码为对 query/key 向量的旋转**，从而让内积 `q·k` 只依赖于相对位置 `m−n`，且天然适合外推。

设 head 维度为 `D`（偶数），把向量按前后两半拆分：

```
x = [x1, x2, ..., x_{D/2}  |  x_{D/2+1}, ..., x_D]
      └──── first half ────┘  └──── second half ────┘
```

对位置 `m`，定义频率 `θ_i = 10000^{-2i/D}`，对每一对 `(x_i, x_{i+D/2})` 做一次二维旋转：

```
o_i      = x_i * cos(mθ_i) − x_{i+D/2} * sin(mθ_i)
o_{i+D/2}= x_{i+D/2} * cos(mθ_i) + x_i * sin(mθ_i)
```

等价地写成“rotate_half”形式（Llama / GPT-NeoX 惯例）：

```
rotate_half(x) = concat(-x_second, x_first)
out = x * cos + rotate_half(x) * sin
```

其中 `cos`/`sin` 通常是 `cat(freqs, freqs)` 形状 `(L, D)`，即前后两半相同。本文件的 kernel 同时支持“前后半相同”（`rope_fwd_kernel`）与“前后半独立”（`single_tensor_rope_kernel`）两种约定。

> **为什么要在 FP32 计算？** BF16 的乘加在长序列、多 head 累加时会有精度损失。本 kernel 把输入 BF16 cast 到 FP32 做旋转运算，再 cast 回 BF16 输出，兼顾显存/带宽与数值精度——这也是文件名 `_fp32` 的由来。

### 1.2 FlashAttention 的核心思想（顺带讲解）

本文件不含 attention 计算，但用户特别要求理解 FlashAttention 思想。FlashAttention（Dao, 2022）是 TileLang 编程模型最典型的应用场景，理解它有助于掌握 TileLang 的设计哲学。其三大支柱：

#### (1) IO-aware：在 SRAM 里算，别老跑 HBM

GPU 显存层级：

| 层级      | 容量    | 带宽       |
|-----------|---------|------------|
| HBM（显存）| 几十 GB | ~1–2 TB/s  |
| SRAM（共享内存）| 每块 ~100–200 KB | ~10–20 TB/s |

标准 attention 把 `N×N` 的注意力矩阵写回 HBM 再读出来，成为瓶颈。FlashAttention 把 Q、K、V **分块（tile）** 装进 SRAM，整块计算只在 SRAM 里流动，最终只把输出写回 HBM。**“少跑 HBM”比“少算 FLOPs”更重要**——这是 IO-aware 的本质。

#### (2) Online Softmax：不物化 N×N 矩阵也能算 softmax

标准 softmax 需要 `exp(x − max)`，而 `max` 需要遍历整行。如果分块处理，怎么在没看到整行时就更新？关键技巧是 **online softmax**（Milakov & Gimelshein, 2018）：

维护每个 query tile 的两个 running 量：
- `m`：到目前为止见过的最大值
- `l`：到目前为止的 exp 求和

每处理一个新的 K/V tile：
```
m_new = max(m_old, m_tile)
l_new = l_old * exp(m_old − m_new) + l_tile * exp(m_tile − m_new)
O_new = O_old * (l_old * exp(m_old − m_new) / l_new) + P_tile * V_tile * exp(m_tile − m_new) / l_new
```

这样不需要把整个 `N×N` 矩阵存下来，分块流式累积即可。**“分块 + 在线归约”是 FlashAttention 的算法核心。**

#### (3) Backward 不存矩阵，重算

前向没存 attention 矩阵，反向怎么拿梯度？答案：**不存，重算**。反向时再用同样的分块算法走一遍，重新生成需要的中间值。用算力换显存——这也让 FlashAttention 的显存从 `O(N²)` 降到 `O(N)`。

#### 与 TileLang 的对应

| FlashAttention 概念 | TileLang 原语 |
|---------------------|---------------|
| 启动 block 网格     | `T.Kernel(num_blocks, threads=...)` |
| 把 tile 装进 SRAM   | `T.alloc_fragment(...)` / `T.alloc_shared(...)` |
| 在 tile 上并行计算  | `T.Parallel(...)` / `T.gemm(...)` |
| 在 tile 间累积      | 普通的 `for` 循环 + fragment 累加 |
| HBM↔SRAM 搬运       | `T.copy(src, dst)` |

本 RoPE kernel 比 attention 简单得多：**没有归约、没有在线 softmax**，但同样用到“kernel 网格 + fragment + 并行循环”这三件套。掌握它们，再读 FlashAttention kernel 就只剩“多了一层循环和 online softmax 更新”而已。

---

## 2. TileLang 语法速查

TileLang 是基于 TVM/TIR 的高层 GPU kernel DSL。核心抽象是：**你写“一个 block 干什么”，它帮你生成 CUDA/Triton 代码并 JIT 编译。**

| 语法 | 含义 |
|------|------|
| `@tilelang.jit(out_idx=[...], pass_configs={...})` | 把一个“生成 prim_func 的 Python 函数”标记为可 JIT 编译的 kernel 工厂。`out_idx` 指明哪些入参是输出 buffer；`pass_configs` 设编译选项。 |
| `tilelang.PassConfigKey.TL_ENABLE_FAST_MATH` | 编译期开关：开启快速数学（近似倒数/平方根等），提升速度。 |
| `@T.prim_func` | 把嵌套函数标记为 TVM 的 prim_func（与 `@tilelang.jit` 配合使用）。 |
| `T.Buffer(shape, dtype)` | 声明一个 buffer（对应一块 GPU 显存），形状 + dtype 用字符串如 `"bfloat16"`/`"float32"`。 |
| `T.Kernel(num_blocks, threads=N)` | 启动 `num_blocks` 个 thread block，每块 `threads` 个线程；`with` 块里 `bx` 是 block 索引。 |
| `T.alloc_fragment(shape, dtype)` | 在 block 内分配 fragment（通常映射到寄存器/共享内存），用于 tile 内的中间计算。 |
| `T.alloc_shared(shape, dtype)` | 显式分配共享内存 tile（本文件没用，FlashAttention 会用）。 |
| `T.Parallel(n)` | “把长度为 `n` 的循环并行分给本 block 的线程” —— 类似 `#pragma omp parallel for`。 |
| `T.unroll(n)` | 把循环完全展开（编译期）。 |
| `T.cast(x, dtype)` | 显式类型转换。 |
| `T.serial(n)` / 普通 `for` | 串行循环（普通 `for ... in range(...)` 也可，会自动串行）。 |

> 约定：buffer 索引用方括号 `Q[b, l, h, d]`；fragment 也是 `frag[d]`。所有计算最终落到对 buffer 的读写。

---

## 3. 逐行讲解

### 3.1 导入与模块头

```python
"""
Rotary Position Embedding
================================================
TileLang implementation of RoPE.

Author: Xinyu Wei
"""

import torch
import tilelang
from tilelang import language as T
```

- 文件 docstring 说明用途与作者。
- `import torch`：用 PyTorch 创建 tensor、做 autograd、跑正确性对比。
- `import tilelang`：DSL 主包。
- `from tilelang import language as T`：把所有 DSL 原语（`T.prim_func`、`T.Kernel`、`T.Parallel` …）收到命名空间 `T` 下，写起来短。

### 3.2 `rope_fwd_kernel`：双 tensor RoPE kernel

这是第一个 kernel，同时旋转 Q 和 K。

```python
@tilelang.jit(
    out_idx=[4, 5],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    })
def rope_fwd_kernel(batch, seq_len, heads_q, heads_k, dim, threads=64):
```

- `@tilelang.jit(...)`：装饰一个“参数化的 kernel 工厂”。被装饰的函数返回一个 `prim_func`，TileLang 会按这组参数（`batch, seq_len, ...`）特化并 JIT 编译成 GPU 代码。
- `out_idx=[4, 5]`：第 4、5 个入参（`Q_out`, `K_out`）是输出 buffer —— TileLang 据此决定哪些 buffer 由 kernel 写、由谁分配。
- `pass_configs={TL_ENABLE_FAST_MATH: True}`：开启快速数学。
- 工厂参数：`batch, seq_len, heads_q, heads_k, dim` 是形状参数，`threads=64` 每块线程数。

```python
    dtype = "bfloat16"
    compute_dtype = "float32"

    assert dim % 2 == 0
    dim_half = dim // 2
```

- `dtype`：输入/输出数据类型 BF16（省显存、省带宽）。
- `compute_dtype`：实际旋转计算用 FP32（保精度）—— 这正是文件名 `_fp32` 的含义。
- `dim % 2 == 0`：RoPE 要求 head 维度为偶数，否则无法配对旋转。
- `dim_half`：前后两半的长度。

```python
    @T.prim_func
    def main(
        Q:   T.Buffer((batch, seq_len, heads_q, dim), dtype),
        K:   T.Buffer((batch, seq_len, heads_k, dim), dtype),
        Cos: T.Buffer((batch, seq_len, dim), dtype),
        Sin: T.Buffer((batch, seq_len, dim), dtype),
        Q_out: T.Buffer((batch, seq_len, heads_q, dim), dtype),
        K_out: T.Buffer((batch, seq_len, heads_k, dim), dtype),
    ):
```

- `@T.prim_func`：真正的 kernel 函数（TIR prim_func）。`main` 的签名同时是 kernel 的入参声明。
- 6 个 `T.Buffer`：Q、K、Cos、Sin 是输入；Q_out、K_out 是输出（与 `out_idx=[4,5]` 对应）。形状用闭包里的工厂参数表达。
- 注意 Cos/Sin 形状是 `(B, L, D)`，即“每个 token 一组 cos/sin”。

```python
        total_tokens = batch * seq_len

        with T.Kernel(total_tokens, threads=threads) as bx:
            b_idx = bx // seq_len
            l_idx = bx % seq_len
```

- `total_tokens = batch * seq_len`：把 batch 和序列维度合并成一个一维网格。**每个 block 处理一个 token**（一个 `(batch, seq)` 位置）。
- `T.Kernel(total_tokens, threads=threads) as bx`：启动 `total_tokens` 个 block，`bx` 是 block 索引。这正是 FlashAttention 里“每个 block 处理一个 tile”的同一套抽象，只是这里 tile = 单个 token 的所有 head。
- `b_idx = bx // seq_len`、`l_idx = bx % seq_len`：从一维索引还原出 batch 索引和序列索引。`//` 整除、`%` 取余。

```python
            cos_frag = T.alloc_fragment((dim,), compute_dtype)
            sin_frag = T.alloc_fragment((dim,), compute_dtype)

            for d in T.Parallel(dim):
                cos_frag[d] = T.cast(Cos[b_idx, l_idx, d], compute_dtype)
                sin_frag[d] = T.cast(Sin[b_idx, l_idx, d], compute_dtype)
```

- `T.alloc_fragment((dim,), compute_dtype)`：分配长度为 `dim` 的 fragment，dtype 为 FP32。**这是把 cos/sin 从 HBM 装进 SRAM/寄存器的“tile”**，对应 FlashAttention 里把 K/V tile 装进共享内存。
- `T.Parallel(dim)`：把这 `dim` 次加载并行分给本 block 的 `threads` 个线程。
- `T.cast(..., compute_dtype)`：把 BF16 的 cos/sin 提升到 FP32。**每个 token 只读一次 cos/sin，之后所有 head 共用** —— 这是经典的“加载到 fragment 复用”优化，避免每个 head 重复跑 HBM。

```python
            for h in T.unroll(heads_q):
                for d in T.Parallel(dim // 2):
                    idx_1 = d
                    idx_2 = d + dim_half

                    q1 = T.cast(Q[b_idx, l_idx, h, idx_1], compute_dtype)
                    q2 = T.cast(Q[b_idx, l_idx, h, idx_2], compute_dtype)

                    c_val = cos_frag[idx_1]
                    s_val = sin_frag[idx_1]

                    o1 = q1 * c_val - q2 * s_val
                    o2 = q2 * c_val + q1 * s_val

                    Q_out[b_idx, l_idx, h, idx_1] = T.cast(o1, dtype)
                    Q_out[b_idx, l_idx, h, idx_2] = T.cast(o2, dtype)
```

- `T.unroll(heads_q)`：在 head 维度完全展开。head 数通常不大（8/32），展开后编译器可更好调度。
- `T.Parallel(dim // 2)`：在“配对维度”上并行，每个线程处理一对 `(idx_1, idx_2)`。
- `idx_1 = d`（前半下标）、`idx_2 = d + dim_half`（后半下标）：把 head 向量拆成前后两半配对。
- `q1`/`q2`：把这对 Q 元素 cast 到 FP32。
- `c_val`/`s_val`：从 fragment 读 cos/sin。**这里用 `idx_1`（前半下标）取**，因为本 kernel 假设 `cos = cat(freqs, freqs)`，前后两半相同，所以 `cos[d] == cos[d+dim_half]`，取前半即可。
- `o1 = q1*c − q2*s`、`o2 = q2*c + q1*s`：正是 §1.1 推导的二维旋转公式。
- 最后 `T.cast(..., dtype)` 写回 BF16 输出。

```python
            for h in T.unroll(heads_k):
                for d in T.Parallel(dim // 2):
                    idx_1 = d
                    idx_2 = d + dim_half

                    k1 = T.cast(K[b_idx, l_idx, h, idx_1], compute_dtype)
                    k2 = T.cast(K[b_idx, l_idx, h, idx_2], compute_dtype)

                    c_val = cos_frag[idx_1]
                    s_val = sin_frag[idx_1]

                    o1 = k1 * c_val - k2 * s_val
                    o2 = k2 * c_val + k1 * s_val

                    K_out[b_idx, l_idx, h, idx_1] = T.cast(o1, dtype)
                    K_out[b_idx, l_idx, h, idx_2] = T.cast(o2, dtype)

    return main
```

- K 的处理与 Q 完全同构，只是 head 数为 `heads_k`、用 `K`/`K_out`。
- `return main`：工厂函数返回 prim_func。`@tilelang.jit` 会在第一次调用时编译它，之后缓存。

### 3.3 `rope_rotary_pos_emb`：Python wrapper

```python
def rope_rotary_pos_emb(q, k, cos, sin):
    B, L, HQ, D = q.shape
    H = k.shape[2]
    kernel_func = rope_fwd_kernel(B, L, HQ, H, D)
    return kernel_func(q, k, cos, sin)
```

- 从 tensor 形状解出 `B, L, HQ, D`，从 `k` 解出 `H`（key 的 head 数）。
- `rope_fwd_kernel(B, L, HQ, H, D)`：调用被 `@tilelang.jit` 装饰的工厂，返回（已编译的）kernel 句柄。
- `kernel_func(q, k, cos, sin)`：直接传 tensor 执行。TileLang 会处理 torch tensor ↔ buffer 的绑定。由于 `out_idx=[4,5]`，它会自动返回 Q_out、K_out 两个 tensor。

### 3.4 `single_tensor_rope_kernel`：单 tensor 通用 kernel

第二个 kernel 更通用：只旋转一个 tensor（典型用于 Q/K 分开调用），且**允许 cos 前后半不同**。

```python
@tilelang.jit(
    out_idx=[3],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    })
def single_tensor_rope_kernel(batch, seq_len, heads, dim, cos_sin_dtype="bfloat16", threads=64):
    x_dtype = "bfloat16"
    compute_dtype = "float32"

    assert dim % 2 == 0
    dim_half = dim // 2
```

- `out_idx=[3]`：只有 `X_out` 一个输出。
- 多了 `cos_sin_dtype` 参数：cos/sin 可能是 BF16 也可能是 FP32（见 wrapper §3.5）。
- `x_dtype` 固定 BF16，`compute_dtype` 固定 FP32。

```python
    @T.prim_func
    def main(
        X:     T.Buffer((batch, seq_len, heads, dim), x_dtype),
        X_cos: T.Buffer((seq_len, dim), cos_sin_dtype),
        X_sin: T.Buffer((seq_len, dim), cos_sin_dtype),
        X_out: T.Buffer((batch, seq_len, heads, dim), cos_sin_dtype),
    ):
```

- 注意 `X_cos`/`X_sin` 形状是 `(seq_len, dim)` —— **没有 batch 维**，假设 cos/sin 在 batch 间共享（这是真实 Qwen3RotaryEmbedding 的行为，见 §3.5 wrapper）。
- `X_out` 的 dtype 跟 cos_sin 走（wrapper 里会再 cast 回 x 的 dtype）。

```python
        total_tokens = batch * seq_len
        with T.Kernel(total_tokens, threads=threads) as bx:
            b_idx = bx // seq_len
            l_idx = bx % seq_len

            cos_lo = T.alloc_fragment((dim_half,), compute_dtype)
            sin_lo = T.alloc_fragment((dim_half,), compute_dtype)
            cos_hi = T.alloc_fragment((dim_half,), compute_dtype)
            sin_hi = T.alloc_fragment((dim_half,), compute_dtype)

            for d in T.Parallel(dim_half):
                cos_lo[d] = T.cast(X_cos[l_idx, d], compute_dtype)
                sin_lo[d] = T.cast(X_sin[l_idx, d], compute_dtype)
                cos_hi[d] = T.cast(X_cos[l_idx, d + dim_half], compute_dtype)
                sin_hi[d] = T.cast(X_sin[l_idx, d + dim_half], compute_dtype)
```

- 同样的网格设计：每 block 一个 token。
- 这次分配 **4 个 fragment**：`cos_lo/sin_lo`（前半）、`cos_hi/sin_hi`（后半），因为本 kernel 支持 cos 前后半独立。
- 加载时 `X_cos[l_idx, ...]`（不带 batch 维），并分别取 `d`（前半）和 `d+dim_half`（后半）。
- 当 `cos = cat(freqs, freqs)` 时 `cos_lo == cos_hi`，结果与第一个 kernel 一致；但写成这样更通用、更直白地对应“前后半各一组 cos/sin”。

```python
            for h in T.unroll(heads):
                for d in T.Parallel(dim_half):
                    x1 = T.cast(X[b_idx, l_idx, h, d], compute_dtype)
                    x2 = T.cast(X[b_idx, l_idx, h, d + dim_half], compute_dtype)

                    o1 = x1 * cos_lo[d] - x2 * sin_lo[d]
                    o2 = x2 * cos_hi[d] + x1 * sin_hi[d]

                    X_out[b_idx, l_idx, h, d] = T.cast(o1, cos_sin_dtype)
                    X_out[b_idx, l_idx, h, d + dim_half] = T.cast(o2, cos_sin_dtype)

    return main
```

- `x1`/`x2`：前半 / 后半元素，提升到 FP32。
- `o1 = x1*cos_lo − x2*sin_lo`：前半旋转（用前半的 cos/sin）。
- `o2 = x2*cos_hi + x1*sin_hi`：后半旋转（用后半的 cos/sin）。
- 写回时 cast 到 `cos_sin_dtype`。

> 对比两个 kernel：`rope_fwd_kernel` 把 cos 前后半视作相同（取 `idx_1`），一次处理 Q+K；`single_tensor_rope_kernel` 显式区分前后半 cos/sin，一次只处理一个 tensor。后者是实际被 `__main__` 调用的版本。

### 3.5 `_single_tensor_rope`：wrapper 与 dtype 适配

```python
def _single_tensor_rope(x, x_cos, x_sin):
    B, L, H, D = x.shape
    if x_cos.ndim == 3:
        assert x_cos.shape[0] == 1 and x_sin.shape[0] == 1, \
            f"Expected batch dim of x_cos/x_sin to be 1 when ndim==3 (shared across batch), \
             but got x_cos.shape={tuple(x_cos.shape)}, x_sin.shape={tuple(x_sin.shape)}"
        x_cos = x_cos[0]
        x_sin = x_sin[0]
    x_cos = x_cos.contiguous()
    x_sin = x_sin.contiguous()
    if x_cos.dtype == torch.float32:
        cos_sin_dtype = "float32"
    else:
        cos_sin_dtype = "bfloat16"
    kernel_func = single_tensor_rope_kernel(B, L, H, D, cos_sin_dtype)
    return kernel_func(x, x_cos, x_sin)
```

- 解 shape：`B, L, H, D`。
- **维度归一化**：如果 cos/sin 是 3 维 `(1, L, D)`，断言 batch 维必须是 1（即跨 batch 共享），然后 `x_cos[0]` 压成 2 维 `(L, D)`，匹配 kernel 的 buffer 形状。注释里提到这匹配真实 `Qwen3RotaryEmbedding` 的行为。
- `.contiguous()`：保证内存连续，TileLang buffer 要求连续内存。
- **dtype 适配**：cos 是 FP32 就传 `"float32"`，否则 `"bfloat16"`。
- 取 kernel 句柄并执行，返回输出 tensor。

### 3.6 `SingleTensorRoPEFunction`：autograd 前向/反向

为了让 RoPE 能进 PyTorch autograd 图（训练需要），定义一个 `torch.autograd.Function`。

```python
class SingleTensorRoPEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, x_cos, x_sin):
        x_dtype = x.dtype
        x_out = _single_tensor_rope(x, x_cos, x_sin)
        ctx.save_for_backward(x_cos, x_sin)
        ctx.x_dtype = x_dtype
        return x_out.to(x_dtype)
```

- `forward`：记下 `x` 的原始 dtype（BF16），调 kernel 算 `x_out`，用 `ctx.save_for_backward` 存住 `x_cos`/`x_sin`（反向要用），最后 `.to(x_dtype)` 确保输出 dtype 与输入一致。
- `ctx.x_dtype = x_dtype`：把 dtype 塞进 ctx，反向取用。

```python
    @staticmethod
    def backward(ctx, dx):
        x_cos, x_sin = ctx.saved_tensors
        dx_out = _single_tensor_rope(dx.contiguous(), x_cos, -x_sin)
        return dx_out.to(ctx.x_dtype), None, None
```

- **反向的数学依据**：RoPE 是一个**正交旋转**。对一对 `(x1, x2)`，旋转矩阵 `R = [[c, −s],[s, c]]` 满足 `R^T R = I`。前向 `o = R x`，反向 `dx = R^T grad = R(−s) grad`（把 `s → −s` 即转置）。所以反向就是**对梯度再做一次 RoPE，但 sin 取负**。
- `dx.contiguous()`：梯度可能非连续，先 contiguous。
- `_single_tensor_rope(dx, x_cos, -x_sin)`：复用同一个 kernel，传 `-x_sin`。
- 返回三元组 `(dx, None, None)`：对应 forward 的三个输入 `(x, x_cos, x_sin)`，cos/sin 不需要梯度。

```python
def single_tensor_rope_autograd(x, x_cos, x_sin):
    """
    Args:
        x:     (B, L, H, D)  - BF16
        x_cos: (B, L, D) - BF16 or FP32
        x_sin: (B, L, D) - BF16 or FP32
    Returns:
        x_out: (B, L, H, D) - same dtype as x (BF16)
    """
    return SingleTensorRoPEFunction.apply(x, x_cos, x_sin)
```

- 对外友好接口，docstring 写清形状约定，调 `Function.apply`。

### 3.7 `rotate_half_torch`：PyTorch 参考实现

```python
def rotate_half_torch(x):
    """Rotates half the hidden dims of the input. testof PyTorch ."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)
```

- Llama 风格的 rotate_half：后半取负放前面、前半放后面。
- 仅用作**正确性参考**，不参与 kernel 运行。

### 3.8 正确性测试 `test_single_tensor_rope_correctness`

```python
def test_single_tensor_rope_correctness():
    torch.manual_seed(42)
    device = "cuda"

    B = 1
    L = 512 * 16       # sequence length
    H = 8
    D = 64

    print(f"\nTesting Single Tensor RoPE: B={B}, L={L}, H={H}, D={D}")
```

- 固定随机种子，`L = 8192`（512×16），单 batch。

```python
    x = torch.randn((B, L, H, D), dtype=torch.bfloat16, device=device)

    freqs = torch.randn((1, L, D // 2), dtype=torch.float32, device=device)
    emb = torch.cat((freqs, freqs), dim=-1)  # (1, L, D)
    cos = emb.cos().to(torch.bfloat16)
    sin = emb.sin().to(torch.bfloat16)
```

- 构造 BF16 输入 `x`。
- 模拟真实 RoPE 的 cos/sin 构造：随机频率 `freqs (1, L, D/2)`，`cat(freqs, freqs)` 得 `(1, L, D)`，取 cos/sin 后转 BF16。这就是 `cos_lo == cos_hi` 的情形。

```python
    def get_abs_err(a, b):
        mask = (a > -1e5) & (b > -1e5)
        if mask.sum() == 0: return 0.0
        return (a[mask] - b[mask]).abs().max().item()

    def get_err_ratio(a, b):
        mask = (a > -1e5) & (b > -1e5)
        if mask.sum() == 0: return 0.0
        err = (a[mask] - b[mask]).square().mean().sqrt().item()
        base = (a[mask]).square().mean().sqrt().item()
        return err / (base + 1e-12)

    def assert_close(prefix, ref, tri, ratio=0.005):
        abs_err = get_abs_err(ref, tri)
        rel_ratio = get_err_ratio(ref, tri)
        msg = f"{prefix} diff: {abs_err:.6f} ratio: {rel_ratio:.6f}"
        print(msg)
        assert rel_ratio < ratio, msg
```

- 三个工具函数：**绝对误差**、**相对误差比**（RMSE / base RMSE）、带阈值的 `assert_close`（默认相对比 < 0.5%）。
- `mask = (a > -1e5) & (b > -1e5)`：过滤掉极端值/inf，避免污染统计。

**Test 1：标准前向**

```python
    x_rot = x.transpose(1, 2).float()              # (B, H, L, D)
    cos_ref = cos.float().unsqueeze(1)              # (B, 1, L, D)
    sin_ref = sin.float().unsqueeze(1)
    x_out_ref = ((x_rot * cos_ref) + (rotate_half_torch(x_rot) * sin_ref)) \
                .transpose(1, 2).contiguous().to(torch.bfloat16)

    x_out_kernel = single_tensor_rope_autograd(x, cos, sin)

    assert_close("Single RoPE FWD", x_out_ref.float(), x_out_kernel.float())
```

- PyTorch 参考：`x * cos + rotate_half(x) * sin`，全程 FP32，最后转 BF16。
- kernel 版本：调 autograd 接口。
- `assert_close` 对比，FWD 容差 0.5%。

**Test 2：不同 head_dim（D=128, H=1）**

```python
    D2 = 128
    H2 = 1
    x2 = torch.randn((B, L, H2, D2), dtype=torch.bfloat16, device=device)
    freqs2 = torch.randn((1, L, D2 // 2), dtype=torch.float32, device=device)
    emb2 = torch.cat((freqs2, freqs2), dim=-1)
    cos2 = emb2.cos().to(torch.bfloat16)
    sin2 = emb2.sin().to(torch.bfloat16)
    ...
    assert_close("Single RoPE (D=128, H=1)", x2_out_ref.float(), x2_out_kernel.float())
```

- 验证 kernel **对 head_dim 的通用性**（D=128 而非 64）。因为 kernel 用 `dim` 参数化、用 `T.Parallel(dim//2)`，任何偶数 D 都能跑。

**Test 3：不同 head 数（H=32）**

```python
    H3 = 32
    x3 = torch.randn((B, L, H3, D), dtype=torch.bfloat16, device=device)
    ...
    assert_close("Single RoPE (H=32)", x3_out_ref.float(), x3_out_kernel.float())
```

- 验证对 head 数的通用性。`T.unroll(heads)` 让 head 数任意。

**Test 4：反向梯度一致性**

```python
    x_grad = torch.randn((B, L, H, D), dtype=torch.float32, device=device, requires_grad=True)
    grad_out = torch.randn((B, L, H, D), dtype=torch.float32, device=device)

    x_rot_ref = x_grad.transpose(1, 2)
    x_out_pt = ((x_rot_ref * cos_ref) + (rotate_half_torch(x_rot_ref) * sin_ref)) \
               .transpose(1, 2).contiguous()
    x_out_pt.backward(grad_out)
    dx_ref = x_grad.grad.clone()

    x_kernel = x_grad.detach().to(torch.bfloat16).requires_grad_(True)
    x_out_ag = single_tensor_rope_autograd(x_kernel, cos, sin)
    x_out_ag.backward(grad_out.to(torch.bfloat16))
    dx_kernel = x_kernel.grad.clone()

    assert_close("Single RoPE BWD dX", dx_ref.float(), dx_kernel.float(), ratio=0.01)
```

- PyTorch 参考：FP32 前向 + `backward(grad_out)` 得 `dx_ref`。
- kernel 版本：把 `x_grad` detach 转 BF16 再 `requires_grad_`，跑 autograd 前向/反向得 `dx_kernel`。
- 反向容差放宽到 1%（BF16 反向会引入额外误差）。
- 这一步验证了 §3.6 “反向 = 用 `-sin` 再做一次 RoPE” 的正确性。

### 3.9 性能基准 `test_single_tensor_rope_performance`

```python
def test_single_tensor_rope_performance():
    print("\n" + "=" * 60)
    print(" Single Tensor RoPE Latency Benchmark")
    print("=" * 60)

    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16

    B = 8
    L = 4096
    H = 8
    D = 128
```

- 基准配置 `B=8, L=4096, H=8, D=128`。

```python
    x = torch.randn((B, L, H, D), dtype=dtype, device=device)
    freqs = torch.randn((1, L, D // 2), dtype=torch.float32, device=device)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(dtype)
    sin = emb.sin().to(dtype)

    _ = _single_tensor_rope(x, cos, sin)
    torch.cuda.synchronize()
```

- 构造输入；先跑一次 kernel 做 **JIT 编译预热**（第一次调用会编译），再 `synchronize` 确保编译完成，避免计入基准。

```python
    def pytorch_baseline():
        x_rot = x.transpose(1, 2)
        cos_u = cos.unsqueeze(1)
        sin_u = sin.unsqueeze(1)
        return ((x_rot * cos_u) + (rotate_half_torch(x_rot) * sin_u)) \
               .transpose(1, 2).contiguous()

    def tilelang_kernel_only():
        return _single_tensor_rope(x, cos, sin)
```

- 两个被测函数：PyTorch 向量化实现 vs TileLang kernel。

```python
    def benchmark(func, name, num_iters=100, num_warmup=20):
        for _ in range(num_warmup):
            func()
        torch.cuda.synchronize()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(num_iters):
            func()
        end_event.record()
        torch.cuda.synchronize()

        avg_time = start_event.elapsed_time(end_event) / num_iters
        print(f"  [{name}] Average Latency: {avg_time:.3f} ms")
        return avg_time
```

- `num_warmup=20`：warmup 让 GPU 进入高频率、缓存稳定。
- 用 `torch.cuda.Event` 计时（比 `time.time` 准，只测 GPU 时间）。
- `elapsed_time` 是毫秒，除以迭代数得平均。

```python
    print("-" * 50)
    t_baseline = benchmark(pytorch_baseline, "PyTorch path")
    t_kernel = benchmark(tilelang_kernel_only, "TileLang  kernel")

    print("-" * 50)
    print(f"  : {t_baseline / t_kernel:.2f}x")
    print("=" * 60)
```

- 打印两者延迟与加速比。实测约 **5.56x**（见 §4）。

### 3.10 Liger 对比测试（已禁用）

```python
# import torch.nn.functional as F
# from liger_kernel.transformers.rope import liger_rotary_pos_emb  # 依赖未安装，已移除

def test_rope_correctness():
    ...
```

- 文件原有两个用 `liger_kernel` 做对比的函数 `test_rope_correctness` / `test_rope_performance`。
- 因运行环境无 `liger_kernel` 且无网络安装，已把第 391–392 行的顶层 import 注释掉。这两个函数本身未被 `__main__` 调用（见 §3.11），所以不影响当前运行。
- 它们的逻辑与 single_tensor 测试类似，只是把参考实现换成 Liger 的 `liger_rotary_pos_emb`。

### 3.11 `__main__` 入口

```python
if __name__ == "__main__":
    test_single_tensor_rope_correctness()

    test_single_tensor_rope_performance()
    # test_rope_correctness()
    # test_rope_performance()
```

- 直接 `python ops/rope_tilelang_fp32.py` 时执行：先跑正确性（4 个子测试），再跑性能基准。
- 下面两个 Liger 对比测试默认注释关闭。

---

## 4. 运行结果

使用 conda 环境 `tile` 执行（已临时移除未安装的 `liger_kernel` 顶层导入）：

**正确性（全部通过）：**

| 测试 | abs diff | rel ratio | 阈值 |
|------|----------|-----------|------|
| 标准 RoPE 前向 | 0.000000 | 0.000000 | 0.5% |
| D=128, H=1 | 0.000000 | 0.000000 | 0.5% |
| H=32, D=64 | 0.000000 | 0.000000 | 0.5% |
| 反向梯度 dX | 0.030275 | 0.002349 | 1% |

**性能（B=8, L=4096, H=8, D=128）：**

| 实现 | 平均延迟 |
|------|----------|
| PyTorch path | 1.938 ms |
| TileLang kernel | 0.348 ms |
| **加速比** | **5.56x** |

---

## 5. 总结与延伸

### 关键设计点

1. **IO 优化**：每个 token 的 cos/sin 只从 HBM 读一次到 fragment，所有 head 复用 —— 这正是 FlashAttention “把 tile 装进 SRAM 复用”思想的最小范例。
2. **精度策略**：BF16 存储与传输、FP32 计算旋转，兼顾带宽与数值精度。
3. **参数化 kernel**：通过工厂函数 + `@tilelang.jit` 把形状特化进编译产物，运行时零开销。
4. **autograd 正确性**：利用 RoPE 的正交性，反向用 `-sin` 复用前向 kernel，无需另写反算 kernel。
5. **一维网格 + 维度还原**：`bx // seq_len` / `bx % seq_len` 是 GPU kernel 里把多维问题压成一维网格的常见套路。

### 从 RoPE 到 FlashAttention

如果把本文件的 kernel 升级成 FlashAttention，会发生什么？

- 网格：从“每个 block 一个 token”变成“每个 block 一个 **query tile**”。
- fragment：从“cos/sin”变成“Q tile、K tile、V tile、O tile、`m`（running max）、`l`（running sum）”。
- 内层循环：从“unroll heads”变成“**循环遍历 K/V tiles**，每个 tile 做一次 `T.gemm` 算 `P=QKᵀ`，再用 online softmax 公式更新 `m, l, O`”。
- 反向：本文件靠旋转正交性直接复用前向；FlashAttention 靠 **重算** 重建中间值。

可以看到，**本 RoPE kernel 已经包含了 TileLang 编程模型的全部基础原语**（`T.Kernel` / `T.alloc_fragment` / `T.Parallel` / `T.unroll` / `T.cast` / `T.Buffer`）。掌握它们之后，再读 FlashAttention 的 TileLang 实现就只是“多一层 K/V tile 循环 + online softmax 更新”而已。

### 参考阅读

- RoPE: Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*, 2021.
- FlashAttention: Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, 2022.
- Online Softmax: Milakov & Gimelshein, *Online normalizer calculation for softmax*, 2018.
- TileLang: DSL 文档与示例（`tilelang` 包）。
