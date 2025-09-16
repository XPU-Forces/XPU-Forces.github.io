---
layout: page
title: triton-x
permalink: /triton-x/
---

### 概述
国产硬件，例如Ascend, MLU逐渐开始不同业务线开始被应用。大多数业务的工程同学具有不同程度的CUDA背景，基于新的国产硬件进行深度开发，比如开发算子，学习成本很高；Triton-x的出发点即是为了降低用户对国产硬件的使用门槛。

Triton-X 会提供两层programming interface:
- Beginner
- Developer

### 方案设计  
triton-x是triton的中间层，是一组IR的集合，依托于MLIR生态，向上衔接triton-ir，向下衔接硬件方言，聚焦于硬件平台无关的性能优化，同时也会感知硬件优化，并且解决硬件差异化的问题，降低软硬件适配成本。

triton-x希望通过抽象hardware info + VISA(unified IR in MLIR)，使triton-x compiler能够在一个硬件无关的IR表示上，根据当前的特定硬件信息，完成对用户triton-x kernel代码的优化，从而进一步简化编写kernel的难度。
整体上，triton-x的interface与最近比较火的一个项目tile-lang 有些接近，这里先暂时引用tile的图，后续有时间了修改（triton-x的整体编程界面与tile-lang类似，但我们不会引入expert层（expert层整体上定位是cuda c的python化，本质还是cuda c/ascend c编程，用户需要深入了解硬件细节，需要做大量fine-grained工作）。
 
![triton-x](/images/triton-x.png "triton-x总体架构图"){:style="width: 100%"}

### Beginner  
用户提供一份比较初级的triton代码（尽可能大的tile size，或者无需提供tile size），将足够大的problem size直接提交给triton-x，由triton-x compiler提供不同异构硬件的性能保证，实现triton代码的复用，以下是两个简单例子。

### Examples  
#### vector_add  
> 原生triton vector_add实现:  
> https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html  

```python
import torch

@torch.compile()
def foo(input):
    res = torch.add(input, input)
    max_val = torch.max(res)
    res -= max_val
    


import torch
import triton
import triton.language as tl
from tritonx import ttx

ttx.set_hardware_backend("NPU")

@triton.autotuning
# 计算x + y，每个block计算tile_x + tile_y
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr): # tl.constexpr表示该参数为常数
    pid = tl.program_id(axis=0) # 获取当前的block id，grid为1d所以axis=0
    block_start = pid * BLOCK_SIZE # 计算当前block要计算的数据的起始地址
    offsets = block_start + tl.arange(0, BLOCK_SIZE) # 计算当前block要计算的数据的所有地址
    mask = offsets < n_elements # 计算mask，防止越界
    x = tl.load(x_ptr + offsets, mask=mask) # load x
    y = tl.load(y_ptr + offsets, mask=mask) # load y
    output = x + y # 计算output
    tl.store(output_ptr + offsets, output, mask=mask) # save output

def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = output.numel()
    block_size = triton.next_power_of_2(triton.cdiv(n_elements, ttx.hardware_info.core_num)) # 直接根据硬件核数，均匀拆分任务
    grid = lambda meta: (triton.cdiv(n_elements, block_size), ) # 计算实际grid size
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=block_size) # launch kernel
    return output
```

#### Matmul  
> 原生triton matmul实现：  
> https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html  

```python
import torch
import triton
import triton.language as tl
from tritonx import ttx

ttx.set_hardware_backend("NPU")

# 计算 MxK @ KxN => MxN，只在M上切分tile
@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    # Matrix dimensions
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
):
    """
    Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # Map program ids `pid` to the dimension of M it should use.
    pid_m = tl.program_id(axis=0)
    # only tile m
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = tl.arange(0, N)
    offs_k = tl.arange(0, K)

    # row major
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    # no mask
    a = tl.load(a_ptrs)
    b = tl.load(b_ptrs)
    accumulator = tl.dot(a, b)
    c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = tl.arange(0, N)
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    # no mask
    tl.store(c_ptrs, c)

def matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    M = A.shape[0]
    N = B.shape[1]
    K = A.shape[1]
    assert A.shape[1] == B.shape[0]
    stride_am = K
    stride_ak = 1   
    stride_bk = N
    stride_bn = 1
    stride_cm = N
    stride_cn = 1
    C = torch.empty([M, N], dtype=A.dtype)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_SIZE_M"]), )
    matmul_kernel[grid](A, B, C, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                             triton.next_power_of_2(triton.cdiv(M, ttx.hardware_info.core_num)))
    return C
```

# 相关文档链接

- [triton-x user guide](https://bd-seed-hhw.github.io/triton-x_user_guide/)