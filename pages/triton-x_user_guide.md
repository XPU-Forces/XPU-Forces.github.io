---
layout: page
title: triton-x user guide
permalink: /triton-x_user_guide/
---
 
[快速试用](https://bd-seed-hhw.github.io/blog/2025/04/10/triton-example-on-npu)

搭建自己的环境请看后续章节：  
### 环境要求  
1. 如果直接在python环境中安装，相关依赖如下: 

|        组件         |     版本范围     |    说明    | 版本获取方式 |
| :-----------------: | :--------------: | :------: | :----------: |
|       Python        |      >= 3.9      |                                           3.9&3.11                                            |              |
|        torch        |      2.3.1       |                                                                                               |              |
|      torch_npu      |   2.3.1.post4    |                 |   [torch_npu下载](https://tosv.byted.org/obj/aicompiler/npu/deberta/torch_npu-2.3.1.post5-cp311-cp311-linux_x86_64.whl)  |
|    Ascend-driver    |     24.1.rc2     | 一般host宿主机上已经安装，通过npu-smi info确认一下或cat /usr/local/Ascend/driver/version.info |              |
| Ascend-cann-toolkit | 8.0.RC3.alpha003 |                      一般host宿主机上已经安装，通过npu-smi info确认一下                       | [cann包下载](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/POC_ZJ/20250416_daily/Ascend-cann-toolkit_8.0.T113_linux-x86_64.run) [kernels包下载](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/POC_ZJ/20250416_daily/Ascend-cann-kernels-910b_8.0.T113_linux-x86_64.run) |

2. 通过镜像使用，里面已经安装好了相关依赖，可以直接安装byted-triton-x(镜像里面也自带了)：  

| 系统 | 镜像 | dockerfile |
|:-----:|:------:|:------:|
| x86_64 | hub.byted.org/aicompiler/npu.debian12:runtime_cann8.0.rc3.alpha003_py3.11_th26 | [x86_64](https://github.com/BD-Seed-HHW/triton-x/blob/develop/tools/docker/npu-debian/Dockerfile.runtime) |
| aarch64 | hub.byted.org/tritonx/runtime-ascend8.0.rc3-ubuntu20.04-aarch64:1.0.0.1 | [aarch64](https://code.byted.org/seed/triton-x/blob/develop/tools/docker/npu/Dockerfile.runtime) |

### triton-x安装
1. 执行下面命令安装byted-triton-x(也即triton-x在bytedance pypi仓库的包名)

```bash
pip install byted-triton-x --index-url https://bytedpypi.byted.org/simple
# 可以采用下面命令确认是否安装成功和查看版本
pip show byted-triton-x
```
### triton-x样例
采用triton-lang语法写triton kernel（如：后面01-vector-add.py样例），然后执行

```bash
# 确认npu状态
npu-smi info
# 运行triton kernel
python 01-vector-add.py
```  

triton kernel样例：01-vector-add.py  

> Vector Addition
> ===============
> 
> In this tutorial, you will write a simple vector addition using Triton.
> 
> In doing so, you will learn about:
> 
> - The basic programming model of Triton.
> 
> - The `triton.jit` decorator, which is used to define Triton kernels.
> 
> - The best practices for validating and benchmarking your custom ops against native > reference implementations.
  

```python
"""

# %%
# Compute Kernel
# --------------

import torch
import torch_npu

import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               # NOTE: `constexpr` so it can be used as a shape value.
               ):
    # There are multiple 'programs' processing different data. We identify which program
    # we are here:
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    # This program will process inputs that are offset from the initial data.
    # For instance, if you had a vector of length 256 and block_size of 64, the programs
    # would each access the elements [0:64, 64:128, 128:192, 192:256].
    # Note that offsets is a list of pointers:
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < n_elements
    # Load x and y from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # Write x + y back to DRAM.
    tl.store(output_ptr + offsets, output, mask=mask)

# %%
# Let's also declare a helper function to (1) allocate the `z` tensor
# and (2) enqueue the above kernel with appropriate grid/block sizes:

def add(x: torch.Tensor, y: torch.Tensor):
    # We need to preallocate the output.
    output = torch.empty_like(x)
    #assert x.is_cuda and y.is_cuda and output.is_cuda
    n_elements = output.numel()
    # The SPMD launch grid denotes the number of kernel instances that run in parallel.
    # It is analogous to CUDA launch grids. It can be either Tuple[int], or Callable(metaparameters) -> Tuple[int].
    # In this case, we use a 1D grid where the size is the number of blocks:
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    # NOTE:
    #  - Each torch.tensor object is implicitly converted into a pointer to its first element.
    #  - `triton.jit`'ed functions can be indexed with a launch grid to obtain a callable GPU kernel.
    #  - Don't forget to pass meta-parameters as keywords arguments.
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    # We return a handle to z but, since `torch.cuda.synchronize()` hasn't been called, the kernel is still
    # running asynchronously at this point.
    return output

# %%
# We can now use the above function to compute the element-wise sum of two `torch.tensor` objects and test its correctness:
torch.manual_seed(0)
size = 98432
x = torch.rand(size, device='npu')
y = torch.rand(size, device='npu')
output_torch = x + y
output_triton = add(x, y)
print(output_torch)
print(output_triton)
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')
```

### 问题反馈和跟踪
triton-x问题跟踪
### 开源代码路径
[github](https://github.com/BD-Seed-HHW/triton-x)  
[code.byted.org](https://code.byted.org/seed/triton-x)