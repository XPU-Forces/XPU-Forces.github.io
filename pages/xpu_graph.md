---
layout: page
title: xpu_graph
permalink: /xpu_graph/
---

### 简介

面向司内异构硬件应用的图层级优化工具：
- To User：实现业务积累的常见优化pattern
- To Developer：支持便利的pattern编写
- To Vendor：暴露公共的优化算子接口

下图是xpu_graph的架构图：
![xpu_graph](/images/xpu_graph.png "triton-x总体架构图"){:style="width: 100%"}

### 安装

xpu_graph 的安装方法如下：  

```bash
pip install https://luban-source.byted.org/repository/scm/Seed.Foundation.xpu_graph_1.0.0.1.tar.gz
```

### 使用

xpu_graph 的使用方法如下：
1. Use as a torch.compile backend  
```python
def foo(x, y):
    z = x + y
    another_z = x + y
    return z, another_z
from xpu_graph.compiler import XpuGraph
compiled_foo = torch.compile(foo, backend=XpuGraph())
compiled_foo(torch.randn(10), torch.randn(10))
```

2. Configure  
```python
@dataclass
class XpuGraphConfig:
    """Configuration for XPU graph execution."""

    is_training: bool  # Must fill, if is_training is True, XpuGraph will work as a training compiler, otherwise a inference compiler
    debug: bool = False
    target: Target = field(default=Target.none) # Target hardware backend
    opt_level: OptLevel = OptLevel.level1
    dump_graph: bool = False
    enable_cache: bool = True
    use_xpu_ops: bool = False  # Use xpu_ops or not
    freeze: bool = (
        # Only take effects when "is_training" is False.
        # Freezing parameter will change model's parameter from inputs into attributes.
        # This may help XpuGraph do better constant folding.
        False
    )
    constant_folding: bool = True

    # Till now we only support configure "mode", because we mainly use "Inductor" as a vendor's compiler.
    # mode must be one of {"cudagraphs", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"},
    # we add a "cudagraphs" option. At this mode, XpuGraph will only enable torch.compile in-tree backend "cudugraphs".
    # https://pytorch.org/docs/stable/torch.compiler_cudagraph_trees.html
    vendor_compiler_config: Optional[Dict[str, Any]] = None
```

