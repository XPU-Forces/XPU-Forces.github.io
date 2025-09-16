---
layout: post
title: "how to build triton-x"
date: 2025-04-28
author: mingfa
---

### 下载源码  
```shell
git clone https://github.com/BD-Seed-HHW/triton-x.git
```

### 构建编译镜像    
```shell  
cd your_triton-x_path
# 修改 tools/docker/npu-debian/common.sh
# base_image="debian:bookworm" 修改为这个开源镜像

# 1. 先编译runtime镜像
bash tools/docker/npu-debian/build.sh

# 2. 再编译developer镜像
bash tools/docker/npu-debian/build.sh -t devel

# 3. 最后运行developer镜像
bash tools/docker/npu-debian/run.sh -n core_number -p port -t devel

# 4. 进入容器后，执行以下命令
bash tools/script/build-triton-x.sh # 不带参数是编译开发者模式
# 如果要发布版本，需要添加参数
# bash tools/script/build-triton-x.sh -t release
```