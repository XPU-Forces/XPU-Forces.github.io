---
layout: post
title: "Triton Example on NPU"
date: 2025-04-10
author: mingfa
---

```shell
# 1. 登入 npu 服务器，如果没有 npu服务器，可以临时使用ssh tiger@2605:340:cd51:602:ac25:ee38:15ff:d6b5
# 2. 如果是自己的服务器，要下载一下镜像, 如果直接通过上面提供的服务器进行试用，此步骤可以忽略
docker login -u your_name hub.byted.org 
docker pull hub.byted.org/tritonx/runtime-ascend8.0.rc3-ubuntu20.04-x86_64:1.0.0.1
# 3. 运行并进入容器，以仿容器重名建议把 ttx-npu 修改成 ttx-npu-yourname
docker run -it --name ttx-npu --shm-size=300g --privileged -e ASCEND_VISIBLE_DEIVCES=1 -e ASCEND_RT_VISIBLE_DEVICES=1 -v /home:/home -v /etc/localtime:/etc/localtime -v /usr/local/Ascend/driver:/usr/local/Ascend/driver -v /usr/local/dcmi:/usr/local/dcmi -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi --net=host hub.byted.org/tritonx/runtime-ascend8.0.rc3-ubuntu20.04-x86_64:1.0.0.1 bash  
# 4. 确认一下npu 的状态
npu-smi info
# 5. 确认一下 triton 版本
pip show byted-triton-x
# 6. 下载测试用例
wget -q https://tosv.byted.org/obj/aicompiler/triton-x/example/LayerNorm.py
wget -q https://tosv.byted.org/obj/aicompiler/triton-x/example/softmax.py
# 7. 执行用例
python LayerNorm.py
python softmax.py

```