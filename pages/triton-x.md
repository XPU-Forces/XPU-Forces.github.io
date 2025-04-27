---
description: 万物皆比特，一切可编码，宇宙是计算。
title: triton
permalink: /triton-x/
---

# triton-x
triton-x是triton的中间层，向上衔接triton-ir，向下衔接硬件方言，聚焦于硬件平台无关的性能优化，同时也会感知硬件优化，并且解决硬件差异化的问题，降低软硬件适配成本。

## triton-x方案设计  
triton-x的核心是linalg-like的triton-x IR，针对DSA架构抽象出一套API。各硬件厂商开放DSA Dialect接口，并且以共建的方式接入triton-x。如下图红色部分为triton-x提供的核心能力，绿色部分是各硬件厂商需要开放的硬件方言。PS：黑色部分是triton社区支持GPU的方案。  
![](/pages/images/triton-x-overview.png "triton-x总体架构图")  
