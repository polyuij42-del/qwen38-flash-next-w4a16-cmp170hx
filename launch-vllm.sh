#!/bin/bash
# Qwen3.8-Flash-Next W4A16 on 2× CMP 170HX — vLLM 启动命令（线上定档，实测）
# 镜像: qwen-flash-sm80:0.1.4 (vLLM fork, 0.1.dev20073+g8e685d198)
# 以 docker 容器方式运行；下面为容器内实际 vllm serve 命令行。
#
# 关键 env（决定"几乎不吃系统 RAM"的核心）：
#   VLLM_PLE_MMAP=1            # PLE(n-gram 增强嵌入)表走 NVMe mmap，按需换页，不常驻 RAM
#   VLLM_PLE_MMAP_RANDOM=1     # 随机读 PLE 表，验证 mmap 换页路径
#   VLLM_PLE_CPU_OFFLOAD=1     # PLE 层放 CPU 侧（非 pinned）
#   VLLM_PLE_GDS=0             # 不启用 GPUDirect Storage
#   CUDA_VISIBLE_DEVICES=0,1

set -e
docker run -d --name qwen-w4a16-pp2 --gpus all \
  -e VLLM_PLE_MMAP=1 -e VLLM_PLE_MMAP_RANDOM=1 \
  -e VLLM_PLE_CPU_OFFLOAD=1 -e VLLM_PLE_GDS=0 \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e VLLM_PP_LAYER_PARTITION=26,22 \
  -v /mnt/data/qwen-flash-data:/mnt/data/qwen-flash-data \
  -p 18430:18430 \
  qwen-flash-sm80:0.1.4 \
  vllm serve /mnt/data/qwen-flash-data/models/Qwen3.8-Flash-Next-W4A16-AutoRound \
    --pipeline-parallel-size 2 \
    --served-model-name qwen3.8-flash-next-w4a16 \
    --max-model-len 262144 \
    --block-size 1680 \
    --mamba-ssm-cache-dtype float32 \
    --max-num-seqs 64 \
    --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.95 \
    --moe-backend auto \
    --enable-prefix-caching \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --trust-remote-code \
    --no-enable-flashinfer-autotune \
    -cc.cudagraph_mode=FULL_AND_PIECEWISE \
    '-cc.cudagraph_capture_sizes=[1,2,4,8,16,24,32,40,48,96,144,192,288,384]' \
    --speculative-config '{"method":"mtp","num_speculative_tokens":5}'
