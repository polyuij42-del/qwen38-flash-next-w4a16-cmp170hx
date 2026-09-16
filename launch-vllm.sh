#!/bin/bash
# Qwen3.8-Flash-Next W4A16 on 2× CMP 170HX — vLLM 启动命令（线上定档，实测）
# 镜像: qwen-flash-sm80:0.1.4 (vLLM fork, 0.1.dev20073+g8e685d198)
# 以 docker 容器方式运行；下面为容器内实际 vllm serve 命令行。
#
# 关键 env（决定"几乎不吃系统 RAM"的核心）：
#   VLLM_PLE_MMAP=1            # PLE(n-gram 增强嵌入)表走 NVMe mmap，按需换页，不常驻 RAM
#   VLLM_PLE_MMAP_RANDOM=1     # PLE 表 mmap 标 MADV_RANDOM 关内核预读 —— ⚠️ 该开关由
#                              #   patched/vllm_ple_mmap.py 提供，镜像原版**不认识**这个 env
#                              #   （线上 2026-09-16 起实跑 =0：整盘预读已固定 16 KiB，
#                              #    实测 0/1 无速度差；=1 更保险，两者都能用）
#   VLLM_PLE_CPU_OFFLOAD=1     # PLE 层放 CPU 侧（非 pinned）
#   VLLM_PLE_GDS=0             # 不启用 GPUDirect Storage（GDS 与 mmap 源码互斥）
#   CUDA_VISIBLE_DEVICES=0,1
#
# 三个运行时只读挂载（缺一不可，详见 README「Runtime patch overrides」）：
#   pp-partition-fix/distributed/utils.py  ← 没有它 26,22 切分直接把 PLE 边车崩死
#   patched/vllm_ple_mmap.py               ← 镜像只支持 FP8 表；本模型 PLE 表是 BF16，
#                                            不挂它启动即 RuntimeError: only FP8 shards…
#   patched/ep_weight_filter.py            ← 加载阶段跳过 102 GB PLE 分片，省一整遍顺序 I/O

set -e

# ❗❗ 硬性要求（配合下面 VLLM_PP_LAYER_PARTITION=26,22）：必须挂载 pp-partition-fix 补丁。
# VLLM_PP_LAYER_PARTITION 是全局 env，PLE offload 边车进程（PleOffloadWorker，自带 pp_size=1）
# 也会继承它 → 原版 get_pp_indices 校验 len(partitions)=2 != pp_size=1 直接抛 ValueError，
# 引擎起不来（症状：卡在 activating、/health 一直 000）。本仓库 pp-partition-fix/ 里的
# utils.py 只改了 get_pp_indices 一处：段数 != pp_size 时告警并退回自动切分（边车拿 (0,48)），
# 主 rank（pp_size=2）的 26/22 切分完全不受影响。
# 启动日志出现 "VLLM_PP_LAYER_PARTITION 26,22 does not match pp_size=1; falling back to
# default layer partitioning" 的 WARNING 是【正常】的——那就是边车在走退回逻辑。
# ⚠️ 若想改用别的切分档，26,22 之和必须 = num_hidden_layers = 48。
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

# ❗❗ 硬性要求：模型盘 read-ahead 必须调到 16 KiB，否则 prefill/decode 与同盘正常读取都跑不动。
# PLE mmap 是小随机读，内核默认 128 KiB 预读会放大 ~8 倍打爆 NVMe 队列。线上值：getra=32 (=16 KiB)。
# 持久化（重启不丢）：/etc/udev/rules.d/99-ple-readahead.rules:
#   ACTION=="add|change", KERNEL=="nvme0n1", ATTR{bdi/read_ahead_kb}="16"
sudo blockdev --setra 32 /dev/nvme0n1
[ "$(sudo blockdev --getra /dev/nvme0n1)" = "32" ] || { echo "readahead != 16KiB"; exit 1; }

docker run -d --name qwen-w4a16-pp2 --gpus all \
  -e VLLM_PLE_MMAP=1 -e VLLM_PLE_MMAP_RANDOM=1 \
  -e VLLM_PLE_CPU_OFFLOAD=1 -e VLLM_PLE_GDS=0 \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e VLLM_PP_LAYER_PARTITION=26,22 \
  -v "$REPO_DIR/pp-partition-fix/distributed/utils.py:/usr/local/lib/python3.12/dist-packages/vllm/distributed/utils.py:ro" \
  -v "$REPO_DIR/patched/vllm_ple_mmap.py:/usr/local/lib/python3.12/dist-packages/vllm_ple_mmap.py:ro" \
  -v "$REPO_DIR/patched/ep_weight_filter.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/ep_weight_filter.py:ro" \
  -v /mnt/data/qwen-flash-data:/mnt/data/qwen-flash-data \
  -p 18430:8000 \
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
