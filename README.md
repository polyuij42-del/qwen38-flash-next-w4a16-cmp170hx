# Qwen3.8-Flash-Next on 2× CMP 170HX 128GB — 104 tok/s Decode / ~7K Prefill / 774 tok/s @ 23 Concurrent

## ⚡ Runs on just 38 GB of home-grade RAM — no ECC, no server, no pinned memory (W4A16 · vLLM)

> ### 104 tok/s single-stream Decode ｜ ~7K tok/s Prefill ｜ 774 tok/s aggregate @ 23 concurrent
>
> **❗ Hard requirement / the whole point: NO big memory needed.**
> - The box has **only 38 GB of RAM** — plain home-grade DDR, **no ECC, not a server, no pinned host memory**.
> - The **169 GB model (incl. a ~102 GB PLE table) barely touches system RAM**: served via **NVMe `mmap` demand-paging**,
>   steady-state footprint **≈ 11 GiB**, load-time peak only **~20 GiB**.
> - For contrast, the closest community 2×170HX build needs a **47.7 GB pinned sidecar + 96 GB RAM**.
>   **This setup reproduces on 1/3 the memory, on an ordinary home PC.**

> Inference stack: **vLLM (forked build, not SGLang)**. This repo lists **every reproducible factor** of the live
> port-18430 service so these numbers can be compared head-to-head with the community
> (e.g. `moriyasujapan/qwen38-flash-next-cmp170hx`). All figures are **measured on real hardware** —
> see the file list at the bottom and [`data/environment.txt`](data/environment.txt).
> （中文说明见下方正文各节。）

---

## 🧠 "Near-zero system RAM" is the headline result

**Bottom line — the only hard requirement on system memory is: home-grade is enough. No big RAM, no pinned memory.**
系统内存的唯一硬性要求：**家用级即可，不需要大内存、不需要 pinned。**

| | This repo (vLLM) | Community reference |
|---|---|---|
| Total system RAM | **38 GB (home DDR, no ECC)** | ~96 GB |
| PLE sidecar residency | **NVMe `mmap` demand-paged (evictable)** | 47.7 GB, **pinned host memory** |
| vllm process PSS (steady) | **≈ 11.3 GiB** | — |
| Load-time peak RSS | **~20 GiB** | — |

The three env vars that make it work (see [`launch-vllm.sh`](launch-vllm.sh)):
`VLLM_PLE_MMAP=1` · `VLLM_PLE_MMAP_RANDOM=1` · `VLLM_PLE_CPU_OFFLOAD=1`.
⇒ **An ordinary home PC with 38 GB RAM runs the 169 GB W4A16 model** — this is the result most worth independently verifying.
(`docker stats` measured: **15.6 GiB / 38 GiB = 41%**; the community build needs ~96 GB RAM.)

---

## 📦 Provenance — how to actually install this (nothing private)

Every piece below is public. Our image contains **no private patches baked in** — we verified the core files are byte-identical to the public repo (md5). The image runs with **three runtime-only read-only file overrides** (`-v` mounts, image untouched) — all three are mandatory here; see the section **⚠️ Runtime patch overrides — three mounts, all mandatory** below:

| Piece | Where to get it |
|---|---|
| vLLM runtime ("the fork") | Public community project [`nguyenthimy2022kg-alt/Qwen-Flash-SM80-170HX`](https://github.com/nguyenthimy2022kg-alt/Qwen-Flash-SM80-170HX). Our `qwen-flash-sm80:0.1.4` is exactly that repo's `Dockerfile`, built 2026-09-12. Core files `src/vllm_ple_mmap.py` · `scripts/apply-overlay.py` · `patches/source-origins.json` are **md5-identical to current public main**. |
| Base image | Pinned inside their Dockerfile: `vllm/vllm-openai:qwen38-flash-next@sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8` (public) |
| Model checkpoint | Hugging Face / ModelScope [`Intel/Qwen3.8-Flash-Next-W4A16-AutoRound`](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound) — **Intel's public model, not built or quantized by us**. Downloaded with `aria2` from ModelScope (download log URL carries `namespace=Intel`); **no local quantization / conversion step of any kind** — on the mmap path the checkpoint is used exactly as published. |
| Card unlock (BAR1 etc.) | [`bayley/cmpunlocker`](https://github.com/bayley/cmpunlocker) — and note our run **does not even need working GPU P2P** (topo = PHB, `p2p = GNS`) |

```bash
git clone https://github.com/nguyenthimy2022kg-alt/Qwen-Flash-SM80-170HX
cd Qwen-Flash-SM80-170HX && docker build -t qwen-flash-sm80:local .
# then edit paths in this repo's launch-vllm.sh and run it
```

**What differs vs the community deployment guide**: model = W4A16 AutoRound (not NVFP4); PLE path = `mmap` demand-paging with `VLLM_PLE_GDS=0` (not GDS); works **without GPU P2P and without 96 GB RAM**; plus the **mandatory 16 KiB disk read-ahead below — which neither guide mentions**.

### ⚠️ Runtime patch overrides — three mounts, all mandatory

`launch-vllm.sh` read-only-mounts **three** files over the image at runtime (no rebuild). Verified against `qwen-flash-sm80:0.1.4` by md5 — all three differ from the image originals, i.e. none of this behavior exists in the plain image:

| Mount (repo file → container path) | md5 | What it does / what happens without it |
|---|---|---|
| `pp-partition-fix/distributed/utils.py` → `vllm/distributed/utils.py` | `4f9faf4a…` | Lets `VLLM_PP_LAYER_PARTITION=26,22` coexist with the PLE offload sidecar. Without it → **startup crash**, see below. |
| `patched/vllm_ple_mmap.py` → `vllm_ple_mmap.py` | `ec71c7bd…` | **W4A16 requires this.** The image's `vllm_ple_mmap.py` supports **FP8 tables only**; this model's PLE table is **BF16** → without the mount: `RuntimeError: PLE mmap: only FP8 shards are supported, got BF16`. The override adds BF16/F16 passthrough (+ non-FP8 scale handling) **and the `VLLM_PLE_MMAP_RANDOM` switch** — the image does not recognize that env at all (grep for `MADV_RANDOM` in the image file: 0 hits), so without this file the env is silently a no-op and every PLE page fault gets kernel read-ahead. |
| `patched/ep_weight_filter.py` → `vllm/model_executor/model_loader/ep_weight_filter.py` | `79d3d2a1…` | With `VLLM_PLE_MMAP=1`, skips the `ple_embedding.ngram_embedding.shard_*` tensors during weight loading. Without it the loader **sequentially reads the whole ~102 GB PLE table once and throws it away** — load time explodes and the page cache the mmap path depends on gets flushed. |

#### 1. `VLLM_PP_LAYER_PARTITION=26,22` requires the bundled `pp-partition-fix/` override — do not remove the `-v` mount

`VLLM_PP_LAYER_PARTITION` is a **global env inherited by every subprocess**, including the PLE CPU-offload sidecar (`PleOffloadWorker`) — a standalone process that builds its own meta model with `pp_size=1`. Stock `vllm/distributed/utils.py::get_pp_indices()` asserts `len(partitions) == pp_size`, so `2 != 1` raises:

```
ValueError: len(partitions)=2 does not match pp_size=1.
```

Engine never starts (symptoms: stuck at "activating", `/health` keeps returning 000, container stays "Up"). The bundled override changes **one function only**: when segment count ≠ `pp_size` it logs a warning and falls back to automatic partitioning for that process (sidecar gets `(0,48)`, identical to a default run); the real PP ranks (`pp_size=2`) still get 26/22 unchanged. Mount it read-only at runtime — no image rebuild:

```bash
-v "$REPO_DIR/pp-partition-fix/distributed/utils.py:/usr/local/lib/python3.12/dist-packages/vllm/distributed/utils.py:ro"
```

The startup log line `VLLM_PP_LAYER_PARTITION 26,22 does not match pp_size=1; falling back to default layer partitioning` (WARNING) is **expected and normal** — that is the sidecar taking the fallback path.

---

## ⚠️ Mandatory: set the model disk read-ahead to **16 KiB** — do not skip this

```bash
sudo blockdev --setra 32 /dev/nvme0n1     # 32 × 512 B = 16 KiB  (kernel default is 128 KiB)
blockdev --getra /dev/nvme0n1             # must print 32
```

The PLE `mmap` path issues many small **random** reads against the model disk. At the kernel default 128 KiB read-ahead every page fault over-fetches ~8×, flooding the NVMe queue — **prefill / decode, and normal reads on the same disk, all stop running properly.** Our live box runs `getra = 32` (16 KiB) on `/dev/nvme0n1` (the disk holding `/mnt/data`).

Persist it across reboots (a reboot silently reverts to 128 KiB — this bit us):

```bash
printf 'ACTION=="add|change", KERNEL=="nvme0n1", ATTR{bdi/read_ahead_kb}="16"\n' | sudo tee /etc/udev/rules.d/99-ple-readahead.rules
sudo udevadm control --reload && sudo udevadm trigger
```

> **If your numbers look terrible, check read-ahead first.** No guide — ours or the community's — mentions this parameter.

---

## 一、可复现清单（17 项，逐项实测）

| # | 项目 | 实测值 | 出处 |
|---|---|---|---|
| 1 | **GPU 型号** | 2 × NVIDIA **CMP 170HX 64GB**（GA100 矿卡，被动散热） | `nvidia-smi` |
| 2 | **HBM 时钟** | **1836 MHz**（HBM2e，已超频） | `nvidia-smi -q -d CLOCK` |
| 3 | **PCIe link** | 实协商 **Gen2 ×4**（Cap Gen2 ×16，降速到 ×4，双卡均如此） | `lspci -vv` LnkSta |
| 4 | **是否 P2P** | **否**（`topo -p2p` = GNS；拓扑 = **PHB**，无 NVLink，跨卡走 host PCIe） | `nvidia-smi topo` |
| 5 | **CPU** | AMD **Ryzen 5 5600X** 6C/12T，单 NUMA | `lscpu` |
| 6 | **系统 RAM 实际占用** | 容器 **15.6 GiB / 38 GiB**；vllm PSS 合计 **≈11.3 GiB**（加载峰值 ~20 GiB 级） | `docker stats` / `smaps_rollup` |
| 7 | **模型 checkpoint / quant** | `Qwen3.8-Flash-Next-W4A16-AutoRound`，**169 GB / 17 分片**；**AutoRound 0.15.0 int4 group=128 sym（W4A16）**；embed/hyper_connection/indexer/mtp.* 保 16-bit | `config.json` |
| 8 | **vLLM commit** | `0.1.dev20073+g8e685d198`（**fork**，commit `g8e685d198`）；镜像 `qwen-flash-sm80:0.1.4` | 容器内 `vllm.__version__` |
| 9 | **CUDA / Triton / PyTorch** | CUDA **13.0** · Triton **3.7.1** · PyTorch **2.13.0+cu130**；driver 610.43.03；host nvcc 12.4 | 容器内 import |
| 10 | **MTP 参数** | `{"method":"mtp","num_speculative_tokens":5}`（K=5，草稿头 1 层挂 PP1） | `--speculative-config` |
| 11 | **prompt 长度** | 13 场景 87–319 tok（见场景表） | `raw-13.jsonl` |
| 12 | **output token 数** | 场景测 700 tok；并发测 256 tok | 脚本参数 |
| 13 | **concurrency** | 单流=1；并发阶梯 1→23（真实同时并发上限 **≈23**） | `conc.py` / 性能报告 |
| 14 | **单路 tok/s** | **104**（c=1，同 prompt 隔离 decode） | 并发报告 |
| 15 | **aggregate tok/s** | **774** @ c=23 | 并发报告 |
| 16 | **prefill tok/s** | 单请求峰值 **6.3K**，2 路聚合 **7.3K** tok/s | speedtest 日志 |
| 17 | **磁盘预读（硬性要求）** | 模型盘 read-ahead 必须调至 **16 KiB**（`blockdev --setra 32`；默认 128 KiB 时 prefill/decode 与同盘正常读取都跑不动） | `blockdev --getra` = 32 |

启动方式：[`launch-vllm.sh`](launch-vllm.sh)（docker + 完整 `vllm serve` 命令行）。

---

## 二、MTP K 值调优（维度一）

同 block 4848、单流 c=1、丢预热轮。**峰值在 K=5，K≥6 净亏**（每步 ms 线性涨、tok/步在 K≥6 饱和）。

| K | 单流 tok/s | 接受率 | tok/步 | ms/步 | KV 容量 |
|---|---|---|---|---|---|
| 0（关MTP） | 56.3 | — | 1.00 | 17.4 | 1,372,040 |
| 3 | 96.8 | 61.9% | 2.86 | 29.6 | 864,104 |
| 4 | 105.6 | 55.5% | 3.22 | 30.5 | 813,865 |
| **5** | **107.9** | 50.8% | 3.54 | 32.9 | 769,147 |
| 6 | 103.3 | 44.0% | 3.63 | 35.3 | 729,088 |
| 7 | 99.1 | 38.8% | 3.71 | 37.5 | 692,994 |
| 8 | 97.8 | 37.3% | 3.99 | 41.0 | 660,306 |

- **K=0 在并发下会挂死引擎**（实测挂 307s、容器自退），不能当"省显存降级档"。
- MTP 净赚：K=0→K=5 单流 **+91.7%**。
- 原始数据：[`results/18430-MTP与block全量实测报告.md`](results/18430-MTP与block全量实测报告.md)。

---

## 三、KV 缓存调优（维度二）：block_size 是容量旋钮、不是速度旋钮

- **block 对速度无影响**（同 K=4 三档 1616/4848/9696 → 105.1/105.6/105.3，±1% 噪声）。
- **block 对 KV 影响巨大**（每翻倍 −45%）；**K 每 +1 KV 掉约 5%**。
- **K≥5 的合法 block 下限是 1680 不是 4848**：白捡 +26% KV。⇒ 定档 **K=5 / block 1680**。

| 组 | K | block | KV 容量 | 相对256K倍数 | 备注 |
|---|---|---|---|---|---|
| M | 0 | 1648 | 1,372,040 | 5.23× | 关MTP，并发挂死 |
| **Q** | **5** | **1680** | **1,047,217** | **3.99×** | 🏆 定档 |
| P | 4 | 1648 | 1,038,968 | 3.96× | 接受率最高 |
| K | 1 | 3232 | 1,032,526 | 3.94× | 被支配 |
| N | 7 | 1680 | 958,181 | 3.66× | |
| G | 4 | 9696 | 568,719 | 2.17× | block过大 |

数据：[`data/kv-block-scan.tsv`](data/kv-block-scan.tsv)。

---

## 四、13 种场景速度（维度三：MTP 接受率高度依赖任务类型）

W4A16 线上档 **K=5 / block 1680**，单流 c=1，`temperature=0`，每类型丢预热轮取均值。
**decode tok/s = completion / decode_time**（不含 prefill）。

| 场景 | prompt tok | out tok | decode tok/s | tok/步 | 接受率 |
|---|---|---|---|---|---|
| JSON 结构化抽取 | 250 | 700 | **124.8** | 4.12 | 62.4% |
| 长文摘要 | 319 | 700 | 116.9 | 3.86 | 57.1% |
| 表格整理 | 204 | 700 | 107.6 | 3.57 | 51.3% |
| 代码修 bug | 216 | 700 | 107.4 | 3.56 | 51.1% |
| 数学多步推理 | 130 | 700 | 106.1 | 3.55 | 51.0% |
| 中译英 | 157 | 447 | 95.7 | 3.92 | 58.4% |
| 代码生成 | 87 | 700 | 99.8 | 3.34 | 46.7% |
| 短问答 | 97 | 339 | 91.5 | 3.00 | 40.0% |
| 代码解释 | 290 | 700 | 87.7 | 2.92 | 38.3% |
| 中文技术写作 | 91 | 700 | 89.6 | 2.97 | 39.3% |
| SQL 生成 | 166 | 700 | 89.0 | 3.00 | 40.1% |
| Agent 多步规划 | 138 | 700 | 78.7 | 2.69 | 33.8% |
| 创意写作 | 98 | 700 | **67.9** | 2.33 | 26.6% |

- **13 场景均值 decode ≈ 97 tok/s，接受率跨度 26.6%→62.4%（2.3 倍）**。
- 规律：格式模板化（JSON/表格）接受率高 → 单流快；自由创作（创意写作/Agent 规划）接受率低 → 慢。
- **这就是为什么"只测一种文本类型"必然得出错误结论** —— prompt 与题材直接决定接受率。
- 逐场景原始记录：[`data/raw-13.jsonl`](data/raw-13.jsonl)，汇总表：[`data/13-scenario-table.tsv`](data/13-scenario-table.tsv)。

---

## 五、并发吞吐阶梯（真实并发上限 ≈ 23）

同 prompt 命中前缀缓存以隔离 decode，max_tokens=256，各 3 轮取中位。
混合架构（mamba/GDN）每序列即使很短也占一整块状态 + block 1680，**同时并发被卡在 ~23**（喂 64 → 23 运行 + 42 排队），与 `max-num-seqs` 设多大无关。

| 并发 C | 聚合 tok/s | 单路 decode tok/s | 状态 |
|---|---|---|---|
| 1 | 96 | **104** | 基线 |
| 4 | 288 | 77 | 低延迟优先 |
| 8 | 472 | 64 | 甜点位 |
| 12 | 587 | 52 | 甜点位 |
| 16 | 630 | 46 | 吞吐/延迟折中 |
| 20 | 716 | 39 | 吞吐优先 |
| **23** | **774** | 36 | 🏆 吞吐达峰 |

- **8–16 路是最佳甜点位**（单路仍 46–64 tok/s）；23 路是物理吞吐上限。
- 超过 23 路只会排队，吞吐不再上升。
- 完整报告（含 CUDA 图档位对并发的影响）：[`results/18430-性能测试报告.html`](results/18430-性能测试报告.html)。

---

## 六、硬件诚实说明（横向对比时请注意）

- **PCIe 只有 Gen2 ×4、且无 P2P/NVLink**：PP2 跨卡的激活传输走 host，不是理想的 128GB 直连。
- **CMP 170HX 是被动散热矿卡**，HBM 超频到 1836 MHz、功耗放宽；温度/降频会影响绝对值，跨轮存在约 −3% 漂移。
- 因此本仓库的绝对 tok/s 受"矿卡 + 降速 PCIe"约束，**真正的亮点是：这种受限硬件 + 38GB RAM，靠 mmap 换页就把 169GB W4A16 模型跑到 ~104/774 tok/s**。

---

## 七、已知坑清单（照本仓库复现前必读，全部实测踩过）

1. **三个 `-v` 运行时挂载缺一不可**（见上文 Runtime patch overrides）——尤其 `patched/vllm_ple_mmap.py`：镜像不支持 BF16 PLE 表，不挂直接崩。
2. **`--moe-backend` 必须 `auto`**，不能 `marlin`。且这个错（还有下一条）都爆在 **8–10 分钟权重加载完之后**，别靠试错，改参数前先自查。
3. **MTP `K` 与 `--block-size` 是绑死的**，block 必须同时满足三条：`≥1646`（mamba page 对齐）、`% capacity == 0`（QSA 环容量：K0→4 / K≤4→8 / K5~8→12）、`% 16 == 0`（attention backend `MultipleOf(16)`，最容易漏）。⇒ **K=5~8 最小合法 block = 1680**；`1656`、`2424` 实测都崩 `No common block size`。加载完成 ≠ 稳了——这条也崩在加载完之后约 5 分钟。
4. **K=0（关 MTP）在并发下会挂死引擎**（实测挂 307 s、容器自退），不能当省显存降级档。
5. **不要写死 `--kv-cache-memory`**，用 `--gpu-memory-utilization 0.95` 走自动分配：写死会跳过全部余量校验。两卡权重差 6.22 GiB（MTP 草稿头挂 PP1），**爆显存永远先爆 PP1**；日志里 vLLM 的 "fully utilize 建议值" 是零余量数字，绝不能照抄。
6. **崩溃后显存可能残留**（容器没了、每张卡还占着几十 GB，下次启动报 Free memory不足）。用 `nvidia-smi --query-compute-apps=pid,used_memory` 拿 PID 后 `kill -9`；**SSH 下永远别用 `pkill -f <模式>`**——它会匹配到 SSH 自己的命令行而把会话打死（要杀就按 PID）。
7. **单流 decode 时第二张卡利用率只有 ~30% 是 PP2 结构使然**（流水线气泡；本机无 P2P 更明显），不是配置错了。只有 prefill 阶段两卡才会同时 90%+；别人晒"单并发双卡 90%"时，先怀疑他采样在 prefill 或根本不是 PP2。
8. **客户端行为两件事**：服务端强制 `temperature=1.0`（请求里传 `temperature=0` 不是贪心，单次结果不可比，基准必须多轮取中位）；`reasoning_effort` 合法值只有 **xhigh / medium / low**，传别的（包括常见的 `high`）直接 HTTP 400。
9. **端口映射是 `18430->8000`**：vLLM 在容器内监听 8000，宿主上映射到 18430。别写 `-p 18430:18430`，那样服务在 18430 上根本连不上（本仓库曾错过，已修）。

---

## 八、复现

```bash
# 1) 启动服务
bash launch-vllm.sh

# 2) 并发阶梯（改并发列表/轮数/max_tokens）
python3 bench/conc.py 1,2,4,8,16,23,32 3 512 mytag \
  # env: SWEEP_BASE=http://127.0.0.1:18430 SWEEP_DIR=./bench SWEEP_MODEL=qwen3.8-flash-next-w4a16

# 3) prefill 探针（长 prompt + max_tokens=1，冷 prompt）
python3 bench/prefill_probe.py 6000 5    # env: PREFILL_BASE=http://127.0.0.1:18430

# 4) 13 场景解析
python3 bench/parse13.py data/raw-13.jsonl
```

真实同时并发看 `GET /metrics → vllm:num_requests_running`（日志 10s 采样会低估）。

---

## 文件清单

| 文件 | 内容 |
|---|---|
| `launch-vllm.sh` | 完整 docker + `vllm serve` 启动命令 |
| `pp-partition-fix/distributed/utils.py` | 运行时覆盖 ①：26,22 切分与 PLE 边车共存（md5 `4f9faf4a…`） |
| `patched/vllm_ple_mmap.py` | 运行时覆盖 ②：BF16/F16 PLE 表 + `MADV_RANDOM`（md5 `ec71c7bd…`） |
| `patched/ep_weight_filter.py` | 运行时覆盖 ③：加载期跳过 102 GB PLE 分片（md5 `79d3d2a1…`） |
| `data/environment.txt` | 16 项环境事实的原始命令输出快照 |
| `data/mtp-k-scan.tsv` | MTP K 扫描数据 |
| `data/kv-block-scan.tsv` | KV ↔ K/block 双因素数据 |
| `data/concurrency-ladder.tsv` | 并发阶梯数据 |
| `data/13-scenario-table.tsv` · `raw-13.jsonl` | 13 场景速度与原始记录 |
| `bench/` | `conc.py` / `prefill_probe.py` / `parse13.py` / `prompts.json` |
| `results/` | MTP+block 全量报告、并发性能报告、全表格 |

---

*实测平台：msi · Ubuntu 26.04 · kernel 7.0.0-30 · 2× CMP 170HX 64GB · vLLM fork。数据日期 2026-09-15 ~ 09-16。*
