# LLST 标准体检 · 思考强度 xhigh 与生成上限 max_tokens 的四轮对照

> 测试对象：`qwen3.8-flash-next-w4a16`（Intel AutoRound W4A16，169 GB）
> 平台：msi · Ubuntu 26.04 · 2× CMP 170HX 64GB · vLLM fork（PP2 + MTP-5 + prefix-cache）· 端口 18430
> 工具：[Local LLM Standard Test (LLST) v1.0](https://github.com/yang2020chen/local-llm-standard-test) —— 102 道锁定题目 + 4 档长上下文性能
> 轮次：`20260918_074107`（截断版，102 题）· `20260918_110933`（xhigh，102 题）· `20260918_122558_aime65536`（AIME24 追测，10 题）· `20260918_131858`（不限制，102 题）
> 数据日期：2026-09-18

---

## 一、一段说明（结论摘要）

同一套锁定题库跑了四轮，变量只有**思考强度**与**生成上限 `max_tokens`**。第一轮（`20260918_074107`）没显式指定思考强度、上限偏小，**综合分只有 67.2**；第二轮把 Qwen3.8 官方的 `reasoning_effort` 钉到 **xhigh** 并按题型放大上限，**综合分跃到 92.0**；第四轮（`20260918_131858`）把上限直接放到 **250000（等于不限制）**，跑满 102 题，**综合分 94.5、截断归零**。

关键在于：**第一轮"模型很笨"完全是假象，根因是 102 题里有 22 题的推理链被 `max_tokens` 砍断**——答案字段为空自然判 0。四轮的截断题数是 **22 → 3 → 1 → 0**，综合分是 **67.2 → 92.0 → 94.5**。

**AIME24 是唯一真正被上限左右的基准，也是这轮最大的收获：0.30 → 0.80 → 0.90 → 1.00（满分）**。那道反复失败的第 4 题（目标答案 385）在 32768 与 65536 上限下都被砍断、判 0；上限放到 250000 后它一次性写满 **94579 个推理 token** 并答对。**它不是"停不下来"，只是真的需要将近 10 万 token 的思考。**

其余四项在第二轮就已零截断，第四轮放开上限对它们**没有增益**：MMLU-Pro 90.5 → 88.1、IFEval 94.7 → 89.5，两项**各只差 1 道题**，属单轮采样波动（服务端强制 `temperature=1.0`），不是能力退化——因为第四轮这两项同样是**零截断**。反过来，放开上限让模型明显**想得更久**：MMLU-Pro 推理 token 总量 46,965 → 77,510（**+65%**），IFEval 18,067 → 26,705（**+48%**），分数却没涨。**想得更多 ≠ 答得更对。**

还有两个指标读法必须说清楚：① `reasoning_effort` 合法值只有 **xhigh / medium / low**（没有 `high`，传了直接 HTTP 400）。② LLST 报表里的 `output_throughput_tps` 是**端到端平均**（输出 token ÷（首字等待 + 解码），把 TTFT 摊进去了），**不等于解码速度**；**真实单请求解码速度 = 1/TPOT**，本轮实测 61～125 tok/s，与监控软件一致。

---

## 二、能力得分对比（102 题）

| 子项 | 题量 | R1 截断版 | R2 xhigh | **R4 不限制** | R4 vs R2 |
|---|---:|---:|---:|---:|---:|
| MMLU-Pro（通用知识+推理） | 42 | 0.8095 | 0.9048 | **0.8810** | −2.4pp |
| IFEval（指令遵循） | 20 / 19 | 0.7000 | 0.9474 | **0.8947** | −5.3pp |
| AIME24（数学推理） | 10 | 0.3000 | 0.8000 | **1.0000** | **+20.0pp** |
| C-EVAL（中文知识+推理） | 20 | 0.8500 | 0.9500 | **0.9500** | 0.0 |
| LiveCodeBench（代码） | 10 | 0.7000 | 1.0000 | **1.0000** | 0.0 |
| **综合（等权平均）** | 102 | **67.2** | **92.0** | **94.5** | **+2.5** |

> IFEval 题量 20→19：有 1 题被上游数据集过滤（未计入分母），如实标注。
> R2 → R4 的两处回落（MMLU-Pro、IFEval）**各只差 1 道题**，且 R4 两项都是零截断——单轮采样波动，不是能力下降。

### AIME24：上限梯度四轮

| AIME24 | `max_tokens` | 得分 | 截断题数 |
|---|---:|---:|---:|
| 第一轮 | 偏小 | 0.30（3/10） | 7 / 10 |
| 第二轮 | 32768 | 0.80（8/10） | 2 / 10 |
| 第三轮 | 65536 | 0.90（9/10） | 1 / 10 |
| **第四轮** | **250000** | **1.00（10/10）** | **0 / 10** |

> 四轮同一套锁定题目、同一 `reasoning_effort=xhigh`，唯一变量是 `max_tokens`。**每放开一档就多拿回一题，直到截断归零、AIME24 满分。**

---

## 三、根因证据：`max_tokens` 截断统计

按预测记录里的 `choices[0].stop_reason == "max_tokens"` 逐条统计：

| 子项 | R1 | R2 | 第三轮（AIME 追测） | R4 不限制 |
|---|---:|---:|---:|---:|
| MMLU-Pro | 6 / 42 | 1 / 42 | — | **0 / 42** |
| IFEval | 4 / 20 | 0 / 20 | — | **0 / 20** |
| AIME24 | **7 / 10** | 2 / 10 | 1 / 10 | **0 / 10** |
| C-EVAL | 2 / 20 | 0 / 20 | — | **0 / 20** |
| LiveCodeBench | 3 / 10 | 0 / 10 | — | **0 / 10** |
| **合计** | **22 / 102** | **3 / 102** | **1 / 10** | **0 / 102** |

第一轮 AIME24 有 **7/10** 被砍断——这就是它只有 0.30 的全部原因。**第四轮 102 题零截断**，是四轮里唯一一次所有回答都自然收尾。

### AIME24 逐题推理量（第四轮，250000 上限）

| 题号 | stop_reason | 推理 token | 目标答案 | 判定 |
|---:|---|---:|---|---|
| 1 | `stop` | 728 | 204 | ✅ |
| 2 | `stop` | 26,306 | 113 | ✅ |
| 3 | `stop` | 12,465 | 371 | ✅ |
| **4** | `stop` | **94,579** | 385 | ✅ |
| 5 | `stop` | 22,001 | 110 | ✅ |
| 6 | `stop` | 4,410 | 104 | ✅ |
| 7 | `stop` | 34,578 | 721 | ✅ |
| 8 | `stop` | 1,524 | 25 | ✅ |
| 9 | `stop` | 2,254 | 809 | ✅ |
| 10 | `stop` | 552 | 116 | ✅ |

> **第 4 题需要 94,579 个推理 token**——它在 32768（R2）和 65536（R3）下都被砍断、答案为空判 0。这一题独占该基准推理总量的 **47%**（全轮 AIME24 共 199,397 token）。
> 十题分布：最小 552、中位数 8,438、最大 94,579 ⇒ **典型长尾**。所以"用一个足够大的全局上限兜底"代价很高（九题被一题拖着一起预留），但确实**能把分拿满，并非不可收敛**。

---

## 四、四轮执行配置（可复现）

| 配置项 | R1 `074107` | R2 `110933` | R3 `122558` | R4 `131858` |
|---|---|---|---|---|
| `reasoning_effort` | 未显式设置 | xhigh | xhigh | xhigh |
| 全局 `max_tokens` | 较小 | 16384 | 250000 | **250000** |
| AIME24 `max_tokens` | — | 32768 | 65536 | 250000（继承） |
| LiveCodeBench | — | 24576 | 24576 | 250000（继承） |
| IFEval | — | 8192 | 8192 | 250000（继承） |
| 执行样本数 | 102 | 102 | 10（仅 AIME） | 102 |
| 截断题数 | 22 | 3 | 1 | **0** |
| 综合分 | 67.2 | 92.0 | 94.0（回填） | **94.5** |
| 耗时 | — | ~48 min | ~23 min | **62 min 20 s** |

> `reasoning_effort` 通过 evalscope 的 `generation_config.extra_body` 透传（GenerateConfig 是 pydantic 模型，顶层字段会被静默丢弃，必须放进 `extra_body` 才会作为顶层请求体转发到 vLLM）。
> **第四轮刻意去掉了所有逐基准 `max_tokens` 覆盖**——否则 AIME24=32768 之类的小上限还在，等于没"不限制"。
> 上限值怎么定：`max_model_len` = 262144，题库最长 prompt 实测仅 **3413**（C-EVAL）⇒ `3413 + 250000 = 253413 < 262144`，安全。

---

## 五、性能分档（4 档长上下文）

`output_throughput_tps` = **端到端平均吞吐**（输出 token ÷（TTFT + 解码），已把首字等待摊进去）。
**真实解码速度请看 `1/TPOT` 那一列**，它才等于监控软件上的 tok/s。

### 第四轮（不限制，`20260918_131858`）

| 输入长度 | 平均 TTFT (ms) | 平均 TPOT (ms) | 端到端吞吐 (t/s)* | **真实解码 ≈ 1/TPOT (tok/s)** | MTP 接受率 |
|---:|---:|---:|---:|---:|---:|
| 512 | 514.9 | 10.97 | 83.6 | 91.2 | 67.5% |
| 4096 | 1921.5 | 8.74 | 80.2 | 114.4 | 71.6% |
| 16384 | 9241.4 | 11.43 | 33.9 | 87.5 | 68.0% |
| 28672 | 30194.4 | 10.17 | 14.5 | 98.3 | 71.5% |

### 第二轮（xhigh，`20260918_110933`）

| 输入长度 | 平均 TTFT (ms) | 平均 TPOT (ms) | 端到端吞吐 (t/s)* | **真实解码 ≈ 1/TPOT (tok/s)** | MTP 接受率 |
|---:|---:|---:|---:|---:|---:|
| 512 | 627.0 | 8.00 | 108.6 | 125.0 | 74.8% |
| 4096 | 4440.9 | 15.14 | 42.0 | 66.1 | 61.8% |
| 16384 | 15766.9 | 16.36 | 21.2 | 61.1 | 66.5% |
| 28672 | 11700.8 | 12.08 | 28.6 | 82.8 | 61.3% |

### 第一轮（截断版，`20260918_074107`）

| 输入长度 | 平均 TTFT (ms) | 平均 TPOT (ms) | 端到端吞吐 (t/s)* | **真实解码 ≈ 1/TPOT (tok/s)** | MTP 接受率 |
|---:|---:|---:|---:|---:|---:|
| 512 | 984.5 | 17.35 | 52.0 | 57.6 | 67.6% |
| 4096 | 3855.9 | 16.92 | 41.0 | 59.1 | 67.1% |
| 16384 | 9110.5 | 18.37 | 27.7 | 54.4 | 68.1% |
| 28672 | 15596.7 | 10.79 | 24.3 | 92.7 | 68.3% |

> \*`output_throughput_tps` 是端到端平均值，**不等于解码速度**——这一点此前容易被误读，512 档因此显示 83.6 而不是 91.2。
> ⚠️ **性能段噪声很大，别当结论用**：每档只有 2 个请求，而且这一段是在跑完 40 分钟能力测试、KV 缓存被大量前缀占满之后紧接着测的。R4 的 28K 档 TTFT（30.2 s）明显高于 R2（11.7 s），512 档反而更快（515 ms vs 627 ms）。要下性能结论应**空载专项重跑**（本仓库第一节的 104 tok/s 单流解码就是那种条件下测的）。

**AIME24 追测轮的性能侧写**（`20260918_122558_aime65536`）：平均单题延迟 **140.6 s**、平均 TTFT **161.3 ms**、平均 TPOT **8.1 ms**、平均输出 **16323 token**、端到端吞吐 **116.09 tok/s**。
> 这次端到端吞吐（116.1）与 `1/TPOT`（≈123.5）**很接近**——因为输出极长（平均 1.6 万 token），100 多毫秒的 TTFT 早被摊薄。**输出越长，端到端平均越接近真实解码速度；只有短输出（如 512 档）才会差出 2 倍。**

---

## 六、交付物清单

| 文件 | 内容 |
|---|---|
| `results/llst/llst-report-four-rounds.png` | **四轮完整报告长图（推荐先看这张）**（2320×8352，含 KPI、五子项三轮对比、AIME24 上限梯度、逐题推理长尾、截断统计、配置、性能双图） |
| `results/llst/dashboard-four-rounds.html` | 上面对应长图的源 HTML（自包含，Chart.js 已内联，离线可开） |
| `results/llst/llst-report-three-rounds.png` | 三轮版长图（2320×8418，存档） |
| `results/llst/dashboard-three-rounds.html` | 三轮版源 HTML（存档） |
| `results/llst/llst-report-xhigh-vs-truncated.png` | 两轮版对比仪表盘长图（2320×3388，含五维雷达，存档） |
| `results/llst/dashboard-xhigh-vs-truncated.html` | 两轮版交互仪表盘（存档） |
| `results/llst/truncation-stats.json` | 四轮逐基准截断统计汇总 + AIME24 推理长度分析 |
| `results/llst/run-20260918_131858-capability.json` | 第四轮（不限制）能力分汇总（原始） |
| `results/llst/run-20260918_131858-performance.json` | 第四轮（不限制）性能分档汇总（原始） |
| `results/llst/run-20260918_131858-capability-manifest.json` | 第四轮能力执行清单（含全部预测文件 sha256） |
| `results/llst/run-20260918_131858-aime24-detail.json` | 第四轮 AIME24 逐题明细（推理 token / 答案 / 对错） |
| `results/llst/run-20260918_122558-aime24-report.json` | 第三轮 AIME24 追测的原始评测报告（evalscope schema v2） |
| `results/llst/run-20260918_122558-aime24-detail.json` | 第三轮 AIME24 逐题明细 |
| `results/llst/run-20260918_110933-{capability,performance}.json` | 第二轮（xhigh）汇总（原始） |
| `results/llst/run-20260918_074107-{capability,performance}.json` | 第一轮（截断版）汇总（原始） |

---

## 七、复现方式

三个协议文件都在 `configs/protocols/`（**新建独立文件，不改原有的**）：

| 协议 | 用途 |
|---|---|
| `standard_test_v1.think.yaml` | xhigh + 16384（R2 用的就是它） |
| `standard_test_v1.unlimited.yaml` | xhigh + 250000，且**无逐基准覆盖**（R4 用的） |

核心片段：

```yaml
capability:
  generation:
    temperature: 0.0
    max_tokens: 250000        # 不限制：262144 - 最长prompt 3413 - 余量
    stream: true
    extra_body:
      reasoning_effort: "xhigh"      # 合法值仅 xhigh / medium / low（无 high，传 high → HTTP 400）
  benchmarks:
    - name: "mmlu_pro"
    - name: "ifeval"                 # 不写 generation: 覆盖 ⇒ 继承 250000
    - name: "aime24"                 # 实测：32K→截断2/10得0.80；64K→1/10得0.90；250K→0截断得1.00
    - name: "ceval"
    - name: "live_code_bench"
```

**只重跑单个基准**（例如只想追测 AIME24，省掉其余 92 题的时间）：把 `cfg["capability"]["benchmarks"]` 过滤成单项后直接调 `run_capability_suite`。
⚠️ 注意：`run_capability_suite` 末尾会用过滤后的执行数去比对**全量清单**的总数，单项运行必然抛 `PROTOCOL_EXECUTION_MISMATCH: Executed 10 samples, expected 102`。**这只是收尾审计报错，评测本身已经跑完、报告已经落盘**——直接从 `capability/aime24/*/reports/**/*.json` 取分即可。

**建议的后台跑法**（长任务必须脱离 SSH 会话）：

```bash
# 在 msi 上
cd /data/home-poly/llst
export MODELSCOPE_CACHE=/data/home-poly/modelscope-cache/datasets LLST_API_KEY=EMPTY PYTHONPATH=/data/home-poly/llst
export PYTHON=/data/home-poly/llst/.venv/bin/python3 EVALSCOPE=/data/home-poly/llst/.venv/bin/evalscope
setsid nohup bash -c './scripts/run_standard_test.sh --protocol configs/protocols/standard_test_v1.unlimited.yaml' \
  > /tmp/llst_unlimited.log 2>&1 < /dev/null & disown
```

---

*实测平台：msi · Ubuntu 26.04 · kernel 7.0.0-30 · 2× CMP 170HX 64GB · vLLM fork（PP2 · MTP-5 · 256K ctx）。四轮：`074107` / `110933` / `122558_aime65536` / `131858`（第四轮 13:18:46 → 14:21:06，共 62 分 20 秒）。数据日期 2026-09-18。*
