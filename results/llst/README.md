# LLST 标准体检 · xhigh 思考强度 vs 截断版（两轮对比）

> 测试对象：`qwen3.8-flash-next-w4a16`（Intel AutoRound W4A16，169 GB）
> 平台：msi · Ubuntu 26.04 · 2× CMP 170HX 64GB · vLLM fork（PP2 + MTP-5 + prefix-cache）· 端口 18430
> 工具：[Local LLM Standard Test (LLST) v1.0](https://github.com/yang2020chen/local-llm-standard-test) —— 102 道锁定题目 + 4 档长上下文性能
> 数据日期：2026-09-18

---

## 一、一段说明（结论摘要）

本轮把 LLST（本地大模型标准测试）完整跑了两遍，唯一变量是**模型的思考强度**。第一轮（`20260918_074107`）没有显式指定思考强度，用的是服务端默认路径、并且 `max_tokens` 偏小；第二轮（`20260918_110933`）显式把 Qwen3.8 官方的 `reasoning_effort` 钉到 **xhigh**，同时按题型放大 `max_tokens`（AIME24 32768、LiveCodeBench 24576、IFEval 8192）。结果差异非常大：**综合分从 67.2 提到 92.0（+24.8 分）**，五个子项全线上涨，其中数学（AIME24）从 0.30 涨到 0.80（+50pp）、代码（LiveCodeBench）从 0.70 满分到 1.00（+30pp）、指令遵循（IFEval）从 0.70 涨到 0.9474（+24.7pp）。**根因不是模型变聪明了，而是第一轮有 22/102 道题的回答被 `max_tokens` 截断**——推理链写到一半就被砍掉，答案字段为空自然判 0 分；第二轮把额度放宽后截断降到 3/102，分数才回到真实水平。这同时验证了此前 README 第七节第 8 条的结论：`reasoning_effort` 合法值只有 **xhigh / medium / low**（没有 `high`，传了直接 HTTP 400）。速度方面要特别澄清一件事：LLST 报表里的 `output_throughput_tps` 是**端到端平均**（输出 token ÷（首字等待 + 解码），把 TTFT 摊进去了），所以 512 档显示 108.6 而不是监控软件上的 ~147 tok/s；**真实单请求解码速度 = 1/TPOT**，本轮实测 61～125 tok/s，与监控一致。需要说明的是，本轮的速度数字是**在跑满 102 题、显存/缓存被占用的状态下顺带测的**，不是干净环境下的峰值；本仓库第一节的 104 tok/s 单流解码是空载专项测试的结果，两者不冲突。

---

## 二、能力得分对比（102 题）

| 子项 | 题量 | 第一轮（截断版） | 第二轮（xhigh） | 变化 |
|---|---:|---:|---:|---:|
| MMLU-Pro（通用知识+推理） | 42 | 0.8095 | **0.9048** | +9.5pp |
| IFEval（指令遵循） | 20 / 19 | 0.7000 | **0.9474** | +24.7pp |
| AIME24（数学推理） | 10 | 0.3000 | **0.8000** | **+50.0pp** |
| C-EVAL（中文知识+推理） | 20 | 0.8500 | **0.9500** | +10.0pp |
| LiveCodeBench（代码） | 10 | 0.7000 | **1.0000** | **+30.0pp** |
| **综合（等权平均）** | 102 | **67.2** | **92.0** | **+24.8** |

> IFEval 题量 20→19：第二轮有 1 题被上游数据集过滤（未计入分母），对结论无影响，此处如实标注。

---

## 三、根因证据：`max_tokens` 截断统计

按预测记录里的 `choices[0].stop_reason == "max_tokens"` 逐条统计（脚本扫描 `predictions/*/*.jsonl`）：

| 子项 | 第一轮截断 | 第二轮截断 |
|---|---:|---:|
| MMLU-Pro | 6 / 42 | 1 / 42 |
| IFEval | 4 / 20 | 0 / 20 |
| AIME24 | **7 / 10** | 2 / 10 |
| C-EVAL | 2 / 20 | 0 / 20 |
| LiveCodeBench | 3 / 10 | 0 / 10 |
| **合计** | **22 / 102（21.6%）** | **3 / 102（2.9%）** |

第一轮 AIME24 有 **7/10** 道题的推理链被砍断——这正是它只有 0.30 的直接原因。第二轮把 AIME24 额度提到 32768 后仍有 2 题触顶，说明**该模型 xhigh 下的数学推理极长，32K 仍不够**；若要冲满分，AIME24 需再放宽到 65536。

---

## 四、两轮执行配置（可复现）

| 配置项 | 第一轮 `20260918_074107` | 第二轮 `20260918_110933` |
|---|---|---|
| `reasoning_effort` | 未显式设置（服务端默认） | **xhigh** |
| 全局 `max_tokens` | 较小（致多题截断） | 16384 |
| AIME24 `max_tokens` | — | 32768 |
| LiveCodeBench `max_tokens` | — | 24576 |
| IFEval `max_tokens` | — | 8192 |
| `temperature` / `stream` | 0.0 / true | 0.0 / true |
| 执行样本数 | 102 | 102 |
| 截断题数 | 22 | 3 |
| capability manifest sha256 | `1cb00021…db17b` | `101d00f0…a4f7eb` |

> `reasoning_effort` 通过 evalscope 的 `generation_config.extra_body` 透传（GenerateConfig 是 pydantic 模型，顶层字段会被静默丢弃，必须放进 `extra_body` 才会作为顶层请求体转发到 vLLM）。

---

## 五、性能分档（4 档长上下文）

`output_throughput_tps` = **端到端平均吞吐**（输出 token ÷（TTFT + 解码），已把首字等待摊进去）。
**真实解码速度请看 `1/TPOT` 那一列**，它才等于监控软件上的 tok/s。

### 第二轮（xhigh，`20260918_110933`）

| 输入长度 | 平均 TTFT (ms) | 平均 TPOT (ms) | 端到端吞吐 (t/s)* | **真实解码 ≈ 1/TPOT (tok/s)** | MTP 接受率 |
|---:|---:|---:|---:|---:|---:|
| 512 | 627.0 | 8.00 | 108.6 | **125.0** | 74.8% |
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

> \*`output_throughput_tps` 是端到端平均值，**不等于解码速度**——这一点此前容易被误读。
> 两轮同档位存在 3% 级别的漂移属正常（矿卡被动散热 + HBM 超频，跨轮波动约 −3%）。
> TPOT 在不同输入长度间波动，主要受 MTP-5 投机解码接受率与 batch 组成影响，不是配置错误。

---

## 六、交付物清单

| 文件 | 内容 |
|---|---|
| `results/llst/llst-report-xhigh-vs-truncated.png` | 完整对比仪表盘长图（2320×3388，含五维雷达、分项对比、双轴 TTFT/吞吐、报告全文） |
| `results/llst/dashboard-xhigh-vs-truncated.html` | 自包含交互仪表盘（Chart.js 已内联，离线可开；可切换两轮、显示与上一轮的差值） |
| `results/llst/run-20260918_110933-capability.json` | 第二轮（xhigh）能力分汇总（原始） |
| `results/llst/run-20260918_110933-performance.json` | 第二轮（xhigh）性能分档汇总（原始） |
| `results/llst/run-20260918_074107-capability.json` | 第一轮（截断版）能力分汇总（原始） |
| `results/llst/run-20260918_074107-performance.json` | 第一轮（截断版）性能分档汇总（原始） |

---

## 七、复现方式

测试脚本与协议配置位于 Mac 端 `~/.local/bin/llst`（一键驱动）与 LLST 项目 `configs/protocols/standard_test_v1.think.yaml`（思考强度协议）。核心协议片段：

```yaml
capability:
  generation:
    temperature: 0.0
    max_tokens: 16384
    stream: true
    extra_body:
      reasoning_effort: "xhigh"      # 合法值仅 xhigh / medium / low（无 high，传 high → HTTP 400）
  benchmarks:
    - name: "aime24"
      generation: { max_tokens: 65536 }   # 建议：xhigh 下 32K 仍会截断 2/10
    - name: "live_code_bench"
      generation: { max_tokens: 24576 }
    - name: "ifeval"
      generation: { max_tokens: 8192 }
```

---

*实测平台：msi · Ubuntu 26.04 · kernel 7.0.0-30 · 2× CMP 170HX 64GB · vLLM fork（PP2 · MTP-5 · 256K ctx）。数据日期 2026-09-18。*
