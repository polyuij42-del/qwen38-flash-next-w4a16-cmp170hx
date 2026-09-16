#!/usr/bin/env python3
"""多任务混合并发测速（MTP/投机解码专用）。

与 pp-bubble-2820.py 的区别：
  1) 每个并发槽取**不同任务类型**的 prompt（原来所有槽都是同一道 LRU 代码题，
     接受率被单一题材带偏）
  2) 直接算 MTP 三项硬指标：ms/步、tok/步、接受率（数据来自 /metrics 差分）

用法:
  python3 conc.py <concs> <rounds> <max_tokens> [tag]
输出:
  CONC<TAB>c<TAB>round<TAB>agg_tok_s<TAB>ms_per_step<TAB>tok_per_step<TAB>accept_pct<TAB>wall
  末尾按 c 汇总（丢弃 round 1 预热）
"""
import json
import os
import re
import sys
import threading
import time
import urllib.request

BASE = os.environ.get("SWEEP_BASE", "http://127.0.0.1:18430")   # 本仓库的 W4A16 服务端口
SWEEP_DIR = os.environ.get("SWEEP_DIR", os.path.dirname(os.path.abspath(__file__)))  # 默认取本脚本目录下的 prompts.json
MODEL = os.environ.get("SWEEP_MODEL", "qwen3.8-flash-next-w4a16")   # 实际会从 /v1/models 自动识别覆盖
CONCS = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["1", "2", "4", "8"])]
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
MAX_TOKENS = int(sys.argv[3]) if len(sys.argv) > 3 else 700
TAG = sys.argv[4] if len(sys.argv) > 4 else "-"

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
COUNT_KEYS = [
    "inter_token_latency_seconds_count", "inter_token_latency_seconds_sum",
    "spec_decode_num_draft_tokens_total", "spec_decode_num_accepted_tokens_total",
    "generation_tokens_total", "request_success_total", "request_failure_total",
    "iteration_tokens_total_count", "iteration_tokens_total_sum",
]
LINE = re.compile(r'^vllm:([A-Za-z0-9_]+)(?:\{(.*)\})?\s+([0-9eE.+-]+)\s*$')
WARM = "你好，请用一句话介绍你自己。"


def snapshot():
    cs = {}
    txt = OPENER.open(BASE + "/metrics", timeout=30).read().decode()
    for raw in txt.splitlines():
        m = LINE.match(raw.strip())
        if m and m.group(1) in COUNT_KEYS:
            cs[m.group(1)] = cs.get(m.group(1), 0.0) + float(m.group(3))
    return cs


def request(prompt, max_tokens):
    payload = {"model": MODEL,
               "messages": [{"role": "user", "content": prompt}],
               "temperature": 0, "max_tokens": max_tokens}
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    r = json.loads(OPENER.open(req, timeout=1800).read().decode())
    return time.perf_counter() - t0, (r.get("usage") or {}).get("completion_tokens", 0)


prompts = json.load(open(os.path.join(SWEEP_DIR, "prompts.json"), encoding="utf-8"))
model = json.load(OPENER.open(BASE + "/v1/models", timeout=20))["data"][0]["id"]
MODEL = model
print("# tag=%s model=%s rounds=%d max_tokens=%d concs=%s" %
      (TAG, model, ROUNDS, MAX_TOKENS, ",".join(map(str, CONCS))), flush=True)

for _ in range(2):
    request(WARM, 48)
print("# 预热完成", flush=True)

rows = []
for c in CONCS:
    for rd in range(1, ROUNDS + 1):
        # 每个槽取不同任务；跨轮再错开起点，避免同一类型总在同一槽位
        picks = [prompts[(c * rd + i) % len(prompts)] for i in range(c)]
        out = [None] * c

        def work(i):
            try:
                out[i] = request(picks[i]["prompt"], MAX_TOKENS)
            except Exception as e:  # noqa: BLE001
                out[i] = (0.0, 0, "ERR %s" % e)

        cs0 = snapshot()
        t0 = time.perf_counter()
        ths = [threading.Thread(target=work, args=(i,)) for i in range(c)]
        [t.start() for t in ths]
        [t.join() for t in ths]
        wall = time.perf_counter() - t0
        cs1 = snapshot()
        d = {k: cs1.get(k, 0.0) - cs0.get(k, 0.0) for k in COUNT_KEYS}
        steps = d.get("inter_token_latency_seconds_count", 0)
        isum = d.get("inter_token_latency_seconds_sum", 0)
        acc = d.get("spec_decode_num_accepted_tokens_total", 0)
        draft = d.get("spec_decode_num_draft_tokens_total", 0)
        comp = d.get("generation_tokens_total", 0)
        ms = isum / steps * 1000 if steps else 0
        tps = (1 + acc / steps) if steps else 0
        rate = acc / draft * 100 if draft else 0
        rows.append((c, rd, wall, comp, steps, ms, tps, rate, d.get("request_success_total", 0)))
        print("CONC\t%d\t%d\t%.1f\t%.2f\t%.3f\t%.1f\t%.2f\t%d"
              % (c, rd, comp / wall if wall else 0, ms, tps, rate, wall,
                 int(d.get("request_success_total", 0))), flush=True)

print()
print("%-4s %6s %10s %9s %9s %9s" % ("c", "有效轮", "聚合tok/s", "ms/步", "tok/步", "接受率"))
print("-" * 56)
for c in CONCS:
    rs = [r for r in rows if r[0] == c and r[1] > 1]
    if not rs:
        continue
    n = len(rs)
    agg = sum(r[3] / r[2] for r in rs if r[2]) / n
    print("%-4d %6d %10.1f %9.2f %9.3f %8.1f%%"
          % (c, n, agg,
             sum(r[5] for r in rs) / n,
             sum(r[6] for r in rs) / n,
             sum(r[7] for r in rs) / n))
