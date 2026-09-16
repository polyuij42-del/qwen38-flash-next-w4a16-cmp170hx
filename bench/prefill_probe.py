#!/usr/bin/env python3
"""Prefill 吞吐探针：长 prompt + max_tokens=1，流式测 TTFT。
用法: PREFILL_BASE=http://127.0.0.1:18430 python3 prefill_probe.py [目标prompt_token数] [reps]
输出: PRE<TAB>rep<TAB>prompt_tokens<TAB>ttft_ms<TAB>prefill_tok_s
"""
import json
import os
import sys
import time
import uuid
import urllib.request

BASE = os.environ.get("PREFILL_BASE", "http://127.0.0.1:18430")
TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 5

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
model = json.load(OPENER.open(BASE + "/v1/models", timeout=20))["data"][0]["id"]

# 用一段可复现的中文技术文本反复拼到目标长度（粗略 1 token ≈ 1.4 汉字）
para = ("本地大模型推理的瓶颈常常落在显存带宽与显存容量而不是算力上。"
        "以混合线性注意力加稀疏注意力的 MoE 架构为例，解码阶段每一步都要读取权重与 KV，"
        "因此单流速度受带宽限制，而并发吞吐受容量与调度限制。以下段落用于填充长度，"
        "不构成任何语义结论，仅为压测提供稳定可复现的 prompt 体量。") * 1
unit = para + "补充说明用于稳定词元计数，避免缓存命中造成的偏差，确保每次都是冷 prompt。"
# 粗估每 unit 约 300 字 ≈ 210 token，按目标放大
repeat = max(1, TARGET // 200)
long_prompt = (unit + " ") * repeat

for rep in range(1, REPS + 1):
    # 每轮前置随机 nonce，强制冷 prompt，避免 prefix cache 让复测虚高
    cold_prompt = "[%s] " % uuid.uuid4().hex + long_prompt
    payload = {"model": model,
               "messages": [{"role": "user", "content": cold_prompt}],
               "temperature": 0, "max_tokens": 1, "stream": True}
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = None
    ptok = None
    for raw in OPENER.open(req, timeout=600):
        line = raw.decode().strip()
        if line.startswith("data:"):
            if ttft is None:
                ttft = time.perf_counter() - t0
            body = line[5:].strip()
            if body and body != "[DONE]":
                try:
                    u = json.loads(body).get("usage")
                    if u and u.get("prompt_tokens"):
                        ptok = u["prompt_tokens"]
                except Exception:
                    pass
    if ttft is None:
        ttft = time.perf_counter() - t0
    # 非流式再取一次 usage 拿准确 prompt_tokens
    if not ptok:
        payload2 = dict(payload); payload2["stream"] = False
        req2 = urllib.request.Request(BASE + "/v1/chat/completions",
                                      data=json.dumps(payload2).encode(),
                                      headers={"Content-Type": "application/json"})
        r2 = json.loads(OPENER.open(req2, timeout=600).read().decode())
        ptok = (r2.get("usage") or {}).get("prompt_tokens", 0)
    tps = ptok / ttft if ttft else 0
    print("PRE\t%d\t%d\t%.1f\t%.1f" % (rep, ptok, ttft * 1000, tps), flush=True)
