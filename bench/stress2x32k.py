#!/usr/bin/env python3
"""Reproduce the reporter's failing case: PP2 + MTP5 + 2 concurrent 32K streams.

Two DISTINCT ~32K-token prompts (so prefix caching cannot collapse the prefill),
fired simultaneously, long outputs. Records TTFT, decode rate, per-stream errors,
plus a GPU sampler and the engine's new log lines.

Env: TARGET_TOK (default 32768), MAXTOK (default 4096)
"""
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

BASE = os.environ.get("SWEEP_BASE", "http://127.0.0.1:18430")
MODEL = "qwen3.8-flash-next-w4a16"
TARGET = int(os.environ.get("TARGET_TOK", "32768"))
MAXTOK = int(os.environ.get("MAXTOK", "4096"))
CONTAINER = os.environ.get("CONTAINER", "qwen-w4a16-pp2")
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

A_TMPL = """第{sec}节 关于 vLLM 分页注意力与块大小调优的工程记录。
观察窗口 {win} 秒，命中前缀缓存比例 {pct}%，活跃序列 {act} 条，显存占用 {mem} MiB。
在该配置下 QSA 环形容量必须整除注意力块大小，否则断言失败。混合架构每条序列即使很短也占用整整一个块状态，
因此同时并发被块数硬性限制。若把块大小从 {b0} 调到 {b1}，KV 容量按比例变化，而解码速度在噪声带内基本不变。
PLE 边车表的随机缺页需要小预读，内核默认预读会把 NVMe 队列打爆，prefill 与同盘读取一起停摆。"""

B_TMPL = """Section {sec}: a code-review transcript. The reviewer notes the counter is not thread-safe, because
`reset` mutates `self.n` without holding the lock while eight worker threads call `inc` one hundred thousand times each.
The proposed fix wraps both mutations in `with self.lock:` and adds a typed return annotation. Latency budget {ms} ms,
throughput target {tp} tokens per second, cache hit rate {hit}%. Block size must satisfy three constraints at once:
at least 1646 for mamba page alignment, a multiple of the QSA ring capacity, and a multiple of sixteen for the attention backend.
Speculative decoding depth five, acceptance length near four, tensor layout pipeline-parallel two. Please audit the ring buffer,
the semaphore handshake, the mmap fault path, and the per-layer embedding sidecar before merging this change."""


def tokenize(text):
    req = urllib.request.Request(BASE + "/tokenize",
                                data=json.dumps({"model": MODEL, "prompt": text}).encode(),
                                headers={"Content-Type": "application/json"})
    with OPENER.open(req, timeout=600) as r:
        return json.loads(r.read())["count"]


def build(tmpl, prefix, per):
    """Append numbered sections until the whole thing tokenizes to >= TARGET."""
    n = 0
    while True:
        n += per
        parts = [prefix] + [tmpl.format(sec=i, win=100 + i * 7, pct=(i * 13) % 100,
                                        act=i % 24 + 1, mem=38000 + i * 17, b0=1680, b1=4848 + i,
                                        ms=40 + i, tp=110 + i % 37, hit=(i * 11) % 100)
                            for i in range(1, n + 1)]
        txt = "\n".join(parts)
        cnt = tokenize(txt)
        if cnt >= TARGET:
            return txt, cnt
        if n > 4000:
            raise SystemExit("cannot reach target length")


PREFIX_A = "以下是一份内部工程日志，请通读后回答最后的问题。\n"
PREFIX_B = "Below is an internal engineering log. Read it, then answer the question at the end.\n"
Q_A = "\n\n【问题】请用不超过 200 字总结这份日志里与块大小、并发上限、预读参数相关的三条结论，并指出哪一条最容易被忽略。"
Q_B = "\n\n[Question] Summarise in under 200 words the three conclusions in this log about block size, concurrency ceiling and read-ahead, and say which one is easiest to overlook."

print("== 构造两个互不相同的 ~%d token prompt ==" % TARGET, flush=True)
a = build(A_TMPL, PREFIX_A, 50)
print("  stream A: %d tokens" % a[1], flush=True)
b = build(B_TMPL, PREFIX_B, 50)
print("  stream B: %d tokens" % b[1], flush=True)

# ---- engine log: we will re-read with --since at the end, so no full-log scan ----
logoff = None
print("  engine log: will be read with --since at the end", flush=True)

results = {}
stop_sampler = threading.Event()
samples = []


def sampler():
    while not stop_sampler.is_set():
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,clocks.sm,power.draw,utilization.gpu,memory.used",
                 "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip()
            samples.append("%.1f | %s" % (time.time(), out.replace("\n", " | ")))
        except Exception:
            pass
        stop_sampler.wait(3)


threading.Thread(target=sampler, daemon=True).start()


def run_stream(tag, prompt, maxtok):
    out = {"tag": tag}
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": maxtok, "transport": None, "stream": True,
            "stream_options": {"include_usage": True}}
    body.pop("transport")
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    out["t_start"] = t0
    ttft = None
    last = None
    content_chars = 0
    think_chars = 0
    usage = None
    stop_reason = None
    try:
        with OPENER.open(req, timeout=3600) as r:
            for raw in r:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                p = line[5:].strip()
                if p == "[DONE]":
                    break
                try:
                    obj = json.loads(p)
                except json.JSONDecodeError:
                    continue
                if obj.get("usage"):
                    usage = obj["usage"]
                for ch in obj.get("choices") or []:
                    d = ch.get("delta") or {}
                    got = d.get("content") or d.get("reasoning_content")
                    if got:
                        now = time.time()
                        if ttft is None:
                            ttft = now - t0
                        last = now
                        if d.get("content"):
                            content_chars += len(d["content"])
                        else:
                            think_chars += len(d["reasoning_content"])
                    if ch.get("finish_reason"):
                        stop_reason = ch["finish_reason"]
        out["ok"] = True
    except urllib.error.HTTPError as e:
        out["ok"] = False
        out["error"] = "HTTP %s: %s" % (e.code, e.read().decode("utf-8", "ignore")[:800])
    except Exception as e:
        out["ok"] = False
        out["error"] = "%s: %s" % (type(e).__name__, str(e)[:800])
    out["t_end"] = time.time()
    out["ttft_ms"] = round(ttft * 1000, 1) if ttft else None
    out["decode_s"] = round(last - (t0 + ttft), 2) if (ttft and last) else None
    out["wall_s"] = round(out["t_end"] - t0, 2)
    out["usage"] = usage
    out["stop_reason"] = stop_reason
    out["content_chars"] = content_chars
    out["reasoning_chars"] = think_chars
    ct = (usage or {}).get("completion_tokens")
    if ct and out["decode_s"]:
        out["decode_tok_s"] = round((ct - 1) / out["decode_s"], 1)
    results[tag] = out
    print("[%s] done ok=%s ttft=%sms wall=%ss %s" %
          (tag, out["ok"], out["ttft_ms"], out["wall_s"],
           out.get("error", "")[:200]), flush=True)


MKEYS = ["request_decode_time_seconds_count", "request_decode_time_seconds_sum",
         "spec_decode_num_draft_tokens_total", "spec_decode_num_accepted_tokens_total",
         "generation_tokens_total", "num_preemptions_total", "request_success_total"]
MLINE = re.compile(r"^vllm:([A-Za-z0-9_]+)(?:\{(.*)\})?\s+([0-9eE.+-]+)\s*$")


def metrics():
    out = {}
    try:
        with OPENER.open(BASE + "/metrics", timeout=30) as r:
            for raw in r:
                m = MLINE.match(raw.decode("utf-8", "ignore").strip())
                if m and m.group(1) in MKEYS:
                    out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(3))
    except Exception:
        pass
    return out


print("\n== 同时发起两路 %d-token prefill + %d-token 输出 ==" % (a[1], MAXTOK), flush=True)
m0 = metrics()
t_launch = time.time()
ths = [threading.Thread(target=run_stream, args=("A", a[0] + Q_A, MAXTOK)),
       threading.Thread(target=run_stream, args=("B", b[0] + Q_B, MAXTOK))]
for t in ths:
    t.start()
for t in ths:
    t.join()

stop_sampler.set()
time.sleep(1)

m1 = metrics()
print("\n== 服务端口径（/metrics 差分，本 2 请求）==", flush=True)
dt = {k: m1.get(k, 0.0) - m0.get(k, 0.0) for k in MKEYS}
print("   request_decode_time_s   = %.3f  (count=%d)" % (dt["request_decode_time_seconds_sum"],
                                                         dt["request_decode_time_seconds_count"]), flush=True)
print("   generation_tokens       = %.0f" % dt["generation_tokens_total"], flush=True)
if dt["request_decode_time_seconds_sum"] > 0:
    print("   真实 decode 聚合 tok/s = %.1f" % (dt["generation_tokens_total"] /
                                               dt["request_decode_time_seconds_sum"]), flush=True)
draft = dt["spec_decode_num_draft_tokens_total"]
acc = dt["spec_decode_num_accepted_tokens_total"]
if draft:
    print("   MTP: drafts=%.0f accepted=%.0f  接受率=%.1f%%" % (draft, acc, acc / draft * 100), flush=True)
print("   num_preemptions_total   = %.0f   ← 0 表示全程无抢占" % dt["num_preemptions_total"], flush=True)
print("   request_success_total   = %.0f" % dt["request_success_total"], flush=True)

print("\n== 汇总 ==", flush=True)
print(json.dumps(results, ensure_ascii=False, indent=1))

print("\n== GPU 采样（前 5 / 后 5 / 峰值）==", flush=True)
def peak(metric):
    vals = []
    for s in samples:
        for part in s.split(" | ")[1].split(" | "):
            f = part.split(", ")
            if len(f) == 5:
                try:
                    vals.append(float(re.sub(r"[^0-9.]", "", f[metric])))
                except ValueError:
                    pass
    return max(vals) if vals else None
print("  samples=%d  peak SM=%s MHz  peak power=%s W  peak util=%s%%  peak mem=%s MiB" %
      (len(samples), peak(1), peak(2), peak(3), peak(4)), flush=True)
for s in samples[:5] + ["   ..."] + samples[-5:]:
    print("   ", s, flush=True)

print("\n== 引擎日志（本次测试窗口内，--since）==", flush=True)
try:
    win = int(time.time() - t_launch) + 30
    r = subprocess.run(["docker", "logs", "--since", "%ds" % win, CONTAINER],
                       capture_output=True, text=True, timeout=180)
    allnew = (r.stderr or "").splitlines() + (r.stdout or "").splitlines()
    print("  窗口 %ds，共 %d 行" % (win, len(allnew)), flush=True)
    bad = [l for l in allnew if re.search(r"Error|Traceback|Illegal|illegal|Xid|died|abort|CUDA|OOM|assert|memory allocation", l)]
    if bad:
        print("  ⚠️ 可疑行 %d 条:" % len(bad), flush=True)
        for l in bad[:40]:
            print("   ", l[:320], flush=True)
    else:
        print("  （无 Error/Traceback/Illegal/Xid/CUDA/OOM/assert 相关行）", flush=True)
    for l in allnew[-8:]:
        print("   …", l[:250], flush=True)
except Exception as e:
    print("  (read engine log failed: %s)" % e, flush=True)
