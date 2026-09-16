#!/usr/bin/env python3
"""把 13 场景 raw.jsonl 聚合成每场景 K4/K5 的速度与接受率表。
decode tok/s = generation_tokens / (inter_token_latency_sum)  —— 纯解码，不含 prefill
每 rep 单独算，丢 rep1 预热，取 rep2/3 平均。
"""
import json
import sys
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "raw-13.jsonl"
# key: (group, type_id) -> list of per-rep dict
per = defaultdict(list)
order = []
names = {}
ptoks = {}
with open(path, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        g = r["group"]
        tid = r["type_id"]
        if (g, tid) not in per and g == "K5":
            order.append(tid)
            names[tid] = r["type_name"]
            ptoks[tid] = r.get("prompt_tokens", 0)
        d = r.get("delta", {})
        comp = d.get("generation_tokens_total", 0) or r.get("completion_tokens", 0)
        istep = d.get("inter_token_latency_seconds_count", 0)
        isum = d.get("inter_token_latency_seconds_sum", 0)
        acc = d.get("spec_decode_num_accepted_tokens_total", 0)
        draft = d.get("spec_decode_num_draft_tokens_total", 0)
        if istep and isum:
            decode_tps = comp / isum
        else:
            decode_tps = comp / r["wall_s"] if r.get("wall_s") else 0
        ms_step = isum / istep * 1000 if istep else 0
        tok_step = (1 + acc / istep) if istep else 0
        rate = acc / draft * 100 if draft else 0
        per[(g, tid)].append(dict(rep=r["rep"], decode_tps=decode_tps,
                                  ms_step=ms_step, tok_step=tok_step,
                                  rate=rate, ptok=r.get("prompt_tokens", 0),
                                  comp=comp))

def agg(g, tid):
    rs = [x for x in per.get((g, tid), []) if x["rep"] > 1]
    if not rs:
        rs = per.get((g, tid), [])
    if not rs:
        return None
    n = len(rs)
    return dict(decode=sum(x["decode_tps"] for x in rs) / n,
                ms=sum(x["ms_step"] for x in rs) / n,
                tok=sum(x["tok_step"] for x in rs) / n,
                rate=sum(x["rate"] for x in rs) / n,
                ptok=rs[0]["ptok"], comp=rs[0]["comp"])

print("== 13 场景（W4A16 · 单流 c=1 · 丢 rep1）==")
print("group\ttype_name\tprompt_tok\tout_tok\tdecode_tok_s\tms_step\ttok_step\taccept%")
for g in ("K4", "K5"):
    for tid in order:
        a = agg(g, tid)
        if not a:
            continue
        print("%s\t%s\t%d\t%d\t%.1f\t%.2f\t%.3f\t%.1f" % (
            g, names[tid], a["ptok"], a["comp"], a["decode"],
            a["ms"], a["tok"], a["rate"]))
    print()

# K5 汇总（线上档）
k5 = [agg("K5", t) for t in order]
k5 = [x for x in k5 if x]
print("K5 单流均值 decode=%.1f 接受率均值=%.1f%%  tok/步均值=%.2f" % (
    sum(x["decode"] for x in k5) / len(k5),
    sum(x["rate"] for x in k5) / len(k5),
    sum(x["tok"] for x in k5) / len(k5)))
