# 推送到 GitHub

在本目录（`qwen38-flash-next-cmp170hx/`）执行：

```bash
git init
git add -A
git commit -m "Qwen3.8-Flash-Next W4A16 on 2x CMP 170HX: benchmarks + env"

# 在 GitHub 先建一个空仓库（名字与下面一致），然后：
git remote add origin git@github.com:polyuij42-del/qwen38-flash-next-w4a16-cmp170hx.git
git branch -M main
git push -u origin main
```

注意：本仓库只含**脚本 + 结果数据 + 文档**，不含模型权重（169 GB 不随仓库分发）。
模型来自 `Qwen3.8-Flash-Next-W4A16-AutoRound`（AutoRound W4A16 量化），路径见 launch-vllm.sh。
