# Qwen3.8-Flash-Next on 8× RTX 4090

> 从 SGLang FP8 到 llama.cpp GGUF：消费级多卡部署、卡数探索、长上下文、Prompt Cache、
> Responses API 与 Codex 接入实测报告。

## 摘要

本报告记录在一台 **8× RTX 4090 24GB、无 NVLink** 的服务器上部署
Qwen3.8-Flash-Next 的完整过程。

最终生产方案：

| 项目 | 最终配置 |
|---|---|
| 推理引擎 | Unsloth llama.cpp Qwen4Exp 分支 |
| 模型 | `unsloth/Qwen3.8-Flash-Next-GGUF` |
| 量化 | `UD-IQ4_XS`，约 93.7GB |
| GPU | 4× RTX 4090，使用同一 NUMA node 内的 GPU |
| 多卡切分 | `--split-mode layer` |
| Context | 131,072 tokens |
| 并发 slots | 1 |
| Prompt Cache | 显式开启 |
| API | OpenAI-compatible Responses / Chat Completions / Completions |
| 512-token Decode | **42.25 tok/s** 三次中位数 |
| 热实例单次复验 | **44.17 tok/s** |

主要结论：

1. RTX 4090 上的 SGLang FP8 路线受 SM89 后端、QSA fallback 和 CUDA graph 限制，
   本机最佳仅为 14.50 tok/s。
2. 参考社区方案切换到 llama.cpp + GGUF 后，4 卡稳定达到 42–44 tok/s。
3. 对无 NVLink 的消费卡，llama.cpp 默认的 **layer split** 明显比实验性的 tensor split 更合适。
4. 3 卡是 `UD-IQ4_XS` 的最低全 GPU 可运行配置，但显存余量太小；4 卡是最佳单副本配置。
5. 8 卡单副本受跨 NUMA/SYS 通信影响，反而比 4 卡慢约 4.3%。
6. 最佳整机利用方式是两个独立 4 卡副本，而不是一个 8 卡副本。
7. 131K context 可以实际检索远距离信息，但冷 prefill 很慢；Prompt Cache 对 Codex 多轮会话至关重要。

所有结果均来自单机实测，不应被视为其他硬件、驱动或模型版本的性能保证。

## 1. 硬件与软件环境

### 1.1 硬件

- GPU：8× NVIDIA GeForce RTX 4090 24GB
- GPU 互联：PCIe，无 NVLink
- GPU 0–3：NUMA node 0，组内 PIX
- GPU 4–7：NUMA node 1，组内 PIX
- 两组 GPU 之间：SYS
- CPU：144 logical CPUs，2 个 NUMA node
- 内存：503GiB
- 存储：NVMe，测试时约 2TiB 可用空间

### 1.2 软件

- 操作系统：Linux
- CUDA Toolkit：12.8
- NVIDIA Driver：610.43.02
- llama.cpp 源码提交：

```text
6c5afc86ae84448ae4d744e357017e2c490ad9c3
```

- 部署日期：2026-08-27

### 1.3 NUMA 选择

单副本优先选择同一 NUMA node 内的四张 GPU，例如 GPU 0–3：

```text
GPU 0 ─┐
GPU 1 ─┼─ NUMA 0 / PIX group
GPU 2 ─┤
GPU 3 ─┘

GPU 4 ─┐
GPU 5 ─┼─ NUMA 1 / PIX group
GPU 6 ─┤
GPU 7 ─┘
```

这可以避免单路生成跨 CPU socket 和 SYS 路径传输。

## 2. 路线选择

### 2.1 最初目标

- 替换原有的 Qwen3.8-27B 服务。
- 部署 Qwen3.8-Flash-Next。
- 探索最低 GPU 数量。
- 找到消费级 RTX 4090 上单用户、低并发场景的最快稳定配置。
- 提供 OpenAI-compatible Responses API，供 Codex 使用。

### 2.2 参考资料

- [SGLang Qwen3.8-Flash-Next Cookbook](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next)
- [Qwen3.8-Flash-Next-Fleet-Deploy](https://github.com/tonyd2wild/Qwen3.8-Flash-Next-Fleet-Deploy)
- [Unsloth llama.cpp Qwen4Exp branch](https://github.com/unslothai/llama.cpp/tree/qwen4exp/qwen3.8-flash-next)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)

社区参考项目中“4× RTX 3090 约 43 tok/s”的方案并不是 SGLang FP8，而是：

- llama.cpp Qwen4Exp 实验分支；
- `UD-IQ3_XXS` GGUF；
- 4-way layer split；
- 131K context；
- 无 speculative decoding。

本次部署沿用该架构思路，但选择质量更高的 `UD-IQ4_XS`。

## 3. SGLang FP8 对照实验

### 3.1 模型与版本

模型：

```text
Qwen/Qwen3.8-Flash-Next-FP8
```

权重总大小：

```text
185,502,232,570 bytes
```

使用支持 PLE offload 的 SGLang 提交：

```text
73a255206f916366c8d26d4022f82ddfb0ab558d
```

### 3.2 RTX 4090 兼容性问题

官方 H200 配置不能直接用于 RTX 4090：

1. `TP8+EP1` 无法加载：专家中间维度被切分为 80，不满足 FP8 `block_n=128` 的对齐要求。
2. `TP8+EP8` 可以加载。
3. 官方 FlashInfer GDN 要求 SM90+；RTX 4090 是 SM89，因此需要 Triton GDN。
4. QSA 默认 fallback 到 FA4 Cute，在 SM89 上出现 MLIR 编译错误。
5. 增加 SM89 QSA correctness fallback 后可以运行。
6. 该 fallback 无法完成 CUDA graph capture，只能 eager 执行。
7. PyTorch SDPA 改写没有带来明显提升。
8. NEXTN eager 有显著收益，但仍不满足目标速度。

### 3.3 SGLang 实测

512 输出 tokens，3 次运行取中位数：

| 配置 | TTFT | Decode |
|---|---:|---:|
| TP8+EP8，eager，无 NEXTN | 0.40s | 6.21 tok/s |
| TP8+EP8，NEXTN eager，8K | 0.49s | 14.50 tok/s |
| TP8+EP8，NEXTN eager，radix cache，32K | 0.50s | 14.27 tok/s |

结论：该路线功能可用，但 RTX 4090 缺少 H200/SM90 对应的高效执行路径，因此不作为最终方案。

## 4. 为什么 llama.cpp GGUF 可行

Qwen3.8-Flash-Next 包含很大的 PLE n-gram embedding table。GGUF/llama.cpp 路线可以通过
mmap 让该表主要保留在 NVMe 和操作系统页缓存中，而不是把整个表加载到 VRAM。

因此模型文件虽然约 93.7GB，实际常驻 GPU 的主要计算权重和运行缓存仍能分布到 3–4 张
24GB 消费卡上。

当前 Qwen4Exp llama.cpp 分支尚未实现该架构的 MTP/NEXTN/DFlash2 speculative decoding，
本报告中的 42–44 tok/s 是纯自回归速度。

## 5. 构建 llama.cpp

建议将本仓库脚本目录设为 `$DEPLOY_ROOT`，将 llama.cpp clone 到其子目录：

```bash
export DEPLOY_ROOT="$PWD"

git clone --depth 1 \
  --branch qwen4exp/qwen3.8-flash-next \
  https://github.com/unslothai/llama.cpp.git \
  "$DEPLOY_ROOT/llama.cpp"
```

针对 RTX 4090 / SM89 构建：

```bash
cmake -S "$DEPLOY_ROOT/llama.cpp" \
  -B "$DEPLOY_ROOT/llama.cpp/build" \
  -DGGML_CUDA=ON \
  -DLLAMA_CURL=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DCMAKE_BUILD_TYPE=Release

cmake --build "$DEPLOY_ROOT/llama.cpp/build" \
  --target llama-server llama-bench \
  -j "$(nproc)"
```

生成的服务程序：

```text
$DEPLOY_ROOT/llama.cpp/build/bin/llama-server
```

## 6. 模型下载与校验

模型仓库：

```text
unsloth/Qwen3.8-Flash-Next-GGUF
```

量化目录：

```text
UD-IQ4_XS
```

本次下载得到的分片：

| 文件 | bytes |
|---|---:|
| `Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf` | 10,946,624 |
| `Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf` | 49,835,229,856 |
| `Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00003.gguf` | 43,836,407,744 |
| **总计** | **93,682,584,224** |

服务返回的模型元数据：

- `n_ctx_train=262144`
- `n_params=176943899520`
- `size=93671559680`
- `ftype=IQ4_XS - 4.25 bpw`

将首分片路径配置到 `MODEL`：

```bash
export MODEL="$DEPLOY_ROOT/models/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf"
```

不要将 GGUF 权重提交到 Git 仓库。

## 7. 关键优化：Layer Split

### 7.1 不推荐 Tensor Split

实验初期曾显式设置：

```text
--split-mode tensor
```

这是实验性的权重/KV tensor parallel 模式，会在层内产生频繁的跨卡通信。对于无 NVLink 的
RTX 4090，PCIe 通信开销会抵消并行收益。

### 7.2 最终选择

最终采用：

```text
--split-mode layer
```

Layer split 主要在设备边界传递 activation，更接近参考 4×3090 方案，也更适合 PCIe 多卡。

这是本次从低速配置恢复到 40+ tok/s 的关键修正。

## 8. 最终启动配置

### 8.1 API Key

生成独立 API key：

```bash
openssl rand -hex 32 > .api-key
chmod 600 .api-key
```

不要将 `.api-key` 提交到 Git、日志、Issue 或示例配置中。

### 8.2 生产启动命令

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

"$DEPLOY_ROOT/llama.cpp/build/bin/llama-server" \
  -m "$MODEL" \
  -ngl 999 \
  --split-mode layer \
  --tensor-split 1,1,1,1 \
  --ctx-size 131072 \
  --batch-size 2048 \
  --ubatch-size 512 \
  --flash-attn on \
  --parallel 1 \
  --host 0.0.0.0 \
  --port 8001 \
  --jinja \
  --chat-template-file "$DEPLOY_ROOT/qwen3.8-flash-next-codex.jinja" \
  --reasoning-format deepseek \
  --alias qwen3.8-flash-next \
  --cache-prompt \
  --metrics \
  --api-key-file "$DEPLOY_ROOT/.api-key"
```

仓库中的 `launch.sh`、`start.sh`、`stop.sh` 和 `status.sh` 对上述参数进行了封装。

### 8.3 选择并发 1 的原因

该部署目标是单用户 Codex 和低延迟生成：

```text
--parallel 1
```

这可以让一个 slot 保留完整的 Prompt Cache，并避免多个请求分割 KV cache 和显存。

如果目标是多用户吞吐，应重新测试 slots、context 分配和 continuous batching，而不是直接沿用本报告数据。

## 9. 卡数探索

### 9.1 汇总

| GPU 配置 | Context | ubatch | 512-token Decode | 稳态 TTFT | 结论 |
|---|---:|---:|---:|---:|---|
| 3×4090，同 NUMA | 131K | 256 | **41.87 tok/s** | 0.304s | 可运行，但显存余量过小 |
| 4×4090，同 NUMA | 131K | 512 | **42.25 tok/s** | **0.159s** | 最佳生产配置 |
| 8×4090，跨 NUMA | 131K | 512 | **40.43 tok/s** | 0.202s | 单路更慢 |
| 4×4090，同 NUMA | 262K | 512 | **41.98 tok/s** | 0.310s | 可用，但余量较少 |

200-token 短输出对照：

| GPU 配置 | Context | Decode 中位数 |
|---|---:|---:|
| 3×4090 | 32K | 42.32 tok/s |
| 4×4090 | 131K | 42.23 tok/s |
| 8×4090 | 131K | 40.41 tok/s |

### 9.2 显存占用

4 卡、131K、压测后：

| GPU | 显存占用 |
|---|---:|
| GPU 0 | 19,086 MiB |
| GPU 1 | 17,004 MiB |
| GPU 2 | 17,404 MiB |
| GPU 3 | 17,082 MiB |

3 卡、131K、`ubatch=256`：

| GPU | 显存占用 |
|---|---:|
| GPU 0 | 23,982 MiB |
| GPU 1 | 22,226 MiB |
| GPU 2 | 21,908 MiB |

3 卡 GPU0 仅剩约 0.6GB。使用 `ubatch=512` 时出现过约 687MiB CUDA compute buffer
分配失败，因此不建议把 3 卡作为无人值守生产配置。

4 卡、262K：

| GPU | 显存占用 |
|---|---:|
| GPU 0 | 20,656 MiB |
| GPU 1 | 18,574 MiB |
| GPU 2 | 18,974 MiB |
| GPU 3 | 18,652 MiB |

### 9.3 卡数结论

- **最低全 GPU 卡数：3 卡。** 需要将 `ubatch` 降至 256，且显存余量很小。
- **最佳单副本：4 卡。** 解码略快，显存安全余量明显更好。
- **8 卡不适合单副本。** 512-token decode 比 4 卡慢约 4.3%。
- **最佳整机吞吐：2×4 卡副本。** 两个 NUMA node 各运行一条独立 lane。
- **2 卡不适合当前性能目标。** 需要明显 CPU offload，会偏离 40 tok/s 目标。

## 10. 基准测试方法

### 10.1 Decode 基准

- OpenAI-compatible `/v1/completions` streaming API
- 固定英文技术提示词
- `temperature=0`
- `ignore_eos=true`
- Decode 速度按首 token 之后计算：

```text
(completion_tokens - 1) / (结束时间 - 首 token 时间)
```

- 200-token 测试用于接近社区参考口径
- 512-token 测试用于确认持续生成速度

运行：

```bash
./benchmark.py --runs 3 --output-tokens 512
```

### 10.2 生产实例复验

最终 systemd 实例启动后，单次 512-token 请求达到：

```text
44.17 tok/s
```

该单次结果用于确认服务化后没有性能回退；正式对比仍使用三次中位数 42.25 tok/s。

## 11. 长上下文实测

### 11.1 测试方法

在输入开头放置随机 8 位 passkey，随后填充到目标 token 数，在末尾要求模型仅输出 passkey。

为了测量完整冷 prefill，请求设置：

```text
cache_prompt=false
```

### 11.2 结果

| 实际输入 | 客户端 TTFT | 服务端 Prefill | Prefill 吞吐 | Decode | Passkey |
|---:|---:|---:|---:|---:|---|
| 7,999 tokens | 6.54s | 6.25s | 1,279.90 tok/s | 40.95 tok/s | 正确 |
| 31,999 tokens | 79.37s | 78.92s | 405.47 tok/s | 36.38 tok/s | 正确 |
| 63,999 tokens | 244.62s | 243.68s | 262.63 tok/s | 30.58 tok/s | 正确 |
| 119,999 tokens | 804.43s | 802.73s | 149.49 tok/s | 23.73 tok/s | 正确 |

### 11.3 结论

- 8K、32K、64K、120K 均能找回输入开头的 passkey。
- 8K 仍可交互；32K 冷 TTFT 已约 79 秒。
- 64K 冷 TTFT 约 4.1 分钟。
- 120K 冷 TTFT 约 13.4 分钟。
- 长 context 不仅影响 prefill，也会让 decode 从 42–44 tok/s 下降到约 23.7 tok/s。
- 131K 是可用容量，不是推荐的日常请求长度。
- 日常交互建议控制在 8K–16K；32K 以上更适合离线任务或重复查询同一长文。
- Passkey 只验证基础远距离信息保持，不等价于完整的长文推理或多跳问答评测。

运行：

```bash
./long_context_benchmark.py
```

## 12. Prompt Cache

### 12.1 显式开启

llama.cpp server 默认开启 prompt cache，但生产命令仍显式设置：

```text
--cache-prompt
```

这样可以避免未来默认值或启动脚本变更造成无意关闭。

### 12.2 Responses API 实测

连续发送两次完全相同的 8K 输入：

| 请求 | 输入 tokens | cached tokens | 总耗时 | 实际 prefill |
|---|---:|---:|---:|---:|
| 第一次 | 8,008 | 0 | 5.49s | 4.27s / 8,008 tokens |
| 第二次 | 8,008 | **8,004** | **0.90s** | 0.129s / 4 new tokens |

第二次请求复用了 99.95% 的输入 token，总耗时下降约 83.6%。

这正是 Codex 多轮会话需要的行为：历史不变时只 prefill 新增消息，而不是每轮重新读取完整
仓库上下文和对话历史。

运行：

```bash
./prompt_cache_benchmark.py
```

### 12.3 Cache 使用边界

- Cache 依赖相同 token 前缀。
- 改变 system prompt、工具定义、消息顺序或序列化方式可能降低命中率。
- `--parallel 1` 让单用户场景更容易保留完整 slot cache。
- 服务重启后，当前内存中的 slot cache 会丢失。
- 本报告没有启用持久化 slot cache。

## 13. OpenAI-Compatible API

### 13.1 基础信息

```text
Base URL: http://127.0.0.1:8001/v1
Model:    qwen3.8-flash-next
```

可用接口包括：

- `/v1/models`
- `/v1/responses`
- `/v1/chat/completions`
- `/v1/completions`

### 13.2 Responses API

```bash
curl -N http://127.0.0.1:8001/v1/responses \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8-flash-next",
    "input": "Reply with OK",
    "stream": true
  }'
```

实际返回包含：

- reasoning item
- message item
- streaming text delta
- usage
- cached token 统计

### 13.3 Chat Completions

```bash
curl -N http://127.0.0.1:8001/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8-flash-next",
    "messages": [
      {"role": "user", "content": "Write a Python LRU cache."}
    ],
    "temperature": 0,
    "max_tokens": 1024,
    "stream": true
  }'
```

模型会生成 reasoning 内容。过小的输出 token 上限可能被 reasoning 消耗，应为最终答案保留足够空间。

## 14. Codex 接入

### 14.1 Codex Profile

建议创建独立 profile，不覆盖现有默认 provider：

```text
~/.codex/qwen-flash.config.toml
```

示例：

```toml
model_provider = "qwen_flash"
model = "qwen3.8-flash-next"
model_reasoning_effort = "medium"
model_context_window = 131072
model_auto_compact_token_limit = 114688
model_catalog_json = "/path/to/deploy/codex-models.json"
disable_response_storage = true

[model_providers.qwen_flash]
name = "Qwen3.8 Flash Next (local)"
base_url = "http://127.0.0.1:8001/v1"
wire_api = "responses"
request_max_retries = 2
stream_max_retries = 2
stream_idle_timeout_ms = 900000

[model_providers.qwen_flash.auth]
command = "/usr/bin/cat"
args = ["/path/to/deploy/.api-key"]
refresh_interval_ms = 0
```

启动：

```bash
codex -p qwen-flash
```

非交互调用：

```bash
codex -p qwen-flash exec "Inspect the repository and run the most relevant tests."
```

### 14.2 Model Catalog

仓库中的 `codex-models.json` 为 Codex 声明：

- 模型 slug：`qwen3.8-flash-next`
- Context：131,072
- 输入模态：text
- Responses API
- Freeform apply-patch tool

如果不提供本地 model catalog，Codex 可能使用 fallback metadata，从而错误估计 context 或工具能力。

### 14.3 Codex 验收

已完成：

- Responses API reasoning + message 返回
- Codex profile 加载
- 131K model metadata 加载
- shell tool call
- Prompt Cache 多轮复用

工具调用测试中，模型成功执行 `pwd` 并返回正确工作目录。

### 14.4 多轮会话的 Chat Template 兼容修复

模型 GGUF 内置模板只允许 `system` / `developer` 消息连续出现在消息列表开头。Codex 的
Responses API 多轮历史可能在 user、assistant 或 tool 消息之后再次插入 developer 消息，原模板会返回：

```text
HTTP 500
Jinja Exception: System message must be at the beginning.
```

Cloudflare 或 Codex 此时可能只显示通用的高负载提示，但根因是 llama.cpp 的模板异常，单纯重启服务不会修复。

仓库中的 `qwen3.8-flash-next-codex.jinja` 做了两项兼容处理：

1. 按原始顺序收集全部 `system` / `developer` 内容，并合并到开头的 system block。
2. 后续渲染消息历史时跳过这些已合并消息，不再抛出顺序异常。

启动时必须显式加载：

```bash
--chat-template-file "$DEPLOY_ROOT/qwen3.8-flash-next-codex.jinja"
```

2026-08-28 实机验证结果：

- 本机 `/v1/responses`：HTTP 200。
- Cloudflare 公网 `/v1/models` 与 `/v1/responses`：HTTP 200。
- 包含中途 developer 消息的请求正常完成，模板异常计数未增加。
- `codex exec -p qwen-flash` 完整链路正常返回。

排查时先确认进程参数中存在 `--chat-template-file`，再检查日志：

```bash
systemctl --user status qwen3.8-flash-next.service --no-pager -l
grep -n "System message must be at the beginning" llama-server.log | tail
```

llama.cpp 仍可能提示跳过 Responses API 中尚不支持的 `custom`、`namespace`、`tool_search` 或
`web_search` 工具类型；这与本节的消息顺序 500 是两个独立问题。

## 15. DFlash2：为什么没有使用

当前 Flash-Next llama.cpp 服务没有使用 DFlash2。

DFlash2 是旧 Qwen3.8-27B SGLang 路线中使用的独立 speculative draft model：

```text
z-lab/Qwen3.8-27B-DFlash2
```

旧路线在同一台机器上的长 context decode 数据：

| 实际 prompt | Qwen3.8-27B + DFlash2 Decode |
|---:|---:|
| 约 61K | 57.54 tok/s |
| 约 112K | 34.22 tok/s |

它不能直接用于当前服务：

1. Draft checkpoint 的目标模型是 Qwen3.8-27B，不是 Qwen3.8-Flash-Next。
2. 当前 Qwen4Exp llama.cpp 分支没有实现该架构的 DFlash2/MTP/NEXTN speculative decode。
3. 不同目标模型之间不能安全共享 speculative draft。

因此当前架构是：

```text
Qwen3.8-Flash-Next IQ4_XS
  + llama.cpp layer split
  + prompt cache
  + Responses API
  + Codex profile
  - DFlash2
  - MTP/NEXTN
```

如果目标是 Flash-Next 模型能力和 4 卡资源占用，使用当前方案；如果目标是超长活跃上下文的
decode 速度，旧 27B + DFlash2 仍有参考价值，但它是另一套模型服务。

## 16. systemd 部署

仓库包含示例 user unit：

```text
qwen3.8-flash-next.service
```

公开发布前，应把 unit 中的部署路径修改为自己的实际路径。

安装：

```bash
systemctl --user link "$DEPLOY_ROOT/qwen3.8-flash-next.service"
systemctl --user daemon-reload
systemctl --user enable --now qwen3.8-flash-next.service
```

允许退出登录后继续运行：

```bash
loginctl enable-linger "$USER"
```

运维：

```bash
systemctl --user status qwen3.8-flash-next.service --no-pager -l
systemctl --user restart qwen3.8-flash-next.service
journalctl --user -u qwen3.8-flash-next.service -f
```

也可以使用仓库脚本：

```bash
./start.sh 4gpu
./status.sh
./stop.sh
```

## 17. Cloudflare Tunnel（可选）

如需公网访问，可以将 Cloudflare Tunnel 的 HTTP origin 指向：

```text
http://127.0.0.1:8001
```

公网客户端仍使用：

```text
https://your-domain.example/v1
```

建议：

- 保留 llama.cpp API key 鉴权。
- 不要在仓库中公开 Tunnel token、API key 或真实内部域名。
- 优先使用 Cloudflare Access、IP 策略或额外反向代理访问控制。
- Codex 使用 streaming Responses API，避免长生成等待到请求结束才返回。
- 验证 `/v1/models` 和 `/v1/responses`，不能只检查 Tunnel 进程状态。

公网流式 Responses API 已在本次部署中完成端到端验证，但公开报告不记录真实域名和凭据。

## 18. 安全与开源发布

不要提交：

- `.api-key`
- `.env*`
- Cloudflare token
- GGUF 权重
- Hugging Face / ModelScope cache
- `llama-server.log`
- PID 文件
- `__pycache__`
- llama.cpp build 目录
- 嵌套的 llama.cpp Git clone

发布前运行：

```bash
git status --short
git ls-files | grep -E '(api-key|\.env|token|\.gguf|llama-server\.log)'
```

并使用 secret scanner 检查 Git 历史，而不仅是工作区。

模型权重和上游源码有各自的许可证。不要把模型文件或 llama.cpp 源码直接重新打包进本仓库，
除非已经核验并遵守对应许可证和分发条款。

## 19. 已知限制

- Qwen4Exp llama.cpp 支持来自实验分支，不是所有主线版本都可直接加载该架构。
- 当前没有 Flash-Next speculative decoding。
- 单 slot 配置面向单用户，不代表高并发吞吐。
- 长 context 冷 prefill 成本很高。
- Passkey 测试不代表完整长上下文推理质量。
- IQ4_XS 是质量、显存和速度折中，没有在本报告中完成系统化质量评测。
- “理论上强于 27B”需要业务评测验证，不能仅凭总参数量判断。
- Cloudflare、反向代理和客户端可能有各自的超时限制。

## 20. 推荐部署矩阵

| 目标 | 推荐方案 |
|---|---|
| 单用户、速度与质量平衡 | 4×4090、IQ4_XS、131K、layer split |
| 最低卡数实验 | 3×4090、ubatch 256，谨慎使用 |
| 原生最大 context 实验 | 4×4090、262K，接受更小显存余量 |
| 整机并发吞吐 | 两个独立 4 卡副本 |
| Codex 多轮会话 | 4 卡 + 131K + prompt cache + Responses API |
| 超长冷文档 | 离线处理，避免实时交互预期 |
| 超长活跃 context decode 优先 | 评估另一套 27B + DFlash2 服务 |

## 21. 文件说明

| 文件 | 用途 |
|---|---|
| `README.md` | 完整部署和实验报告 |
| `launch.sh` | 3/4/8 GPU llama-server 启动入口 |
| `start.sh` | 后台或 systemd 启动 |
| `stop.sh` | 停止服务 |
| `status.sh` | 服务和 GPU 状态 |
| `benchmark.py` | 短输出 decode 基准 |
| `long_context_benchmark.py` | 长 context prefill 与 passkey 测试 |
| `prompt_cache_benchmark.py` | Responses API cache 命中测试 |
| `codex-models.json` | Codex 本地模型 metadata |
| `qwen3.8-flash-next-codex.jinja` | 兼容 Codex 多轮 developer/system 消息的模板 |
| `qwen3.8-flash-next.service` | systemd user unit 示例 |

## 22. 复现检查清单

- [ ] 核对 GPU 拓扑与 NUMA node
- [ ] 准备足够 NVMe 空间
- [ ] 构建 SM89 CUDA llama.cpp
- [ ] 下载并校验 3 个 IQ4_XS 分片
- [ ] 创建独立 `.api-key`
- [ ] 修改模型路径和 systemd 部署路径
- [ ] 首先启动 4 卡 layer split
- [ ] 验证 `/v1/models`
- [ ] 验证 `/v1/responses` streaming
- [ ] 确认进程已加载 Codex 兼容 chat template
- [ ] 验证中途 developer 消息不会触发 HTTP 500
- [ ] 运行 200/512-token decode benchmark
- [ ] 连续请求验证 Prompt Cache
- [ ] 根据业务长度测试 8K/32K context
- [ ] 配置 Codex profile
- [ ] 验证至少一次 shell tool call
- [ ] 公网发布前运行 secret scan

## 致谢

本部署主要参考了 SGLang 官方文档、Unsloth 的 Qwen4Exp llama.cpp 实现，以及
`tonyd2wild/Qwen3.8-Flash-Next-Fleet-Deploy` 的 RTX 3090 实践报告。

特别有价值的社区结论是：Qwen3.8-Flash-Next 在消费级 GPU 上的高性能路径并不一定是
官方数据中心 GPU 配置的直接缩小版；GGUF mmap、PCIe 拓扑和多卡切分方式可能比理论 FLOPS
更决定实际单 token 延迟。
