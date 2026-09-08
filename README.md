# Qwen3.8-Flash-Next on RTX 4090 48GB

在一台 **8 x RTX 4090 48GB、无 NVLink** 的服务器上，探索
Qwen3.8-Flash-Next 面向单用户 Codex 的低延迟部署路线。

本文所有新数据均为 2026-09-08 实机结果。上一台 8 x 24GB 机器的数据仅作为历史对照。

## 结论

当前最佳生产路线：

| 项目 | 配置 |
|---|---|
| 引擎 | Unsloth llama.cpp MTP PR #144 |
| 固定提交 | `a9e9c3c5fed8a0bb5cc617532d0d16b8f59c13e0` |
| 模型 | `unsloth/Qwen3.8-Flash-Next-GGUF` |
| 量化 | `UD-IQ4_XS`，93.68GB |
| GPU | **2 x RTX 4090 48GB**，同 NUMA、同 PIX 组 |
| 切分 | `--split-mode layer` |
| Context | **262,144** |
| MTP | shared Q8_0，`draft=4` |
| 并发 | 1 slot |
| API | OpenAI-compatible Responses / Chat Completions / Completions |
| 512-token decode | **123.61 tok/s**，三次中位数 |
| 英文单 prompt 峰值 | **126.09 tok/s**，`draft=5` |

最重要的发现：

1. 48GB 卡让 `UD-IQ4_XS` 在 2 卡上全 GPU 运行，2 卡比 3/4 卡更快。
2. MTP 已可用。旧报告中“llama.cpp 不支持 Flash-Next MTP”的结论已经过期。
3. 面向中英文和代码生成的稳健默认是 `draft=4`；`draft=5` 只在本次英文 prompt 上略快。
4. 真实 250K 请求完成并正确检索 passkey，不是只验证服务能分配 262K KV cache。
5. 该机器所有 GPU 间 PCIe P2P 不可用，row split 无法加载；layer split 是正确路线。
6. MTP 适合单并发低延迟，不应直接用于高并发吞吐服务。

## 实验环境

### 硬件

- GPU：8 x NVIDIA GeForce RTX 4090，49,140MiB/卡，SM89
- Driver：580.173.02
- PCIe：Gen4 x16
- CPU：2 x Intel Xeon Gold 6462C，64 物理核 / 128 线程
- RAM：1TiB
- GPU 0-3：NUMA 0
- GPU 4-7：NUMA 1
- GPU 4/5 与 GPU 6/7 分别为 PIX 组
- 无 NVLink
- CUDA P2P read/write 与 PCIe P2P 均不可用

测试期间 GPU 0-2 有其他任务，因此所有新基准只使用 GPU 4-7，并将进程绑定到：

```text
CPU 32-63,96-127 / NUMA 1
```

仓库启动器默认选择 GPU 4、5。其他机器可以通过 `GPUS` 和 `CPU_SET` 显式覆盖。

### 软件

- CUDA Toolkit：12.8
- llama.cpp：`0.3.0-dev (build 10794, commit a9e9c3c5f)`
- CUDA architecture：89
- uv：0.12.10
- cmake：4.4.3
- ninja：1.13.2

## 为什么选择 llama.cpp GGUF

Qwen3.8-Flash-Next 包含很大的 PLE n-gram embedding table。GGUF + mmap 可以让该表主要
保留在系统内存和页缓存中，计算权重则 offload 到 GPU。

本机 2 卡 131K、无 MTP 时显存约为：

| GPU | 显存 |
|---|---:|
| GPU 4 | 35,697MiB |
| GPU 5 | 34,089MiB |

最终 2 卡 262K + MTP `draft=4` 实例约为 39,493MiB / 42,263MiB，仍有可用余量。

单卡无法全 GPU 加载 IQ4_XS：即使 context 降到 32K，仍尝试分配约 61.2GiB CUDA
权重缓冲。将 32 层 offload 到单卡后可以运行，但只有 17.83 tok/s。

## 与参考路线的关系

本次参考了：

- [A-XIANGQAQ/Qwen3.8-Flash-Next-4090](https://github.com/A-XIANGQAQ/Qwen3.8-Flash-Next-4090)
- [TomPython/Qwen-3.8-Flash-Next](https://github.com/TomPython/Qwen-3.8-Flash-Next)
- [Unsloth Qwen3.8 Flash Next GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)
- [Unsloth llama.cpp MTP PR #144](https://github.com/unslothai/llama.cpp/pull/144)

A-XIANGQAQ 的单 48GB 卡方案和 TomPython 的双卡方案使用 NVFP4 + Lsglang/lk_moe
CPU-GPU 混合推理，适合卡数受限场景。其公开 decode 约为 36-40 tok/s 和 34.8 tok/s。

本机目标是“卡数不限、单流尽可能快”，因此选择 2 x 48GB 全 GPU GGUF + MTP。
这些结果不是同一量化和同一引擎下的严格横向质量评测。

## 使用 uv 和国内源

所有 Python 环境由 `uv` 管理。引导脚本使用清华 PyPI 为主源、阿里云 PyPI 为备用源，
uv cache 也保存在项目目录内：

```bash
./tools/bootstrap.sh
```

如果系统尚无 uv，脚本只使用系统 Python 从上述国内 PyPI 安装固定版本 uv；随后所有
虚拟环境和 Python 包安装均由 uv 完成。

## 下载模型

默认使用 ModelScope 国内镜像：

```bash
./tools/download-model.sh
```

切换到 hf-mirror：

```bash
DOWNLOAD_BASE=https://hf-mirror.com/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/main \
  ./tools/download-model.sh
```

下载器支持 HTTP Range、分块内断点续传、低速超时和已有分块复用。默认只使用一个连接，
避免代理或镜像在高并发下载时重置 TLS。

国内镜像不可用时，可在当前 shell 临时设置标准代理环境变量后重试。不要把代理地址、
凭据或环境文件提交到仓库。

模型文件：

| 文件 | bytes | SHA256 |
|---|---:|---|
| `UD-IQ4_XS/...00001-of-00003.gguf` | 10,946,624 | `5ce89370...e71a4` |
| `UD-IQ4_XS/...00002-of-00003.gguf` | 49,835,229,856 | `577a38a2...aaa7` |
| `UD-IQ4_XS/...00003-of-00003.gguf` | 43,836,407,744 | `d4634e6d...8833` |
| `MTP/...shared-Q8_0.gguf` | 2,786,568,256 | `5ff54097...e6` |

完整校验值位于 `model-checksums.sha256`：

```bash
sha256sum --check model-checksums.sha256
```

模型、分块和下载缓存均被 Git 忽略。

## 构建 llama.cpp

```bash
./tools/build-llama.sh
```

脚本固定到 Unsloth MTP PR #144 的完整 commit，并针对 SM89 构建
`llama-server` 和 `llama-bench`。

不能使用旧的 `qwen4exp/qwen3.8-flash-next` commit `eaf9376` 运行 shared MTP。
该版本虽然显示 `--spec-type draft-mtp`，但缺少 shared tensor borrowing，实际会因
`token_embd.weight not found` 退出。

shared MTP 在新版本启动时会先输出一条内存预估警告：

```text
borrow_shared_tensor: this model is a draft head without its own token_embd.weight
failed to measure the memory of the extra model, fitting without it
```

随后若日志出现 `loading draft model`、`model loaded` 和 `draft acceptance`，MTP
就是正常工作的。这是 shared sidecar 单独预估时无法借用主模型 tensor 的已知行为。

## 启动

生成本地 API key：

```bash
openssl rand -out .api-key -hex 32
chmod 600 .api-key
```

最佳配置直接启动：

```bash
./launch.sh
```

等价关键参数：

```text
GPU 4,5
--split-mode layer
--ctx-size 262144
--parallel 1
--spec-type draft-mtp
--spec-draft-model models/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf
--spec-draft-n-max 4
--cache-prompt
```

常用覆盖：

```bash
# 选择其他两张物理 GPU 和对应 NUMA CPU
GPUS=0,1 CPU_SET=0-31,64-95 ./launch.sh 2gpu

# 131K、关闭 MTP
CONTEXT=131072 SPEC_MTP=0 ./launch.sh 2gpu

# 英文固定输出可实验 draft=5
DRAFT_N=5 ./launch.sh 2gpu
```

默认仅监听 `127.0.0.1:8001`。确需局域网监听时显式设置 `HOST=0.0.0.0`，并确保
`.api-key` 存在及防火墙策略正确。

启动器会检查目标 GPU 上是否已有 compute process；检测到忙碌 GPU 时默认拒绝启动。
只有确认可以共享时才使用 `ALLOW_BUSY_GPUS=1` 覆盖。

后台运行：

```bash
./start.sh
./status.sh
./stop.sh
```

需要登录后自动启动时，安装用户级 systemd 服务。安装脚本会把当前仓库的绝对路径写入
生成的单元文件，因此仓库不必位于家目录：

```bash
./tools/install-service.sh
```

## 卡数实测

131K context、layer split、512 输出 tokens、greedy、3 次中位数：

| 配置 | MTP | Decode | 相对 2 卡无 MTP |
|---|---|---:|---:|
| 2 x 48GB | 关闭 | **78.95 tok/s** | 1.00x |
| 3 x 48GB | 关闭 | 75.89 tok/s | 0.96x |
| 4 x 48GB | 关闭 | 75.46 tok/s | 0.96x |
| 2 x 48GB | draft=5 | **126.09 tok/s** | 1.60x |
| 3 x 48GB | draft=5 | 120.92 tok/s | 1.53x |
| 4 x 48GB | draft=5 | 114.71 tok/s | 1.45x |

无 MTP 显存：

| 配置 | 各卡显存 |
|---|---|
| 2 卡 | 35,697 / 34,089MiB |
| 3 卡 | 25,239 / 23,545 / 23,227MiB |
| 4 卡 | 19,991 / 17,901 / 18,301 / 17,979MiB |

layer split 在设备边界传递 activation。增加卡数不会让单 token 的各层并行执行，反而增加
设备边界和同步，因此能放下模型的最少全 GPU 卡数通常最快。

## MTP draft 长度

2 卡、131K、固定英文技术 prompt：

| draft | Decode 中位数 | 接受率 | 相对无 MTP |
|---:|---:|---:|---:|
| 关闭 | 78.95 | - | 1.00x |
| 1 | 100.56 | 84.78% | 1.27x |
| 2 | 110.90 | 74.82% | 1.40x |
| 3 | 118.12 | 67.73% | 1.50x |
| 4 | 123.61 | 62.81% | 1.57x |
| 5 | **126.09** | 57.62% | **1.60x** |
| 6 | 118.88 | 49.31% | 1.51x |

不同 prompt 的 512-token 单次复验：

| prompt | draft=4 | draft=5 |
|---|---:|---:|
| 英文技术说明 | 123.61（3 次中位数） | **126.09**（3 次中位数） |
| 中文技术说明 | **121.08** | 118.34 |
| Python 代码生成 | **122.13** | 121.76 |

三类 prompt 等权平均几乎相同，但 draft=4 的接受率更高、波动更小，所以生产默认选择 4。

MTP 的输出仍由主模型验证，不改变 greedy 结果。温度升高会降低接受率。高并发时 draft
会占用本可用于 target batching 的计算资源，不能根据单并发结果推断高并发吞吐。

## Split mode

`layer` 是本机唯一推荐模式。

2 卡 `row` 对照在加载阶段直接失败：

```text
device CUDA0 does not support split buffers
```

这与本机所有 GPU 间 P2P 不可用一致。不要在无 NVLink/P2P 的消费卡上把 row/tensor split
当作默认路线。

## Context 与 Prompt Cache

2 卡、262K、MTP draft=4 的冷 passkey 检索：

| 实际输入 | TTFT | 末端 decode | passkey |
|---:|---:|---:|---|
| 7,999 | 3.41s | 121.27 tok/s | 正确 |
| 31,999 | 13.48s | 69.39 tok/s | 正确 |
| 119,999 | 81.04s | 55.81 tok/s | 正确 |
| 249,999 | 273.63s | 54.49 tok/s | 正确 |

250K 证明模型能处理接近原生 262K 上限的真实请求，但 4.6 分钟冷 TTFT 不适合交互。
日常 Codex 应依赖 prompt cache 和自动压缩，而不是反复冷 prefill 全历史。

相同 8K Responses 请求：

| 请求 | 输入 | cached | 总耗时 |
|---|---:|---:|---:|
| 第一次 | 8,008 | 0 | 3.82s |
| 第二次 | 8,008 | 8,004 | **0.414s** |

第二次复用 99.95% 输入，总耗时下降约 89.2%。

## API 与 Codex

```text
Base URL: http://127.0.0.1:8001/v1
Model:    qwen3.8-flash-next
```

Responses API 示例：

```bash
curl http://127.0.0.1:8001/v1/responses \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.8-flash-next","input":"Reply with OK"}'
```

Codex provider 示例：

```toml
model_provider = "qwen_flash"
model = "qwen3.8-flash-next"
model_reasoning_effort = "medium"
model_context_window = 262144
model_auto_compact_token_limit = 229376
model_catalog_json = "/path/to/Qwen3.8-Flash-Next-4090-Deploy/codex-models.json"
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
args = ["/path/to/Qwen3.8-Flash-Next-4090-Deploy/.api-key"]
refresh_interval_ms = 0
```

仓库的 `qwen3.8-flash-next-codex.jinja` 会将多轮历史中的 system/developer 消息按原顺序
归并到开头，避免内置模板抛出：

```text
Jinja Exception: System message must be at the beginning.
```

本机已验证：

- Responses API 返回 reasoning + message
- assistant 历史后再次出现 developer 消息：HTTP 200，正确返回 `amber`
- function_call / function_call_output 后再次出现 developer 消息：HTTP 200
- Prompt Cache 返回 cached token 统计

## 基准方法

`benchmark.py` 使用 streaming `/v1/completions`：

- `temperature=0`
- `ignore_eos=true`
- 默认生成 512 tokens
- 客户端 decode = 首 token 之后的 tokens / 时间
- 同时记录服务端 `predicted_per_second`
- MTP 时记录 `draft_n`、`draft_n_accepted` 和接受率

```bash
UV_CACHE_DIR=$PWD/.cache/uv ./.tools/uv run --no-project \
  python benchmark.py --runs 3 --output-tokens 512
```

客户端与服务端速率在所有正式测试中基本一致。表中默认使用三次客户端中位数，不使用单次
峰值。

## 24GB 机器历史对照

上一台 8 x RTX 4090 24GB 上，相同 `UD-IQ4_XS` 的历史最佳为：

| 配置 | Context | Decode |
|---|---:|---:|
| 4 卡 layer split，无 MTP | 131K | 42.25 tok/s |
| 8 卡 layer split，无 MTP | 131K | 40.43 tok/s |

48GB 机器的 2 卡无 MTP 已达到 78.95 tok/s，启用 MTP 后达到约 122 tok/s。不要把旧机器
“4 卡最佳”和“MTP 不可用”的结论继续用于当前部署。

## 文件

| 文件 | 用途 |
|---|---|
| `launch.sh` | 1/2/3/4/8 卡、NUMA、MTP 启动入口 |
| `start.sh` / `stop.sh` / `status.sh` | 后台服务管理 |
| `tools/bootstrap.sh` | uv + 国内 PyPI 环境 |
| `tools/build-llama.sh` | 固定 MTP commit 的 SM89 构建 |
| `tools/download-model.sh` | ModelScope/hf-mirror 模型下载 |
| `tools/install-service.sh` | 按当前路径安装用户级 systemd 服务 |
| `tools/range-download.sh` | 可靠 Range 断点续传 |
| `model-checksums.sha256` | 模型完整 SHA256 |
| `benchmark.py` | Decode、服务端 timing、MTP 接受率 |
| `long_context_benchmark.py` | 冷长上下文 passkey |
| `prompt_cache_benchmark.py` | Responses Prompt Cache |
| `qwen3.8-flash-next-codex.jinja` | Codex 多轮模板兼容 |
| `codex-models.json` | Codex 本地模型目录 |

## 限制

- IQ4_XS 是速度、质量和显存的折中，本报告没有完成系统化质量评测。
- passkey 只验证远距离信息保持，不等价于长文多跳推理。
- 单并发 MTP 数据不能代表多用户吞吐。
- MTP 仍来自实验 PR，升级 llama.cpp 后必须重新验证 shared sidecar、API 和速度。
- 262K 是容量上限，不是推荐每轮都填满的日常输入长度。
- 服务默认只监听 localhost；公网暴露需要额外鉴权、TLS 和访问控制。

## 发布检查

不要提交：

- API key、代理地址或凭据
- GGUF 权重及下载分块
- Python 虚拟环境与 uv cache
- llama.cpp clone/build
- 服务日志、PID、原始基准输出

```bash
git status --short
git ls-files | grep -E '(api-key|\.env|token|\.gguf|llama-server\.log)' || true
git diff --check
```
