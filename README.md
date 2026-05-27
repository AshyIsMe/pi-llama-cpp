# pi-llama-cpp

Pi provider extension for local Qwen 3.6 / Qwen 3.6 MTP and Qwen 3.5 GGUF models using
[llama.cpp](https://github.com/ggml-org/llama.cpp) binary releases and Unsloth
quantized models.

Inspired by https://github.com/mitsuhiko/pi-ds4.

It registers these `/model` entries:

- `llama-cpp/qwen3.6-27b`
- `llama-cpp/qwen3.6-35b-a3b`
- `llama-cpp/qwen3.6-27b-mtp`
- `llama-cpp/qwen3.6-35b-a3b-mtp`
- `llama-cpp/qwen3.5-0.8b`
- `llama-cpp/qwen3.5-2b`
- `llama-cpp/qwen3.5-4b`
- `llama-cpp/qwen3.5-9b`
- `llama-cpp/qwen3.5-27b`
- `llama-cpp/qwen3.5-35b-a3b`
- `llama-cpp/qwen3.5-122b-a10b`
- `llama-cpp/qwen3.5-397b-a17b`

On first use it:

1. detects available llama.cpp backends (`metal`, `cuda`, `rocm`, `vulkan`, `cpu`),
2. asks you to choose if multiple accelerated backends are plausible,
3. downloads the matching latest llama.cpp binary release,
4. downloads the configured Unsloth GGUF quant from Hugging Face,
5. starts `llama-server` on `127.0.0.1:8080`, and
6. keeps it alive while Pi clients have active leases.

A bundled watchdog stops `llama-server` after the last Pi process exits.

## Install

```sh
pi install https://github.com/your-name/pi-llama-cpp
```

For local development:

```sh
~/src/pi-llama-cpp/install-pi-extension-local.sh
```

Then restart pi or run `/reload`.

## Runtime layout

State lives under `~/.pi/llama-cpp`:

- `runtime/<backend>/` — downloaded llama.cpp binary release
- `models/` — downloaded GGUF files
- `clients/` — active Pi process leases
- `settings.json` — optional configuration
- `log` — download/server/watchdog log

Use `/llama-cpp` in Pi to show current status and the log path.

## Configuration

Create `~/.pi/llama-cpp/settings.json`. Environment variables with the same
names win over settings-file values. Settings may use env var names,
camel-case keys without `LLAMA_CPP_`, or lower snake-case keys.

Important options:

- `LLAMA_CPP_BACKEND`: force `metal`, `cuda`, `rocm`, `vulkan`, or `cpu`.
- `LLAMA_CPP_RELEASE_URL`: force a specific llama.cpp release zip URL.
- `LLAMA_CPP_SERVER_BINARY`: use an existing `llama-server` binary.
- `LLAMA_CPP_SERVER_ARGS`: extra args appended to `llama-server`, default `--parallel 1 --timeout 600`. The extension always starts llama.cpp with `--reasoning off` before extra args so local models run with thinking disabled by default.
- `LLAMA_CPP_N_GPU_LAYERS`: default `999`.
- `LLAMA_CPP_CTX`: default context size, default `131072`.
- `LLAMA_CPP_MODEL_QUANT`: default GGUF filename match, default `Q4_K_M`.
- Per-model repo/file overrides use the model key, e.g. `LLAMA_CPP_QWEN36_27B_REPO`, `LLAMA_CPP_QWEN36_35B_A3B_MTP_FILE`, or `LLAMA_CPP_QWEN35_122B_A10B_REPO`.
- Legacy `LLAMA_CPP_QWEN36_REPO` / `LLAMA_CPP_QWEN36_FILE` still apply to `qwen3.6-27b`.
- Legacy `LLAMA_CPP_QWEN35_REPO` / `LLAMA_CPP_QWEN35_FILE` still apply to `qwen3.5-35b-a3b`.

Built-in repo defaults use Unsloth GGUF repos such as `unsloth/Qwen3.6-27B-MTP-GGUF` and `unsloth/Qwen3.5-122B-A10B-GGUF`.

If Unsloth publishes any variant under a different repo name, set the repo/file values
explicitly. If `*File` is null, the extension lists the HF repo and picks the
first `.gguf` containing the quant string, falling back to `Q4_K_M` or the first
GGUF file.

## Requirements

- `curl`, `tar`, and `unzip`
- enough disk and RAM/VRAM for your chosen GGUF quant
- optional backend tools for detection: `nvidia-smi`, `rocminfo`/`rocm-smi`, or
  `vulkaninfo`

macOS arm64 prefers Metal automatically. Non-interactive modes pick the first
available backend in this order: `metal`, `cuda`, `rocm`, `vulkan`, `cpu`.
