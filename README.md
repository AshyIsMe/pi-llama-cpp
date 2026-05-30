# pi-llama-cpp

Pi provider extension for local Qwen 3.6 / Qwen 3.6 MTP and Qwen 3.5 GGUF models using
the official [llama.app](https://llama.app/) runtime for llama.cpp and Unsloth
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
3. uses a configured local llama.cpp checkout/build, an existing `llama` app on your machine, or installs the official llama.app runtime,
4. downloads the configured Unsloth GGUF quant from Hugging Face,
5. starts `llama serve` (or a configured `llama-server`) on `127.0.0.1:8080`, and
6. keeps it alive while Pi clients have active leases.

A bundled watchdog stops the local llama server process after the last Pi process exits.

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

- `runtime/llama-app/<backend>/` — managed llama.app install home, when the app is not already installed
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
- `LLAMA_CPP_LLAMA_APP_BINARY`: use an existing `llama` app binary directly.
- `LLAMA_CPP_LLAMA_APP_HOME`: managed install home for llama.app, default `~/.pi/llama-cpp/runtime/llama-app/<backend>`.
- `LLAMA_CPP_LLAMA_APP_INSTALL_URL`: installer URL, default `https://llama.app/install.sh`.
- `LLAMA_CPP_SERVER_BINARY`: use an existing `llama-server` binary directly instead of llama.app.
- `LLAMA_CPP_RELEASE_URL`: deprecated; direct llama.cpp release downloads are no longer supported.
- `LLAMA_CPP_SOURCE_DIR`: path to a local llama.cpp checkout/fork. When set, the extension builds/uses it instead of llama.app.
- `LLAMA_CPP_BUILD_DIR`: local CMake build directory. Absolute paths are used as-is; relative paths are resolved under `LLAMA_CPP_SOURCE_DIR`. Defaults to `build` under `LLAMA_CPP_SOURCE_DIR`.
- `LLAMA_CPP_BUILD_POLICY`: `auto` (default; build only when no `llama-server` is found), `always`, or `never`.
- `LLAMA_CPP_CMAKE_ARGS`: extra args appended to `cmake -S ... -B ...`; `-DCMAKE_BUILD_TYPE=Release` and backend flags like `-DGGML_CUDA=ON` are added automatically before these args.
- `LLAMA_CPP_BUILD_ARGS`: extra args appended to `cmake --build ... --parallel`.
- `LLAMA_CPP_SERVER_ARGS`: args appended to `llama serve` / `llama-server`, default `--parallel 1 --timeout 600`.
- `LLAMA_CPP_SERVER_EXTRA_ARGS`: additional server CLI args appended after `LLAMA_CPP_SERVER_ARGS`; useful for local fork-specific flags. In `settings.json`, `serverArgs`, `serverExtraArgs`, `cmakeArgs`, and `buildArgs` may be either shell-style strings or arrays of strings.

The extension always starts the local llama.cpp server with `--reasoning off` before extra args so local models run with thinking disabled by default.
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

Example local fork configuration:

```json
{
  "backend": "cuda",
  "sourceDir": "~/src/llama.cpp-fork",
  "buildDir": "build-pi",
  "buildPolicy": "auto",
  "cmakeArgs": "-DGGML_CUDA_FA_ALL_QUANTS=ON",
  "serverExtraArgs": ["--flash-attn", "on"]
}
```

If you already built the fork yourself, either set `serverBinary` to the exact
`llama-server` path or set `buildDir` to the build directory and
`buildPolicy` to `never`.

## Requirements

- `curl`
- `cmake` when building from `LLAMA_CPP_SOURCE_DIR`
- Windows currently needs `LLAMA_CPP_SERVER_BINARY` or `LLAMA_CPP_LLAMA_APP_BINARY` because `llama.app/install.sh` is Unix-only.
- enough disk and RAM/VRAM for your chosen GGUF quant
- optional backend tools for detection: `nvidia-smi`, `rocminfo`/`rocm-smi`, or
  `vulkaninfo`

macOS arm64 prefers Metal automatically. Non-interactive modes pick the first
available backend in this order: `metal`, `cuda`, `rocm`, `vulkan`, `cpu`.
