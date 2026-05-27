---
name: llama-cpp-local-qwen
description: Notes for using local llama.cpp Qwen 3.6, Qwen 3.6 MTP, and Qwen 3.5 GGUF models from the pi-llama-cpp provider. Use when the selected model ID starts with llama-cpp/qwen3.6- or llama-cpp/qwen3.5-.
---

# llama.cpp local Qwen notes

When using `llama-cpp/qwen3.6-*` or `llama-cpp/qwen3.5-*`, the model is served locally by llama.cpp from GGUF files. Downloads and startup can take a long time on first use. Prefer concise prompts when memory is tight, and avoid assuming cloud-model latency.
