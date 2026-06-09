# Models

Specifications for the four SLMs integrated in this framework. All use llama.cpp with GGUF quantization.

## Model Registry

| Model | Parameters | GGUF File | Size | Template | GPU |
|-------|-----------|-----------|------|----------|-----|
| Qwen2 | 0.5B | `qwen2.5-0.5b-instruct-q4_0.gguf` | 409 MB | ChatML | No |
| Qwen3 | 0.6B | `Qwen3-0.6B-Q4_0.gguf` | 409 MB | ChatML | No |
| TinyLlama | 1.1B | `tinyllama-1.1b-chat-v1.0.Q4_0.gguf` | 608 MB | Llama2-style | No |
| Phi3 | 3.8B | `Phi-3-mini-4k-instruct-q4.gguf` | 2.2 GB | Phi-3 custom | **Yes** |

**Default location:** the framework resolves GGUF files from the sibling thesis repo at `../SLMs-master-thesis/Tesis/Codigo/models/gguf/` (no re-download needed). Override with `SLM_GGUF_DIR` or place copies in local `models/gguf/` (not tracked in git).

## Per-Model Details

### Qwen2.5-0.5B-Instruct

| Spec | Value |
|------|-------|
| Architecture | Qwen2.5 (Alibaba) |
| Context | 32K tokens (framework uses 2048) |
| Quantization | Q4_0 |
| Speed (CPU) | ~80–100 words/sec |
| `n_gpu_layers` | 0 |

ChatML template: `<|im_start|>system/user/assistant` blocks.

### Qwen3-0.6B

| Spec | Value |
|------|-------|
| Architecture | Qwen3 (Alibaba) |
| Context | 40K tokens (framework uses 2048) |
| Quantization | Q4_0 |
| Speed (CPU) | ~80–100 words/sec |
| `n_gpu_layers` | 0 |
| Special | Thinking tags — disabled via `/nothink` suffix + formatter strip |

ChatML template (same as Qwen2).

### TinyLlama-1.1B-Chat

| Spec | Value |
|------|-------|
| Architecture | Llama2-based |
| Context | 2K tokens |
| Quantization | Q4_0 |
| Speed (CPU) | ~60–80 words/sec |
| `n_gpu_layers` | 0 |

Llama2-style: `<|system|>`, `[INST]` user/assistant blocks.

### Phi-3-mini-4k-Instruct

| Spec | Value |
|------|-------|
| Architecture | Phi-3 (Microsoft) |
| Context | 4K tokens (framework uses 4096) |
| Quantization | Q4 |
| Speed (CPU) | ~0.1 words/sec — **unusable** |
| Speed (GPU) | ~18 words/sec |
| `n_gpu_layers` | **-1** (all layers on GPU) |

Phi-3 template: `<|system|>`, `<|user|>`, `<|assistant|>` tokens.

## GPU Configuration

### When to Use GPU

| Model size | `n_gpu_layers` | Rationale |
|-----------|---------------|-----------|
| < 1 GB | 0 | Already fast on CPU; GPU overhead negates benefit |
| 1–2 GB | 0 | Marginal GPU benefit (< 1.5×) |
| > 2 GB | -1 | CPU too slow (Phi-3: 73× speedup with GPU) |

### Phi-3 GPU Requirement

Phi-3 at 3.8B parameters generates ~0.1 words/sec on CPU (~16 min for 100 words).
GPU offloading (`n_gpu_layers=-1`) is required for practical runtimes.

| Environment | Backend | Speed | Notes |
|-------------|---------|-------|-------|
| Local Mac (M1/M2) | Metal (CUDA build with Metal) | ~18 words/sec | Default for local development |
| ClusterUY | CUDA (Tesla P100) | ~18 words/sec | Verified smoke test 2026-06-09 |

Empirical results on M2 Mac:

| Metric | CPU | GPU (Metal) | Improvement |
|--------|-----|-------------|-------------|
| Load time | 2.4s | 4.4s | Slower (one-time) |
| Generation (10 tokens) | 36.8s | 0.5s | **73× faster** |

### Verification

```bash
python -c "from llama_cpp import Llama; \
llm = Llama('models/gguf/Phi-3-mini-4k-instruct-q4.gguf', n_gpu_layers=-1, verbose=True)"
```

Look for: `offloaded 32/32 layers to GPU`

## Downloading GGUF Files

| Model | HuggingFace Source |
|-------|-------------------|
| Qwen2 | `Qwen/Qwen2.5-0.5B-Instruct-GGUF` |
| Qwen3 | `ggml-org/Qwen3-0.6B-GGUF` |
| TinyLlama | `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF` |
| Phi3 | `microsoft/Phi-3-mini-4k-instruct-gguf` |

Use Q4_0 quantization for models < 1 GB; Q4 for Phi-3.

## Adding a New Model

1. Download GGUF file to `models/gguf/`
2. Create wrapper in `src/slm_experiments/models/wrappers/` extending `llamacpp.py`
3. Implement `_format_prompt()`, `_get_stop_tokens()`, `_extract_response()`
4. Register in `phase1/configs.py` model registry
5. Test CPU speed; enable GPU if generation > 5s for 30 tokens
6. Update this document

## Hardware Requirements

**Minimum (CPU models):** Apple M1/M2 or equivalent, 8 GB RAM, 2 GB storage

**GPU (Phi-3, local):** Apple M1/M2 with Metal, 16 GB unified memory, ~3 GB GPU memory available

**GPU (Phi-3, ClusterUY):** Tesla P100 (16 GB VRAM) via Singularity + CUDA. See [clusteruy.md](clusteruy.md).
