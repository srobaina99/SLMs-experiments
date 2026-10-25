# ============================================================
# slm-thesis — Phase 2 generation image (llama-cpp + CEFR-SP runtime)
#
# Multi-stage: compile llama-cpp-python for ClusterUY P100 (sm_60), then
# install CEFR-SP Python deps and prefetch bert-base-cased. The ~1.2 GB
# CEFR-SP checkpoint is NOT baked (gitignored); bind-mount at run time:
#   -v /path/to/level_estimator.ckpt:/workspace/data/cefr_sp/level_estimator.ckpt:ro
#
# TSAR ModernBERT pins stay in Dockerfile.eval (this image does not
# prefetch them). Do not install flash-attn (P100 is sm_60).
#
# Build (from repo root):
#   ./scripts/clusteruy/build_gen_image.sh
#   # or:
#   docker build --platform=linux/amd64 \
#     -f scripts/clusteruy/Dockerfile \
#     -t srobaina99/slm-thesis:latest .
# ============================================================

# ============================================================
# Stage 1: builder — compile llama-cpp-python with CUDA support
# for P100 (sm_60). Uses the devel image because nvcc / CUDA
# headers are required; nothing from this stage ships except the wheel.
# ============================================================
FROM --platform=linux/amd64 nvidia/cuda:12.1.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-dev \
        python3-pip \
        build-essential \
        cmake \
        ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Make the CUDA driver stubs discoverable at link time. The devel image ships
# /usr/local/cuda/lib64/stubs/libcuda.so, but:
#   (a) libggml-cuda.so has a DT_NEEDED entry for libcuda.so.1 (the SONAME of
#       the real driver), not libcuda.so, so we need a libcuda.so.1 symlink.
#   (b) The stubs dir isn't on ld's default search path; we add it via LDFLAGS
#       (-L and -rpath-link) so the linker can satisfy transitive deps when
#       producing the auxiliary mtmd binaries.
RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so \
           /usr/local/cuda/lib64/stubs/libcuda.so.1

ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LIBRARY_PATH}
ENV LDFLAGS="-L/usr/local/cuda/lib64/stubs -Wl,-rpath-link,/usr/local/cuda/lib64/stubs"

# 60 = P100 — the only GPU on ClusterUY. Multi-arch builds take 6× longer.
ENV CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=60"
ENV FORCE_CMAKE=1

# Build llama-cpp-python as a reusable wheel so stage 2 just installs it.
RUN pip wheel --no-cache-dir --no-deps --no-binary=llama-cpp-python \
        -w /wheels llama-cpp-python

# ============================================================
# Stage 2: runtime — llama-cpp-python + CEFR-SP scoring deps.
#
# CEFR-SP runs in ExperimentPipeline during generation (default ON).
# Needs torch + Lightning + bert-base-cased. The Zenodo ckpt is bind-mounted.
# matplotlib / seaborn omitted: cluster jobs pass --no-plot.
# ============================================================
FROM --platform=linux/amd64 nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3-pip \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /workspace

# Install the prebuilt CUDA-enabled llama-cpp-python wheel from stage 1.
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl && rm -rf /wheels

# Generation runtime deps (textstat / NLTK readability + content-word paths).
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        pandas==2.1.4 \
        tqdm==4.66.1 \
        textstat==0.7.4 \
        nltk==3.9.1

# Torch MUST come from the cu121 index with an exact pin that still ships sm_60
# (P100). Unpinned PyPI torch floats to CUDA 13 / sm_70+ and breaks ClusterUY.
# transformers pinned exact on 4.x; do not float to 5.x.
# Arch probe: torch.cuda.get_arch_list() is empty under some driver-less builds
# (incl. this CUDA 12.1 runtime image); fall back to _cuda_getArchFlags().
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu121 \
        torch==2.4.1 && \
    python -c "import torch; \
flags=(torch._C._cuda_getArchFlags() or '').split(); \
arches=torch.cuda.get_arch_list() or flags; \
assert torch.__version__.startswith('2.4.1'), torch.__version__; \
assert str(torch.version.cuda).startswith('12.1'), torch.version.cuda; \
assert 'sm_60' in arches, f'P100 unsupported: arches={arches!r} flags={flags!r}'; \
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'arch_list', arches)" && \
    pip install --no-cache-dir \
        transformers==4.55.4 \
        pytorch-lightning==2.4.0 && \
    python -c "import torch; \
flags=(torch._C._cuda_getArchFlags() or '').split(); \
arches=torch.cuda.get_arch_list() or flags; \
assert torch.__version__.startswith('2.4.1'), torch.__version__; \
assert 'sm_60' in arches, f'P100 unsupported after deps: arches={arches!r} flags={flags!r}'; \
print('post-deps torch', torch.__version__, 'cuda', torch.version.cuda, 'arch_list', arches)"

COPY scripts/clusteruy/prefetch_tsar_models.py ./scripts/clusteruy/prefetch_tsar_models.py

ENV HF_HOME=/opt/hf_home
ENV NLTK_DATA=/usr/share/nltk_data

# Prefetch bert-base-cased only (CEFR-SP encoder base). Skip TSAR ModernBERT.
RUN python scripts/clusteruy/prefetch_tsar_models.py \
        --hf-home /opt/hf_home \
        --include-cefr-sp-base \
        --skip-tsar

# Pre-populate NLTK data so no network call is needed on the compute node.
#   cmudict — textstat flesch_kincaid / gunning_fog / spache
#   punkt + averaged_perceptron_tagger — textstat / content-word POS
#   punkt_tab — NLTK 3.8+ sentence tokenizer (CEFR-SP sent_tokenize)
#   wordnet — content-word lemmatizer (metrics.py downloads at import if missing)
#   averaged_perceptron_tagger_eng — NLTK 3.8+ POS tagger name
RUN python -m nltk.downloader -d /usr/share/nltk_data \
        cmudict punkt punkt_tab averaged_perceptron_tagger \
        averaged_perceptron_tagger_eng wordnet && \
    python -c "from pathlib import Path; import zipfile; \
root=Path('/usr/share/nltk_data');\
[zipfile.ZipFile(z).extractall(z.parent) \
 for kind in ('corpora','tokenizers','taggers') \
 for z in (root/kind).glob('*.zip') if not (z.parent/z.stem).exists()]"

# Runtime defaults: no Hub fetch on air-gapped ClusterUY nodes.
ENV TRANSFORMERS_OFFLINE=1
ENV HF_HUB_OFFLINE=1

WORKDIR /workspace
CMD ["bash"]
