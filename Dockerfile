# AI inference server, serving image. CPU only: we measured 638-676 ms warm per short utterance
# on a laptop CPU, and personal-adapter training is asynchronous (17.6 min on CPU vs 18 s on an
# H100, same selected epoch, byte-identical adapter), so no GPU is required to serve or to train.
#
# This image contains CODE ONLY. Three things are deliberately NOT baked in:
#   - model weights       whisper-small downloads on first start, or mount an HF cache (see below)
#   - the prompt pool     derived from AI-Hub 013, whose redistribution conditions are unchecked
#   - adapters            LoRA weights trained on AI-Hub audio, i.e. derived data
# A published image is a distribution channel. None of the above may travel in one.
# Built and run 2026-10-03: 1.61 GB, builds in ~2.5 min, boots with ASR_ENGINE=mock, and serves
# the REAL engine when an HF cache is mounted at /data/hf (base_revision confirmed
# 973afd24..., source cache_snapshot_path; first transcribe 3.9 s inside the container).
FROM python:3.11-slim-bookworm

# Versions measured working together 2026-10-03 (macOS arm64 + this image): the serving path and
# the 2026-09-26 adapters load under these. Pinned because an image must be reproducible; the dev
# requirements.txt stays unpinned on purpose.
ARG TORCH=2.14.1
ARG TRANSFORMERS=5.18.0
ARG PEFT=0.21.2
ARG NUMPY=2.4.6

# The default PyPI wheel for linux is the CUDA build (multi-GB). This image serves on CPU.
# The CPU index publishes the wheel as 2.14.1+cpu; `==2.14.1` matches it because PEP 440 ignores
# a candidate's local version label when the specifier has none. Verified against the index
# listing: torch-2.14.1+cpu-cp311-cp311-manylinux_2_28_x86_64.whl exists.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch==${TORCH}" \
 && pip install --no-cache-dir "transformers==${TRANSFORMERS}" "peft==${PEFT}" \
      "numpy==${NUMPY}" safetensors \
 && useradd --system --create-home --uid 10001 appuser

WORKDIR /app
COPY --chown=appuser:appuser demo/ /app/

# Writable state. Mount volumes over these: adapters are derived data and must outlive the image.
ENV ENROLL_DIR=/data/enroll \
    JOB_DIR=/data/jobs \
    ASR_ADAPTERS=/data/adapters \
    PROMPT_POOL=/data/script_pool.json \
    HF_HOME=/data/hf \
    HOST=0.0.0.0 \
    PORT=8000
RUN mkdir -p /data/enroll /data/jobs /data/adapters /data/hf && chown -R appuser:appuser /data

USER appuser
EXPOSE 8000
# No HEALTHCHECK: the first model load takes longer than a sensible interval and a restart loop
# during load would be worse than no check. Probe GET /v1/health from the orchestrator instead.
ENTRYPOINT ["python3", "server.py"]
