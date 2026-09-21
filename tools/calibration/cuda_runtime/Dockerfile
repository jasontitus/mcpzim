# CUDA 13 supports Blackwell; immutable official PyTorch base, linux/amd64.
FROM pytorch/pytorch@sha256:bfbb4a2b4fdba0fefdb428ea737e626d61bb3daf74a16e1ff935bdb03aa7c3f0
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled \
    OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
WORKDIR /opt/zimfo-runtime
RUN python -m venv --without-pip --system-site-packages /opt/zimfo-venv
ENV PATH="/opt/zimfo-venv/bin:${PATH}"
COPY requirements.lock ./requirements.lock
RUN python -m pip install --no-cache-dir --no-deps --require-hashes --only-binary=:all: -r requirements.lock \
    && python -m pip check
COPY .context/sources/ /opt/upstream/
COPY upstream_check.py verify_sources.py ./
COPY tests/ ./tests/
RUN python verify_sources.py /opt/upstream \
    && python -m pytest -q tests \
    && python upstream_check.py --stage gsq --device cpu --report /opt/build-evidence/gsq-cpu.json \
    && python upstream_check.py --stage rco --device cpu --report /opt/build-evidence/rco-cpu.json \
    && python -m pip freeze > /opt/build-evidence/pip-freeze.txt
COPY .context/application/ /opt/zimfo-app/
COPY .context/prism/ /opt/prism/
ENV PYTHONPATH="/opt/zimfo-app" ZIMFO_PRISM_SOURCE="/opt/prism"
RUN python -m pytest -q -p no:cacheprovider /opt/zimfo-app/solver/tests \
    && python -m pytest -q -p no:cacheprovider /opt/zimfo-app/tests \
    && python -m solver.run --help \
    && python -m solver.job --help \
    && python -c "from google.cloud import storage; import verify_baseline, encoding_inventory; from packing.export_qwen import load_converter; load_converter('/opt/prism')"
LABEL org.opencontainers.image.title="Zimfo pinned GSQ/RCO runtime preparation" \
      org.opencontainers.image.description="CPU integration checked; whole-model and CUDA validation required"
CMD ["python", "upstream_check.py", "--help"]
