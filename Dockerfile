# Transformers NLP Suite image: runs the REST API by default, the demo UIs on request.
#
# Dependencies are installed into a virtualenv at /opt/venv, which is copied into
# the runtime stage and readable by the unprivileged runtime user. (Installing with
# `pip --user` as root put packages in /root/.local, which the runtime user cannot
# read, so nothing could be imported.)

# Stage 1: build the virtualenv
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

WORKDIR /build
COPY requirements.txt ./

# CPU-only PyTorch first: the default index ships CUDA builds that add several GB
# and are useless in this CPU image. requirements.txt then reuses this torch.
RUN pip install --upgrade pip setuptools wheel \
    && pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install -r requirements.txt \
    && python -m spacy download en_core_web_sm

# Stage 2: runtime
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 nlpuser

COPY --from=builder /opt/venv /opt/venv

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app
COPY --chown=nlpuser:nlpuser . .

# Directories the runtime user writes to. Named volumes mounted on these paths
# (logs, Hugging Face cache) start with this ownership, so downloads and log files work.
RUN mkdir -p /app/logs /home/nlpuser/.cache/huggingface \
    && chown nlpuser:nlpuser /app /app/logs \
    && chown -R nlpuser:nlpuser /home/nlpuser/.cache

USER nlpuser

# 8000: REST API. 7860: combined demo UI app (`python launch_ui.py all`).
# Individual UIs use 7860-7869 (see ui/catalog.py).
EXPOSE 8000 7860

CMD ["python", "launch_ui.py", "api", "--host", "0.0.0.0"]
