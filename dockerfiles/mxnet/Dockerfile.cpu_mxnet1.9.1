# Use a minimal Python base image
FROM python:3.8-slim

LABEL maintainer="Changjae Lee <changjae@buffalo.edu>" \
      build_date="${BUILD_DATE}" \
      vcs_ref="${VCS_REF}" \
      vcs_url="${VCS_URL}" \
      version="${VERSION}" \
      architecture="${ARCH}"

ARG TZ=America/New_York

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies and set timezone
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git build-essential ca-certificates wget curl libopenblas-dev && \
    ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata && \
    rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir mxnet==1.9.1 numpy==1.23.1 opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-grpc grpcio opentelemetry-exporter-otlp-proto-http httpio aenum requests tqdm torchvision==0.9.0 scipy chardet psycopg "psycopg[binary]" Pika opencv-python-headless 

# Clone py-mlmodelscope repository
RUN git clone https://github.com/xlab-ub/py-mlmodelscope.git /py-mlmodelscope

WORKDIR /py-mlmodelscope
ENTRYPOINT ["python", "run_mlmodelscope.py", "--standalone", "false", "--agent", "mxnet", "--env_file", "true"]