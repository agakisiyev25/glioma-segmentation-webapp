FROM python:3.11-slim

# CPU build by default. For GPU, build with:
#   --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu126
# (docker-compose.gpu.yml does this and requests the GPU).
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
# 0 = demo mode only (small image, no PyTorch).
ARG INSTALL_INFERENCE=1
# 1 = also install HD-BET for uploads that are not already skull-stripped.
ARG INSTALL_HDBET=0

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements-inference.txt constraints.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# torch AND torchvision are installed first, pinned, from the PyTorch index;
# see the note in requirements-inference.txt.
RUN if [ "$INSTALL_INFERENCE" = "1" ]; then \
      pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 --index-url "$TORCH_INDEX" \
      && pip install --no-cache-dir -c constraints.txt -r requirements-inference.txt; \
    fi
RUN if [ "$INSTALL_INFERENCE" = "1" ] && [ "$INSTALL_HDBET" = "1" ]; then \
      pip install --no-cache-dir -c constraints.txt hd-bet; \
    fi

COPY app ./app
COPY scripts ./scripts

ENV GLIOMA_DATA_DIR=/data \
    NNUNET_MODEL_DIR=/models \
    GLIOMA_HOST=0.0.0.0 \
    GLIOMA_PORT=8000 \
    PYTHONUNBUFFERED=1
VOLUME ["/data"]
EXPOSE 8000
CMD ["python", "-m", "app"]
