FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Build/runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    build-essential \
    cmake \
    wget \
    tar \
    && rm -rf /var/lib/apt/lists/*

# Install ONNX Runtime (CPU)
ARG ORT_VERSION=1.18.1
RUN wget -q "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz" \
    && tar -xzf "onnxruntime-linux-x64-${ORT_VERSION}.tgz" \
    && cp -r "onnxruntime-linux-x64-${ORT_VERSION}/include"/* /usr/local/include/ \
    && cp -r "onnxruntime-linux-x64-${ORT_VERSION}/lib"/* /usr/local/lib/ \
    && ldconfig \
    && rm -rf "onnxruntime-linux-x64-${ORT_VERSION}"* 

# Copy source
COPY . /app

# Build
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build -j"$(nproc)"

# Default command: one-shot inference CLI (override as needed)
ENTRYPOINT ["/app/build/engine/chess_engine_cli"]
CMD ["--depth", "4", "--model", "/app/models/chess_eval.onnx"]
