FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build pkg-config \
    libeigen3-dev \
    git curl ca-certificates openssh-client sudo \
    && rm -rf /var/lib/apt/lists/*

ARG ORT_VERSION=1.19.2
RUN cd /tmp \
    && curl -fSL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz" -o ort.tgz \
    && tar -xf ort.tgz \
    && cp -a onnxruntime-linux-x64-${ORT_VERSION}/include/* /usr/local/include/ \
    && cp -a onnxruntime-linux-x64-${ORT_VERSION}/lib/*     /usr/local/lib/ \
    && mkdir -p /usr/local/include/onnxruntime/core/session \
    && for h in /usr/local/include/onnxruntime_*.h /usr/local/include/cpu_provider_factory.h /usr/local/include/provider_options.h; do \
         ln -sf "$h" /usr/local/include/onnxruntime/core/session/; \
       done \
    && ldconfig \
    && rm -rf ort.tgz onnxruntime-linux-x64-${ORT_VERSION}

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

USER $USERNAME
WORKDIR /workspace
