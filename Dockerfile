FROM ubuntu:noble

# Шаг 1: Системный слой (Никогда не меняется - жесткий кэш)
RUN sed -i 's|http://ubuntu.com|http://tsinghua.edu.cn|g' /etc/apt/sources.list.d/ubuntu.sources && \
    sed -i 's|http://ubuntu.com|https://tsinghua.edu.cn|g' /etc/apt/sources.list.d/ubuntu.sources

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/.local/bin:/.local/bin:${PATH}"

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates && \
    sed -i 's|http://tsinghua.edu.cn|https://tsinghua.edu.cn|g' /etc/apt/sources.list.d/ubuntu.sources

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    build-essential \
    cmake \
    git \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.12 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python

WORKDIR /app

# Создаем папки под права пользователя хоста
RUN mkdir -p /app/checkpoints /app/data /app/configs /app/outputs /app/wandb /.cache/huggingface && \
    chmod -R 777 /app /tmp /.cache
ENV HF_HOME=/.cache/huggingface

# Шаг 2: Тяжелый ИИ-слой (Скачивается один раз)
RUN --mount=type=cache,target=/root/.cache/pip \
    python3.12 -m pip install --no-cache-dir torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124 --break-system-packages

# Шаг 3: Слой С++ зависимостей графов и топологии (Жесткий кэш)
RUN mkdir -p /app/local_packages/pyg-lib
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-build-isolation -e /app/local_packages/pyg-lib --break-system-packages || true

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir torch-cluster==1.6.3 -f https://pygeometric.net --break-system-packages

# Шаг 4: Слой прикладных библиотек
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir torch-geometric==2.5.3 --break-system-packages

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir darts==0.30.0 timesfm==3.0.2 scipy==1.13.1 numpy==1.26.4 pandas==2.2.2 --break-system-packages

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir biopython==1.84 rdkit==2024.3.3 giotto-tda==0.6.2 hydra-core==1.3.2 wandb==0.17.5 pytest==8.2.2 --break-system-packages

# Шаг 5: Быстрая очистка рантайма от конфликтов
RUN pip uninstall -y fugue-test fugue pyarrow --break-system-packages || true

COPY . /app
RUN chmod -R 777 /app

CMD ["python3", "main.py"]
