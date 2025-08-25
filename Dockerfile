# Imagen base con CUDA + cuDNN (también corre en CPU si no hay GPU)
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

# Sistema + Python y utilidades
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip git build-essential curl ffmpeg libsm6 libxext6 ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Instalar uv (Astral) de forma oficial y añadirlo al PATH
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
 && echo 'export PATH="/root/.local/bin:$PATH"' >> /root/.bashrc
ENV PATH="/root/.local/bin:${PATH}"

# Directorio de trabajo
WORKDIR /app

# Copiar todo el proyecto (respetando .dockerignore)
COPY . .

# Instalar dependencias (respetará tu uv.lock si existe)
# Si no tienes lock, puedes quitar --frozen
RUN uv sync --frozen

# Crear carpetas de datos y resultados
RUN mkdir -p data runs checkpoints

# Puertos útiles (TensorBoard / UIs)
EXPOSE 6006 7860 8501

# Comando por defecto: entrenamiento
CMD ["make", "train"]
