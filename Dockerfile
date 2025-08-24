# Imagen base con CUDA + cuDNN, pero funciona también en CPU si no hay GPU
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Instalar Python y utilidades
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip git build-essential curl ffmpeg libsm6 libxext6 \
 && rm -rf /var/lib/apt/lists/*

# Instalar uv (gestor de dependencias rápido)
RUN pip install uv

# Directorio de trabajo
WORKDIR /app

# Copiar todo el proyecto (lo que no esté en .dockerignore)
COPY . .


# Instalar dependencias (PyTorch con soporte CUDA, usará CPU si no hay GPU)
RUN uv sync --frozen

# Crear carpetas de datos y resultados
RUN mkdir -p data runs checkpoints

# Puerto para TensorBoard
EXPOSE 6006

# Comando por defecto: entrenamiento
CMD ["make", "train"]
