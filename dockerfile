# Usa una imagen base con CUDA 11.8 y Python 3.10
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Configura entorno
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Instala Python y herramientas básicas
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    && rm -rf /var/lib/apt/lists/*

# Crea enlace simbólico para python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Establece directorio de trabajo
WORKDIR /app

# Copia requirements.txt y instala dependencias (PyTorch con CUDA aparte)
COPY requirements.txt .

# Instala PyTorch con CUDA 11.8
RUN pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu118

# Instala el resto de dependencias
RUN pip install -r requirements.txt

# Copia el código fuente
COPY . .

# Expone el puerto de FastAPI
EXPOSE 8000

# Comando de inicio
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]