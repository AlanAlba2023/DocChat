# ============================================
# app/core/config.py - Configuración del proyecto
# ============================================

import os
import torch
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """
    Configuración central del proyecto usando pydantic-settings.
    Lee variables de entorno desde un archivo .env automáticamente.
    """

    # ============================================
    # DeepSeek API (LLM principal)
    # ============================================
    DEEPSEEK_API_KEY: Optional[str] = None
    DEEPSEEK_API_BASE: str = "https://api.deepseek.com/v1"
    DEEPSEEK_MODEL: str = "deepseek-chat"  # o "deepseek-reasoner"

    # ============================================
    # Hugging Face (alternativa gratuita para LLM)
    # ============================================
    HUGGINGFACEHUB_API_TOKEN: Optional[str] = None

    # ============================================
    # Configuración del sistema
    # ============================================
    CHROMA_PERSIST_DIR: str = "./chroma_db"   # Directorio persistente de ChromaDB
    EMBEDDINGS_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    TEMP_UPLOAD_DIR: str = "./temp_uploads"   # Directorio para archivos temporales
    MAX_FILE_SIZE_MB: int = 50                # Tamaño máximo de PDF en MB

    # ============================================
    # Detección automática de GPU (información)
    # ============================================
    @property
    def DEVICE(self) -> str:
        """Detecta si hay GPU disponible y retorna 'cuda' o 'cpu'."""
        return "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def GPU_NAME(self) -> str:
        """Retorna el nombre de la GPU si está disponible."""
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        return "No GPU disponible"

    @property
    def GPU_MEMORY_GB(self) -> float:
        """Retorna la memoria total de la GPU en GB."""
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 1024**3
        return 0.0

    # ============================================
    # Verificación de que el LLM está configurado
    # ============================================
    @property
    def LLM_PROVIDER(self) -> str:
        if self.DEEPSEEK_API_KEY:
            return "deepseek"
        elif self.HUGGINGFACEHUB_API_TOKEN:
            return "huggingface"
        else:
            return "none"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

# Instancia global de configuración para importar en toda la app
settings = Settings()

# Mostrar información al iniciar (opcional, útil para depuración)
if settings.DEVICE == "cuda":
    print(f"🚀 GPU detectada: {settings.GPU_NAME} ({settings.GPU_MEMORY_GB:.2f} GB)")
else:
    print("💻 Ejecutando en CPU (sin GPU)")
print(f"🔧 Proveedor LLM: {settings.LLM_PROVIDER}")
print(f"📁 ChromaDB persistente en: {settings.CHROMA_PERSIST_DIR}")