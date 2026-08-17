# ============================================
# app/models/schemas.py - Modelos Pydantic para API
# ============================================

from pydantic import BaseModel, Field
from typing import Optional, List, Any

# ============================================
# Modelos para /upload
# ============================================

class UploadResponse(BaseModel):
    """Respuesta al subir un documento."""
    status: str = Field(..., description="Estado de la operación (success/error)")
    message: str = Field(..., description="Mensaje descriptivo")
    chunks: int = Field(..., description="Número de fragmentos indexados")
    document_id: str = Field(..., description="ID único del documento indexado")

# ============================================
# Modelos para /ask
# ============================================

class QuestionRequest(BaseModel):
    """Petición para hacer una pregunta."""
    question: str = Field(..., description="Pregunta del usuario", min_length=1)
    session_id: Optional[str] = Field(None, description="ID de sesión para mantener contexto")

class SourceDocument(BaseModel):
    """Información de la fuente de una respuesta."""
    source: str = Field(..., description="Nombre del documento fuente")
    page: Optional[int] = Field(None, description="Número de página (si está disponible)")
    relevance: Optional[float] = Field(None, description="Puntuación de relevancia (0-1)")

class AnswerResponse(BaseModel):
    """Respuesta a una pregunta."""
    answer: str = Field(..., description="Respuesta generada por el sistema")
    sources: List[SourceDocument] = Field(default_factory=list, description="Fuentes utilizadas")
    tool_used: str = Field(..., description="Herramienta que usó el agente (rag/calculator/...)")

# ============================================
# Modelos para /health
# ============================================

class HealthResponse(BaseModel):
    """Estado de salud del sistema."""
    status: str = Field(..., description="Estado general (healthy/degraded)")
    vectorstore: str = Field(..., description="Estado de ChromaDB (online/offline)")
    llm: str = Field(..., description="Estado del LLM (online/offline)")
    documents_indexed: int = Field(..., description="Número de documentos indexados")
    gpu_available: bool = Field(..., description="Indica si hay GPU disponible")

# ============================================
# Modelos internos (no expuestos directamente)
# ============================================

class DocumentMetadata(BaseModel):
    """Metadatos de un documento interno."""
    document_id: str
    original_filename: str
    chunks: int
    uploaded_at: str
    file_size: int