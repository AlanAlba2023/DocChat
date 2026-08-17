# ============================================
# app/main.py - Punto de entrada FastAPI
# ============================================

import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Importaciones de servicios (los crearemos después)
from app.services.rag_engine import RAGEngine
from app.services.agent import AgentService
from app.services.llm_factory import LLMFactory
from app.core.config import settings
from app.models.schemas import (
    QuestionRequest,
    AnswerResponse,
    UploadResponse,
    HealthResponse,
    SourceDocument
)

# ============================================
# Configuración de la aplicación
# ============================================

app = FastAPI(
    title="DocChat API",
    description="API para consulta inteligente de documentos usando RAG y Agentes Autónomos",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS - Permitir peticiones desde cualquier origen (para desarrollo)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# Dependencias globales (inyectables)
# ============================================

# Instanciamos los servicios una sola vez al iniciar la app
# FastAPI las inyectará en los endpoints que las soliciten

def get_rag_engine() -> RAGEngine:
    """Retorna una instancia del motor RAG (singleton)."""
    return RAGEngine()

def get_agent_service() -> AgentService:
    """Retorna una instancia del servicio de agente."""
    llm = LLMFactory.create_llm()  # Usa la fábrica para crear el LLM
    rag_engine = get_rag_engine()
    return AgentService(llm=llm, rag_engine=rag_engine)

# ============================================
# Modelos de respuesta (definidos en schemas.py)
# ============================================

# Ya importados arriba: QuestionRequest, AnswerResponse, UploadResponse, HealthResponse

# ============================================
# ENDPOINTS
# ============================================

@app.get("/", tags=["Root"])
async def root():
    """Endpoint raíz con información básica."""
    return {
        "message": "DocChat API - Sistema RAG con Agentes Autónomos",
        "docs": "/docs",
        "redoc": "/redoc",
        "version": "1.0.0"
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Verifica el estado de la API y de los servicios subyacentes."""
    rag = get_rag_engine()
    try:
        # Verificamos que ChromaDB responde (si hay documentos indexados)
        collection_count = rag.get_collection_count()
        vectorstore_online = True
    except Exception:
        vectorstore_online = False
        collection_count = 0

    # Verificar LLM (hacemos una llamada ligera)
    llm_online = False
    try:
        llm = LLMFactory.create_llm()
        # Pequeña prueba: invocar con una pregunta simple
        test_response = llm.invoke("Hola, responde con 'OK' si estás funcionando.")
        if "OK" in test_response:
            llm_online = True
    except Exception:
        llm_online = False

    return HealthResponse(
        status="healthy" if vectorstore_online and llm_online else "degraded",
        vectorstore="online" if vectorstore_online else "offline",
        llm="online" if llm_online else "offline",
        documents_indexed=collection_count,
        gpu_available=settings.DEVICE == "cuda"
    )

@app.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED, tags=["Documents"])
async def upload_document(
    file: UploadFile = File(..., description="Archivo PDF a indexar"),
    rag_engine: RAGEngine = Depends(get_rag_engine)
):
    """
    Sube y indexa un documento PDF en la base de datos vectorial.

    - **file**: Archivo PDF (máximo 50 MB por defecto)
    - Retorna el número de fragmentos indexados y un ID único del documento.
    """
    # Validar extensión
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Solo se permiten archivos PDF"
        )

    # Validar tamaño (usando settings)
    file.file.seek(0, 2)  # Ir al final del archivo
    file_size = file.file.tell()
    file.file.seek(0)  # Volver al inicio
    if file_size > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"El archivo excede el tamaño máximo de {settings.MAX_FILE_SIZE_MB} MB"
        )

    # Guardar temporalmente el archivo
    temp_dir = Path(settings.TEMP_UPLOAD_DIR)
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / f"{uuid.uuid4()}_{file.filename}"

    try:
        # Escribir el contenido
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Indexar con el motor RAG
        doc_id = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        chunks = rag_engine.add_document(
            file_path=str(temp_path),
            document_id=doc_id,
            metadata={"original_filename": file.filename}
        )

        return UploadResponse(
            status="success",
            message="Documento indexado correctamente",
            chunks=len(chunks),
            document_id=doc_id
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al procesar el documento: {str(e)}"
        )
    finally:
        # Limpiar archivo temporal
        if temp_path.exists():
            os.remove(temp_path)

@app.post("/ask", response_model=AnswerResponse, tags=["Questions"])
async def ask_question(
    request: QuestionRequest,
    agent_service: AgentService = Depends(get_agent_service)
):
    """
    Realiza una pregunta al sistema. El agente decidirá si usar RAG, calculadora u otras herramientas.

    - **question**: Texto de la pregunta
    - **session_id** (opcional): Para mantener contexto en conversaciones futuras
    """
    try:
        # Ejecutar el agente
        result = agent_service.ask(
            question=request.question,
            session_id=request.session_id
        )

        # Construir respuesta
        return AnswerResponse(
            answer=result["answer"],
            sources=[
                SourceDocument(
                    source=src.get("source", ""),
                    page=src.get("page"),
                    relevance=src.get("relevance")
                )
                for src in result.get("sources", [])
            ],
            tool_used=result.get("tool_used", "unknown")
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al procesar la pregunta: {str(e)}"
        )

# ============================================
# Manejo de excepciones globales (opcional)
# ============================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Personaliza la respuesta de errores HTTP."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "status": "error",
            "timestamp": datetime.now().isoformat()
        }
    )

# ============================================
# Para ejecutar directamente (solo desarrollo)
# ============================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )