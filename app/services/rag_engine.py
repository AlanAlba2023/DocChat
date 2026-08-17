# ============================================
# app/services/rag_engine.py - Motor RAG
# ============================================

import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from app.core.config import settings


class RAGEngine:
    """
    Motor RAG que maneja embeddings, indexación y búsqueda en ChromaDB.
    """

    def __init__(self):
        """Inicializa el motor con el modelo de embeddings y ChromaDB."""
        self.embeddings = self._load_embeddings()
        self.vectorstore = self._load_vectorstore()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""],
            length_function=len,
        )

    def _load_embeddings(self) -> HuggingFaceEmbeddings:
        """
        Carga el modelo de embeddings de Hugging Face.
        Usa el modelo especificado en settings.EMBEDDINGS_MODEL.
        """
        return HuggingFaceEmbeddings(
            model_name=settings.EMBEDDINGS_MODEL,
            model_kwargs={"device": settings.DEVICE},  # 'cuda' o 'cpu'
            encode_kwargs={"normalize_embeddings": True},
        )

    def _load_vectorstore(self) -> Chroma:
        """
        Carga la base de datos vectorial ChromaDB desde el directorio persistente.
        Si no existe, la crea vacía.
        """
        persist_dir = settings.CHROMA_PERSIST_DIR
        # Si el directorio no existe, Chroma lo crea automáticamente
        return Chroma(
            persist_directory=persist_dir,
            embedding_function=self.embeddings,
            collection_name="documents",
        )

    def add_document(self, file_path: str, document_id: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Indexa un documento PDF en ChromaDB.

        Args:
            file_path: Ruta del archivo PDF.
            document_id: ID único para el documento.
            metadata: Metadatos adicionales (ej. nombre original).

        Returns:
            Lista de fragmentos indexados con sus metadatos.
        """
        # 1. Cargar el PDF
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        # 2. Dividir en fragmentos (chunks)
        chunks = self.text_splitter.split_documents(documents)

        # 3. Añadir metadatos a cada fragmento
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "document_id": document_id,
                "chunk_index": i,
                "source": metadata.get("original_filename", file_path),
                "page": chunk.metadata.get("page", 0),
            })

        # 4. Generar embeddings y guardar en ChromaDB
        self.vectorstore.add_documents(chunks)
        self.vectorstore.persist()

        # 5. Retornar información de los fragmentos (para la respuesta)
        return [
            {
                "chunk_index": i,
                "page": chunk.metadata.get("page", 0),
                "text": chunk.page_content[:200] + "..." if len(chunk.page_content) > 200 else chunk.page_content,
            }
            for i, chunk in enumerate(chunks)
        ]

    def retrieve(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """
        Busca fragmentos relevantes para una consulta.

        Args:
            query: Texto de la consulta.
            top_k: Número de fragmentos a recuperar.

        Returns:
            Lista de fragmentos con metadatos.
        """
        results = self.vectorstore.similarity_search_with_score(query, k=top_k)

        return [
            {
                "text": doc.page_content,
                "metadata": doc.metadata,
                "score": score,
            }
            for doc, score in results
        ]

    def get_collection_count(self) -> int:
        """
        Retorna el número total de fragmentos indexados en la colección.
        """
        try:
            # ChromaDB no tiene un método directo para contar, usamos la colección
            collection = self.vectorstore._collection
            return collection.count()
        except Exception:
            # Si falla, intentamos contar con una búsqueda vacía
            try:
                # Alternativa: obtener todos los IDs y contar
                all_ids = self.vectorstore.get()["ids"]
                return len(all_ids)
            except Exception:
                return 0

    def clear(self) -> None:
        """
        Elimina todos los documentos de la base de datos (para pruebas).
        """
        try:
            # Obtener todos los IDs y eliminarlos
            all_ids = self.vectorstore.get()["ids"]
            if all_ids:
                self.vectorstore.delete(ids=all_ids)
                self.vectorstore.persist()
        except Exception:
            pass