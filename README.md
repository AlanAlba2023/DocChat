# 📚 DocChat - RAG System with Generative AI

> **Sistema de consulta inteligente sobre documentación técnica usando RAG (Retrieval-Augmented Generation) y Agentes Autónomos**

[![Python 3.10+](https://www.python.org/downloads/)
[![FastAPI](https://fastapi.tiangolo.com/)
[![LangChain](https://www.langchain.com/)
[![Docker](https://www.docker.com/)

---

## 🎯 Descripción del Proyecto

**DocChat** es una API inteligente que permite a los usuarios subir documentos PDF y realizar preguntas sobre su contenido. El sistema utiliza:

- **RAG (Retrieval-Augmented Generation)** para recuperar información precisa de los documentos
- **Agentes Autónomos** que deciden si responder con RAG, hacer cálculos o usar otras herramientas
- **Embeddings open-source** de Hugging Face para vectorización gratuita y eficiente

### 🎬 Demo Rápida

```bash
# Subir un documento
curl -X POST http://localhost:8000/upload -F "file=@documento.pdf"

# Hacer una pregunta
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d '{"question": "¿Qué dice el documento sobre el plazo de entrega?"}'
