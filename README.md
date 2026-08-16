# 📚 DocChat - RAG System with Generative AI

> **Sistema de consulta inteligente sobre documentación técnica usando RAG (Retrieval-Augmented Generation) y Agentes Autónomos**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-green.svg)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-0.2.0-orange.svg)](https://www.langchain.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)

---

## 🎯 Descripción del Proyecto

**DocChat** es una API inteligente que permite a los usuarios subir documentos PDF y realizar preguntas sobre su contenido. El sistema utiliza:

- **RAG (Retrieval-Augmented Generation)** para recuperar información precisa de los documentos
- **Agentes Autónomos** que deciden si responder con RAG, hacer cálculos o usar otras herramientas
- **Embeddings open-source** de Hugging Face para vectorización gratuita y eficiente

### 🎬 Demo Rápida
# Subir un documento
curl -X POST http://localhost:8000/upload -F "file=@documento.pdf"

# Hacer una pregunta
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d '{"question": "¿Qué dice el documento sobre el plazo de entrega?"}'

```bash
🏗️ Arquitectura del Sistema

┌─────────────┐      ┌──────────────┐      ┌─────────────────┐
│   Usuario   │─────▶│  FastAPI     │─────▶│   Agente IA    │
│  (Cliente)  │      │  (Endpoint)  │      │   (LangChain)  │
└─────────────┘      └──────────────┘      └────────┬────────┘
                                                     │
                              ┌──────────────────────┼──────────────────────┐
                              │                      │                      │
                              ▼                      ▼                      ▼
                    ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
                    │  Herramienta 1  │  │  Herramienta 2  │  │  Herramienta 3  │
                    │  RAG + ChromaDB │  │  Calculadora    │  │  (Futuro)       │
                    └─────────────────┘  └─────────────────┘  └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  Embeddings HF  │
                    │ all-MiniLM-L6-v2│
                    └─────────────────┘

Flujo de datos:

El usuario sube un PDF → Se divide en chunks → Se vectoriza → Se guarda en ChromaDB

El usuario hace una pregunta → El agente decide qué herramienta usar

Si usa RAG → Busca chunks similares → Genera respuesta con el LLM

Si usa calculadora → Ejecuta la operación y devuelve el resultado
```


## 🛠️ Stack Tecnológico

El proyecto se construye sobre las siguientes tecnologías clave:

- **API**: FastAPI (v0.111.0) – Elegido por su alto rendimiento, documentación automática OpenAPI y soporte nativo para asincronía.

- **Orquestación**: LangChain (v0.2.0) – Framework estándar para construir agentes y cadenas RAG, con amplia comunidad y ejemplos.

- **Embeddings**: Hugging Face `all-MiniLM-L6-v2` – Modelo gratuito, eficiente y sin dependencia de APIs de pago, con buena relación calidad/rendimiento.

- **Vector DB**: ChromaDB (v0.5.0) – Base de datos vectorial ligera, persistente en disco, ideal para desarrollo y pruebas sin infraestructura en la nube.

- **LLM**: Azure OpenAI (GPT-4o mini) – Ofrece respuestas precisas y de alta calidad. *Opcionalmente sustituible por Hugging Face Inference API o modelos locales con llama-cpp-python.*

- **Contenedor**: Docker (20.10+) – Garantiza portabilidad y reproducibilidad del entorno en cualquier sistema.

- **Lenguaje**: Python (3.10+) – Por su maduro ecosistema de librerías para IA y procesamiento de datos.

## 📋 Requisitos Previos
Python 3.10 o superior

- Docker (opcional, para despliegue)

- Cuenta en Azure OpenAI (o alternativa gratuita)

- Token de Hugging Face (gratuito)

## 🔑 Variables de Entorno
- Crea un archivo .env en la raíz del proyecto:
   .env

##📁 Estructura del Proyecto

docchat/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI y endpoints
│   ├── core/
│   │   ├── config.py        # Configuraciones y env vars
│   │   └── dependencies.py  # Dependencias inyectables
│   ├── services/
│   │   ├── rag_engine.py    # Embeddings + ChromaDB
│   │   ├── agent.py         # Agente LangChain
│   │   └── llm_factory.py   # Fábrica de LLMs (Azure/HF)
│   └── models/
│       ├── schemas.py       # Pydantic models
│       └── document.py      # Modelo de documento
├── chroma_db/               # Base de datos vectorial (persistente)
├── tests/
│   ├── test_api.py
│   ├── test_rag.py
│   └── evaluation.json      # Dataset de prueba
├── scripts/
│   ├── ingest.py            # Script para indexar documentos
│   └── evaluate.py          # Evaluador de respuestas
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md


## 🧠 Decisiones Técnicas Clave
Decisión	Alternativa	Razón
- Embeddings de Hugging Face	Azure OpenAI Embeddings	💰 Costo cero. all-MiniLM-L6-v2 ofrece buena relación calidad/rendimiento
- ChromaDB persistente	Pinecone / Milvus	🏠 No requiere infraestructura en la nube. Fácil desarrollo y pruebas
- Agente con LangChain	LlamaIndex	🎯 Mayor flexibilidad para herramientas múltiples y razonamiento
- FastAPI	Flask / Django	⚡ Asíncrono por defecto, documentación automática, tipado fuerte
- Docker📦 Reproducibilidad garantizada en cualquier entorno

##🔮 Roadmap Futuro
□ Soporte para múltiples formatos: Word, Excel, Markdown
□ Memoria conversacional: El agente recordará el historial de la conversación
□ Cache de respuestas: Respuestas frecuentes en Redis para reducir costos
□ Evaluación automática: Integración con MLflow para tracking de experimentos
□ Web UI: Interfaz gráfica con Streamlit o Gradio
□ CI/CD: GitHub Actions para pruebas automáticas en cada push


🤝 Contribución
Fork el proyecto

Crea tu rama: git checkout -b feature/nueva-funcionalidad

Commit: git commit -m 'Añade nueva funcionalidad'

Push: git push origin feature/nueva-funcionalidad

Abre un Pull Request

📄 Licencia
Este proyecto está bajo la licencia MIT. Ver archivo LICENSE para más detalles.

✨ Agradecimientos
LangChain por su increíble framework

Hugging Face por modelos open-source de calidad

FastAPI por hacer las APIs tan elegantes

📬 Contacto
Autor: [Alan Alba]

GitHub: @AlanAlba2023

LinkedIn: (https://www.linkedin.com/in/alan-alba1991/)

Email: albaa9053@gmail.com

⭐ Si este proyecto te fue útil, no olvides darle una estrella en GitHub!
