# ============================================
# app/services/llm_factory.py - Fábrica de LLMs (solo Hugging Face)
# ============================================

from typing import Optional
from langchain_community.llms import HuggingFaceEndpoint
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from app.core.config import settings


class LLMFactory:
    """
    Fábrica para crear instancias del LLM usando Hugging Face Inference API.
    """

    @staticmethod
    def create_llm(
        temperature: float = 0.7,
        max_tokens: int = 512,
        streaming: bool = False,
    ) -> Optional[HuggingFaceEndpoint]:
        """
        Crea y retorna una instancia del LLM de Hugging Face.

        Args:
            temperature: Controla la creatividad (0.0 = determinista, 1.0 = creativo).
            max_tokens: Número máximo de tokens a generar.
            streaming: Si es True, habilita streaming de la respuesta.

        Returns:
            Una instancia de HuggingFaceEndpoint o None si no hay token configurado.
        """
        # Verificar que el token de Hugging Face está configurado
        if not settings.HUGGINGFACEHUB_API_TOKEN:
            print("⚠️ No se encontró HUGGINGFACEHUB_API_TOKEN en el archivo .env")
            print("   El agente no podrá funcionar sin un LLM.")
            return None

        return HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.3",  # Modelo gratuito y de buena calidad
            huggingfacehub_api_token=settings.HUGGINGFACEHUB_API_TOKEN,
            task="text-generation",
            temperature=temperature,
            max_new_tokens=max_tokens,
            streaming=streaming,
            callbacks=[StreamingStdOutCallbackHandler()] if streaming else None,
        )