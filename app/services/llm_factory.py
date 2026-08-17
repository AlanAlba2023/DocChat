# ============================================
# app/services/llm_factory.py - Fábrica de LLMs (SOLO DeepSeek)
# ============================================

from typing import Optional
from langchain_openai import ChatOpenAI

from app.core.config import settings


class LLMFactory:
    """
    Fábrica para crear instancias del LLM usando DeepSeek API.
    """

    @staticmethod
    def create_llm(
        temperature: float = 0.7,
        max_tokens: int = 512,
        streaming: bool = False,
    ) -> Optional[ChatOpenAI]:
        """
        Crea y retorna una instancia del LLM de DeepSeek.

        Args:
            temperature: Controla la creatividad (0.0 = determinista, 1.0 = creativo).
            max_tokens: Número máximo de tokens a generar.
            streaming: Si es True, habilita streaming de la respuesta.

        Returns:
            Una instancia de ChatOpenAI configurada para DeepSeek.
        """
        if not settings.DEEPSEEK_API_KEY:
            print("⚠️ No se encontró DEEPSEEK_API_KEY en el archivo .env")
            print("   El agente no podrá funcionar sin un LLM.")
            return None

        return ChatOpenAI(
            model=settings.DEEPSEEK_MODEL,
            api_key=settings.DEEPSEEK_API_KEY,
            base_url=settings.DEEPSEEK_API_BASE,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=streaming,
        )