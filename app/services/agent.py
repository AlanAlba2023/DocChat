# ============================================
# app/services/agent.py - Agente Autónomo
# ============================================

import re
from typing import List, Dict, Any, Optional

from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

from app.core.config import settings
from app.services.rag_engine import RAGEngine


class AgentService:
    """
    Servicio que orquesta el agente con herramientas RAG y calculadora.
    """

    def __init__(self, llm, rag_engine: RAGEngine):
        """
        Inicializa el agente con el LLM y el motor RAG.

        Args:
            llm: Instancia del LLM (creado por LLMFactory).
            rag_engine: Instancia del motor RAG.
        """
        self.llm = llm
        self.rag_engine = rag_engine
        self.tools = self._create_tools()
        self.agent_executor = self._create_agent()

    def _create_tools(self) -> List[Tool]:
        """
        Crea las herramientas que el agente puede usar:
        - RAG: Búsqueda en documentos.
        - Calculadora: Operaciones matemáticas.
        """
        tools = [
            Tool(
                name="RAG Search",
                func=self._rag_search,
                description="Útil para buscar información en los documentos indexados. " 
                            "Entrada: una pregunta en lenguaje natural sobre el contenido de los documentos.",
            ),
            Tool(
                name="Calculator",
                func=self._calculator,
                description="Útil para realizar operaciones matemáticas. " 
                            "Entrada: una expresión matemática simple (ej. '2 + 2', '10 * 5'). "
                            "No uses esta herramienta para preguntas sobre documentos.",
            ),
        ]
        return tools

    def _rag_search(self, query: str) -> str:
        """
        Herramienta RAG: Busca fragmentos relevantes en ChromaDB y los devuelve como texto.

        Args:
            query: La pregunta del usuario.

        Returns:
            Texto con los fragmentos más relevantes.
        """
        try:
            # Recuperar fragmentos relevantes
            results = self.rag_engine.retrieve(query, top_k=4)

            if not results:
                return "No se encontró información relevante en los documentos."

            # Formatear los fragmentos en un texto legible
            formatted_results = []
            for i, result in enumerate(results, 1):
                text = result["text"]
                source = result["metadata"].get("source", "Documento desconocido")
                page = result["metadata"].get("page", "N/A")
                score = result["score"]

                formatted_results.append(
                    f"Fuente {i} (Documento: {source}, Página: {page}, Relevancia: {score:.2f}):\n{text}\n"
                )

            return "\n".join(formatted_results)

        except Exception as e:
            return f"Error al buscar en los documentos: {str(e)}"

    def _calculator(self, expression: str) -> str:
        """
        Herramienta Calculadora: Evalúa una expresión matemática simple.

        Args:
            expression: Expresión matemática (ej. "2 + 2 * 3").

        Returns:
            Resultado de la operación.
        """
        # Limpiar la expresión: eliminar espacios y caracteres no permitidos
        expression = expression.replace(" ", "")
        # Solo permitir números, operadores básicos y paréntesis
        if not re.match(r'^[\d+\-*/().]+$', expression):
            return "Error: La expresión contiene caracteres no permitidos."

        try:
            # Evaluar la expresión (con restricciones de seguridad)
            result = eval(expression)
            return f"El resultado de {expression} es: {result}"
        except Exception as e:
            return f"Error al calcular: {str(e)}"

    def _create_agent(self) -> AgentExecutor:
        """
        Crea el agente con LangChain usando el enfoque ReAct.

        Returns:
            Un AgentExecutor listo para ejecutar consultas.
        """
        # 1. Definir el prompt del agente (ReAct)
        template = """
        Eres un asistente útil y preciso que responde preguntas usando herramientas disponibles.

        Tienes acceso a las siguientes herramientas:

        {tools}

        Usa el siguiente formato para responder:

        Question: la pregunta del usuario
        Thought: piensa qué herramienta necesitas usar
        Action: el nombre de la herramienta a usar (RAG Search o Calculator)
        Action Input: la entrada para la herramienta
        Observation: el resultado de la herramienta
        ... (este proceso puede repetirse si es necesario)
        Thought: ahora sé la respuesta final
        Final Answer: la respuesta final al usuario

        Reglas importantes:
        - Si la pregunta es sobre el contenido de los documentos, usa "RAG Search".
        - Si la pregunta es matemática, usa "Calculator".
        - Si no estás seguro, usa "RAG Search" para buscar en los documentos.
        - Si no encuentras información en los documentos, di que no lo sabes.

        ¡Comienza!

        Question: {input}
        Thought: {agent_scratchpad}
        """

        prompt = PromptTemplate.from_template(template)

        # 2. Crear el agente ReAct
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt,
        )

        # 3. Crear el ejecutor del agente (con memoria si se desea)
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
        )

        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=memory,
            verbose=True,  # Muestra el razonamiento en consola (útil para depuración)
            handle_parsing_errors=True,  # Maneja errores de parsing del agente
            max_iterations=3,  # Límite de iteraciones para evitar bucles
        )

        return agent_executor

    def ask(self, question: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Procesa una pregunta del usuario usando el agente.

        Args:
            question: La pregunta del usuario.
            session_id: ID de sesión para mantener contexto (futuro).

        Returns:
            Diccionario con:
                - answer: Respuesta generada.
                - sources: Lista de fuentes utilizadas.
                - tool_used: Herramienta que usó el agente.
        """
        try:
            # Ejecutar el agente
            result = self.agent_executor.invoke({"input": question})

            # Extraer la respuesta
            answer = result.get("output", "No se pudo generar una respuesta.")

            # Determinar qué herramienta usó (si es posible)
            # El agente no devuelve explícitamente la herramienta usada,
            # pero podemos intentar inferirlo de la salida o dejarlo fijo.
            # Para simplificar, dejamos "rag" como predeterminado si hay documentos.
            tool_used = "rag"
            if "calculator" in question.lower():
                tool_used = "calculator"

            # Extraer fuentes (si el agente usó RAG, podemos simularlas)
            # En una implementación más avanzada, el agente devolvería metadatos.
            sources = []
            try:
                # Si hay resultados de RAG, usamos los metadatos
                rag_results = self.rag_engine.retrieve(question, top_k=3)
                for res in rag_results:
                    sources.append({
                        "source": res["metadata"].get("source", "Documento desconocido"),
                        "page": res["metadata"].get("page"),
                        "relevance": res["score"],
                    })
            except Exception:
                pass

            # Si no hay fuentes, devolvemos lista vacía
            if not sources:
                sources = []

            return {
                "answer": answer,
                "sources": sources,
                "tool_used": tool_used,
            }

        except Exception as e:
            return {
                "answer": f"Error al procesar la pregunta: {str(e)}",
                "sources": [],
                "tool_used": "error",
            }